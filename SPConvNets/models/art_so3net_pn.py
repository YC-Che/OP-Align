import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
import time
from collections import OrderedDict
import json
import vgtk
import pytorch3d
import SPConvNets.utils as M
import vgtk.so3conv.functional as L
from SPConvNets.utils.lietorch import ExpSO3
import vgtk.so3conv as sptk
from SPConvNets.models.art_encoder import ResnetPointnet
from pytorch3d.ops.knn import knn_gather, knn_points
from SPConvNets.utils.chamfer import chamfer_distance

class ArtSO3ConvModel(nn.Module):
    def __init__(self, params):
        super(ArtSO3ConvModel, self).__init__()
        self.backbone = nn.ModuleList()
        for block_param in params['backbone']:
            self.backbone.append(M.BasicSO3ConvBlock(block_param))
        self.outblock = ArtOutBlock(params['outblock'])
        self.pnblock = ResnetPointnet(nmask=params['pn']['nmask'], dim=3, njoint = params['pn']['njoint'], rotation_range = params['pn']['rotation_range'], joint_type = params['joint_type'])
        self.rcnblock = DecoderFCWithPVP(point=params['recon']['point'])
        self.na_in = params['na']
        self.n_mask = params['pn']['nmask']
        self.color_w = params['color_w']

    def forward(self, x, c):
        B = x.shape[0]
        x = M.preprocess_input(x, self.na_in, False)
        shape_input = x.xyz
        shape_color = c.permute(0,2,1)
        x._feats = shape_color.unsqueeze(-1).repeat(1,1,1,self.na_in)

        x_list = []
        for block_i, block in enumerate(self.backbone):
            x = block(x)
            x_list.append(x)
        global_feat, r_attn, r_base, t_base = self.outblock(x_list) #B,256,   B,R   B,R,3,3
        #Reconstruction
        shape_cano, shape_var, shape_cano_color = self.rcnblock(global_feat) #B,3,N
        #Rotation Selection
        base_idx, base_distance = self.anchor_based_distance(
            r_base, t_base, shape_input, shape_cano, shape_color, shape_cano_color, color_weight=self.color_w
            )
        r_select, t_select = r_base[torch.arange(B), base_idx], t_base[torch.arange(B), base_idx]
        shape_align = torch.bmm(r_select.permute(0,2,1), shape_input) - t_select.unsqueeze(-1)
        #Segmentation, Alignment
        ret = list(self.pnblock(shape_align, shape_cano, augmentation=True))
        p1_align, c1, j_coor_1, j_drct_1, j_angl_1, p2_align, c2, j_coor_2, j_drct_2, j_angl_2 = ret
        #Segmentation Exchange
        c1, c2 = self.segmentation_order(
            p1_align, shape_cano, c1, c2, shape_color, shape_cano_color, color_w=self.color_w, common_part_w=5
            )

        ret = {
            'R_attn': r_attn,
            'R_base': r_base,
            'T_base': t_base,
            'R_select': r_select,
            'T_select': t_select,
            'R_distance': base_distance,
            'R_idx': base_idx,

            'S_input': shape_input,
            'S_color': shape_color,
            'S_align': shape_align,
            'S_align_part': p1_align,
            'S_seg': c1,
            'S_joint': j_coor_1,
            'S_drct': j_drct_1,
            'S_angl': j_angl_1,
            #'S_joint_var': j_coor_translation_1,
            #'S_drct_var': j_drct_translation_1,

            "I_cano": shape_cano,
            "I_align_part": p2_align,
            'I_color': shape_cano_color,
            'I_seg': c2,
            'I_joint': j_coor_2,
            'I_drct': j_drct_2,
            'I_angl': j_angl_2,
            'I_shape_var': shape_var,
            #'I_joint_var': j_coor_translation_2,
            #'I_drct_var': j_drct_translation_2,
            
            #'E_vote_center': vote_center,
            #'E_vote_joint': vote_joint,
            #'E_slot': slot,
            #'E_base_slot': self.slot_base,
            #'E_point': f_point,
        }
        return ret
    
    def anchor_based_distance(self, R_base, T_base, input, cano, input_c, cano_c, color_weight = 0):
        with torch.no_grad():
            B,R = R_base.shape[0], R_base.shape[1]

            input_tmp = input.unsqueeze(1).repeat(1,R,1,1)
            input_tmp = torch.einsum("brij, brjk -> brik", R_base.permute(0,1,3,2), input_tmp)#B,R,3,N
            input_tmp = input_tmp - T_base.unsqueeze(-1)
            cano_tmp = cano.unsqueeze(1).repeat(1,R,1,1)#B,R,3,M

            if color_weight != 0:
                input_tmp = torch.cat(
                    [input_tmp, color_weight * input_c.unsqueeze(1).expand(-1,R,-1,-1)],
                    dim=2)#B,R,6,N
                cano_tmp = torch.cat(
                    [cano_tmp, color_weight * cano_c.unsqueeze(1).expand(-1,R,-1,-1)],
                    dim=2)#B,R,6,M
                #use_color = True
            
            knn = knn_points(
                input_tmp.permute(0,1,3,2).reshape(B*R,-1,3),
                cano_tmp.permute(0,1,3,2).reshape(B*R,-1,3),
                norm=1, K=1)
            dist = knn.dists[...,0].reshape(B,R,-1)#B,R,N
            R_distance = dist.mean(-1)
            

            #R_distance = torch.ones_like(R_base[:,:,0,0])#B,R
            #for r in range(R):
            #    cur_R = R_base[:,r]
            #    input_tmp = torch.bmm(cur_R.permute(0,2,1), input)
            #    input_tmp = input_tmp - T_base[:,r].unsqueeze(-1)
            #
            #    if use_color:
            #        input_tmp = torch.cat([input_tmp, input_c], dim=1)
            #    knn = knn_points(input_tmp.permute(0,2,1), cano_tmp.permute(0,2,1), norm=1, K=1)
            #    dist, idx = knn.dists[...,0], knn.idx[...,0]#B,N, B,N
            #    R_distance[:,r] = dist.mean(-1)
            base_idx = torch.argmin(R_distance, dim=-1)

        R_distance = R_distance.detach()
        base_idx = base_idx.detach()
        return base_idx, R_distance
    
    def segmentation_order(self, align, recon, align_p, recon_p, align_c, recon_c, color_w = 0, common_part_w = 1):
        #part_align: B,P,3,N
        #recon: B,3,N
        B,P = align.shape[0], align.shape[1]
        if P == 2:
            #Not nessesary for 2 part object
            #order = [[0,1], [1,0]]
            order = [[0,1]]
        else:
            order = [#多いね。。。
                [0,1,0,2], #[0,1,2,0], [1,0,0,2], [1,0,2,0],
                #[0,2,0,1], #[0,2,1,0], [2,0,0,1], [2,0,1,0],
                #[1,0,1,2], #[1,0,2,1], [0,1,1,2], [0,1,2,1],
                #[1,2,1,0], #[1,2,0,1], [2,1,1,0], [2,1,0,1],
                #[2,1,2,0], #[2,1,0,2], [1,2,2,0], [1,2,0,2],
                #[2,0,2,1], #[2,0,1,2], [0,2,2,1], [0,2,1,2],
                ]
        with torch.no_grad():
            if color_w > 0 :
                align_color = color_w * align_c.unsqueeze(1).expand(-1,P,-1,-1)
                align_full = torch.cat([align, align_color], dim=2)
                recon_color = color_w * recon_c
                recon_full = torch.cat([recon, recon_color], dim=1)
            else:
                align_full, recon_full = align, recon

            d_list= []
            for p in range(P):
                cur_align = align_full[:, p]
                knn = knn_points(cur_align.permute(0,2,1), recon_full.permute(0,2,1), norm=1, K=1)
                dist = knn.dists[...,0]#B,N
                d_list.append(dist)
            d_list = torch.stack(d_list, dim=1)#B,P,N


        order_align_p = torch.stack([align_p[:, order[o]] for o in range(len(order))], dim=1)#B,O,P,N
        order_recon_p = torch.stack([recon_p[:, order[o]] for o in range(len(order))], dim=1)#B,O,P,M
        order_d = torch.zeros_like(order_align_p[:,:,0,0])#B,O

        for o in range(len(order)):
            cur_align_p = order_align_p[:,o]
            cur_d = torch.mean(d_list * cur_align_p, dim=-1)#B,P
            if P == 4:
                cur_d[:,[0,2]] = 0.5 * cur_d[:,[0,2]]
                common_part_d = torch.norm(align_full[:, 0] - align_full[:, 2], dim=1).mean(-1)#B
                cur_d = torch.cat([cur_d, common_part_w * common_part_d.unsqueeze(1)], dim=1)#B,P+1
            order_d[:,o] = cur_d.sum(-1)
        order_idx = torch.argmin(order_d, dim=-1)#B

        align_p = order_align_p[torch.arange(B), order_idx]
        recon_p = order_recon_p[torch.arange(B), order_idx]
        if P==4:
            align_p, recon_p = align_p[:,[0,1,3]], recon_p[:,[0,1,3]]
        return align_p, recon_p

    def chamfer(self, x, y, single_direction=False, dcd=0):
        cd_left, cd_right = chamfer_distance(
            x.permute(0,2,1), y.permute(0,2,1),
            single_directional = single_direction
        )
        
        if dcd != 0:
            cd_left = 1 - torch.exp(-dcd * cd_left)
            cd_right = 1 - torch.exp(-dcd * cd_left)
        
        if single_direction:
            return cd_left
        else:
            return cd_left, cd_right

class ArtOutBlock(nn.Module):
    def __init__(self, params, norm=None):
        """outblock for se(3)-invariant descriptor learning"""
        super(ArtOutBlock, self).__init__()

        c_in = 256
        mlp = params['mlp']
        c_out = 256
        na = params['kanchor']
        self.permute_soft = params['permute_soft']
        self.nmask = params['nmask']
        self.L = len(mlp)
        self.pointnet, self.pooling_bn = nn.ModuleList(), nn.ModuleList()
        for i in range(self.L):
            self.pointnet.append(sptk.PointnetSO3Conv(mlp[i][-1],mlp[i][-1],na))
            self.pooling_bn.append(nn.BatchNorm1d(mlp[i][-1]))
        self.pn_linear = nn.Conv1d(sum([mlp[i][-1] for i in range(self.L)]), c_in, 1)
        self.pn_bn = nn.BatchNorm1d(c_in)
        self.feat_layer = nn.Sequential(
            nn.Conv1d(c_in*na, 512, 1),    #, bias=False
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            )
        self.attention_layer = nn.Sequential(
            nn.Conv1d(512, 128, 1),    #, bias=False
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 1, 1)
            )
        self.regressor_layer = nn.Sequential(
            nn.Conv1d(512, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 7, 1)
            )
        self.out_layer = nn.Sequential(
            nn.Linear(512, c_out*2),
            nn.BatchNorm1d(c_out*2),
            nn.ReLU(inplace=True),
            nn.Linear(c_out*2,c_out)
            )

        trace_idx_ori, trace_idx_rot = sptk.get_relativeV_index()
        self.register_buffer("trace_idx_ori", torch.tensor(trace_idx_ori.swapaxes(0,1), dtype=torch.long), persistent=False)  # 12*60
        self.register_buffer("trace_idx_rot", torch.tensor(trace_idx_rot.swapaxes(0,1), dtype=torch.long), persistent=False)
        self.register_buffer('anchors', torch.from_numpy(L.get_anchorsV()), persistent=False)
        # self.out_norm = nn.BatchNorm1d(c_out, affine=True)

    def forward(self, x_list):
        
        nb, nc, np, na = x_list[-1].feats.shape
        
        x_feat = self._pooling(x_list)   # bcpa -> bca
        # x_feat = self.pn_linear(x_feat)     # bca
        x_feat = x_feat[:,:,self.trace_idx_ori].flatten(1,2) #   b[ca]r
        x_feat = self.feat_layer(x_feat)# b, 512, r

        residual = self.regressor_layer(x_feat).permute(0,2,1)
        residual = residual.reshape(nb, 60, 7)#b,r,7
        #residual[:,:,:3] = torch.sqrt(torch.softmax(residual[:,:,:3], dim=-1))
        res_D = residual[:,:,:3] / torch.norm(residual[:,:,:3], dim=-1,keepdim=True)
        res_N = (torch.sigmoid(residual[:,:,3])-0.5) * torch.pi / 5
        res_V = res_D * res_N.unsqueeze(-1)
        res_R = ExpSO3(res_V.reshape(-1,3)).reshape(nb,60,3,3).contiguous()
        res_T = residual[:,:,4:]
        pred_R = torch.einsum('btij, btjk->btik', self.anchors.unsqueeze(0), res_R)# br33

        x_attn = self.attention_layer(x_feat).squeeze(1) # br
        x_attn = F.softmax(x_attn, dim=-1)           # br

        x_feat_max_r = torch.max(x_feat, dim=-1).values
        x_out = self.out_layer(x_feat_max_r)        # b c_out

        return x_out, x_attn, pred_R, res_T
    
    def _pooling(self, x_list):
        # [nb, nc, na]
        x_out_list = []
        for i in range(self.L):
            x_out = self.pointnet[i](x_list[i])
            x_out = self.pooling_bn[i](x_out)
            x_out = F.relu(x_out)
            x_out_list.append(x_out)

        x_out_list = torch.cat(x_out_list, dim=1)#B,C,A
        x_ML_out = self.pn_linear(x_out_list)
        x_ML_out = self.pn_bn(x_ML_out)

        return x_ML_out


class DecoderFCWithPVP(nn.Module):
    def __init__(self, n_features=(256, 256), latent_dim=256, bn=True, point=1024):
        super(DecoderFCWithPVP, self).__init__()
        self.n_features = list(n_features)
        self.latent_dim = latent_dim
        self.output_pts = point


        model = []
        prev_nf = self.latent_dim
        for idx, nf in enumerate(self.n_features): # n_features is used for constructing layers
            fc_layer = nn.Linear(prev_nf, nf) # Linear layer
            model.append(fc_layer)

            if bn:
                bn_layer = nn.BatchNorm1d(nf) # batch norm
                model.append(bn_layer)

            act_layer = nn.LeakyReLU(inplace=True)
            model.append(act_layer)
            prev_nf = nf

        self.model = nn.Sequential(*model)
        self.fc_coor = nn.Sequential(
            nn.Linear(self.n_features[-1], self.n_features[-1]),
            nn.BatchNorm1d(self.n_features[-1]),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.n_features[-1], self.output_pts * 3)
            )
        self.fc_color = nn.Sequential(
            nn.Linear(self.n_features[-1], self.n_features[-1]),
            nn.BatchNorm1d(self.n_features[-1]),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.n_features[-1], self.output_pts * 3)
            )
        self.y_base = torch.nn.init.xavier_uniform_(nn.Parameter(torch.zeros(self.output_pts, 3)))
        #self.c_base = torch.nn.init.xavier_uniform_(nn.Parameter(torch.zeros(self.output_pts, 3)))
     
    def forward(self, x):
        #recon
        B = x.shape[0]
        y = self.model(x)

        y_translation = self.fc_coor(y).reshape(B, self.output_pts, 3)
        y_coor = self.y_base.unsqueeze(0).repeat(B,1,1) + y_translation
        y_coor = y_coor.permute(0,2,1)

        y_color = self.fc_color(y).reshape(B, self.output_pts, 3)
        #y_color = self.c_base.unsqueeze(0).repeat(B,1,1) + y_color
        y_color = y_color.permute(0,2,1)

        return y_coor, y_translation, y_color



# Full Version
def build_model(opt,
                #mlps=[[64,64], [128,128], [256,256], [256]],
                mlps=[[32,32], [64,64], [128,128], [256]],
                # mlps=[[128,128], [256,256], [512,512], [256]],
                # mlps=[[16,16], [32,32], [64,64],[256]],
                # mlps=[[32,32,32,32], [64,64,64,64], [128,128,128,128],[256]],
                # mlps=[[32,], [64,], [128,],[256]],
                # mlps=[[32,32], [64,64], [128,128],[128,128],[256]],
                out_mlps=[256],
                strides=[2,2,2,2,2],
                # strides=[2,2,2,2,2],
                initial_radius_ratio = 0.2,
                sampling_ratio = 0.4,
                sampling_density = 0.5,
                kernel_density = 1,
                kernel_multiplier = 2,
                input_radius = 1.0,
                sigma_ratio= 0.5, # 0.1
                xyz_pooling = None,
                so3_pooling = "max",
                to_file=None):


    device = opt.device
    input_num = opt.model.input_num
    dropout_rate = opt.model.dropout_rate
    temperature = opt.train_loss.temperature
    so3_pooling =  opt.model.flag
    na = 1 if opt.model.kpconv else opt.model.kanchor
    feat_all_anchors = opt.model.feat_all_anchors
    anchor_ab_loss = opt.train_loss.anchor_ab_loss
    fc_on_concat = opt.model.fc_on_concat
    sym_kernel = not opt.model.no_sym_kernel
    check_equiv = opt.debug_mode == 'check_equiv'
    drop_xyz = opt.model.drop_xyz
    permute_soft = opt.model.permute_soft


    if input_num > 1024:
        sampling_ratio /= (input_num / 1024)
        strides[0] = int(2 * (input_num / 1024))
        print("Using sampling_ratio:", sampling_ratio)
        print("Using strides:", strides)

    params = {'name': 'Invariant ZPConv Model',
              'backbone': [],
              'na': na,
              'check_equiv': check_equiv,
              }

    dim_in = 3

    # process args
    n_layer = len(mlps)
    stride_current = 1
    ### stride_multipliers: in each layer, the number of points is reduced by half, 
    ### and the wanted neighboring points around each center is doubled.
    stride_multipliers = [stride_current]
    for i in range(n_layer):
        stride_current *= 2 # strides[i]
        stride_multipliers += [stride_current]

    ### num_centers: number of output points of an layer, also the number of input points of the next layer
    num_centers = [int(input_num / multiplier) for multiplier in stride_multipliers]
    ### The calculation of radius_ratio: 
    ### initial_radius_ratio: the normalized radius (to the input_radius) of the first layer
    ### sampling_density: 0.5, a coefficient to convert space to radius (in 3D, it should be 1/3 strictly speaking, 
    ### using 0.5 could cause the radius and thus the ball too large for multiplier > 1, but the compuation of 
    ### neighbor is not affected).
    ### radius_ratio: normalized radius (to the input_radius) for ball query when finding neighbor points around centers
    radius_ratio = [initial_radius_ratio * multiplier**sampling_density for multiplier in stride_multipliers]
    # radius_ratio = [0.25, 0.5]
    ### radii: actual radius of ball query. input_radius: radius of the input point cloud
    radii = [r * input_radius for r in radius_ratio]
    # Compute sigma
    # weighted_sigma = [sigma_ratio * radii[i]**2 * stride_multipliers[i] for i in range(n_layer + 1)]
    weighted_sigma = [sigma_ratio * radii[0]**2]

    for idx, s in enumerate(strides):
        weighted_sigma.append(weighted_sigma[idx] * 2)
    ### weighted_sigma: ratio in denominator for computing the weight of an input point to a kernel point

    for i, block in enumerate(mlps):
        block_param = []
        for j, dim_out in enumerate(block):
            lazy_sample = i != 0 or j != 0
            stride_conv = i == 0 or xyz_pooling != 'stride'
            # TODO: WARNING: Neighbor here did not consider the actual nn for pooling. Hardcoded in vgtk for now.
            ### The calculation of neighbor: 
            ### num_centers: number of points
            ### radius_ratio[i]**(1/sampling_density) = initial_radius_ratio**(1/sampling_density) * multiplier: 
            ### the normalized space of a neighborhood
            ### num_centers[i] * radius_ratio[i]**(1/sampling_density): number of points in a neighborhood
            ### sampling_ratio: as named
            ### neighbor: number of points to sample in a neighborhood
            neighbor = int(sampling_ratio * num_centers[i] * radius_ratio[i]**(1/sampling_density))
            # if i==0 and j==0:
            #    neighbor *= int(input_num/1024)
            kernel_size = 1
            if j == 0:
                # stride at first (if applicable), enforced at first layer
                inter_stride = strides[i]
                nidx = i if i == 0 else i+1
                if stride_conv:
                    ### coming from the last layer where the points are twice as many, thus neighbor size is also doubled
                    neighbor *= 2 # = 2 * int(sampling_ratio * num_centers[i] * radius_ratio[i]**(1/sampling_density))
                    # kernel_size = 1 # if inter_stride < 4 else 3
            else:
                inter_stride = 1
                nidx = i+1

            print(f"At block {i}, layer {j}!")
            print(f'neighbor: {neighbor}')
            print(f'stride: {inter_stride}')
            sigma_to_print = weighted_sigma[nidx]**2 / 3
            print(f'sigma: {sigma_to_print}')
            print(f'radius ratio: {radius_ratio[nidx]}')

            # one-inter one-intra policy
            # block_type = 'inter_block' if na<60 else 'separable_block'
            if na == 60:
                block_type = 'separable_block' 
            elif na == 12:
                block_type = 'separable_s2_block'
            elif na < 60:
                block_type = 'inter_block'
            else:
                raise ValueError(f"na={na} not supported.")
            conv_param = {
                'type': block_type,
                'args': {
                    'dim_in': dim_in,
                    'dim_out': dim_out,
                    'kernel_size': kernel_size,
                    'stride': inter_stride,
                    'radius': radii[nidx],
                    'sigma': weighted_sigma[nidx],
                    'n_neighbor': neighbor,
                    'lazy_sample': lazy_sample,
                    'dropout_rate': dropout_rate,
                    'multiplier': kernel_multiplier,
                    'activation': 'leaky_relu',
                    'pooling': xyz_pooling,
                    'kanchor': na,
                    'norm': 'BatchNorm2d',
                    # 'sym_kernel': sym_kernel,
                }
            }
            if na == 12:
                conv_param['args']['sym_kernel'] = sym_kernel
            block_param.append(conv_param)
            dim_in = dim_out

        params['backbone'].append(block_param)

    params['outblock'] = {
        'dim_in': dim_in,
        'mlp': mlps,
        'fc': [64],
        'kanchor':na,
        'permute_soft':permute_soft,
        'nmask': opt.equi_settings.nmasks
    }

    params['recon'] = {
        'nmask': opt.equi_settings.nmasks,
        'point': opt.model.input_num,
    }

    params['pn'] = {
        'nmask': opt.equi_settings.nmasks,
        'njoint': opt.equi_settings.njoints,
        'rotation_range': opt.model.rotation_range,
    }

    params['color_w'] = opt.model.color_cd_w
    params['joint_type'] = opt.model.joint_type
    if params['joint_type'] not in ['r', 'p']:
        raise NotImplementedError

    if to_file is not None:
        with open(to_file, 'w') as outfile:
            json.dump(params, outfile, indent=4)

    model = ArtSO3ConvModel(params).to(device)
    return model

def build_model_from(opt, outfile_path=None):
    return build_model(opt, to_file=outfile_path)
