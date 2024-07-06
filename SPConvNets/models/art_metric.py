import torch
import numpy as np
import math
import copy
from SPConvNets.utils.chamfer import chamfer_distance
from pytorch3d.ops.knn import knn_gather, knn_points
import torch.nn.functional as F
from torch_cluster import fps
from vgtk.spconv.functional import batched_index_select
from vgtk.functional.rotation import so3_mean
from scipy.spatial.transform import Rotation as R
from SPConvNets.utils.lietorch import ExpSO3, drct_rotation

class Art_Metric(torch.nn.Module):
    def __init__(self, shape_type, nmask, rigid_w=0.5, color_w=0.5, prob_threshold = 0.1, non_valid_exist=False, rigid_halve=5000):
        """For multi-class classification. """
        super(Art_Metric, self).__init__()
        self.shape_type = shape_type
        self.nmask = nmask
        self.prob_threshold = prob_threshold
        self.rigid_w = rigid_w
        self.color_w = color_w
        self.standardization_list = []
        self.non_valid_exist = non_valid_exist
        self.register_buffer("mean_rotation", torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(nmask,1,1))
        self.register_buffer("mean_translation", torch.zeros(3, dtype=torch.float32).unsqueeze(0).repeat(nmask,1))
        self.iter_count = 0
        self.rigid_halve = rigid_halve


    def forward(self, pred, label=None, mode='train'):
        R_attn = pred['R_attn'].contiguous()                # B,R
        R_base = pred['R_base'].contiguous()                # B,R,3,3
        T_base = pred['T_base'].contiguous()                # B,R,3
        R_select = pred['R_select'].contiguous()            #B,3,3
        T_select = pred['T_select'].contiguous()            #B,3
        base_distance = pred['R_distance'].contiguous()     # B,R
        base_idx = pred['R_idx'].contiguous()               # B,R

        S_input = pred['S_input'].contiguous()              # B,3,N
        S_align = pred['S_align'].contiguous()              # B,3,N
        S_align_2 = pred['S_align_part'].contiguous()       # B,2J,3,N
        S_color = pred['S_color'].contiguous()              # B,3,N
        S_joint = pred['S_joint'].contiguous()              # B,J,3
        S_drct = pred['S_drct'].contiguous()                # B,J,3
        S_angl = pred['S_angl'].contiguous()                # B,J,2
        S_segmentation = pred['S_seg'].contiguous()         #B,P,N

        I_cano = pred['I_cano'].contiguous()                # B,3,M
        I_align_2 = pred['I_align_part'].contiguous()       # B,P,3,M
        I_color = pred['I_color'].contiguous()              # B,3,M
        I_joint = pred['I_joint'].contiguous()              # B,J,3
        I_drct = pred['I_drct'].contiguous()                # B,J,3
        I_angl = pred['I_angl'].contiguous()                # B,J,2
        I_shape_var = pred["I_shape_var"].contiguous()      # B,3,M
        I_segmentation = pred['I_seg'].contiguous()         #B,P,M

        if len(label) == 3:
            gt_pose, gt_joint, gt_segmentation = label
        else:
            gt_pose, gt_joint, gt_segmentation, gt_expand, gt_center = label

        B,R,P = R_base.shape[0], R_base.shape[1], S_segmentation.shape[1]


        #Align Input
        I_cano_input = torch.bmm(R_select, I_cano + T_select.unsqueeze(-1))
        S_drct_input = torch.bmm(R_select, S_drct.permute(0,2,1)).permute(0,2,1)
        S_joint_input = torch.bmm(R_select, S_joint.permute(0,2,1) + T_select.unsqueeze(-1)).permute(0,2,1)

        if mode == 'train':
            self.iter_count += 1
            if self.iter_count % self.rigid_halve == 0:
                self.rigid_w /= 2
            loss = 0
            #Reconstruction Loss
            rigid_loss = 10 * self.loss_recon_all_shape(S_align, I_cano, dcd=30, dcd_inv=120) #rigid loss
            art_loss = 10 * self.loss_recon_part_align(S_align_2, I_cano, S_segmentation, I_segmentation, S_color, I_color, dcd=30, dcd_inv=120, color_w=self.color_w)#articulation loss
            loss += self.rigid_w * rigid_loss
            loss += (1-self.rigid_w) * art_loss
            #Shape Fix Loss
            loss += 200 * self.loss_gather(I_cano, K=64)# cano shape local density
            loss += 10 * self.loss_cano_variance_reg(I_shape_var, dcd=60)# cano shape variance
            loss += 1 * self.loss_base_part_reg(I_cano)#shape center offset
            #Joint Fix
            loss += 10 * self.loss_cano_joint_reg(I_joint, I_drct, I_angl)# cano joint fix
            loss += 0.1 * self.loss_joint_closest_reg(I_joint, I_cano, k=8, dcd=30)# cano joint closest
            loss += 0.1 * self.loss_joint_closest_reg(S_joint, S_align, k=8, dcd=30)
            if P == 3:
                drct_align_r =  drct_rotation(S_drct, I_drct)
                loss += 1 * ((drct_align_r[:,0] - drct_align_r[:,1]) ** 2).mean()#drct align remain the same
                #loss += 10 * ((torch.norm(S_joint[:,0] - S_joint[:,1], dim=-1) - torch.norm(I_joint[:,0] - I_joint[:,1], dim=-1)) ** 2).mean()#Joint relative distance remain the same
                #loss += 10 * ((torch.norm(S_joint[:,0] - I_joint[:,0], dim=-1) - torch.norm(S_joint[:,1] - I_joint[:,1], dim=-1)) ** 2).mean()#Joint relative distance remain the same
                #loss += 1 * (torch.norm(S_drct[:,0] - S_drct[:,1], dim=-1) ** 2).mean()# drct same
                #loss += 1 * (torch.norm(I_drct[:,0] - I_drct[:,1], dim=-1) ** 2).mean()# drct same
                loss += 10 * (torch.norm(S_align_2[:,0] - S_align_2[:,2], dim=1) ** 2).mean()#Align part remain the same

            #Reg loss
            loss += 1 * self.loss_attn(R_attn, base_distance)
            loss += 1 * (torch.norm(T_select, dim=-1)**2).mean()#T offset
            loss += 10 * self.loss_prob_reg(I_segmentation, S_segmentation, threshold = self.prob_threshold, min_prob = 1e-2)

            attn_acc = torch.argmax(R_attn, dim=-1) == base_idx
            '''
            torch.save([
                #shape
                S_input.detach().cpu(),
                S_align.detach().cpu(),
                S_align_2.detach().cpu(),
                I_cano.detach().cpu(),
                I_cano_input.detach().cpu(),
                #segmentation
                S_segmentation.detach().cpu(),
                I_segmentation.detach().cpu(),
                #joint
                S_joint.detach().cpu(),
                S_drct.detach().cpu(),
                S_angl.detach().cpu(),
                I_joint.detach().cpu(),
                I_drct.detach().cpu(),
                I_angl.detach().cpu(),
                attn_acc.detach().cpu()
                ], './test/1.pt')
            '''
            return loss
        
        elif mode == 'standardization':
            gt_pose,_,_ = self.standardization_order(gt_pose, gt_joint, gt_segmentation, S_segmentation, self.non_valid_exist)
            gt_pose_align = self.standardization_pose_align(gt_pose, R_select, T_select, S_joint, S_drct, S_angl, I_joint, I_drct, I_angl)
            self.standardization_list.append(gt_pose_align)
            return

        elif mode == 'test':
            gt_pose, gt_joint, gt_segmentation = self.standardization_order(gt_pose, gt_joint, gt_segmentation, S_segmentation, self.non_valid_exist)
            pred_rotation_align, pred_trans_align = self.evaluation_pose_align(R_select, T_select, S_joint, S_drct, S_angl, I_joint, I_drct, I_angl)
            pred_pose_align = torch.eye(4, device=gt_pose.device, dtype=gt_pose.dtype).reshape(1,1,4,4).repeat(B,P,1,1)
            pred_pose_align[:,:,:3,:] = torch.concat([pred_rotation_align, pred_trans_align.unsqueeze(-1)], dim=-1)

            rotation_error, trans_error = self.evaluation_pose(gt_pose, pred_rotation_align, pred_trans_align, gt_expand)#B,P,  B,P
            segmentation_error = self.evaluation_segmentation(gt_segmentation, S_segmentation, self.non_valid_exist)#B,P
            joint_error, drct_error = self.evaluation_joint(gt_joint[:,:,:3], gt_joint[:,:,3:], S_joint_input, S_drct_input, gt_expand)#B,J
            viz_f = {
                'input': S_input.detach().cpu(),
                'input_align': S_align.detach().cpu(),
                'input_align_2': S_align_2.detach().cpu(),
                'input_seg': S_segmentation.detach().cpu(),
                'input_joint': S_joint_input.detach().cpu(),
                'input_drct': S_drct_input.detach().cpu(),
                'input_angl': S_angl.detach().cpu(),

                'gt_pose': gt_pose.detach().cpu(),
                'pred_pose':pred_pose_align.detach().cpu(),
                'gt_expand': gt_expand.detach().cpu(),
                'gt_center': gt_center.detach().cpu(),

                'recon': I_cano.detach().cpu(),
                'recon_seg': I_segmentation.detach().cpu(),
                'recon_joint': I_joint.detach().cpu(),
                'recon_drct': I_drct.detach().cpu(),
                'recon_angl': I_angl.detach().cpu(),

            }
            return segmentation_error, joint_error, drct_error, rotation_error, trans_error, viz_f
        
        else:
            raise NotImplementedError
    

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
        
    def loss_attn(self, attn, total_cd):
        loss = attn * total_cd
        loss = loss.sum(dim=-1).mean()
        return loss
    
    def loss_recon_part_align(self, align, recon, align_p, recon_p, align_c, recon_c, dcd=30, dcd_inv=120, color_w = 0):
        #part_align: B,P,3,N
        #recon: B,3,N
        B,P = align.shape[0], align.shape[1]
        if color_w > 0 :
            align_color = color_w * align_c.unsqueeze(1).expand(-1,P,-1,-1)
            align_full = torch.cat([align, align_color], dim=2)
            recon_color = color_w * recon_c
            recon_full = torch.cat([recon, recon_color], dim=1)
        else:
            align_full, recon_full = align, recon
        
        d_list, d_inv_list = [], []  
        for p in range(P):
            if P==4 and p == 0:
                cur_align = 0.5 * (align_full[:, 0] + align_full[:, 2])
            elif P==4 and p == 2:
                continue
            else:
                cur_align = align_full[:, p]
            d = self.chamfer(cur_align, recon_full, single_direction=True, dcd=dcd)
            d_list.append(d)
            d_inv = self.chamfer(recon_full, cur_align, single_direction=True, dcd=dcd_inv)
            d_inv_list.append(d_inv)
        d_list = torch.stack(d_list, dim=1)
        d_inv_list = torch.stack(d_inv_list, dim=1)
        
        #if P == 4:#Eyeglasses
        #    d_list = torch.stack([0.5 * (d_list[:,0] + d_list[:,2]), d_list[:,1], d_list[:,3]], dim=1)
        #    d_inv_list = torch.stack([0.5 * (d_inv_list[:,0] + d_inv_list[:,2]), d_inv_list[:,1], d_inv_list[:,3]], dim=1)


        aling_w, recon_w = align_p, recon_p
        d_min = torch.min(d_list, dim=1).values#B,N
        d_mean = torch.sum(d_list * aling_w, dim=1)
        
        d_inv_min = torch.min(d_inv_list, dim=1).values#B,M
        d_inv_mean = torch.sum(d_inv_list * recon_w, dim=1)#B,M

        loss_d = d_mean
        loss_d_inv = (dcd / dcd_inv) * (d_inv_mean)

        loss = loss_d.mean() + loss_d_inv.mean()
        return loss
    
    def loss_recon_all_shape(self, align, recon, dcd=30, dcd_inv=120):
        d = self.chamfer(align, recon, single_direction=True, dcd=dcd)
        d_inv = self.chamfer(recon, align, single_direction=True, dcd=dcd_inv)
        d_inv = (dcd / dcd_inv) * d_inv
        loss = d.mean() + d_inv.mean()
        return loss
    
    def loss_gather(self, input, K = 64):
        B = input.shape[0]
        loss_connection, loss_var = 0, 0
            
        part_p = input.permute(0,2,1) # B,-1,3
        loss = 0

        part_knn_d = knn_points(part_p, part_p, K=K+1)
        part_knn_d = torch.sqrt(part_knn_d.dists[..., 1:])
        for i in range(K):
            loss += torch.var(part_knn_d[...,i], dim=-1).mean()
            
        loss = loss / K
        return loss
    
    def loss_cano_joint_reg(self, joint, drct, angl):
        d_a = (angl ** 2).mean()

        m_j = torch.mean(joint, dim=0)
        d_j = torch.mean(torch.norm(joint - m_j.unsqueeze(0), dim=-1) ** 2)

        m_d = torch.mean(drct, dim=0)
        d_d = torch.mean(torch.norm(drct - m_d.unsqueeze(0), dim=-1) ** 2)

        return d_a + d_d# + d_j
    
    def loss_joint_closest_reg(self, I_joint, I_cano, k=8, dcd=30):
        B,N = I_cano.shape[0], I_cano.shape[-1]
        cano_d = I_cano.detach().clone()
        dist_cano = knn_points(I_joint, cano_d.permute(0,2,1), K=k).dists
        loss_cano = torch.sqrt(dist_cano).mean(-1)
        loss_cano = 1 - torch.exp(-dcd * (loss_cano**2))

        return loss_cano.mean()
    
    def loss_joint_closest_reg_2(self, I_joint, I_cano, I_segmentation, k=16, dcd=30, eps=1e-1):
        B,N = I_cano.shape[0], I_cano.shape[-1]
        cano_d = I_cano.detach().clone()
        close_idx = knn_points(cano_d.permute(0,2,1), cano_d.permute(0,2,1), K=k).idx#B,N,K
        close_idx = close_idx[:,:,1:]
        prob_list = []
        for b in range(B):
            cur_idx = close_idx[b]#N,K
            cur_prob = I_segmentation[b, :,cur_idx.flatten()].reshape(-1,N,k-1)#P,N,K
            prob_list.append(cur_prob)
        prob_list = torch.stack(prob_list, dim=0)#B,P,N,K
        var_list = torch.var(prob_list, dim=-1)#B,P,N
        var_list = var_list[:,1:,:]#B,J,N
        distance_list = torch.norm(I_cano.unsqueeze(1) - I_joint.unsqueeze(-1), dim=-2) ** 2#B,J,N
        distance_list = 1 - torch.exp(-dcd * distance_list)
        #loss = (distance_list * (var_list + eps)).mean()
        loss = (distance_list * eps).mean()
        return loss
  
    def loss_base_part_reg(self, cano):

        base_mean = cano.mean(-1)
        distance_2 = torch.norm(base_mean, dim=-1) ** 2

        return distance_2.mean()
    
    def loss_cano_variance_reg(self, shape_v, dcd=100):
        d_s = torch.norm(shape_v, dim=-1) ** 2 #B,M
        d_s = 1 - torch.exp(-dcd * d_s)
        d_s = d_s.mean()

        return d_s

    def loss_align_variance_reg(self, joint_v, drct_v, drct):
        d_j = torch.norm(joint_v, dim=-1)
        d_j = (d_j ** 2).mean()
        d_d = torch.norm(drct_v, dim=-1)
        d_d = (d_d ** 2).mean()
        d_n = 10 * torch.mean((torch.norm(drct, dim=-1)-1) ** 2)
        return d_j + d_d + d_n

    def loss_prob_reg(self, prob, real_prob, threshold = 0.1, min_prob=1e-3):
        prob_mean = prob.mean(-1) # B,P
        loss_threshold = (threshold - prob_mean) * (threshold > prob_mean)
        loss_threshold = loss_threshold.mean()

        loss_prob_min = torch.clamp(min_prob - torch.min(prob, dim=1).values, min=0).mean()
        loss_prob_real_min = torch.clamp(min_prob - torch.min(real_prob, dim=1).values, min=0).mean()

        return loss_threshold + loss_prob_real_min + loss_prob_min

    def standardization_pose_align(self, gt_pose, R_base, T_base, S_joint, S_drct, S_angl, I_joint, I_drct, I_angl):
        B,P,J = gt_pose.shape[0], gt_pose.shape[1], S_angl.shape[1]
        gt_pose[:,:,:3,:3] = torch.einsum("bpij, bpjk -> bpik", R_base.permute(0,2,1).unsqueeze(1), gt_pose[:,:,:3,:3])
        gt_pose[:,:,:3,3] = gt_pose[:,:,:3,3] - T_base.unsqueeze(1)

        #rigid transformation
        res_drct = drct_rotation(S_drct, I_drct)#B,P-1,3,3
        res_angl = ExpSO3(I_drct.unsqueeze(2) * (I_angl - S_angl).unsqueeze(-1)) #B,P-1,2,3,3

        #joint base transformation
        t_align, r_align = [], []
        order = [0,1, 0,2] if P == 3 else [0,1]
        for j in range(J):
            for a in range(2):
                part_idx = order[2*j + a]
                new_t = gt_pose[:,part_idx,:3,3].unsqueeze(-1).clone()#B,3,1
                new_t = new_t - S_joint[:,j].unsqueeze(-1)
                new_t = torch.bmm(res_drct[:,j], new_t)
                new_t = torch.bmm(res_angl[:,j,a], new_t)
                new_t = new_t + I_joint[:,j].unsqueeze(-1)
                t_align.append(new_t.squeeze(-1))
                new_r = gt_pose[:,part_idx, :3,:3]#B,3,3
                new_r = torch.bmm(res_drct[:,j], new_r)
                new_r = torch.bmm(res_angl[:,j,a], new_r)
                r_align.append(new_r)
        t_align = torch.stack(t_align, dim=1)#B,2j,3
        r_align = torch.stack(r_align, dim=1)#B,2j,3,3

        #return to part base
        if P == 3:
            t_align[:,0] = 0.5*(t_align[:,0] + t_align[:,2])
            t_align = t_align[:,:3]
            r_align[:,0] = so3_mean(r_align[:,[0,2]])
            r_align = r_align[:,:3]

        gt_pose[:,:,:3,:3] = r_align
        gt_pose[:,:,:3,3] = t_align
        return gt_pose
    
    def standardization_ransac(self):
        T, s, r_th, t_th = 100, 5, 10, 0.25
        self.standardization_list = torch.cat(self.standardization_list, dim=0)#B,P,4,4
        B, P = self.standardization_list.shape[0], self.standardization_list.shape[1]
        self.mean_rotation = torch.zeros_like(self.standardization_list[0,:,:3,:3])
        self.mean_translation = torch.zeros_like(self.standardization_list[0,:,:3,-1])
        best_r, best_t = 0, 0
        for t in range(T):
            cur_idx = torch.tensor(np.random.choice(B, size=s, replace=False))
            cur_rotation = self.standardization_list[cur_idx,:,:3,:3]
            cur_translation = self.standardization_list[cur_idx,:,:3,3]
            cur_mean_rotation = so3_mean(cur_rotation.permute(1,0,2,3)).unsqueeze(0)#1,P,3,3
            cur_mean_translation = torch.mean(cur_translation, dim=0).unsqueeze(0)#1,P,3
            r_error, t_error = self.evaluation_pose(self.standardization_list, cur_mean_rotation, cur_mean_translation)
            r_performance, t_performance = (r_error <= r_th).sum(), (t_error <= t_th).sum()
            if best_r < r_performance:
                self.mean_rotation = cur_mean_rotation.squeeze(0)
                best_r = r_performance
            if best_t < t_performance:
                self.mean_translation = cur_mean_translation.squeeze(0)
                best_t = t_performance
        #Clear this standardiazation
        self.standardization_list = []
        return best_r / (B*P), best_t / (B*P)
    
    def standardization_order(self, gt_pose, gt_joint, gt_seg, pred_seg, noise_exist=False):
        B,P = pred_seg.shape[0], pred_seg.shape[1]
        if P == 2:
            order = [[0,1], [1,0]]
        elif P == 3:
            order = [[0,1,2], [0,2,1]]

        iou_list = []
        for o in order:
            cur_seg = torch.argmax(pred_seg[:,o], dim=1)#B,N

            if not noise_exist:
                acc = cur_seg == gt_seg #B,N
                iou = acc.sum(-1) / acc.shape[-1]#B
            else:
                iou = torch.zeros_like(pred_seg[:,0,0])#B
                for b in range(B):
                    valid_idx = gt_seg[b] != 0
                    valid_cur_seg = cur_seg[b, valid_idx]
                    valid_gt_seg = gt_seg[b, valid_idx] - 1
                    acc = valid_cur_seg == valid_gt_seg
                    iou[b] = acc.sum() / acc.shape[-1]
            iou_list.append(iou)
        iou_list = torch.stack(iou_list, dim=1)#B,O
        best_order = torch.argmax(iou_list, dim=-1)

        for b in range(B):
            #Segmentation exchange
            if not noise_exist:
                cur_order = order[best_order[b]]
                gt_seg[b] = torch.argmax(F.one_hot(gt_seg[b], num_classes=P)[:, cur_order], dim=-1)
            else:
                cur_order = copy.copy(order[best_order[b]])
                for i in range(len(cur_order)):
                    cur_order[i] += 1
                cur_order.insert(0,0)
                gt_seg[b] = torch.argmax(F.one_hot(gt_seg[b], num_classes=P+1)[:, cur_order], dim=-1)
            #Joint Exchange
            if best_order[b] == 1 and P == 3:
                gt_joint[b] = gt_joint[b, [1,0]]
            #Pose Exchange
            gt_pose[b] = gt_pose[b, order[best_order[b]]]
                # unsymm object
            if self.shape_type in ['safe', 'oven', 'washing_machine']:
                pass
                # laptop HOI4D, laptop Motion
            elif best_order[b] == 1 and self.shape_type in ['laptop_h', 'laptop_m']:
                additional_r = torch.tensor([[[0,0,1],[0,-1,0],[1,0,0]]]).reshape(1,3,3).to(gt_pose.device).to(gt_pose.dtype)
                gt_pose[b,:,:3,:3] = torch.einsum('pij, pjk -> pik', gt_pose[b,:,:3,:3], additional_r)
                # eyeglasses Motion
            elif best_order[b] == 1 and self.shape_type == 'eyeglasses':
                additional_r = torch.tensor([[1,0,0],[0,-1,0],[0,0,-1]]).reshape(1,3,3).to(gt_pose.device).to(gt_pose.dtype)
                gt_pose[b,:,:3,:3] = torch.einsum('pij, pjk -> pik', gt_pose[b,:,:3,:3], additional_r)
                #scissor Real
            elif best_order[b] == 1 and self.shape_type == 'scissor_output':
                additional_r = torch.tensor([[1,0,0],[0,-1,0],[0,0,-1]]).reshape(1,3,3).to(gt_pose.device).to(gt_pose.dtype)
                gt_pose[b,:,:3,:3] = torch.einsum('pij, pjk -> pik', gt_pose[b,:,:3,:3], additional_r)
            elif best_order[b] == 1 and self.shape_type == 'basket_output':
                additional_r = torch.tensor([[-1,0,0],[0,-1,0],[0,0,1]]).reshape(1,3,3).to(gt_pose.device).to(gt_pose.dtype)
                gt_pose[b,:,:3,:3] = torch.einsum('pij, pjk -> pik', gt_pose[b,:,:3,:3], additional_r)
            elif best_order[b] == 1 and self.shape_type in ['laptop_output', 'suitcase_output']:
                additional_r = torch.tensor([[-1,0,0],[0,1,0],[0,0,-1]]).reshape(1,3,3).to(gt_pose.device).to(gt_pose.dtype)
                gt_pose[b,:,:3,:3] = torch.einsum('pij, pjk -> pik', gt_pose[b,:,:3,:3], additional_r)
        
        return gt_pose, gt_joint, gt_seg
            
    def evaluation_joint(self, gt_joint, gt_drct, pred_joint, pred_drct, gt_expand=None):
        B,J = gt_joint.shape[0], gt_joint.shape[1]
        gt_drct = gt_drct / torch.norm(gt_drct, dim=-1, keepdim=True)
        pred_drct = pred_drct / torch.norm(pred_drct, dim=-1, keepdim=True)

        inner = torch.abs((pred_drct * gt_drct).sum(-1)) #B,J
        theta = torch.acos(torch.clamp(inner, min=1e-8, max=1-1e-8))
        theta = theta * 180 / torch.pi
        residual = pred_joint - gt_joint
        distance_on_drct = (residual * gt_drct).sum(-1)
        distance = torch.norm(residual,dim=-1) ** 2 - distance_on_drct ** 2
        distance = torch.sqrt(distance)

        if gt_expand != None:
            distance /= gt_expand.reshape(-1,1)

        return distance, theta

    def evaluation_segmentation(self, gt_seg, pred_seg, noise_exist=False):
        B,P = pred_seg.shape[0], pred_seg.shape[1]
        hard_seg = torch.argmax(pred_seg, dim=1)#B,N
        iou_list = torch.zeros_like(pred_seg[:,:,0])#B,P

        for b in range(B):
            for p in range(P):
                if noise_exist:
                    valid_idx = gt_seg[b] != 0
                    valid_hard_seg = hard_seg[b, valid_idx]
                    valid_gt_seg = gt_seg[b, valid_idx]-1
                else:
                    valid_hard_seg = hard_seg[b]
                    valid_gt_seg = gt_seg[b]
                intersection = torch.logical_and(valid_hard_seg == p, valid_gt_seg == p).sum(-1)
                union = torch.logical_or(valid_hard_seg == p, valid_gt_seg == p).sum(-1)
                iou = intersection / union
                if torch.isnan(iou):
                    print('detect Nan in Segmentation')


                '''
                if self.shape_type in ['laptop_m', 'laptop_h', 'scissor', 'eyeglasses']:
                    if self.shape_type in ['laptop_m', 'laptop_h', 'scissor']:
                        order = [1,0]
                    elif self.shape_type in ['eyeglasses']:
                        order = [0,2,1]
                    else:
                        raise NotImplementedError
                    valid_gt_seg2 = torch.argmax(F.one_hot(valid_gt_seg, num_classes=P)[:,order], dim=-1)
                    intersection2 = torch.logical_and(valid_hard_seg == p, valid_gt_seg2 == p).sum(-1)
                    union2 = torch.logical_or(valid_hard_seg == p, valid_gt_seg2 == p).sum(-1)
                    iou2 = intersection2 / union2
                    iou = iou2 if iou2 > iou else iou
                '''
                iou_list[b,p] = iou
        return iou_list
    
    def evaluation_pose(self, gt_pose, pred_rotation, pred_translation, gt_expand=None):
        B,P = gt_pose.shape[0], gt_pose.shape[1]

        t_error = torch.norm(gt_pose[:,:,:3,3] - pred_translation, dim=-1)#B,P
        u, s, vh = torch.linalg.svd(pred_rotation)
        R = u @ vh
        R = torch.einsum('bpij, bpjk -> bpik', gt_pose[:,:,:3,:3], R.permute(0,1,3,2))
        cos_theta = 0.5 * (torch.diagonal(R, offset=0, dim1=2, dim2=3).sum(-1) - 1)
        #cos_theta = torch.abs(cos_theta)
        r_error = torch.acos(torch.clamp(cos_theta, -1+1e-10, 1-1e-10)) * 180 / torch.pi

        '''
        if self.shape_type in ['laptop_m', 'laptop_h', 'scissor', 'eyeglasses']:
            if self.shape_type in ['laptop_m', 'laptop_h', 'scissor']:
                order = [1,0]
            elif self.shape_type in ['eyeglasses']:
                order = [0,2,1]
            else:
                raise NotImplementedError
            t_error2 = torch.norm(gt_pose[:,order,:3,3] - pred_translation, dim=-1)#B,P
            R = torch.einsum('bpij, bpjk -> bpik', gt_pose[:,order,:3,:3], pred_rotation.permute(0,1,3,2))
            cos_theta = 0.5 * (torch.diagonal(R, offset=0, dim1=2, dim2=3).sum(-1) - 1)
            #cos_theta = torch.abs(cos_theta)
            r_error2 = torch.acos(torch.clamp(cos_theta, -1+1e-10, 1-1e-10)) * 180 / torch.pi

            t_error[t_error2.sum(-1) < t_error.sum(-1)] = t_error2[t_error2.sum(-1) < t_error.sum(-1)]
            r_error[t_error2.sum(-1) < t_error.sum(-1)] = r_error2[t_error2.sum(-1) < t_error.sum(-1)]
        '''
        if gt_expand != None:
            t_error /= gt_expand.reshape(-1,1)
        
        return r_error, t_error
    
    def evaluation_pose_align(self, R_base, T_base, S_joint, S_drct, S_angl, I_joint, I_drct, I_angl):
        B,J = S_angl.shape[0], S_angl.shape[1]
        base_rotation = self.mean_rotation.unsqueeze(0).repeat(B,1,1,1).clone().detach()#B,P,3,3
        base_translation = self.mean_translation.unsqueeze(0).repeat(B,1,1).clone().detach()#B,P,3
        part_order = [0,1,0,2] if J == 2 else [0,1]

        res_drct = drct_rotation(I_drct, S_drct)#B,J,3,3
        res_angl = ExpSO3(S_drct.unsqueeze(2) * (S_angl - I_angl).unsqueeze(-1)) #B,J,2,3,3
        t_align, r_align = [], []

        for j in range(J):
            for a in range(2):
                part_idx = part_order[2*j+a]
                new_t = base_translation[:, part_idx].unsqueeze(-1).clone()#B,3,1
                new_t = new_t - I_joint[:,j].unsqueeze(-1)
                new_t = torch.bmm(res_drct[:,j], new_t)
                new_t = torch.bmm(res_angl[:,j,a], new_t)
                new_t = new_t + S_joint[:,j].unsqueeze(-1)
                t_align.append(new_t.squeeze(-1))
                new_r = base_rotation[:,part_idx].clone()
                new_r = torch.bmm(res_drct[:,j], new_r)
                new_r = torch.bmm(res_angl[:,j,a], new_r)
                r_align.append(new_r)
        t_align = torch.stack(t_align, dim=1)#B,2j,3
        r_align = torch.stack(r_align, dim=1)#B,2j,3,3
        #return to part base
        if J == 2:
            t_align[:,0] = 0.5*(t_align[:,0] + t_align[:,2])
            t_align = t_align[:,:3]
            r_align[:,0] = so3_mean(r_align[:,[0,2]])
            r_align = r_align[:,:3]
        t_align = t_align + T_base.unsqueeze(1)
        r_align = torch.einsum("bpij, bpjk -> bpik", R_base.unsqueeze(1), r_align)

        return r_align, t_align

def refine_r(r):
                B = r.shape[0]
                for b in range(B):
                    #svd
                    r[b] = r[b] / torch.norm(r[b], dim=0, keepdim=True)
                    u, s, vh = torch.linalg.svd(r[b])
                    r[b] = u @ vh
                    r[b] /= torch.norm(r[b], dim=0, keepdim=True)
                    #Gram–Schmidt orthonormalization with order (Y,X,Z)
                    r[b,:,0] = r[b,:,0] - torch.inner(r[b,:,0], r[b,:,1]) * r[b,:,1]
                    r[b,:,0] /= torch.norm(r[b,:,0], dim=0)
                    r[b,:,2] = r[b,:,2] - torch.inner(r[b,:,2], r[b,:,1]) * r[b,:,1]
                    r[b,:,2] /= torch.norm(r[b,:,2], dim=0)
                    r[b,:,2] = r[b,:,2] - torch.inner(r[b,:,2], r[b,:,0]) * r[b,:,0]
                    r[b,:,2] /= torch.norm(r[b,:,2], dim=0)

                    error_t = r[b] @ r[b].T - torch.eye(3, device=r.device)
                    error = torch.det(r[b]) - 1
                return r