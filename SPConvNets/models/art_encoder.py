import torch
import torch.nn as nn
import torch.nn.functional as F
from vgtk.spconv.functional import batched_index_select
import vgtk.so3conv as sptk

from SPConvNets.utils.lietorch import ExpSO3, drct_rotation, slerp
from vgtk.functional.rotation import so3_mean
from SPConvNets.models.net_bank.vanilla_pn import *
from SPConvNets.models.net_bank.slot_attention import SlotAttention

class ResnetPointnet(nn.Module):
    """PointNet-based encoder network with ResNet blocks.

    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
    """

    def __init__(self, nmask, njoint, dim=3, hidden_dim=256, rotation_range=180, joint_type='r'):
        super().__init__()
        self.nmask = nmask
        self.njoint = njoint
        self.joint_type = joint_type
        self.rotation_range = rotation_range / 180 * torch.pi
        self.fc_pos = nn.Linear(dim, 2*hidden_dim)
        self.block_0 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.block_1 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.block_2 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.block_3 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.block_4 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.bn_0 = nn.BatchNorm1d(hidden_dim)
        self.bn_1 = nn.BatchNorm1d(hidden_dim)
        self.bn_2 = nn.BatchNorm1d(hidden_dim)
        self.bn_3 = nn.BatchNorm1d(hidden_dim)
        self.bn_4 = nn.BatchNorm1d(hidden_dim)

        self.fc_s = nn.Sequential(
            nn.Conv1d(hidden_dim, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 32, 1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, nmask, 1),
            )
        self.fc_j = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.Linear(128, self.njoint*8),
            )
        self.fc_c = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 2),
            )
        
        self.actvn = nn.ReLU()
        self.pool = maxpool
        self.joint_base = nn.Parameter(torch.zeros(self.njoint, 3))
        self.drct_base = nn.Parameter(torch.ones((self.njoint,3)) / (3**0.5))
        '''
        self.sample_fc = nn.ModuleList()
        self.sample_bn = nn.ModuleList()
        for i in range(len(sample_dim)):
            self.sample_fc.append(nn.Linear(sample_dim[i]*12, hidden_dim))
            self.sample_bn.append(nn.BatchNorm1d(hidden_dim))
        
        trace_idx_ori, trace_idx_rot = sptk.get_relativeV_index()
        self.register_buffer("trace_idx_ori", torch.tensor(trace_idx_ori.swapaxes(0,1), dtype=torch.long))  # 12*60
        self.register_buffer("trace_idx_rot", torch.tensor(trace_idx_rot.swapaxes(0,1), dtype=torch.long))
        '''
    def forward(self, p1, p2, augmentation=False):
        B, D, N = p1.size()
        p = torch.stack([p1,p2],dim=1)#B,2,3,N
        p = p.reshape(2*B, 3, N)
        net = p.permute(0,2,1)

        net = self.fc_pos(net)
        net = self.block_0(net)
        net = self.bn_0(net.permute(0,2,1)).permute(0,2,1)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)
        net = self.block_1(net)
        net = self.bn_1(net.permute(0,2,1)).permute(0,2,1)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)
        net = self.block_2(net)
        net = self.bn_2(net.permute(0,2,1)).permute(0,2,1)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)
        net = self.block_3(net)
        net = self.bn_3(net.permute(0,2,1)).permute(0,2,1)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)
        net = self.block_4(net)
        net = self.bn_4(net.permute(0,2,1)).permute(0,2,1)
        net = net.permute(0,2,1)#2B,D,N
        pool = self.pool(net, dim=-1)#2B,D

        s = self.fc_s(net)
        s = F.softmax(s, dim=1) #2B,P,N
        s_1 = s.reshape(B,2,self.nmask,-1)[:,0]
        s_2 = s.reshape(B,2,self.nmask,-1)[:,1]

        j = self.fc_j(pool).reshape(-1, 8, self.njoint) #2B,8,P-1
        #j_coor_translation = j[:,:3].permute(0,2,1)#2B,P-1,3
        #j_coor = self.joint_base.unsqueeze(0) + j_coor_translation#2B,P-1,3
        j_coor = j[:,:3].permute(0,2,1)#2B,P-1,3
        #j_drct_translation = j[:,3:6].permute(0,2,1)#2B,P-1,3
        #j_drct_translation_mtx = ExpSO3(j_drct_translation)
        #drct_base_n = self.drct_base / torch.norm(self.drct_base, dim=-1, keepdim=True)
        #j_drct = torch.einsum('bpij, bpjk -> bpik', j_drct_translation_mtx, drct_base_n.reshape(1,self.nmask-1,3,1)).squeeze(-1)#2B,P-1,3
        j_drct = (j[:,3:6] / torch.norm(j[:,3:6], dim=1, keepdim=True)).permute(0,2,1)#2B,J,3
        j_angl = self.rotation_range * (torch.sigmoid(j[:,6:,:]) - 0.5).permute(0,2,1)#2B,J,2

        j_coor = j_coor.reshape(B,2,self.njoint,3)
        j_drct = j_drct.reshape(B,2,self.njoint,3)
        j_angl = j_angl.reshape(B,2,self.njoint,2)
        j_coor_1 = j_coor[:,0]
        j_coor_2 = j_coor[:,1]
        j_drct_1 = j_drct[:,0]
        j_drct_2 = j_drct[:,1]
        j_angl_1 = j_angl[:,0]
        j_angl_2 = j_angl[:,1]

        if augmentation:
            noise_1 = 1e-2 * (torch.rand_like(j_drct_1) - 0.5)
            noise_2 = 1e-2 * (torch.rand_like(j_drct_2) - 0.5)
            j_drct_1 = j_drct_1 + noise_1
            j_drct_2 = j_drct_2 + noise_2
            j_drct_1 = j_drct_1 / torch.norm(j_drct_1, dim=-1, keepdim=True)
            j_drct_2 = j_drct_2 / torch.norm(j_drct_2, dim=-1, keepdim=True)

        res_drct_1 = drct_rotation(j_drct_1, j_drct_2)#B,J,3,3
        res_angl_1 = ExpSO3(j_drct_2.unsqueeze(2) * (j_angl_2 - j_angl_1).unsqueeze(-1)) #B,J,2,3,3 For r joint only
        res_trans_1 = j_drct_2.unsqueeze(2) * (j_angl_2 - j_angl_1).unsqueeze(-1)#B,J,2,3 For p joint only
        res_drct_2 = drct_rotation(j_drct_2, j_drct_1)#B,J,3,3
        res_angl_2 = ExpSO3(j_drct_1.unsqueeze(2) * (j_angl_1 - j_angl_2).unsqueeze(-1)) #B,J,2,3,3
        res_trans_2 = j_drct_1.unsqueeze(2) * (j_angl_1 - j_angl_2).unsqueeze(-1)#B,J,2,3 For p joint only

        p1_align = []
        for p in range(self.njoint):
            for a in range(2):
                new_p1 = p1.clone()#B,3,N
                new_p1 = new_p1 - j_coor_1[:,p].unsqueeze(-1)
                new_p1 = torch.bmm(res_drct_1[:,p], new_p1)
                if self.joint_type == 'r':
                    new_p1 = torch.bmm(res_angl_1[:,p,a], new_p1)
                elif self.joint_type == 'p':
                    new_p1 = new_p1 + res_trans_1[:,p,a].unsqueeze(-1)
                new_p1 = new_p1 + j_coor_2[:,p].unsqueeze(-1)
                p1_align.append(new_p1)
        p1_align = torch.stack(p1_align, dim=1)#B,2J,3,N

        p2_align = []
        for p in range(self.njoint):
            for a in range(2):
                new_p2 = p2.clone()#B,3,N
                new_p2 = new_p2 - j_coor_2[:,p].unsqueeze(-1)
                new_p2 = torch.bmm(res_drct_2[:,p], new_p2)
                if self.joint_type == 'r':
                    new_p2 = torch.bmm(res_angl_2[:,p,a], new_p2)
                elif self.joint_type == 'p':
                    new_p2 = new_p2 + res_trans_2[:,p,a].unsqueeze(-1)
                new_p2 = new_p2 + j_coor_1[:,p].unsqueeze(-1)
                p2_align.append(new_p2)
        p2_align = torch.stack(p2_align, dim=1)#B,J,3,N

        return p1_align, s_1, j_coor_1, j_drct_1, j_angl_1, p2_align, s_2, j_coor_2, j_drct_2, j_angl_2
