import os
import json
from typing import Any
import numpy as np
import math
import sys
import torch
import copy
from torch.utils import data
import open3d as o3d
from glob import glob
from scipy.spatial.transform import Rotation as sciR

class RealDataset(data.Dataset):
    def __init__(
            self, root="./dataset/", npoints=1024, split='train', nmask=2, shape_type="laptop", partial = 0, args=None):
        super(RealDataset, self).__init__()

        self.root = root
        self.npoints = npoints
        self.shape_type = shape_type
        self.split = split
        self.args = args
        self.shape_root = os.path.join(self.root, shape_type, self.split)
        self.instance_name_list = glob(os.path.join(self.shape_root, '*', '*', 'input_2', '*.npz'))
        if len(self.instance_name_list) == 0:
            # Check the category name!
            raise NotImplementedError
        
        self.instance_list = []
        for i in range(len(self.instance_name_list)):
            f = self.getitem(i)
            self.instance_list.append(f)
    
    def __len__(self):
        return len(self.instance_list)
    
    def __getitem__(self, index):
        
        f = copy.deepcopy(self.instance_list[index])

        #Sampling to Input Number
        sample_idx = np.random.choice(f['pc'].shape[0], self.npoints,replace=f['pc'].shape[0] < self.npoints)
        pc = f['pc'][sample_idx,:]
        color = f['color'][sample_idx,:]
        label = f['label'][sample_idx]
        pose = f['pose_segs']
        expand = f['expand']
        center = f['center']
        part_pv_point = f['part_pv_point']
        part_axis = f['part_axis']

        #augmentation
        if self.split == 'train':
            noise = torch.randn_like(pc) * 1e-3 if np.random.rand(1) >= 0.5 else torch.zeros_like(pc)
            rotation = torch.tensor(sciR.random().as_matrix(), dtype=pc.dtype) if np.random.rand(1) >= 0.5 else torch.eye(3, dtype=pc.dtype)

            pc = torch.mm(rotation, pc.permute(1,0)).permute(1,0) + noise
            pose[:,:3,:3] = torch.einsum('bij, bjk -> bik', rotation.unsqueeze(0), pose[:,:3,:3])
            pose[:,:3,-1] = torch.mm(rotation, pose[:,:3,-1].permute(1,0)).permute(1,0)
            part_axis = torch.mm(rotation, part_axis.permute(1,0)).permute(1,0)
            part_pv_point = torch.mm(rotation, part_pv_point.permute(1,0)).permute(1,0)

        output = {
            'pc': pc,
            'color': color,
            'label': label,
            'part_pv_point': part_pv_point,
            'part_axis': part_axis,
            'pose_segs': pose,
            'expand': expand,
            'center': center,
            'idx': torch.tensor(index, dtype=torch.int64),
            'name': self.instance_name_list[index],
            'sample_idx': sample_idx,
        }

        return output

    def getitem(self, index):
        npz_path = self.instance_name_list[index]
        npz = np.load(npz_path, allow_pickle=True)['arr_0'].item()
        try:
            part_r = npz['part'][:,:9].reshape(-1,3,3)
            part_t = npz['part'][:,9:12]
            part_s = npz['part'][:,12:]
            joint_t = npz['joint'][:,:3]
            joint_d = npz['joint'][:,3:6]
        except:
            print('Load Fail at',npz_path)
            return self.getitem(index+1)

        #Capture detected valid points
        mask = np.logical_and(npz['pc'].sum(-1) != 0, npz['detection'])
        pc = npz['pc'][mask]
        color = npz['color'][mask]
        #pc = npz['pc'][npz['pc'].sum(-1) != 0,:]
        #color = npz['rgb'][npz['pc'].sum(-1) != 0,:]
        seg = npz['segmentation'][mask]

        #Normalize point cloud center
        mean_center = pc.mean(0)
        pc = pc -  mean_center.reshape(1,3)
        part_t = part_t -  mean_center.reshape(1,3)
        joint_t = joint_t -  mean_center.reshape(1,3)

        #Normalize point cloud size
        mean_center_distance = np.linalg.norm(pc, axis=-1).mean()
        pc = pc / mean_center_distance * 0.5
        part_t = part_t / mean_center_distance * 0.5
        part_s = part_s / mean_center_distance * 0.5
        joint_t = joint_t / mean_center_distance * 0.5
        
        part_rt = np.concatenate([part_r, part_t.reshape(-1,3,1)],axis=-1)
        part_rt = np.concatenate([part_rt, np.array([[[0,0,0,1]]]).repeat(part_rt.shape[0],0)], axis=1)

        instance_file = {
            'pc': torch.tensor(pc, dtype=torch.float32),
            'color': torch.tensor(color, dtype=torch.float32),
            'label': torch.tensor(seg, dtype=torch.int64), # Part Segmentation
            'part_pv_point': torch.tensor(joint_t, dtype=torch.float32), # Joint Position
            'part_axis': torch.tensor(joint_d, dtype=torch.float32), # Joint Direction
            'pose_segs': torch.tensor(part_rt, dtype=torch.float32), # Part Transformation
            'expand': torch.tensor(0.5 / mean_center_distance, dtype=torch.float32), # Normalize scale
            'center': torch.tensor(mean_center, dtype=torch.float32), # Normalize translation
            #'label': torch.zeros((pc.shape[0]), dtype=torch.int64), # Part Segmentation
            #'part_pv_point': torch.zeros((1,3), dtype=torch.float32), # Joint Position
            #'part_axis': torch.zeros((1,3), dtype=torch.float32), # Joint Direction
            #'pose_segs': torch.eye(4, dtype=torch.float32), # Part Transformation
            }
        return instance_file

    def augmentation_noise(self, pc, scale = 1e-3):
        noise = torch.randn_like(pc)
        noise *= scale
        return pc + noise
