import os
import json
from typing import Any
import numpy as np
import math
import sys
import torch
from torch.utils import data
import open3d as o3d
from glob import glob

class LightDataset(data.Dataset):
    def __init__(
            self, root="./data/pc", npoints=1024, split='train', nmask=2, shape_type="laptop", partial = 0, args=None):
        super(LightDataset, self).__init__()

        self.root = root
        self.npoints = npoints
        self.shape_type = shape_type
        self.split = split
        self.partial = partial
        self.args = args
        if self.partial:
            self.shape_root = os.path.join(self.root, 'partial', shape_type)
        else:
            self.shape_root = os.path.join(self.root, 'full', shape_type)
        self.instance_list = glob(os.path.join(self.shape_root, split, '*.pt'))

        if len(self.instance_list) == 0:
            print("Check the category name!")
            raise NotImplementedError
        
        return
    
    def __len__(self):
        return len(self.instance_list)
    
    def __getitem__(self, index):
        instance_path = self.instance_list[index]
        instance_file = torch.load(instance_path)
        #Dummy color
        instance_file['color'] = 0.5 * torch.ones_like(instance_file['pc'])

        #Normalize point cloud center / scale
        pc = instance_file['pc']
        pc = (pc -  pc.mean(0, keepdims=True))
        mean_center_distance = np.linalg.norm(pc, axis=-1).mean()
        #print(instance_file['idx'].item(), mean_center_distance)
        instance_file['pc'] = instance_file['pc'] / mean_center_distance * 0.25
        instance_file['pose_segs'][:,:3,-1] = instance_file['pose_segs'][:,:3,-1] / mean_center_distance * 0.25
        instance_file['part_pv_point'] = instance_file['part_pv_point'] / mean_center_distance * 0.25


        return instance_file

