

import os
# import os
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
import os.path
import json
import numpy as np
import math
import sys
import torch
import vgtk.so3conv.functional as L
# import vgtk.pc as pctk
from scipy.spatial.transform import Rotation as sciR
from SPConvNets.datasets.part_transform import revoluteTransform
from SPConvNets.models.model_util import *
import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
# import provider
from torch.utils import data
from SPConvNets.models.common_utils import *
from SPConvNets.datasets.data_utils import *
import scipy.io as sio
import copy
import trimesh
import pyrender
# from model.utils import farthest_point_sampling

# padding 1
def padding_1(pos):
    pad = np.array([1.], dtype=np.float).reshape(1, 1)
    # print(pos.shape, pad.shape)
    return np.concatenate([pos, pad], axis=1)

# normalize point-cloud
def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

# decod rotation info
def decode_rotation_info(rotate_info_encoding):
    if rotate_info_encoding == 0:
        return []
    rotate_vec = []
    if rotate_info_encoding <= 3:
        temp_angle = np.reshape(np.array(np.random.rand(3)) * np.pi, (3, 1))
        if rotate_info_encoding == 1:
            line_vec = np.concatenate([
                np.cos(temp_angle), np.zeros_like(temp_angle), np.sin(temp_angle),
            ], axis=-1)
        elif rotate_info_encoding == 2:
            line_vec = np.concatenate([
                np.cos(temp_angle), np.sin(temp_angle), np.zeros_like(temp_angle)
            ], axis=-1)
        else:
            line_vec = np.concatenate([
                np.zeros_like(temp_angle), np.cos(temp_angle), np.sin(temp_angle)
            ], axis=-1)
        return [line_vec[0], line_vec[1], line_vec[2]]
    elif rotate_info_encoding <= 6:
        base_rotate_vec = [np.array([1.0, 0.0, 0.0], dtype=np.float),
                           np.array([0.0, 1.0, 0.0], dtype=np.float),
                           np.array([0.0, 0.0, 1.0], dtype=np.float)]
        if rotate_info_encoding == 4:
            return [base_rotate_vec[0], base_rotate_vec[2]]
        elif rotate_info_encoding == 5:
            return [base_rotate_vec[0], base_rotate_vec[1]]
        else:
            return [base_rotate_vec[1], base_rotate_vec[2]]
    else:
        return []

def rotate_by_vec_pts(un_w, p_x, bf_rotate_pos):

    def get_zero_distance(p, xyz):
        k1 = np.sum(p * xyz).item()
        k2 = np.sum(xyz ** 2).item()
        t = -k1 / (k2 + 1e-10)
        p1 = p + xyz * t
        # dis = np.sum(p1 ** 2).item()
        return np.reshape(p1, (1, 3))

    w = un_w / np.sqrt(np.sum(un_w ** 2, axis=0))
    # w = np.array([0, 0, 1.0])
    w_matrix = np.array(
        [[0, -float(w[2]), float(w[1])], [float(w[2]), 0, -float(w[0])], [-float(w[1]), float(w[0]), 0]]
    )

    rng = 0.25
    offset = 0.1

    effi = np.random.uniform(-rng, rng, (1,)).item()
    # effi = effis[eff_id].item()
    if effi < 0:
        effi -= offset
    else:
        effi += offset
    theta = effi * np.pi
    # rotation_matrix = np.exp(w_matrix * theta)

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    # rotation_matrix = np.eye(3) + w_matrix * sin_theta + (w_matrix ** 2) * (1. - cos_theta)
    rotation_matrix = np.eye(3) + w_matrix * sin_theta + (w_matrix.dot(w_matrix)) * (1. - cos_theta)

    # bf_rotate_pos = pcd_points[sem_label_to_idxes[rotate_idx][rotate_idx_inst]]

    trans = get_zero_distance(p_x, un_w)

    af_rotate_pos = np.transpose(np.matmul(rotation_matrix, np.transpose(bf_rotate_pos - trans, [1, 0])), [1, 0]) + trans

    # af_rotate_pos = rotation_matrix.dot((bf_rotate_pos - trans).T).T + trans
    return af_rotate_pos, rotation_matrix, np.reshape(trans, (3, 1))

def inv_triangles_to_seg_idx(triangles_to_seg_idx):
    # triangles_to_seg_idx: N_tri
    seg_idx_to_triangles_idxes = {}
    for i_t in range(triangles_to_seg_idx.shape[0]):
        cur_tri_seg_idx = int(triangles_to_seg_idx[i_t].item())
        if cur_tri_seg_idx not in seg_idx_to_triangles_idxes:
            seg_idx_to_triangles_idxes[cur_tri_seg_idx] = [i_t]
        else:
            seg_idx_to_triangles_idxes[cur_tri_seg_idx].append(i_t)
    for seg_idx in seg_idx_to_triangles_idxes:
        seg_idx_to_triangles_idxes[seg_idx] = np.array(seg_idx_to_triangles_idxes[seg_idx], dtype=np.long) # N_seg_tri
    return seg_idx_to_triangles_idxes

def reindex_triangles_vertices(vertices, triangles):
    vert_idx_to_new_idx = {}
    new_idx = 0
    for i_tri in range(triangles.shape[0]):
        cur_tri = triangles[i_tri]
        v1, v2, v3 = int(cur_tri[0].item()), int(cur_tri[1].item()), int(cur_tri[2].item())
        for v in [v1, v2, v3]:
            if v not in vert_idx_to_new_idx:
                vert_idx_to_new_idx[v] = new_idx
                new_idx = new_idx + 1
    new_vertices = np.zeros((new_idx, 3), dtype=np.float)
    for vert_idx in vert_idx_to_new_idx:
        new_vert_idx = vert_idx_to_new_idx[vert_idx]
        new_vertices[new_vert_idx] = vertices[vert_idx]
    new_triangles = np.zeros_like(triangles)
    for i_tri in range(triangles.shape[0]):
        cur_tri = triangles[i_tri]
        v1, v2, v3 = int(cur_tri[0].item()), int(cur_tri[1].item()), int(cur_tri[2].item())
        new_v1, new_v2, new_v3 = vert_idx_to_new_idx[v1], vert_idx_to_new_idx[v2], vert_idx_to_new_idx[v3]
        cur_new_tri = np.array([new_v1, new_v2, new_v3], dtype=np.long)
        new_triangles[i_tri] = cur_new_tri
    # new_triangles: tot_n_tri x 3
    # new_vertices: tot_n_vert x 3
    return new_vertices, new_triangles

def ndc_depth_to_buffer(z, near, far):  # z in [-1, 1]
    return 2 * near * far / (near + far - z * (far - near))


def buffer_depth_to_ndc(d, near, far):  # d in (0, +
    return ((near + far) - 2 * near * far / np.clip(d, a_min=1e-6, a_max=1e6)) / (far - near)



def create_partial_pts(mean_pose=np.array([0, 0, -1.8]), std_pose=np.array([0.2, 0.2, 0.15]),
                   yfov=np.deg2rad(60), pw=640, ph=480, near=0.1, far=10, upper_hemi=False, vertices=None, triangles=None, seg_label_to_triangles=None, seg_transformation_mtx=None, render_img=True, no_transformation=False):
    # seg_transformation_mtx: n_seg x 4 x 4
    # if vertices is not None and triangles is not None:
    #     assert seg_label_to_triangles is not None and seg_transformation_mtx is not None
    #     m = trimesh.base.Trimesh(vertices=vertices, faces=triangles)
    # else:
    #     m = trimesh.load(read_path)

    # centralization
    # c = np.mean(m.vertices, axis=0)
    # trans = np.eye(4)
    # trans[:3, 3] = -c
    # m.apply_transform(trans) # pose of each part --- should be remembered and transformed for the further use
    # for further
    # centralize
    # seg_transformation_mtx[:, :3, 3] = seg_transformation_mtx[:, :3, 3] + np.reshape(trans, (1, 3))
    seg_label_to_depth_buffer = {}
    seg_label_to_new_pose = {}
    seg_label_to_idxes = {}
    tot_depth_buffer = []
    tot_n = 0
    seg_label_to_pts = {}
    seg_label_to_no_rot_pts = {}

    if not no_transformation:

        axis_angle = np.random.randint(0, 50, (3,))

        x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float)
        y_axis = np.array([0.0, 1.0, 0.0], dtype=np.float)
        z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float)

        ''' Random angle strategy 1 '''
        # x_angle = -1.0 * float(
        #     axis_angle[0].item()) / 100. * np.pi  # [-0.5, 0] -> sampled rotation angle around the x-axis
        # # y_angle = float(axis_angle[0].item()) / 50. * np.pi - 0.5 * np.pi
        # # z_angle = float(axis_angle[0].item()) / 50. * np.pi - 0.5 * np.pi
        # # ''' Strategy 1 -- '''
        # y_angle = 1.0 * float(axis_angle[1].item()) / 100. * np.pi  # - 0.5 * np.pi
        # # z_angle = 1.0 * float(axis_angle[2].item()) / 100. * np.pi  # - 0.5 * np.pi
        # z_angle = 0.0  # - 0.5 * np.pi
        ''' Random angle strategy 1 '''

        ''' Random angle strategy 2 '''
        x_angle = 0.0 # -1.0 * float(axis_angle[0].item()) / 100. * np.pi  # [-0.5, 0] -> sampled rotation angle
        # around the x-axis
        ''' Washing machine '''
        x_angle = -0.2 * np.pi # washing...
        # y_angle = 1.0 * float(axis_angle[1].item()) / 100. * np.pi - 0.25 * np.pi  # - 0.5 * np.pi
        y_angle = 1.0 * float(axis_angle[1].item()) / 300. * np.pi # - 0.25 * np.pi  # - 0.5 * np.pi
        # z_angle = 1.0 * float(axis_angle[2].item()) / 100. * np.pi  # - 0.5 * np.pi
        # z_angle = 0.0  # - 0.5 * np.pi
        # z_angle = 1.0 * float(axis_angle[2].item()) / 100. * np.pi - 0.25 * np.pi
        z_angle = 1.0 * float(axis_angle[2].item()) / 300. * np.pi # - 0.25 * np.pi
        ''' Washing machine '''

        ''' Oven --- v1 '''
        axis_angle = np.random.randint(0, 100, (3,))
        x_angle = 0.0 * np.pi
        y_angle = 1.0 * float(axis_angle[1].item()) / 100. * np.pi - 0.5 * np.pi
        z_angle = 1.0 * float(axis_angle[2].item()) / 100. * np.pi - 0.5 * np.pi
        ''' Oven --- v1 '''

        ''' Oven --- v2 (small range of view change) --- oven --- no axis/pv p '''
        axis_angle = np.random.randint(0, 100, (3,))
        x_angle = 0.0
        x_angle = 1.0 * float(axis_angle[2].item()) / 800. * np.pi  # - 0.5 * np.pi
        # y_angle = -1.0 * float(axis_angle[1].item()) / 400. * np.pi - 1.0 / 6.0 * np.pi
        y_angle = 1.0 * float(axis_angle[1].item()) / 400. * np.pi - 1.0 / 8.0 * np.pi
        # y_angle = 0.0
        # z_angle = -1.0 * float(axis_angle[2].item()) / 400. * np.pi
        # z_angle = 1.0 * float(axis_angle[2].item()) / 400. * np.pi - 1.0 / 8.0 * np.pi  # - 0.5 * np.pi
        z_angle = 1.0 * float(axis_angle[2].item()) / 800. * np.pi  # - 0.5 * np.pi
        # z_angle = 0.0  # - 0.5 * np.pi
        ''' Oven --- v2 (small range of view change) --- oven --- no axis/pv p '''

        ''' Oven --- v2 (small range of view change) --- oven --- axis/pv p -- ok '''
        axis_angle = np.random.randint(0, 100, (3,))
        x_angle = 0.0
        # x_angle = 1.0 * float(axis_angle[2].item()) / 800. * np.pi  # - 0.5 * np.pi
        # y_angle = -1.0 * float(axis_angle[1].item()) / 400. * np.pi - 1.0 / 6.0 * np.pi
        y_angle = 1.0 * float(axis_angle[1].item()) / 400. * np.pi - 1.0 / 8.0 * np.pi
        # y_angle = 0.0
        # z_angle = -1.0 * float(axis_angle[2].item()) / 400. * np.pi
        # z_angle = 1.0 * float(axis_angle[2].item()) / 400. * np.pi - 1.0 / 8.0 * np.pi  # - 0.5 * np.pi
        # z_angle = 1.0 * float(axis_angle[2].item()) / 800. * np.pi  # - 0.5 * np.pi
        z_angle = 0.0  # - 0.5 * np.pi
        ''' Oven --- v2 (small range of view change) --- oven --- axis/pv p -- ok'''

        ''' Washing machine --- v1 --- no axis/pv p '''
        # # with sel-mode = 29 --> iou ~ 52+... global alignment: /share/xueyi/ckpt/playground/model_20220430_10:56:55/ckpt/playground_net_Iter1600.pth
        # axis_angle = np.random.randint(0, 100, (3,))
        # # x_angle = 0.0
        # x_angle = 1.0 / 8.0 * np.pi
        # # x_angle = -1.0 / 8.0 * np.pi
        # # x_angle = 1.0 * float(axis_angle[1].item()) / 400. * np.pi - 1.0 / 8.0 * np.pi
        # # x_angle = 1.0 * float(axis_angle[1].item()) / 800. * np.pi - 1.0 / 16.0 * np.pi
        # # x_angle = 1.0 * float(axis_angle[1].item()) / 1600. * np.pi + 1.0 / 16.0 * np.pi
        #
        # # x_angle = -1.0 * float(axis_angle[1].item()) / 1600. * np.pi - 1.0 / 16.0 * np.pi # if we want to change x-angle...
        #
        # # x_angle = 1.0 * float(axis_angle[2].item()) / 800. * np.pi  # - 0.5 * np.pi
        # # y_angle = -1.0 * float(axis_angle[1].item()) / 400. * np.pi - 1.0 / 6.0 * np.pi
        #
        # # y_angle = 1.0 * float(axis_angle[1].item()) / 400. * np.pi - 1.0 / 8.0 * np.pi # for global alignment
        # # y_angle = 1.0 * float(axis_angle[1].item()) / 1600. * np.pi + 1.0 / 16.0 * np.pi
        #
        # # y_angle = 1.0 * float(axis_angle[1].item()) / 800. * np.pi - 1.0 / 16.0 * np.pi
        # # y_angle = 1.0 * float(axis_angle[1].item()) / 1600. * np.pi + 1.0 / 16.0 * np.pi
        #
        # y_angle = 1.0 / 8.0 * np.pi
        # # y_angle = 0.0
        # # z_angle = -1.0 * float(axis_angle[2].item()) / 400. * np.pi
        # # z_angle = 1.0 * float(axis_angle[2].item()) / 400. * np.pi - 1.0 / 8.0 * np.pi  # - 0.5 * np.pi # for global alignment
        # # z_angle = 1.0 * float(axis_angle[2].item()) / 800. * np.pi - 1.0 / 16.0 * np.pi  # - 0.5 * np.pi # for global alignment
        # # z_angle = 1.0 * float(axis_angle[2].item()) / 800. * np.pi  # - 0.5 * np.pi
        # # z_angle = 0.0  # - 0.5 * np.pi
        # z_angle = 0.0
        ''' Washing machine --- v1 --- no axis/pv p '''

        ''' eyeglasses '''
        # axis_angle = np.random.randint(0, 100, (3,))
        # x_angle = 1.0 * float(axis_angle[0].item()) / 100. * np.pi  - 0.5 * np.pi # [-0.5, 0] -> sampled rotation angle around the x-axis
        # # y_angle = float(axis_angle[0].item()) / 50. * np.pi - 0.5 * np.pi
        # # z_angle = float(axis_angle[0].item()) / 50. * np.pi - 0.5 * np.pi
        # # ''' Strategy 1 -- '''
        # y_angle = 1.0 * float(axis_angle[1].item()) / 100. * np.pi  - 0.5 * np.pi
        # z_angle = 1.0 * float(axis_angle[2].item()) / 100. * np.pi  - 0.5 * np.pi
        # # z_angle = 0.0  # - 0.5 * np.pi
        ''' eyeglasses '''


        # y_angle = 0.0
        # z_angle = 0.0
        ''' Random angle strategy 2 '''

        ''' Oven --- v2 (small range of view change) --- oven --- axis/pv p for vis only '''
        # axis_angle = np.random.randint(0, 100, (3,))
        # x_angle =  1.0 * float(axis_angle[0].item()) / 200. * np.pi - 1.0 / 4.0 * np.pi
        # # x_angle = 1.0 * float(axis_angle[2].item()) / 800. * np.pi  # - 0.5 * np.pi
        # # y_angle = -1.0 * float(axis_angle[1].item()) / 400. * np.pi - 1.0 / 6.0 * np.pi
        # y_angle = 1.0 * float(axis_angle[1].item()) / 200. * np.pi - 1.0 / 4.0 * np.pi
        # # y_angle = 0.0
        # # z_angle = -1.0 * float(axis_angle[2].item()) / 400. * np.pi
        # # z_angle = 1.0 * float(axis_angle[2].item()) / 400. * np.pi - 1.0 / 8.0 * np.pi  # - 0.5 * np.pi
        # # z_angle = 1.0 * float(axis_angle[2].item()) / 800. * np.pi  # - 0.5 * np.pi
        # z_angle =  1.0 * float(axis_angle[2].item())  / 200. * np.pi - 1.0 / 4.0 * np.pi  # - 0.5 * np.pi
        ''' Oven --- v2 (small range of view change) --- oven --- axis/pv p for vis only '''

        ''' Oven --- v2 (small range of view change) --- oven --- axis/pv p for vis only '''
        axis_angle = np.random.randint(0, 100, (3,))
        # x_angle = 1.0 * float(axis_angle[0].item()) / 400. * np.pi - 1.0 / 8.0 * np.pi
        # x_angle = 1.0 * float(axis_angle[2].item()) / 800. * np.pi  # - 0.5 * np.pi
        # y_angle = -1.0 * float(axis_angle[1].item()) / 400. * np.pi - 1.0 / 6.0 * np.pi
        y_angle = 1.0 * float(axis_angle[1].item()) / 400. * np.pi - 1.0 / 8.0 * np.pi
        # y_angle = 0.0
        # z_angle = -1.0 * float(axis_angle[2].item()) / 400. * np.pi
        # z_angle = 1.0 * float(axis_angle[2].item()) / 400. * np.pi - 1.0 / 8.0 * np.pi  # - 0.5 * np.pi
        # z_angle = 1.0 * float(axis_angle[2].item()) / 800. * np.pi  # - 0.5 * np.pi
        # z_angle = 1.0 * float(axis_angle[2].item()) / 400. * np.pi - 1.0 / 8.0 * np.pi  # - 0.5 * np.pi
        ''' Oven --- v2 (small range of view change) --- oven --- axis/pv p for vis only '''

        ''' Washing_machine --- v2 (small range of view change) --- oven --- axis/pv p for vis only '''
        axis_angle = np.random.randint(0, 100, (3,))
        # x_angle = 1.0 * float(axis_angle[0].item()) / 400. * np.pi - 1.0 / 8.0 * np.pi
        x_angle = 1.0 * float(axis_angle[0].item()) / 1600. * np.pi + 1.0 / 16.0 * np.pi
        # x_angle = 0.0
        # x_angle = 1.0 * float(axis_angle[2].item()) / 800. * np.pi  # - 0.5 * np.pi
        # y_angle = -1.0 * float(axis_angle[1].item()) / 400. * np.pi - 1.0 / 6.0 * np.pi
        # y_angle = 1.0 * float(axis_angle[1].item()) / 400. * np.pi - 1.0 / 8.0 * np.pi
        # y_angle = 1.0 * float(axis_angle[1].item()) / 800. * np.pi - 1.0 / 16.0 * np.pi
        y_angle = 0.0
        # z_angle = -1.0 * float(axis_angle[2].item()) / 400. * np.pi
        # z_angle = 1.0 * float(axis_angle[2].item()) / 400. * np.pi - 1.0 / 8.0 * np.pi  # - 0.5 * np.pi
        # z_angle = 1.0 * float(axis_angle[2].item()) / 800. * np.pi  # - 0.5 * np.pi
        # z_angle = 1.0 * float(axis_angle[2].item()) / 400. * np.pi - 1.0 / 8.0 * np.pi  # - 0.5 * np.pi
        z_angle = 0.0
        ''' Washing_machine --- v2 (small range of view change) --- oven --- axis/pv p for vis only '''

        # axis_angle = np.random.randint(0, 100, (3,))
        # x_angle = 0.0
        # # x_angle = 1.0 * float(axis_angle[2].item()) / 800. * np.pi  # - 0.5 * np.pi
        # # y_angle = -1.0 * float(axis_angle[1].item()) / 400. * np.pi - 1.0 / 6.0 * np.pi
        # y_angle = 1.0 * float(axis_angle[1].item()) / 400. * np.pi - 1.0 / 8.0 * np.pi
        # # y_angle = 0.0
        # # z_angle = -1.0 * float(axis_angle[2].item()) / 400. * np.pi
        # # z_angle = 1.0 * float(axis_angle[2].item()) / 400. * np.pi - 1.0 / 8.0 * np.pi  # - 0.5 * np.pi
        # # z_angle = 1.0 * float(axis_angle[2].item()) / 800. * np.pi  # - 0.5 * np.pi
        # z_angle = 0.0  # - 0.5 * np.pi
        # rotation = np.eye(3)

        axis_angle = np.random.randint(0, 100, (3,))
        x_angle = 1.0 * float(axis_angle[0].item()) / 100. * np.pi  - 0.5 * np.pi # [-0.5, 0] -> sampled rotation angle around the x-axis
        # y_angle = float(axis_angle[0].item()) / 50. * np.pi - 0.5 * np.pi
        # z_angle = float(axis_angle[0].item()) / 50. * np.pi - 0.5 * np.pi
        # ''' Strategy 1 -- '''
        y_angle = 1.0 * float(axis_angle[1].item()) / 100. * np.pi  - 0.5 * np.pi
        z_angle = 1.0 * float(axis_angle[2].item()) / 100. * np.pi  - 0.5 * np.pi
        # z_angle = 0.0  # - 0.5 * np.pi

        axis_angle = np.random.randint(0, 100, (3,))
        x_angle = 1.0 * float(axis_angle[0].item()) / 100. * np.pi  - 0.5 * np.pi # [-0.5, 0] -> sampled rotation angle around the x-axis
        # y_angle = float(axis_angle[0].item()) / 50. * np.pi - 0.5 * np.pi
        # z_angle = float(axis_angle[0].item()) / 50. * np.pi - 0.5 * np.pi
        # ''' Strategy 1 -- '''
        y_angle = 1.0 * float(axis_angle[1].item()) / 100. * np.pi  - 0.5 * np.pi
        z_angle = 1.0 * float(axis_angle[2].item()) / 100. * np.pi  - 0.5 * np.pi
        # z_angle = 0.0  # - 0.5 * np.pi

        x_mtx = compute_rotation_matrix_from_axis_angle(x_axis, x_angle)
        y_mtx = compute_rotation_matrix_from_axis_angle(y_axis, y_angle)
        z_mtx = compute_rotation_matrix_from_axis_angle(z_axis, z_angle)

        rotation = np.matmul(y_mtx, np.matmul(x_mtx, z_mtx))

        # rotation_angle = sciR.random().as_matrix()
        # rotation_matrix = rotation_angle[:3, :3]
        # R1 = rotation_matrix
        # rotation = R1

        # rotation = np.eye(3)

        # rotation = trimesh.transformations.random_rotation_matrix()[:3, :3]
        # while upper_hemi and (rotation[1, 2] < 0 or rotation[2, 2] < 0):
        #     rotation = trimesh.transformations.random_rotation_matrix()[:3, :3]

        # rotation = trimesh.transformations.random_rotation_matrix()[:3, :3]
        # while upper_hemi and (rotation[1, 1] < 0 or rotation[2, 1] < 0):
        #     rotation = trimesh.transformations.random_rotation_matrix()[:3, :3]

        randd = np.random.randn(3)
        pose_trans = mean_pose # + randd * std_pose
    else:
        rotation = np.eye(3)
        # pose_trans = np.zeros((3,))
        pose_trans = mean_pose
    # pose_trans = pose_trans * 0.0
    glb_pose = {'rotation': rotation, 'trans': pose_trans}
    for seg_label in seg_label_to_triangles:
        # cur_seg_triangles: n_tri
        cur_seg_triangles = seg_label_to_triangles[seg_label]
        cur_seg_triangles = triangles[cur_seg_triangles]
        # n_tri --> indexes of triangles...
        cur_seg_vertices, cur_seg_triangles = reindex_triangles_vertices(vertices, cur_seg_triangles)
        m = trimesh.base.Trimesh(vertices=cur_seg_vertices, faces=cur_seg_triangles, validate=True)
        trimesh.repair.fix_inversion(m, multibody=True)
        ''' Setup scene '''
        scene = pyrender.Scene()

        ''' Add mesh to the scene '''
        # add mesh to the node
        mesh = pyrender.Mesh.from_trimesh(m)
        node = pyrender.Node(mesh=mesh, matrix=np.eye(4))
        # add node to the scene
        scene.add_node(node)

        camera_pose = np.eye(4)
        camera = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=pw / ph, znear=near, zfar=far)
        projection = camera.get_projection_matrix()
        scene.add(camera, camera_pose)
        r = pyrender.OffscreenRenderer(pw, ph)

        # print(f"cur_seg_label: {seg_label}, projection matrix: {projection}")

        ''' Set pose '''
        pose = np.eye(4)  # init pose
        # pose[:3, 3] = mean_pose + randd * std_pose  # translation?
        pose[:3, 3] = pose_trans  # translation?
        # rotation = trimesh.transformations.random_rotation_matrix()[:3, :3]
        # while upper_hemi and (rotation[1, 2] < 0 or rotation[2, 2] < 0):
        #     rotation = trimesh.transformations.random_rotation_matrix()[:3, :3]
        pose[:3, :3] = rotation # rotation matrix

        cur_seg_art_rot = seg_transformation_mtx[seg_label, :3, :3]
        cur_seg_art_trans = seg_transformation_mtx[seg_label, :3, 3]
        cur_seg_rot = np.matmul(rotation, cur_seg_art_rot)
        cur_seg_trans = np.matmul(rotation, np.reshape(cur_seg_art_trans, (3, 1))) + np.reshape(pose[:3, 3], (3, 1))
        cur_seg_pose = np.eye(4)
        cur_seg_pose[:3, :3] = cur_seg_rot
        cur_seg_pose[:3, 3] = cur_seg_trans[:, 0]

        ''' Get new pose for this seg... '''
        seg_label_to_new_pose[seg_label] = cur_seg_pose

        scene.set_pose(node, pose)  # set pose for the node # the rotation?  --> rotate

        depth_buffer = r.render(scene, flags=pyrender.constants.RenderFlags.DEPTH_ONLY)  # render the depth buffer
        # pts = backproject(depth_buffer, projection, near, far, from_image=False)
        mask = depth_buffer > 0  # get the mask for valied area

        seg_label_to_depth_buffer[seg_label] = depth_buffer

        cur_seg_pts_idxes = np.array([ii for ii in range(tot_n, tot_n + depth_buffer.shape[0])], dtype=np.long)
        tot_n = tot_n + depth_buffer.shape[0]
        tot_depth_buffer.append(depth_buffer)
        seg_label_to_idxes[seg_label] = cur_seg_pts_idxes

        depth = depth_buffer

        proj_inv = np.linalg.inv(projection)
        height, width = depth.shape
        non_zero_mask = (depth > 0)
        idxs = np.where(non_zero_mask)
        depth_selected = depth[idxs[0], idxs[1]].astype(np.float32).reshape((1, -1))

        d = depth_selected
        z = buffer_depth_to_ndc(d, near, far)

        grid = np.array([idxs[1] / width * 2 - 1, 1 - idxs[0] / height * 2])  # ndc [-1, 1]

        ones = np.ones_like(z)
        # depth
        pts = np.concatenate((grid, z, ones), axis=0) * d  # before dividing by w, w = -z_world = d

        pts = proj_inv @ pts  # back project points
        pts = np.transpose(pts)  # transpose points

        pts = pts[:, :3]

        # inv transform points
        # no_rot_pts = np.matmul(np.transpose(cur_seg_rot, (1, 0)), np.transpose(pts - np.transpose(cur_seg_trans, (1, 0)), (1, 0)))
        # no_rot_pts = np.transpose(no_rot_pts, (1, 0))
        # # pts = pts - np.mean(pts, axis=0, keepdims=True)
        seg_label_to_pts[seg_label] = pts
        # seg_label_to_no_rot_pts[seg_label] = no_rot_pts
    return seg_label_to_pts, seg_label_to_new_pose, glb_pose



class MotionDataset(data.Dataset):
    def __init__(
            self, root="./data/MDV02", npoints=512, split='train', nmask=10, shape_type="laptop", args=None, global_rot=0
    ):
        super(MotionDataset, self).__init__()

        self.root = root
        self.npoints = npoints
        self.shape_type = shape_type
        self.shape_root = os.path.join(self.root, shape_type)
        self.args = args

        self.mesh_fn = "summary.obj"
        self.surface_to_seg_fn = "sfs_idx_to_dof_name_idx.npy"
        self.attribute_fn = "motion_attributes.json"

        self.global_rot = global_rot
        self.split = split
        self.rot_factor = self.args.equi_settings.rot_factor
        self.pre_compute_delta = self.args.equi_settings.pre_compute_delta
        self.use_multi_sample = self.args.equi_settings.use_multi_sample
        self.n_samples = self.args.equi_settings.n_samples if self.use_multi_sample == 1 else 1
        self.partial = self.args.equi_settings.partial
        self.est_normals = self.args.equi_settings.est_normals



        if self.pre_compute_delta == 1 and self.split == 'train':
            # self.use_multi_sample = False
            self.use_multi_sample = 0
            self.n_samples = 1

        #print(f"no_articulation: {self.no_articulation}, equi_settings: {self.args.equi_settings.no_articulation}")

        self.train_ratio = 0.9

        self.shape_folders = []

        shape_idxes = os.listdir(self.shape_root)
        shape_idxes = sorted(shape_idxes)
        shape_idxes = [tmpp for tmpp in shape_idxes if tmpp[0] != "."]

        # if self.shape_type == "oven":
        #     # not_ok_shape_idxes = [2, 3, 4, 6, 9, 15, 18, 23, 24, 28, 33, 34, 38]
        #     not_ok_shape_idxes = [32, 35, 33, 42, 38, 36, 6, 41, 4, 15]
        #     not_ok_shape_idxes = [4, 35, 38, 6, 15] # 10, 38, 31, 6, 15, 4, 35
        #     not_ok_shape_idxes = [10, 38, 31, 6, 15, 4, 35]
        #     not_ok_shape_idxes = [32, 35, 33, 18, 10, 21, 36, 31, 6, 24, 15, 40, 13, 38, 4] + [42, 30, 41, 22, 37, 39, 28, 29, 27, 26] + [5, 2, 20, 11, 28, 17, 10, 36, 9, 30, 8, 1, 6, 24, 15, 14, 7, 12, 16]
        #     # not_ok_shape_idxes = []
        #     not_ok_shape_idxes = ["%.4d" % iii for iii in not_ok_shape_idxes]
        #     not_ok_shape_idx_to_va = {str(iii): 1 for iii in not_ok_shape_idxes}
        #     shape_idxes = [si for si in shape_idxes if si not in not_ok_shape_idx_to_va]
        # elif self.shape_type == "washing_machine":
        #     not_ok_shape_idxes = [61, 59, 50, 57, 56, 58, 60, 2, 7, 38, 9, 31, 62, 53, 30, 1, 39, 52, 55, 46, 41, 40, 18,]
        #     not_ok_shape_idxes = [50, 58, 2, 18, 29, 10, 7, 38, 9, 31, 30, 1, 39, 52, 46, 41, 40, 13]
        #     not_ok_shape_idxes = [50, 58, 34, 2, 18, 27, 11, 29, 45, 28, 10, 21, 7, 38, 9, 31, 62, 30, 37, 8, 1, 6, 39, 52, 46, 41, 24, 12, 40, 13, 61, 59, 32, 35, 59] + [4, 44] + [55, 25] + [56, 51, 5, 20, 42, 44, 54, 55] + [33, 5, 16, 43, 55, 48, 23, 15, 47, 25] + [57, 3, 60, 19] # the last two are for depth...
        #     not_ok_shape_idxes = ["%.4d" % iii for iii in not_ok_shape_idxes]
        #     not_ok_shape_idx_to_va = {str(iii): 1 for iii in not_ok_shape_idxes}
        #     shape_idxes = [si for si in shape_idxes if si not in not_ok_shape_idx_to_va]
        # elif self.shape_type == "eyeglasses":
        #     not_ok_shape_idxes = [42]
        #     not_ok_shape_idxes = [42, 20, 10]
        #     not_ok_shape_idxes = [42, 20, 10, 3, 2, 9, 1] # some are added due to their large category-level distribution gap
        #     not_ok_shape_idxes = [3, 33, 5, 2, 20, 11, 29, 16, 42, 28, 10, 19, 7, 38, 9, 30, 8, 1, 6, 41, 23, 15, 12, 13, 14]
        #     not_ok_shape_idxes = [3, 33, 5, 2, 20, 11, 29, 16, 42, 28, 10, 19, 7, 38, 9, 30, 8, 1, 6, 41, 23, 15, 12, 13, 14, 4, 32, 28, 17] + [27, 36] + [40]
        #     not_ok_shape_idxes = ["%.4d" % iii for iii in not_ok_shape_idxes]
        #     not_ok_shape_idx_to_va = {str(iii): 1 for iii in not_ok_shape_idxes}
        #     shape_idxes = [si for si in shape_idxes if si not in not_ok_shape_idx_to_va]
        # elif self.shape_type == "laptop":
        #     not_ok_shape_idxes = [83, 23]
        #     not_ok_shape_idxes = [86, 83, 23, 73] + [3, 62, 1, 14]
        #     mode_1_idxes = [61, 50, 35, 34, 29, 72, 54, 8, 55, 79, 77, 85, 76]
        #     mode_3_idxes = [57, 3, 51, 60, 33, 45, 17, 81, 86, 7, 31, 30, 63, 15, 71, 49, 40, 14, 77, 80, 81]
        #     not_ok_shape_idxes = ["%.4d" % iii for iii in not_ok_shape_idxes]
        #     not_ok_shape_idx_to_va = {str(iii): 1 for iii in not_ok_shape_idxes}
        #     shape_idxes = [si for si in shape_idxes if si not in not_ok_shape_idx_to_va]

        train_nns = int(len(shape_idxes) * self.train_ratio)

        if self.split == "train":
            shape_idxes = shape_idxes[:train_nns]
            # shape_idxes = random.shuffle(shape_idxes)
            random.shuffle(shape_idxes)
        else:
            shape_idxes = shape_idxes[train_nns:]

        self.shape_idxes = shape_idxes
        print(f"split: {self.split}, shape_idxes: {self.shape_idxes}")

        self.anchors = L.get_anchors(args.model.kanchor)
        equi_anchors = L.get_anchors(60)

        if self.global_rot == 2:
            rotation_angle = sciR.random().as_matrix()
            rotation_matrix = rotation_angle[:3, :3]
            R1 = rotation_matrix
            self.common_R = R1
        elif self.global_rot == 3:
            self.common_R = equi_anchors[7]

        # self.anchors = torch.from_numpy(L.get_anchors(args.model.kanchor)).cuda()
        self.kanchor = args.model.kanchor


    def get_trans_encoding_to_trans_dir(self):
        trans_dir_to_trans_mode = {
            (0, 1, 2): 1,
            (0, 2): 2,
            (0, 1): 3,
            (1, 2): 4,
            (0,): 5,
            (1,): 6,
            (2,): 7,
            (): 0
        }
        self.trans_mode_to_trans_dir = {trans_dir_to_trans_mode[k]: k for k in trans_dir_to_trans_mode}
        self.base_transition_vec = [
            np.array([1.0, 0.0, 0.0], dtype=np.float),
            np.array([0.0, 1.0, 0.0], dtype=np.float),
            np.array([0.0, 0.0, 1.0], dtype=np.float),
        ]

    def get_test_data(self):
        return self.test_data

    def reindex_data_index(self):
        old_idx_to_new_idx = {}
        ii = 0
        for old_idx in self.data:
            old_idx_to_new_idx[old_idx] = ii
            ii += 1
        self.new_idx_to_old_idx = {old_idx_to_new_idx[k]: k for k in old_idx_to_new_idx}

    def reindex_shape_seg(self, shape_seg):
        old_seg_to_new_seg = {}
        ii = 0
        for i in range(shape_seg.shape[0]):
            old_seg = int(shape_seg[i].item())
            if old_seg not in old_seg_to_new_seg:
                old_seg_to_new_seg[old_seg] = ii
                ii += 1
            new_seg = old_seg_to_new_seg[old_seg]
            shape_seg[i] = new_seg
            # old_seg_to_new_seg[old_seg] = ii
            # ii += 1
        return shape_seg

    def transit_pos_by_transit_vec(self, trans_pos):
        tdir = np.random.uniform(-1.0, 1.0, (3,))
        tdir = tdir / (np.sqrt(np.sum(tdir ** 2)).item() + 1e-9)
        trans_scale = np.random.uniform(1.0, 2.0, (1,)).item()
        # for flow test...
        trans_pos_af_pos = trans_pos + tdir * 0.1 * trans_scale * 2
        return trans_pos_af_pos, tdir * 0.1 * trans_scale * 2

    def transit_pos_by_transit_vec_dir(self, trans_pos, tdir):
        # tdir = np.zeros((3,), dtype=np.float)
        # axis_dir = np.random.choice(3, 1).item()
        # tdir[int(axis_dir)] = 1.
        trans_scale = np.random.uniform(0.0, 1.0, (1,)).item()
        trans_pos_af_pos = trans_pos + tdir * 0.1 * trans_scale
        return trans_pos_af_pos, tdir * 0.1 * trans_scale

    def get_random_transition_dir_scale(self):
        tdir = np.random.uniform(-1.0, 1.0, (3,))
        tdir = tdir / (np.sqrt(np.sum(tdir ** 2)).item() + 1e-9)
        trans_scale = np.random.uniform(1.0, 2.0, (1,)).item()
        return tdir, trans_scale

    def decode_trans_dir(self, trans_encoding):
        trans_dir = self.trans_mode_to_trans_dir[trans_encoding]
        return [self.base_transition_vec[d] for ii, d in enumerate(trans_dir)]

    def get_rotation_from_anchor(self):
        ii = np.random.randint(0, self.kanchor, (1,)).item()
        ii = int(ii)
        R = self.anchors[ii]
        return R

    def get_whole_shape_by_idx(self, index):
        shp_idx = self.shape_idxes[index + 1]
        cur_folder = os.path.join(self.shape_root, shp_idx)

        cur_mesh_fn = os.path.join(cur_folder, self.mesh_fn)
        cur_surface_to_seg_fn = os.path.join(cur_folder, self.surface_to_seg_fn)
        cur_motion_attributes_fn = os.path.join(cur_folder, self.attribute_fn)

        cur_vertices, cur_triangles = load_vertices_triangles(cur_mesh_fn)
        cur_triangles_to_seg_idx, seg_idx_to_triangle_idxes = load_triangles_to_seg_idx(cur_surface_to_seg_fn)
        cur_motion_attributes = load_motion_attributes(cur_motion_attributes_fn)

        sampled_pcts, pts_to_seg_idx, seg_idx_to_sampled_pts = sample_pts_from_mesh(cur_vertices, cur_triangles,
                                                                                    cur_triangles_to_seg_idx,
                                                                                    npoints=self.npoints)
        sampled_pcts = torch.from_numpy(sampled_pcts).float()
        return sampled_pcts

    def get_shape_by_idx(self, index):
        shp_idx = self.shape_idxes[index + 1]
        cur_folder = os.path.join(self.shape_root, shp_idx)

        cur_mesh_fn = os.path.join(cur_folder, self.mesh_fn)
        cur_surface_to_seg_fn = os.path.join(cur_folder, self.surface_to_seg_fn)
        cur_motion_attributes_fn = os.path.join(cur_folder, self.attribute_fn)

        cur_vertices, cur_triangles = load_vertices_triangles(cur_mesh_fn)
        cur_triangles_to_seg_idx, seg_idx_to_triangle_idxes = load_triangles_to_seg_idx(cur_surface_to_seg_fn)
        cur_motion_attributes = load_motion_attributes(cur_motion_attributes_fn)

        sampled_pcts, pts_to_seg_idx, seg_idx_to_sampled_pts = sample_pts_from_mesh(cur_vertices, cur_triangles,
                                                                                    cur_triangles_to_seg_idx,
                                                                                    npoints=self.npoints)

        # get points for each segmentation/part
        tot_transformed_pts = []
        pts_nns = []
        for i_seg in range(len(cur_motion_attributes)):
            cur_seg_motion_info = cur_motion_attributes[i_seg]
            cur_seg_pts_idxes = np.array(seg_idx_to_sampled_pts[i_seg], dtype=np.long)
            cur_seg_pts = sampled_pcts[cur_seg_pts_idxes]
            pts_nns.append(cur_seg_pts.shape[0])

            tot_transformed_pts.append(cur_seg_pts)
        maxx_nn_pt = max(pts_nns)
        res_pts = []
        for i, trans_pts in enumerate(tot_transformed_pts):
            cur_seg_nn_pt = trans_pts.shape[0]
            cur_seg_center_pt = np.mean(trans_pts, axis=0, keepdims=True)
            if cur_seg_nn_pt < maxx_nn_pt:
                cur_seg_trans_pts = np.concatenate(
                    [trans_pts] + [cur_seg_center_pt for _ in range(maxx_nn_pt - cur_seg_nn_pt)], axis=0
                )
                res_pts.append(np.reshape(cur_seg_trans_pts, (1, maxx_nn_pt, 3)))
            else:
                res_pts.append(np.reshape(trans_pts, (1, maxx_nn_pt, 3)))

        res_pts = np.concatenate(res_pts, axis=0)
        res_pts = torch.from_numpy(res_pts).float()
        return res_pts

    def refine_triangle_idxes_by_seg_idx(self, seg_idx_to_triangle_idxes, cur_triangles):
        res_triangles = []
        cur_triangles_to_seg_idx = []
        for seg_idx in seg_idx_to_triangle_idxes:
            # if seg_idx == 0:
            #     continue
            cur_triangle_idxes = np.array(seg_idx_to_triangle_idxes[seg_idx], dtype=np.long)
            cur_seg_triangles = cur_triangles[cur_triangle_idxes]
            res_triangles.append(cur_seg_triangles)
            cur_triangles_to_seg_idx += [seg_idx for _ in range(cur_triangle_idxes.shape[0])]
        res_triangles = np.concatenate(res_triangles, axis=0)
        cur_triangles_to_seg_idx = np.array(cur_triangles_to_seg_idx, dtype=np.long)
        return res_triangles, cur_triangles_to_seg_idx

    def __getitem__(self, index):

        # n_samples_per_instance = 100

        nparts = None
        if self.shape_type == "eyeglasses":
            nparts = 2
            nparts = None

        shape_index, sample_index = index // self.n_samples, index % self.n_samples
        # print(index, shape_index, sample_index)
        # shp_idx = self.shape_idxes[index]
        # print(index, shape_index, sample_index)
        shp_idx = self.shape_idxes[shape_index]
        shp_idx_int, sample_idx_int = int(shp_idx), sample_index
        # shp_idx = self.shape_idxes[3]
        # shp_idx = self.shape_idxes[1]

        # shp_idx = self.shape_idxes[0]
        cur_folder = os.path.join(self.shape_root, shp_idx)

        cur_mesh_fn = os.path.join(cur_folder, self.mesh_fn)
        cur_surface_to_seg_fn = os.path.join(cur_folder, self.surface_to_seg_fn)
        cur_motion_attributes_fn = os.path.join(cur_folder, self.attribute_fn)

        # vertices & triangles --- any problem...
        cur_vertices, cur_triangles = load_vertices_triangles(cur_mesh_fn)
        #
        cur_triangles_to_seg_idx, seg_idx_to_triangle_idxes = load_triangles_to_seg_idx(cur_surface_to_seg_fn, nparts=nparts)
        cur_triangles, cur_triangles_to_seg_idx = self.refine_triangle_idxes_by_seg_idx(seg_idx_to_triangle_idxes, cur_triangles)
        # cur_motion_attributes = load_motion_attributes(cur_motion_attributes_fn, ex_none=True)


        seg_idx_to_triangle_idxes = inv_triangles_to_seg_idx(cur_triangles_to_seg_idx)


        cur_motion_attributes = load_motion_attributes(cur_motion_attributes_fn)

        sampled_pcts, pts_to_seg_idx, seg_idx_to_sampled_pts = sample_pts_from_mesh(cur_vertices, cur_triangles, cur_triangles_to_seg_idx, npoints=self.npoints)

        # transformed_vertices: n_vertices x 3
        # transformed_vertices = np.zeros((cur_vertices.shape[0], 3), dtype=np.float)

        ''' Centralize points in the real canonical state ''' # sampeld pcts...
        boundary_pts = [np.min(sampled_pcts, axis=0), np.max(sampled_pcts, axis=0)]
        center_pt = (boundary_pts[0] + boundary_pts[1]) / 2
        length_bb = np.linalg.norm(boundary_pts[0] - boundary_pts[1])

        # all normalize into 0
        sampled_pcts = (sampled_pcts - center_pt.reshape(1, 3)) / length_bb

        boundary_pts = [np.min(cur_vertices, axis=0), np.max(cur_vertices, axis=0)]
        center_pt = (boundary_pts[0] + boundary_pts[1]) / 2
        length_bb = np.linalg.norm(boundary_pts[0] - boundary_pts[1]) # leng
        # todo: change the normalization strategy to pre-normalization
        cur_vertices = (cur_vertices - center_pt.reshape(1, 3)) / length_bb # center pt and length bb
        ''' Centralize points in the real canonical state '''

        ''' Global rotation for complete point clouds '''
        # ''' Add global rotation '''
        # if self.global_rot == 1 and (not (self.split == "train" and self.pre_compute_delta == 1)):
        #     if self.args.equi_settings.rot_anchors == 1:
        #         # just use a matrix from rotation anchors
        #         R1 = self.get_rotation_from_anchor()
        #     else:
        #         rotation_angle = sciR.random().as_matrix()
        #         rotation_matrix = rotation_angle[:3, :3]
        #         R1 = rotation_matrix
        # elif self.global_rot == 2:
        #     R1 = self.common_R
        # else:
        #     R1 = np.eye(3, dtype=np.float32)
        # ''' Add global rotation '''
        ''' Global rotation for complete point clouds '''

        part_state_rots, part_ref_rots = [], []
        part_ref_trans = []

        part_state_trans_bbox = []
        part_ref_trans_bbox = []

        part_axis = []
        part_pv_offset = []
        part_pv_point = []
        part_angles = []

        ''' The following code aims to get articulation changed shape '''
        transformed_vertices = np.zeros((cur_vertices.shape[0], 3), dtype=np.float)
        canon_transformed_vertices = np.zeros((cur_vertices.shape[0], 3), dtype=np.float)


        tot_transformed_pts = []
        tot_transformation_mtx = []
        tot_transformation_mtx_segs = []
        tot_canon_transformation_mtx_segs = []
        canon_transformed_pts = []
        rot_1 = False
        for i_seg in range(len(cur_motion_attributes)):
            if nparts is not None and i_seg >= nparts:
                continue
            cur_seg_motion_info = cur_motion_attributes[i_seg]
            if self.shape_type == "eyeglasses" and i_seg == 1:
                cur_seg_motion_info = cur_motion_attributes[i_seg + 1]
            if self.shape_type == "eyeglasses" and i_seg == 2:
                cur_seg_motion_info = cur_motion_attributes[i_seg - 1]

            # todo: the consistency between seg_idx_to_triangle_idxes and other seg_labels?
            cur_seg_tri_idxes = seg_idx_to_triangle_idxes[i_seg]
            cur_seg_tri = cur_triangles[cur_seg_tri_idxes]
            cur_seg_tri_v1, cur_seg_tri_v2, cur_seg_tri_v3 = cur_vertices[cur_seg_tri[:, 0]], cur_vertices[
                cur_seg_tri[:, 1]], cur_vertices[cur_seg_tri[:, 2]]
            cur_seg_tri_vertices = np.concatenate([cur_seg_tri_v1, cur_seg_tri_v2, cur_seg_tri_v3], axis=0)
            cur_seg_tri_vertices_idxes = np.concatenate([cur_seg_tri[:, 0], cur_seg_tri[:, 1], cur_seg_tri[:, 2]],
                                                        axis=0)

            # Get pts indexes for this segmentation
            cur_seg_pts_idxes = np.array(seg_idx_to_sampled_pts[i_seg], dtype=np.long)
            cur_seg_pts = sampled_pcts[cur_seg_pts_idxes]
            # if self.shape_type == "eyeglasses" and i_seg == 2:
            #     cur_seg_pts = (cur_seg_pts - np.reshape(np.array(cur_seg_motion_info["center"]), (1, 3))) * 1.5 + np.reshape(np.array(cur_seg_motion_info["center"]), (1, 3))
            #     sampled_pcts[cur_seg_pts_idxes] = cur_seg_pts

            # rotation
            if cur_seg_motion_info["motion_type"] == "rotation" and (rot_1 == False or self.shape_type == "eyeglasses"):
                center = cur_seg_motion_info["center"]
                axis = cur_seg_motion_info["direction"]

                if self.use_multi_sample == 0: # multi-sample?
                    if self.shape_type in ['laptop', 'eyeglasses']:
                        # if self.shape_type == 'eyeglasses' and nparts is not None:
                        if self.shape_type == 'eyeglasses' and nparts is None:
                            theta = (np.random.uniform(0.05, 1., (1,)).item() * np.pi) * self.rot_factor
                            theta = -1. * theta
                            # theta = (np.random.uniform(-0.35, 0.35, (1,)).item() * np.pi)
                            # theta = (np.random.uniform(0., 1., (1,)).item() * np.pi - np.pi / 2.) * self.rot_factor
                        else:
                            theta = (np.random.uniform(0., 1., (1,)).item() * np.pi - np.pi / 2.) * self.rot_factor
                    elif self.shape_type in ['oven', 'washing_machine']:
                        theta = (np.random.uniform(0.5, 1., (1,)).item() * np.pi) * self.rot_factor
                        # theta = (np.random.uniform(0., 135. / 180., (1,)).item() * np.pi)  # * self.rot_factor
                        # theta = (((135. / 180.) / self.n_samples) * sample_index + 0.05) * np.pi  # * self.rot_factor
                    else:
                        theta = (np.random.uniform(0., 1., (1,)).item() * np.pi) * self.rot_factor
                else:
                    if self.shape_type in ['laptop', 'eyeglasses']:
                        # if self.shape_type == 'eyeglasses' and nparts is not None:
                        if self.shape_type == 'eyeglasses' and nparts is None:
                            # theta = (np.random.uniform(0., 1., (1,)).item() * np.pi) * self.rot_factor
                            # theta = -1. * theta
                            a_rot_idx = sample_index // 10
                            b_rot_idx = sample_index % 10
                            mult_factor = 0.45 if self.split == 'train' else 0.35
                            if i_seg == 1:
                                # theta = (0.1 * (a_rot_idx ) * np.pi) * 0.4
                                # theta = (0.1 * (a_rot_idx ) * np.pi) * 0.45 # - 0.05
                                theta = (0.1 * (a_rot_idx ) * np.pi) * mult_factor # - 0.05
                                # theta = (0.1 * (a_rot_idx ) * np.pi) * 0.4 # - 0.05
                                # theta = (0.1 * (a_rot_idx ) * np.pi) * 0.35 # - 0.05
                                # theta = (0.1 * (a_rot_idx ) * np.pi) * 0.30 # - 0.05
                            elif i_seg == 2:
                                # theta = (0.1 * (b_rot_idx ) * np.pi) * 0.4
                                # theta = (0.1 * (b_rot_idx ) * np.pi) * 0.45
                                theta = (0.1 * (b_rot_idx ) * np.pi) * mult_factor
                                # theta = (0.1 * (b_rot_idx ) * np.pi) * 0.4
                                # theta = (0.1 * (b_rot_idx ) * np.pi) * 0.35
                                # theta = (0.1 * (b_rot_idx ) * np.pi) * 0.30
                            # theta = -1. * theta
                            # if self.split != 'train':
                            #     theta = theta + 0.05 * np.pi
                            # theta = (np.random.uniform(0., 1., (1,)).item() * np.pi - np.pi / 2.) * self.rot_factor
                        else:
                            # theta = (np.random.uniform(0., 1., (1,)).item() * np.pi - np.pi / 2.) * self.rot_factor

                            # theta = (0.5 / self.n_samples * sample_index * np.pi) # * self.rot_factor
                            theta = (0.5 / self.n_samples * sample_index * np.pi) - 0.05 * np.pi
                            # theta = (0.40 / self.n_samples * sample_index * np.pi) # * self.rot_factor
                            # theta = (0.35 / self.n_samples * sample_index * np.pi) # * self.rot_factor
                            theta = -1. * theta
                    elif self.shape_type in ['oven', 'washing_machine']:
                        # theta = (np.random.uniform(0.5, 1., (1,)).item() * np.pi) * self.rot_factor
                        # theta = (((1.0 - 0.5) / self.n_samples) * sample_index + 0.05) * np.pi # * self.rot_factor
                        # theta = (((135. / 180.) / self.n_samples) * sample_index + 0.15) * np.pi # * self.rot_factor
                        # theta = (((115. / 180.) / self.n_samples) * sample_index + 30. / 180.) * np.pi # * self.rot_factor
                        # theta = (((120. / 180.) / self.n_samples) * sample_index + 30. / 180.) * np.pi # * self.rot_factor
                        ##### would cause global symmetry if we use the following setting #####
                        if self.shape_type == 'washing_machine':
                            if self.split == 'train':
                                # theta = (((90. / 180.) / self.n_samples) * sample_index + 45. / 180.) * np.pi
                                # theta = (((75. / 180.) / self.n_samples) * sample_index + 45. / 180.) * np.pi
                                theta = (((60. / 180.) / self.n_samples) * sample_index + 60. / 180.) * np.pi
                            else:
                                # theta = (((10. / 180.) / self.n_samples) * sample_index + 90. / 180.) * np.pi # * self.rot_factor
                                # theta = (((90. / 180.) / self.n_samples) * sample_index + 45. / 180.) * np.pi # * self.rot_factor
                                # theta = (((75. / 180.) / self.n_samples) * sample_index + 45. / 180.) * np.pi
                                theta = (((60. / 180.) / self.n_samples) * sample_index + 60. / 180.) * np.pi
                        # theta = (((70. / 180.) / self.n_samples) * sample_index + 45. / 180.) * np.pi # * self.rot_factor
                        else:
                            # theta = (((50. / 180.) / self.n_samples) * sample_index + 45. / 180.) * np.pi # * self.rot_factor
                            # theta = (((80. / 180.) / self.n_samples) * sample_index + 45. / 180.) * np.pi # * self.rot_factor
                            theta = (((70. / 180.) / self.n_samples) * sample_index + 45. / 180.) * np.pi # * self.rot_factor

                            # ''' For pics only... '''
                            # theta = (((85. / 180.) / self.n_samples) * sample_index + 30. / 180.) * np.pi  # * self.rot_factor

                        # theta = (((135. / 180.) / self.n_samples) * sample_index + 0.20) * np.pi # * self.rot_factor
                        # theta = (((135. / 180.) / self.n_samples) * sample_index) * np.pi # * self.rot_factor
                    else:
                        theta = (np.random.uniform(0., 1., (1,)).item() * np.pi) * self.rot_factor

                # Get transformed center (the center of the axis?)

                # Change the center point
                center = (center - center_pt) / length_bb
                # axis = (axis - center_pt) / length_bb
                ''' Transform points via revolute transformation '''
                # rot_pts, transformation_mtx = revoluteTransform(cur_seg_pts, center, axis, theta)
                # transformation_mtx = np.transpose(transformation_mtx, (1, 0))
                # rot_pts[:, :3] = np.matmul(np.reshape(R1, (1, 3, 3)), np.reshape(rot_pts[:, :3], (rot_pts.shape[0], 3, 1)))[:, :3, 0]
                # transformation_mtx[:3] = np.matmul(R1, transformation_mtx[:3])
                ''' Transform points via revolute transformation '''

                # part_axis.append(np.reshape(axis, (1, 3)))
                part_angles.append(theta)

                center_pt_offset = center - np.sum(axis * center, axis=0, keepdims=True) * axis
                center_pt_offset = np.sqrt(np.sum(center_pt_offset ** 2, axis=0))
                # part_pv_offset.append(np.reshap)
                part_pv_offset.append(center_pt_offset)
                # part_pv_point.append(np.reshape(center, (1, 3)))

                rot_cur_seg_tri_vertices, transformation_mtx = revoluteTransform(cur_seg_tri_vertices, center, axis, theta)
                transformation_mtx = np.transpose(transformation_mtx, (1, 0))

                ''' Rotate points along the y-axis '''
                y_axis = np.array([0.0, 1.0, 0.0], dtype=np.float) #
                y_center = np.array([0.0, 0.0, 0.0], dtype=np.float)
                y_theta = -0.5 * np.pi
                rot_cur_seg_tri_vertices, y_transformation_mtx = revoluteTransform(rot_cur_seg_tri_vertices[:, :3], y_center, y_axis, y_theta)
                y_transformation_mtx = np.transpose(y_transformation_mtx, (1, 0))
                transformation_mtx[:3, :] = np.matmul(y_transformation_mtx[:3, :3], transformation_mtx[:3, :])
                cur_part_axis = np.matmul(y_transformation_mtx[:3, :3], np.reshape(axis, (3, 1)))
                cur_part_axis = np.reshape(cur_part_axis, (1, 3))
                part_axis.append(cur_part_axis)

                cur_pv_point = np.matmul(y_transformation_mtx[:3, :3], np.reshape(center, (3, 1)))
                cur_pv_point = np.reshape(cur_pv_point, (1, 3))
                part_pv_point.append(cur_pv_point)
                ''' Rotate points along the y-axis '''

                ''' X-axis transformation '''
                # y_axis = np.array([1.0, 0.0, 0.0], dtype=np.float)
                # y_center = np.array([0.0, 0.0, 0.0], dtype=np.float)
                # y_theta = -0.5 * np.pi
                #
                # rot_cur_seg_tri_vertices, y_transformation_mtx = revoluteTransform(rot_cur_seg_tri_vertices[:, :3], y_center, y_axis, y_theta)
                # y_transformation_mtx = np.transpose(y_transformation_mtx, (1, 0))
                # transformation_mtx[:3, :] = np.matmul(y_transformation_mtx[:3, :3], transformation_mtx[:3, :])
                ''' X-axis transformation '''

                transformed_vertices[cur_seg_tri_vertices_idxes] = rot_cur_seg_tri_vertices[:, :3]

                # todo: we need to get the boudning box related transformation after the partial render?
                # todo: part_state_trans_bbox is not set here, to be set later
                ''' Get points state translations with centralized boudning box '''
                # # rot_pts: n_pts_part x 3
                # # rot pts minn; rot pts maxx
                # rot_pts_minn = np.min(rot_pts[:, :3], axis=0)
                # rot_pts_maxx = np.max(rot_pts[:, :3], axis=0)
                # rot_pts_bbox_center = (rot_pts_minn + rot_pts_maxx) / 2.
                # # print(f"transformation_mtx: {transformation_mtx[:3, 3].shape}, rot_pts_bbox_center: {rot_pts_bbox_center.shape}")
                # cur_part_state_trans_bbox = transformation_mtx[:3, 3] - rot_pts_bbox_center
                # part_state_trans_bbox.append(np.reshape(cur_part_state_trans_bbox, (1, 3)))
                ''' Get points state translations with centralized boudning box '''


                ''' Set canonical angle via shape type '''
                canon_theta = 0.5 * np.pi
                if self.shape_type in ["laptop", "eyeglasses"]:
                    canon_theta = 0.0
                    if self.shape_type == "eyeglasses":
                        canon_theta = 0.10 * np.pi
                if self.shape_type == "laptop":
                    # canon_theta = (0.25 / self.n_samples * sample_index * np.pi)
                    canon_theta = (0.25 * np.pi)
                    canon_theta = canon_theta * (-1.0)

                # canon_rot_pts, canon_transformation_mtx = revoluteTransform(cur_seg_pts, center, axis, canon_theta)
                # canon_transformation_mtx: 4 x 4 transformation matrix...
                canon_rot_vertices, canon_transformation_mtx = revoluteTransform(cur_seg_tri_vertices, center, axis, canon_theta)
                canon_transformation_mtx = np.transpose(canon_transformation_mtx, (1, 0))
                # y_axis = np.array([0.0, 1.0, 0.0], dtype=np.float)
                # y_center = np.array([0.0, 0.0, 0.0], dtype=np.float)
                # y_theta = -0.5 * np.pi
                canon_rot_vertices, y_transformation_mtx = revoluteTransform(canon_rot_vertices[:, :3], y_center, y_axis, y_theta)
                y_transformation_mtx = np.transpose(y_transformation_mtx, (1, 0))
                canon_transformation_mtx[:3, :] = np.matmul(y_transformation_mtx[:3, :3], canon_transformation_mtx[:3, :])

                ''' X-axis transformation '''
                # y_axis = np.array([1.0, 0.0, 0.0], dtype=np.float)
                # y_center = np.array([0.0, 0.0, 0.0], dtype=np.float)
                # y_theta = -0.5 * np.pi
                #
                # canon_rot_vertices, y_transformation_mtx = revoluteTransform(canon_rot_vertices[:, :3], y_center, y_axis, y_theta)
                # y_transformation_mtx = np.transpose(y_transformation_mtx, (1, 0))
                # canon_transformation_mtx[:3, :] = np.matmul(y_transformation_mtx[:3, :3], canon_transformation_mtx[:3, :])
                ''' X-axis transformation '''

                ''' Get points state translations with centralized boudning box '''
                # and we can set the canon trans bbox here...
                # rot_pts: n_pts_part x 3; # canon rot pts
                canon_rot_pts_minn = np.min(canon_rot_vertices[:, :3], axis=0)
                canon_rot_pts_maxx = np.max(canon_rot_vertices[:, :3], axis=0)
                canon_rot_pts_bbox_center = (canon_rot_pts_minn + canon_rot_pts_maxx) / 2.
                cur_part_ref_trans_bbox = canon_transformation_mtx[:3, 3] - canon_rot_pts_bbox_center
                part_ref_trans_bbox.append(np.reshape(cur_part_ref_trans_bbox, (1, 3)))


                # transformation_mtx = np.transpose(transformation_mtx, (1, 0))
                transformation_mtx = np.reshape(transformation_mtx, (1, 4, 4))

                # cur_rot_mtx = transformation_mtx[0, :3, :3]
                # dot_product = np.dot(cur_rot_mtx, np.transpose(cur_rot_mtx, [1, 0]))
                # traces = dot_product[0, 0].item() + dot_product[1, 1].item() + dot_product[2, 2].item()
                # traces = float(traces) - 1.0; traces = traces / 2.
                # print(f"current traces: {traces}")

                # todo: transformation matrices for points are not set here, to be set later...
                # tot_transformation_mtx += [transformation_mtx for _ in range(cur_seg_pts.shape[0])]
                tot_transformation_mtx_segs.append(transformation_mtx)
                tot_canon_transformation_mtx_segs.append(np.reshape(canon_transformation_mtx, (1, 4, 4)))
                # todo: transformation matrices for points are not set here, to be set later...

                part_state_rots.append(np.reshape(transformation_mtx[0, :3, :3], (1, 3, 3)))

                # canon_transformation_mtx = np.transpose(canon_transformation_mtx, (1, 0))
                part_ref_rots.append(np.reshape(canon_transformation_mtx[:3, :3], (1, 3, 3)))
                part_ref_trans.append(np.reshape(canon_transformation_mtx[:3, 3], (1, 3)))

                if self.pre_compute_delta == 1 and self.split == "train":
                    # tot_transformed_pts.append(canon_rot_pts[:, :3])
                    transformed_vertices[cur_seg_tri_vertices_idxes] = canon_rot_vertices[:, :3]
                else:
                    # tot_transformed_pts.append(rot_pts[:, :3])
                    transformed_vertices[cur_seg_tri_vertices_idxes] = rot_cur_seg_tri_vertices[:, :3]
                # canon_transformed_pts.append(canon_rot_pts[:, :3])
                canon_transformed_vertices[cur_seg_tri_vertices_idxes] = canon_rot_vertices[:, :3]
                rot_1 = True

            else:

                y_axis = np.array([0.0, 1.0, 0.0], dtype=np.float)
                y_center = np.array([0.0, 0.0, 0.0], dtype=np.float)
                y_theta = -0.5 * np.pi

                cur_seg_tri_vertices, y_transformation_mtx = revoluteTransform(cur_seg_tri_vertices[:, :3], y_center, y_axis, y_theta)
                y_transformation_mtx = np.transpose(y_transformation_mtx, (1, 0))


                transformation_mtx = np.zeros((4, 4), dtype=np.float)
                transformation_mtx[0, 0] = 1.
                transformation_mtx[1, 1] = 1.
                transformation_mtx[2, 2] = 1.
                # tot_transformed_pts.append(rot_pts)
                # transformation_mtx[:3] = np.matmul(R1, transformation_mtx[:3])

                ''' Mult with y... '''
                transformation_mtx[:3, :] = np.matmul(y_transformation_mtx[:3, :3], transformation_mtx[:3, :])
                ''' Mult with y... '''

                ''' X-axis transformation '''
                # y_axis = np.array([1.0, 0.0, 0.0], dtype=np.float)
                # y_center = np.array([0.0, 0.0, 0.0], dtype=np.float)
                # y_theta = -0.5 * np.pi
                #
                # cur_seg_tri_vertices, y_transformation_mtx = revoluteTransform(cur_seg_tri_vertices[:, :3], y_center, y_axis, y_theta)
                # y_transformation_mtx = np.transpose(y_transformation_mtx, (1, 0))
                # transformation_mtx[:3, :] = np.matmul(y_transformation_mtx[:3, :3], transformation_mtx[:3, :])
                ''' X-axis transformation '''


                # rot_pts: n_pts_part x 3
                # Get part-ref-trans-bbox from triagnle vertices...
                cur_seg_pts_minn = np.min(cur_seg_tri_vertices[:, :3], axis=0)
                cur_seg_pts_maxx = np.max(cur_seg_tri_vertices[:, :3], axis=0)
                cur_seg_pts_bbox_center = (cur_seg_pts_minn + cur_seg_pts_maxx) / 2.
                cur_part_ref_trans_bbox = -1. * cur_seg_pts_bbox_center
                # Get part-ref-trans-bbox from triagnle vertices...

                canon_transformed_pts.append(cur_seg_pts) # transformed canonical points...

                canon_transformed_vertices[cur_seg_tri_vertices_idxes] = cur_seg_tri_vertices[:, :3]
                transformed_vertices[cur_seg_tri_vertices_idxes] = cur_seg_tri_vertices[:, :3]

                # rot_pts = np.zeros_like(cur_seg_pts)
                # rot_pts[:, :] = cur_seg_pts[:, :]
                # # rot_pts = cur_seg_pts
                # rot_pts[:, :3] = np.matmul(np.reshape(R1, (1, 3, 3)),
                #                            np.reshape(rot_pts[:, :3], (rot_pts.shape[0], 3, 1)))[:, :3, 0]


                transformation_mtx = np.reshape(transformation_mtx, (1, 4, 4))

                # tot_transformation_mtx += [transformation_mtx for _ in range(cur_seg_pts.shape[0])]
                tot_transformation_mtx_segs.append(transformation_mtx)
                tot_canon_transformation_mtx_segs.append(transformation_mtx)

                part_state_rots.append(np.reshape(transformation_mtx[0, :3, :3], (1, 3, 3)))

                canon_transformation_mtx = np.zeros((4, 4), dtype=np.float)
                canon_transformation_mtx[0, 0] = 1.
                canon_transformation_mtx[1, 1] = 1.
                canon_transformation_mtx[2, 2] = 1.
                # part_ref_rots.append(np.reshape(canon_transformation_mtx[:3, :3], (1, 3, 3)))
                part_ref_rots.append(np.reshape(transformation_mtx[0, :3, :3], (1, 3, 3)))
                # part_ref_trans.append(np.zeros((1, 3), dtype=np.float))
                part_ref_trans.append(np.reshape(transformation_mtx[0, :3, 3], (1, 3)))

                ''' Get points state translations with centralized boudning box '''
                # todo: part state rot can still not be set here... to be set later
                # # rot_pts: n_pts_part x 3
                # rot_pts_minn = np.min(rot_pts, axis=0)
                # rot_pts_maxx = np.max(rot_pts, axis=0)
                # rot_pts_bbox_center = (rot_pts_minn + rot_pts_maxx) / 2.
                # cur_part_state_trans_bbox = -1. * rot_pts_bbox_center # state trans bbox
                # part_state_trans_bbox.append(np.reshape(cur_part_state_trans_bbox, (1, 3)))

                ''' Ref transformation bbox '''
                # # rot_pts: n_pts_part x 3
                # rot_pts_minn = np.min(cur_seg_pts, axis=0)
                # rot_pts_maxx = np.max(rot_pts, axis=0)
                # rot_pts_bbox_center = (rot_pts_minn + rot_pts_maxx) / 2.
                # cur_part_state_trans_bbox = -1. * rot_pts_bbox_center
                part_ref_trans_bbox.append(np.reshape(cur_part_ref_trans_bbox, (1, 3)))

        # tot_transformed_pts = np.concatenate(tot_transformed_pts, axis=0)
        # canon_transformed_pts = np.concatenate(canon_transformed_pts, axis=0)
        # print(tot_transformed_pts.shape)

        # here, important variables are transformed_vertices and canon_transformed_vertices

        part_axis = np.concatenate(part_axis, axis=0)
        # part_axis = np.matmul(np.reshape(R1, (1, 3, 3)), np.reshape(part_axis, (part_axis.shape[0], 3, 1)))
        # part_axis = np.reshape(part_axis, (part_axis.shape[0], 3))
        part_pv_offset = np.array(part_pv_offset)
        part_pv_point = np.concatenate(part_pv_point, axis=0)

        ''' Use fake initial pose '''
        # gt_pose = np.zeros((self.npoints, 4, 4), dtype=np.float)
        # gt_pose[:, 0, 0] = 1.; gt_pose[:, 1, 1] = 1.; gt_pose[:, 2, 2] = 1.

        ''' Use GT transformation matrix as initial pose '''
        # tot_transformation_mtx = np.concatenate(tot_transformation_mtx, axis=0)
        tot_transformation_mtx_segs = np.concatenate(tot_transformation_mtx_segs, axis=0)
        tot_canon_transformation_mtx_segs = np.concatenate(tot_canon_transformation_mtx_segs, axis=0)

        part_state_rots = np.concatenate(part_state_rots, axis=0)
        part_ref_rots = np.concatenate(part_ref_rots, axis=0)
        part_ref_trans = np.concatenate(part_ref_trans, axis=0)
        ''' Get part_state_trans_bbox and part_ref_trans_bbox '''
        # part_state_trans_bbox = np.concatenate(part_state_trans_bbox, axis=0)
        part_ref_trans_bbox = np.concatenate(part_ref_trans_bbox, axis=0)

        # todo: render more than one times...

        # boundary_pts = [np.min(transformed_vertices, axis=0), np.max(transformed_vertices, axis=0)]
        # center_pt = (boundary_pts[0] + boundary_pts[1]) / 2
        # length_bb = np.linalg.norm(boundary_pts[0] - boundary_pts[1])
        #
        # # center_transformed_vertices = np.mean(transformed_vertices, axis=0,  keepdims=True)
        # transformed_vertices = (transformed_vertices - center_pt.reshape(1, 3)) / length_bb
        # tot_transformation_mtx_segs[:, :3, 3] = (tot_transformation_mtx_segs[:, :3, 3] - center_pt.reshape(1, 3)) / length_bb


        # ins_num - # create partial points --> with global rotation and translation from one viewpoint
        seg_label_to_pts, seg_label_to_new_pose, glb_pose = create_partial_pts(mean_pose=np.array([0, 0, -1.8]), std_pose=np.array([0.2, 0.2, 0.15]), yfov=np.deg2rad(60), pw=640, ph=480, near=0.1, far=10, upper_hemi=True, vertices=transformed_vertices, triangles=cur_triangles, seg_label_to_triangles=seg_idx_to_triangle_idxes, seg_transformation_mtx=tot_transformation_mtx_segs, render_img=False)

        ''' Transform and render partial pc '''
        canon_seg_label_to_pts, canon_seg_label_to_new_pose, canon_glb_pose = create_partial_pts(mean_pose=np.array([0, 0, -1.8]),
                                                                     std_pose=np.array([0.2, 0.2, 0.15]),
                                                                     yfov=np.deg2rad(60), pw=640, ph=480, near=0.1,
                                                                     far=10, upper_hemi=True,
                                                                     vertices=canon_transformed_vertices,
                                                                     triangles=cur_triangles,
                                                                     seg_label_to_triangles=seg_idx_to_triangle_idxes,
                                                                     seg_transformation_mtx=tot_canon_transformation_mtx_segs,
                                                                     render_img=False, no_transformation=True)

        tot_transformed_full_vertices = []
        tot_transformed_pts = []
        pts_to_seg_idx = []
        tot_transformation_mtx = []
        # seg label to
        canon_transformed_pts = []
        part_state_trans_bbox = np.zeros_like(part_ref_trans_bbox)
        glb_rotation, glb_trans = glb_pose['rotation'], glb_pose['trans']
        part_axis = np.matmul(glb_rotation,  np.transpose(part_axis, (1, 0)))
        part_axis = np.transpose(part_axis, (1, 0))
        # global rotation: [3, 3] xx [
        part_pv_point = np.matmul(glb_rotation, np.transpose(part_pv_point, (1, 0)))
        # get transformed part pv point
        part_pv_point = np.transpose(part_pv_point, (1, 0)) + np.reshape(glb_trans, (1, 3))
        part_pv_offset = part_pv_point - np.sum(part_pv_point * part_axis, axis=-1, keepdims=True) * part_axis
        part_pv_offset = np.sqrt(np.sum(part_pv_offset ** 2, axis=-1))

        # print(f"seg_label_to_pts: {len(seg_label_to_pts)}, seg_label_to_new_pose: {len(seg_label_to_new_pose)}")

        # todo: set parT_state_trans_bbox, transformation_mtx_segs, part_state_rots, tot_transformed_pts, tot_full_transformed_vertices
        # todo: so what about the canon_transformed_vertices? --- should we also render it to partial observed? Oh yeah, otherwise where can you get points for the canonically transformed shape from?
        for seg_label in seg_label_to_pts: #

            cur_seg_trans_pts = seg_label_to_pts[seg_label]
            cur_seg_pose = seg_label_to_new_pose[seg_label]
            tot_transformed_pts.append(cur_seg_trans_pts) # partial transformed points
            pts_to_seg_idx += [seg_label for _ in range(cur_seg_trans_pts.shape[0])]

            canon_transformed_pts.append(canon_seg_label_to_pts[seg_label])

            ''' Get triangle idxes for this segmentation '''
            cur_seg_tri_idxes = seg_idx_to_triangle_idxes[seg_label]
            cur_seg_tri = cur_triangles[cur_seg_tri_idxes]
            cur_seg_tri_v1, cur_seg_tri_v2, cur_seg_tri_v3 = cur_vertices[cur_seg_tri[:, 0]], cur_vertices[
                cur_seg_tri[:, 1]], cur_vertices[cur_seg_tri[:, 2]]
            cur_seg_tri_vertices = np.concatenate([cur_seg_tri_v1, cur_seg_tri_v2, cur_seg_tri_v3], axis=0)
            cur_seg_tri_vertices_idxes = np.concatenate([cur_seg_tri[:, 0], cur_seg_tri[:, 1], cur_seg_tri[:, 2]],
                                                        axis=0)

            cur_seg_rot, cur_seg_trans = cur_seg_pose[:3, :3], cur_seg_pose[:3, 3]
            rot_cur_seg_tri_vertices = np.matmul(cur_seg_rot, np.transpose(cur_seg_tri_vertices, (1, 0))) + np.reshape(cur_seg_trans, (3, 1))
            rot_cur_seg_tri_vertices = np.transpose(rot_cur_seg_tri_vertices, (1, 0))

            tot_transformed_full_vertices.append(rot_cur_seg_tri_vertices)

            # cur_seg_pose: 3 x 3; 3

            cur_seg_pts_minn = np.min(rot_cur_seg_tri_vertices, axis=0)
            cur_seg_pts_maxx = np.max(rot_cur_seg_tri_vertices, axis=0)
            cur_seg_pts_bbox_center = (cur_seg_pts_minn + cur_seg_pts_maxx) / 2.
            cur_seg_trans_bbox = cur_seg_trans - cur_seg_pts_bbox_center

            part_state_trans_bbox[seg_label] = cur_seg_trans_bbox
            part_state_rots[seg_label] = cur_seg_rot
            tot_transformation_mtx_segs[seg_label] = cur_seg_pose
            # tot_transformation_mtx.append([cur_seg_pose for _ in range(cur_seg_trans_pts.shape[0])])
            tot_transformation_mtx += [np.reshape(cur_seg_pose, (1, 4, 4)) for _ in range(cur_seg_trans_pts.shape[0])]

        tot_transformed_pts = np.concatenate(tot_transformed_pts, axis=0)
        tot_transformed_full_vertices = np.concatenate(tot_transformed_full_vertices, axis=0)
        # center_transformed_full_vertices = np.mean(tot_transformed_full_vertices, axis=0, keepdims=True)
        tot_transformation_mtx = np.concatenate(tot_transformation_mtx, axis=0)
        canon_transformed_pts = np.concatenate(canon_transformed_pts, axis=0)
        pts_to_seg_idx = np.array(pts_to_seg_idx, dtype=np.long)

        gt_pose = tot_transformation_mtx
        # tot_transformation_mt

        # boundary_pts = [np.min(tot_transformed_pts, axis=0), np.max(tot_transformed_pts, axis=0)]
        # center_pt = (boundary_pts[0] + boundary_pts[1]) / 2
        # length_bb = np.linalg.norm(boundary_pts[0] - boundary_pts[1])
        #
        # # all normalize into 0
        # tot_transformed_pts = (tot_transformed_pts - center_pt.reshape(1, 3)) / length_bb


        if self.global_rot >= 0:
            ''' Point normalization via bounding box centralization '''
            # af_glb_boundary_pts = [np.min(tot_transformed_pts, axis=0), np.max(tot_transformed_pts, axis=0)]
            # af_glb_center_pt = (af_glb_boundary_pts[0] + af_glb_boundary_pts[1]) / 2
            # length_bb = np.linalg.norm(af_glb_boundary_pts[0] - af_glb_boundary_pts[1])
            # tot_transformed_pts = (tot_transformed_pts - af_glb_center_pt.reshape(1, 3)) / length_bb
            ''' Point normalization via bounding box centralization '''

            ''' Point normalization via bounding box centralization '''
            # af_glb_boundary_pts = [np.min(tot_transformed_full_vertices, axis=0), np.max(tot_transformed_full_vertices, axis=0)]
            # af_glb_center_pt = (af_glb_boundary_pts[0] + af_glb_boundary_pts[1]) / 2
            # length_bb = np.linalg.norm(af_glb_boundary_pts[0] - af_glb_boundary_pts[1])
            # tot_transformed_pts = (tot_transformed_pts - af_glb_center_pt.reshape(1, 3)) / length_bb
            ''' Point normalization via bounding box centralization '''

            ''' Point normalization via centralization '''
            af_glb_center_pt = np.mean(tot_transformed_full_vertices, axis=0)
            tot_transformed_pts = (tot_transformed_pts - af_glb_center_pt.reshape(1, 3))
            ''' Point normalization via centralization '''


            # latest work? aiaiaia...

            # all normalize into 0
            # sampled_pcts = (sampled_pcts - center_pt.reshape(1, 3)) / length_bb

            gt_pose[:, :3, 3] = gt_pose[:, :3, 3] - af_glb_center_pt # transformation matrix...
            tot_transformation_mtx_segs[:, :3, 3] = tot_transformation_mtx_segs[:, :3, 3] - af_glb_center_pt
            part_pv_point = part_pv_point - np.reshape(af_glb_center_pt, (1, 3))
            part_pv_offset = part_pv_point - np.sum(part_pv_point * part_axis, axis=-1, keepdims=True) * part_axis
            part_pv_offset = np.sqrt(np.sum(part_pv_offset ** 2, axis=-1))


        cur_pc = torch.from_numpy(tot_transformed_pts.astype(np.float32)).float()
        tot_transformed_pts = torch.from_numpy(tot_transformed_pts.astype(np.float32)).float()
        cur_label = torch.from_numpy(pts_to_seg_idx).long()
        tot_label = torch.from_numpy(pts_to_seg_idx).long()
        cur_pose = torch.from_numpy(gt_pose.astype(np.float32))
        cur_pose_segs = torch.from_numpy(tot_transformation_mtx_segs.astype(np.float32))
        # cur_ori_pc = torch.from_numpy(sampled_pcts.astype(np.float32)).float()
        cur_canon_transformed_pts = torch.from_numpy(canon_transformed_pts.astype(np.float32)).float()
        cur_part_state_rots = torch.from_numpy(part_state_rots.astype(np.float32)).float()
        cur_part_ref_rots = torch.from_numpy(part_ref_rots.astype(np.float32)).float()
        cur_part_ref_trans = torch.from_numpy(part_ref_trans.astype(np.float32)).float()
        cur_part_axis = torch.from_numpy(part_axis.astype(np.float32)).float()
        cur_part_pv_offset = torch.from_numpy(part_pv_offset.astype(np.float32)).float()
        cur_part_pv_point = torch.from_numpy(part_pv_point.astype(np.float32)).float()
        part_angles = np.array(part_angles, dtype=np.float32) # .float()
        part_angles = torch.from_numpy(part_angles.astype(np.float32)).float()

        # if self.est_normals == 1:
        #     cur_normals = estimate_normals(cur_pc)
        #     cur_canon_normals = estimate_normals(cur_canon_transformed_pts)

        # part_state_trans_bbox part_ref_trans_bbox
        cur_part_state_trans_bbox = torch.from_numpy(part_state_trans_bbox.astype(np.float32)).float()
        cur_part_ref_trans_bbox = torch.from_numpy(part_ref_trans_bbox.astype(np.float32)).float()


        full_tot_transformed_pts = tot_transformed_pts.clone()
        fps_idx = farthest_point_sampling(cur_pc.unsqueeze(0), n_sampling=self.npoints)
        fps_idx = fps_idx[:self.npoints]
        fps_idx_oorr = farthest_point_sampling(cur_pc.unsqueeze(0), n_sampling=4096)

        tot_transformed_pts = tot_transformed_pts[fps_idx_oorr]
        tot_label = tot_label[fps_idx_oorr]
        cur_pc = cur_pc[fps_idx]
        cur_label = cur_label[fps_idx]
        cur_pose = cur_pose[fps_idx]

        # cur_pose = cur_pose[fps_idx]
        ##### Get original transformed point cloud #####
        # cur_ori_pc = cur_ori_pc[fps_idx]
        ##### Get original transformed point cloud #####

        canon_fps_idx = farthest_point_sampling(cur_canon_transformed_pts.unsqueeze(0), n_sampling=self.npoints)
        canon_fps_idx = canon_fps_idx[:self.npoints]
        canon_fps_idx_oorr = farthest_point_sampling(cur_canon_transformed_pts.unsqueeze(0), n_sampling=4096)
        cur_oorr_canon_transformed_pts = cur_canon_transformed_pts[canon_fps_idx_oorr]
        cur_canon_transformed_pts = cur_canon_transformed_pts[canon_fps_idx]

        # if self.est_normals == 1:
        #     cur_normals = cur_normals[fps_idx]
        #     cur_canon_normals = cur_canon_normals[fps_idx]

        idx_arr = np.array([index], dtype=np.long)
        idx_arr = torch.from_numpy(idx_arr).long()
        shp_idx_arr = torch.tensor([shp_idx_int], dtype=torch.long).long()
        sampled_idx_arr = torch.tensor([sample_idx_int], dtype=torch.long).long()

        rt_dict = {
            'pc': cur_pc.contiguous(),
            'af_pc': cur_pc.contiguous().transpose(0, 1).contiguous(),
            #'ori_pc': full_tot_transformed_pts.contiguous().transpose(0, 1).contiguous(),
            'canon_pc': cur_canon_transformed_pts, #.contiguous().transpose(0, 1).contiguous(),
            'oorr_pc': tot_transformed_pts.contiguous().transpose(0, 1).contiguous(), # 3 x oorr_N
            'oorr_canon_pc': cur_oorr_canon_transformed_pts.contiguous(),
            'label': cur_label,
            'oorr_label': tot_label, # oorr_N
            'pose': cur_pose,
            'pose_segs': cur_pose_segs,
            'part_state_rots': cur_part_state_rots,
            'part_ref_rots': cur_part_ref_rots,
            'part_ref_trans': cur_part_ref_trans,
            'part_axis': cur_part_axis,  # get ground-truth axis
            'idx': idx_arr,
            'shp_idx': shp_idx_arr,
            'sampled_idx': sampled_idx_arr,
            'part_state_trans_bbox': cur_part_state_trans_bbox,
            'part_ref_trans_bbox': cur_part_ref_trans_bbox,
            'part_pv_offset': cur_part_pv_offset,
            'part_pv_point': cur_part_pv_point,
            'part_angles': part_angles,
        }
        active_sum = torch.count_nonzero(rt_dict['label']==1)
        static_sum = torch.count_nonzero(rt_dict['label']==0)
        if active_sum <= 50 or static_sum <= 150:
            return self.__getitem__(index)
        else:
            torch.save(rt_dict, 'data/pc/partial/washing_machine_2/test/' + str(rt_dict['idx'].item()) + '.pt')
            return rt_dict
        
    def __len__(self):
        return len(self.shape_idxes) * self.n_samples

    def get_num_moving_parts_to_cnt(self):
        return self.num_mov_parts_to_cnt

    def reset_num_moving_parts_to_cnt(self):
        self.num_mov_parts_to_cnt = {}


if __name__ == '__main__':
    pass