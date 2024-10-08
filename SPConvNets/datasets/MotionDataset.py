

import os
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
        #rotation range
        self.rot_factor = self.args.equi_settings.rot_factor
        #default 0 (exclude laptop sometimes 2), validation 1
        self.pre_compute_delta = self.args.equi_settings.pre_compute_delta
        self.use_multi_sample = self.args.equi_settings.use_multi_sample
        self.n_samples = self.args.equi_settings.n_samples if self.use_multi_sample == 1 else 1
        self.partial = self.args.equi_settings.partial
        self.est_normals = self.args.equi_settings.est_normals
        self.add_normal_noise = self.args.equi_settings.add_normal_noise


        if self.pre_compute_delta == 1 and self.split == 'train':
            # self.use_multi_sample = False
            self.use_multi_sample = 0
            self.n_samples = 1

        self.train_ratio = 0.9

        self.shape_folders = []

        shape_idxes = os.listdir(self.shape_root)
        shape_idxes = sorted(shape_idxes)
        shape_idxes = [tmpp for tmpp in shape_idxes if tmpp[0] != "."]

        train_nns = int(len(shape_idxes) * self.train_ratio)

        if self.split == "train":
            shape_idxes = shape_idxes[:train_nns]
            # shape_idxes = random.shuffle(shape_idxes)
            random.shuffle(shape_idxes)
        else:
            shape_idxes = shape_idxes[train_nns:]

        self.shape_idxes = shape_idxes

        self.anchors = L.get_anchors(args.model.kanchor)
        equi_anchors = L.get_anchors(60)

        if self.global_rot == 2:
            rotation_angle = sciR.random().as_matrix()
            rotation_matrix = rotation_angle[:3, :3]
            R1 = rotation_matrix
            self.common_R = R1
        elif self.global_rot == 3:
            self.common_R = equi_anchors[7]

        self.kanchor = args.model.kanchor

    def get_test_data(self):
        return self.test_data


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

        cur_vertices, cur_triangles = load_vertices_triangles(cur_mesh_fn)
        cur_triangles_to_seg_idx, seg_idx_to_triangle_idxes = load_triangles_to_seg_idx(cur_surface_to_seg_fn, nparts=nparts)
        cur_triangles, cur_triangles_to_seg_idx = self.refine_triangle_idxes_by_seg_idx(seg_idx_to_triangle_idxes, cur_triangles)
        # cur_motion_attributes = load_motion_attributes(cur_motion_attributes_fn, ex_none=True)
        cur_motion_attributes = load_motion_attributes(cur_motion_attributes_fn) #

        sampled_pcts, pts_to_seg_idx, seg_idx_to_sampled_pts = sample_pts_from_mesh(cur_vertices, cur_triangles, cur_triangles_to_seg_idx, npoints=self.npoints)

        ''' Centralize points in the real canonical state '''
        boundary_pts = [np.min(sampled_pcts, axis=0), np.max(sampled_pcts, axis=0)]
        center_pt = (boundary_pts[0] + boundary_pts[1]) / 2
        length_bb = np.linalg.norm(boundary_pts[0] - boundary_pts[1])
        # all normalize into 0
        sampled_pcts = (sampled_pcts - center_pt.reshape(1, 3)) / length_bb
        ''' Centralize points in the real canonical state '''

        ''' Add global rotation '''
        if self.global_rot == 1 and (not (self.split == "train" and self.pre_compute_delta == 1)):
            if self.args.equi_settings.rot_anchors == 1:
                # just use a matrix from rotation anchors
                R1 = self.get_rotation_from_anchor()
            else:
                rotation_angle = sciR.random().as_matrix()
                rotation_matrix = rotation_angle[:3, :3]
                R1 = rotation_matrix
                x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float)
                y_axis = np.array([0.0, 1.0, 0.0], dtype=np.float)
                z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float)
                ''' Oven --- v2 (small range of view change) --- oven --- axis/pv p -- ok '''
                axis_angle = np.random.randint(-50, 50, (3,))
                x_angle = 1.0 * float(axis_angle[0].item()) / 100. * np.pi
                y_angle = -1.0 * float(axis_angle[1].item()) / 100. * np.pi
                z_angle = 1.0 * float(axis_angle[2].item()) / 100. * np.pi
                ''' Oven --- v2 (small range of view change) --- oven --- axis/pv p -- ok'''
                x_mtx = compute_rotation_matrix_from_axis_angle(x_axis, x_angle)
                y_mtx = compute_rotation_matrix_from_axis_angle(y_axis, y_angle)
                z_mtx = compute_rotation_matrix_from_axis_angle(z_axis, z_angle)
                rotation = np.matmul(z_mtx, np.matmul(y_mtx, x_mtx))
                R1 = rotation
        elif self.global_rot == 2:
            R1 = self.common_R
        else:
            R1 = np.eye(3, dtype=np.float32)
        ''' Add global rotation '''

        part_state_rots, part_ref_rots = [], []
        part_ref_trans = []

        part_state_trans_bbox = []
        part_ref_trans_bbox = []

        part_axis = []
        part_pv_offset = []
        part_pv_point = []
        part_angles = []


        tot_transformed_pts = []
        tot_transformation_mtx = []
        tot_transformation_mtx_segs = []
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
            # Get pts indexes for this segmentation
            cur_seg_pts_idxes = np.array(seg_idx_to_sampled_pts[i_seg], dtype=np.long)
            cur_seg_pts = sampled_pcts[cur_seg_pts_idxes]
            # if self.shape_type == "eyeglasses" and i_seg == 2:
            #     cur_seg_pts = (cur_seg_pts - np.reshape(np.array(cur_seg_motion_info["center"]), (1, 3))) * 1.5 + np.reshape(np.array(cur_seg_motion_info["center"]), (1, 3))
            #     sampled_pcts[cur_seg_pts_idxes] = cur_seg_pts
            if cur_seg_motion_info["motion_type"] == "rotation" and (rot_1 == False or self.shape_type == "eyeglasses"):
                center = cur_seg_motion_info["center"]
                axis = cur_seg_motion_info["direction"]

                if self.use_multi_sample == 0:
                    if self.shape_type in ['laptop', 'eyeglasses']:
                        # if self.shape_type == 'eyeglasses' and nparts is not None:
                        if self.shape_type == 'eyeglasses' and nparts is None:
                            theta = (np.random.uniform(0.05, 1., (1,)).item() * np.pi) * self.rot_factor
                            theta = -1. * theta
                        else:
                            theta = (np.random.uniform(0., 1., (1,)).item() * np.pi - np.pi / 2.) * self.rot_factor
                    elif self.shape_type in ['oven', 'washing_machine']:
                        theta = (np.random.uniform(0.5, 1., (1,)).item() * np.pi) * self.rot_factor
                    else:
                        theta = (np.random.uniform(0., 1., (1,)).item() * np.pi) * self.rot_factor
                else:
                    if self.shape_type in ['laptop', 'eyeglasses']:
                        if self.shape_type == 'eyeglasses' and nparts is None:
                            a_rot_idx = sample_index // 10
                            b_rot_idx = sample_index % 10
                            mult_factor = 0.45 if self.split == 'train' else 0.35
                            if i_seg == 1:
                                theta = (0.1 * (a_rot_idx ) * np.pi) * mult_factor
                            elif i_seg == 2:
                                theta = (0.1 * (b_rot_idx ) * np.pi) * mult_factor
                        else:
                            theta = -(0.5 / self.n_samples * sample_index * np.pi) + 0.1 * np.pi
                            #theta = (0.5 / self.n_samples * sample_index * np.pi) - 0.05 * np.pi
                            #theta = -1. * theta
                    elif self.shape_type in ['oven', 'washing_machine']:
                        if self.shape_type == 'washing_machine':
                            theta = (((90. / 180.) / self.n_samples) * sample_index + 45. / 180.) * np.pi
                        else:
                            theta = (((80. / 180.) / self.n_samples) * sample_index + 45. / 180.) * np.pi
                    else:
                        theta = (np.random.uniform(0., 1., (1,)).item() * np.pi) * self.rot_factor

                # Get transformed center (the center of the axis?)
                center = (center - center_pt) / length_bb
                # axis = (axis - center_pt) / length_bb

                part_angles.append(theta)

                part_axis.append(np.reshape(axis, (1, 3)))

                center_pt_offset = center - np.sum(axis * center, axis=0, keepdims=True) * axis
                center_pt_offset = np.sqrt(np.sum(center_pt_offset ** 2, axis=0))
                # part_pv_offset.append(np.reshap)
                part_pv_offset.append(center_pt_offset)
                part_pv_point.append(np.reshape(center, (1, 3)))


                rot_pts, transformation_mtx = revoluteTransform(cur_seg_pts, center, axis, theta)
                transformation_mtx = np.transpose(transformation_mtx, (1, 0))

                rot_pts[:, :3] = np.matmul(np.reshape(R1, (1, 3, 3)), np.reshape(rot_pts[:, :3], (rot_pts.shape[0], 3, 1)))[:, :3, 0]
                transformation_mtx[:3] = np.matmul(R1, transformation_mtx[:3])
                ''' Transform points via revolute transformation '''

                ''' Get points state translations with centralized boudning box '''
                rot_pts_minn = np.min(rot_pts[:, :3], axis=0)
                rot_pts_maxx = np.max(rot_pts[:, :3], axis=0)
                rot_pts_bbox_center = (rot_pts_minn + rot_pts_maxx) / 2.
                # print(f"transformation_mtx: {transformation_mtx[:3, 3].shape}, rot_pts_bbox_center: {rot_pts_bbox_center.shape}")
                cur_part_state_trans_bbox = transformation_mtx[:3, 3] - rot_pts_bbox_center
                part_state_trans_bbox.append(np.reshape(cur_part_state_trans_bbox, (1, 3)))


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
                #
                canon_rot_pts, canon_transformation_mtx = revoluteTransform(cur_seg_pts, center, axis, canon_theta)
                canon_transformation_mtx = np.transpose(canon_transformation_mtx, (1, 0))

                ''' Get points state translations with centralized boudning box '''
                # rot_pts: n_pts_part x 3
                canon_rot_pts_minn = np.min(canon_rot_pts[:, :3], axis=0)
                canon_rot_pts_maxx = np.max(canon_rot_pts[:, :3], axis=0)
                canon_rot_pts_bbox_center = (canon_rot_pts_minn + canon_rot_pts_maxx) / 2.
                cur_part_ref_trans_bbox = canon_transformation_mtx[:3, 3] - canon_rot_pts_bbox_center
                part_ref_trans_bbox.append(np.reshape(cur_part_ref_trans_bbox, (1, 3)))


                # transformation_mtx = np.transpose(transformation_mtx, (1, 0))
                transformation_mtx = np.reshape(transformation_mtx, (1, 4, 4))

                tot_transformation_mtx += [transformation_mtx for _ in range(cur_seg_pts.shape[0])]
                tot_transformation_mtx_segs.append(transformation_mtx)

                part_state_rots.append(np.reshape(transformation_mtx[0, :3, :3], (1, 3, 3)))

                # canon_transformation_mtx = np.transpose(canon_transformation_mtx, (1, 0))
                part_ref_rots.append(np.reshape(canon_transformation_mtx[:3, :3], (1, 3, 3)))
                part_ref_trans.append(np.reshape(canon_transformation_mtx[:3, 3], (1, 3)))

                if self.pre_compute_delta == 1 and self.split == "train":
                    tot_transformed_pts.append(canon_rot_pts[:, :3])
                else:
                    tot_transformed_pts.append(rot_pts[:, :3])
                canon_transformed_pts.append(canon_rot_pts[:, :3])
                rot_1 = True
            else:
                # rot_pts: n_pts_part x 3
                cur_seg_pts_minn = np.min(cur_seg_pts, axis=0)
                cur_seg_pts_maxx = np.max(cur_seg_pts, axis=0)
                cur_seg_pts_bbox_center = (cur_seg_pts_minn + cur_seg_pts_maxx) / 2.
                cur_part_ref_trans_bbox = -1. * cur_seg_pts_bbox_center
                canon_transformed_pts.append(cur_seg_pts)

                rot_pts = np.zeros_like(cur_seg_pts)
                rot_pts[:, :] = cur_seg_pts[:, :]
                # rot_pts = cur_seg_pts
                rot_pts[:, :3] = np.matmul(np.reshape(R1, (1, 3, 3)),
                                            np.reshape(rot_pts[:, :3], (rot_pts.shape[0], 3, 1)))[:, :3, 0]
                transformation_mtx = np.zeros((4, 4), dtype=np.float)
                transformation_mtx[0, 0] = 1.; transformation_mtx[1, 1] = 1.; transformation_mtx[2, 2] = 1.
                tot_transformed_pts.append(rot_pts)
                transformation_mtx[:3] = np.matmul(R1, transformation_mtx[:3])

                transformation_mtx = np.reshape(transformation_mtx, (1, 4, 4))

                tot_transformation_mtx += [transformation_mtx for _ in range(cur_seg_pts.shape[0])]
                tot_transformation_mtx_segs.append(transformation_mtx)


                part_state_rots.append(np.reshape(transformation_mtx[0, :3, :3], (1, 3, 3)))

                canon_transformation_mtx = np.zeros((4, 4), dtype=np.float)
                canon_transformation_mtx[0, 0] = 1.
                canon_transformation_mtx[1, 1] = 1.
                canon_transformation_mtx[2, 2] = 1.
                part_ref_rots.append(np.reshape(canon_transformation_mtx[:3, :3], (1, 3, 3)))
                part_ref_trans.append(np.zeros((1, 3), dtype=np.float))

                ''' Get points state translations with centralized boudning box '''
                # rot_pts: n_pts_part x 3
                rot_pts_minn = np.min(rot_pts, axis=0)
                rot_pts_maxx = np.max(rot_pts, axis=0)
                rot_pts_bbox_center = (rot_pts_minn + rot_pts_maxx) / 2.
                cur_part_state_trans_bbox = -1. * rot_pts_bbox_center # state trans bbox
                part_state_trans_bbox.append(np.reshape(cur_part_state_trans_bbox, (1, 3)))

                ''' Ref transformation bbox '''
                # # rot_pts: n_pts_part x 3
                # rot_pts_minn = np.min(cur_seg_pts, axis=0)
                # rot_pts_maxx = np.max(rot_pts, axis=0)
                # rot_pts_bbox_center = (rot_pts_minn + rot_pts_maxx) / 2.
                # cur_part_state_trans_bbox = -1. * rot_pts_bbox_center
                part_ref_trans_bbox.append(np.reshape(cur_part_ref_trans_bbox, (1, 3)))

        ''' Concatenate part axis '''
        # part_axis: n_part x 3; part_axis: n_part x 3 --> part axis...
        part_axis = np.concatenate(part_axis, axis=0)
        part_axis = np.matmul(np.reshape(R1, (1, 3, 3)), np.reshape(part_axis, (part_axis.shape[0], 3, 1)))
        part_axis = np.reshape(part_axis, (part_axis.shape[0], 3))

        part_pv_offset = np.array(part_pv_offset)

        part_pv_point = np.concatenate(part_pv_point, axis=0)
        part_pv_point = np.matmul(np.reshape(R1, (1, 3, 3)), np.reshape(part_pv_point, (part_pv_point.shape[0], 3, 1)))
        part_pv_point = np.reshape(part_pv_point, (part_pv_point.shape[0], 3))

        tot_transformed_pts = np.concatenate(tot_transformed_pts, axis=0)
        canon_transformed_pts = np.concatenate(canon_transformed_pts, axis=0)
        # print(tot_transformed_pts.shape)

        ''' Use fake initial pose '''
        # gt_pose = np.zeros((self.npoints, 4, 4), dtype=np.float)
        # gt_pose[:, 0, 0] = 1.; gt_pose[:, 1, 1] = 1.; gt_pose[:, 2, 2] = 1.

        ''' Use GT transformation matrix as initial pose '''
        tot_transformation_mtx = np.concatenate(tot_transformation_mtx, axis=0)
        tot_transformation_mtx_segs = np.concatenate(tot_transformation_mtx_segs, axis=0)

        part_state_rots = np.concatenate(part_state_rots, axis=0)
        part_ref_rots = np.concatenate(part_ref_rots, axis=0)
        part_ref_trans = np.concatenate(part_ref_trans, axis=0)
        ''' Get part_state_trans_bbox and part_ref_trans_bbox '''
        part_state_trans_bbox = np.concatenate(part_state_trans_bbox, axis=0)
        part_ref_trans_bbox = np.concatenate(part_ref_trans_bbox, axis=0)
        
        
        gt_pose = tot_transformation_mtx
        # tot_transformation_mt

        if self.global_rot >= 0:
            af_glb_boundary_pts = [np.min(tot_transformed_pts, axis=0), np.max(tot_transformed_pts, axis=0)]
            af_glb_center_pt = (af_glb_boundary_pts[0] + af_glb_boundary_pts[1]) / 2
            # length_bb = np.linalg.norm(af_glb_boundary_pts[0] - af_glb_boundary_pts[1])

            af_glb_center_pt = np.mean(tot_transformed_pts, axis=0)

            # latest work? aiaiaia...

            # all normalize into 0
            # sampled_pcts = (sampled_pcts - center_pt.reshape(1, 3)) / length_bb
            tot_transformed_pts = (tot_transformed_pts - af_glb_center_pt.reshape(1, 3))
            # tot_transformed_pts = (tot_transformed_pts - af_glb_center_pt.reshape(1, 3)) / length_bb

            gt_pose[:, :3, 3] = gt_pose[:, :3, 3] - af_glb_center_pt
            tot_transformation_mtx_segs[:, :3, 3] = tot_transformation_mtx_segs[:, :3, 3] - af_glb_center_pt

            part_pv_point = part_pv_point - np.reshape(af_glb_center_pt, (1, 3))
            part_pv_offset = part_pv_point - np.sum(part_pv_point * part_axis, axis=-1, keepdims=True) * part_axis
            part_pv_offset = np.sqrt(np.sum(part_pv_offset ** 2, axis=-1))

            # part ref trans and centralize bbox?

        # if self.partial == 1:
        #     tot_transformed_pts[:, 2] = 0.

        cur_pc = torch.from_numpy(tot_transformed_pts.astype(np.float32)).float()
        tot_transformed_pts = torch.from_numpy(tot_transformed_pts.astype(np.float32)).float()
        cur_label = torch.from_numpy(pts_to_seg_idx).long()
        tot_label = torch.from_numpy(pts_to_seg_idx).long()
        cur_pose = torch.from_numpy(gt_pose.astype(np.float32))
        cur_pose_segs = torch.from_numpy(tot_transformation_mtx_segs.astype(np.float32))
        cur_ori_pc = torch.from_numpy(sampled_pcts.astype(np.float32)).float()
        cur_canon_transformed_pts = torch.from_numpy(canon_transformed_pts.astype(np.float32)).float()
        cur_part_state_rots = torch.from_numpy(part_state_rots.astype(np.float32)).float()
        cur_part_ref_rots = torch.from_numpy(part_ref_rots.astype(np.float32)).float()
        cur_part_ref_trans = torch.from_numpy(part_ref_trans.astype(np.float32)).float()
        cur_part_axis = torch.from_numpy(part_axis.astype(np.float32)).float()
        cur_part_pv_offset = torch.from_numpy(part_pv_offset.astype(np.float32)).float()
        cur_part_pv_point = torch.from_numpy(part_pv_point.astype(np.float32)).float()
        part_angles = np.array(part_angles, dtype=np.float32)  # .float()
        part_angles = torch.from_numpy(part_angles.astype(np.float32)).float()

        if self.est_normals == 1:
            cur_normals = estimate_normals(cur_pc)
            cur_canon_normals = estimate_normals(cur_canon_transformed_pts)

        # part_state_trans_bbox part_ref_trans_bbox
        cur_part_state_trans_bbox = torch.from_numpy(part_state_trans_bbox.astype(np.float32)).float()
        cur_part_ref_trans_bbox = torch.from_numpy(part_ref_trans_bbox.astype(np.float32)).float()


        fps_idx = farthest_point_sampling(cur_pc.unsqueeze(0), n_sampling=self.npoints)
        #fps_idx_oorr = farthest_point_sampling(cur_pc.unsqueeze(0), n_sampling=4096)

        #tot_transformed_pts = tot_transformed_pts[fps_idx_oorr]
        #tot_label = tot_label[fps_idx_oorr]
        cur_pc = cur_pc[fps_idx]
        cur_label = cur_label[fps_idx]
        cur_pose = cur_pose[fps_idx]

        # cur_pose = cur_pose[fps_idx]
        cur_ori_pc = cur_ori_pc[fps_idx] # ori_pc
        #cur_oorr_canon_transformed_pts = cur_canon_transformed_pts[fps_idx_oorr]
        cur_canon_transformed_pts = cur_canon_transformed_pts[fps_idx]
        if self.est_normals == 1:
            cur_normals = cur_normals[fps_idx]
            cur_canon_normals = cur_canon_normals[fps_idx]

        idx_arr = np.array([index], dtype=np.long)
        idx_arr = torch.from_numpy(idx_arr).long()
        shp_idx_arr = torch.tensor([shp_idx_int], dtype=torch.long).long()
        sampled_idx_arr = torch.tensor([sample_idx_int], dtype=torch.long).long()

        # pc: N x 3 --> how to get corrupted pc?
        if self.add_normal_noise > 0.0:
            cur_generated_normal_noise = np.random.normal(loc=0.0, scale=self.add_normal_noise, size=(cur_pc.size(0), 3))
            cur_generated_normal_noise = torch.from_numpy(cur_generated_normal_noise).float()
            cur_pc = cur_pc + cur_generated_normal_noise

        rt_dict = {
            'pc': cur_pc.contiguous(),
            'af_pc': cur_pc.contiguous(),
            'ori_pc': cur_ori_pc.contiguous(),
            'canon_pc': cur_canon_transformed_pts, #.contiguous().transpose(0, 1).contiguous(),
            #'oorr_pc': tot_transformed_pts.contiguous(), # 3 x oorr_N
            #'oorr_canon_pc': cur_oorr_canon_transformed_pts.contiguous(),
            'label': cur_label,
            #'oorr_label': tot_label, # oorr_N
            'pose': cur_pose,
            'pose_segs': cur_pose_segs,
            'part_state_rots': cur_part_state_rots,
            'part_ref_rots': cur_part_ref_rots,
            'part_ref_trans': cur_part_ref_trans,
            'part_axis': cur_part_axis, # get ground-truth axis
            'idx': idx_arr,
            'shp_idx': shp_idx_arr,
            'sampled_idx': sampled_idx_arr,
            'part_state_trans_bbox': cur_part_state_trans_bbox,
            'part_ref_trans_bbox': cur_part_ref_trans_bbox,
            'part_pv_offset': cur_part_pv_offset,
            'part_pv_point': cur_part_pv_point,
            'part_angles': part_angles,
        }
        if self.est_normals == 1:
            rt_dict['cur_normals'] = cur_normals.contiguous()
            rt_dict['cur_canon_normals'] = cur_canon_normals.contiguous()
        torch.save(rt_dict, 'data/pc/full/oven_2/test/' + str(rt_dict['idx'].item()) + '.pt')
        #np.savez_compressed('test/'+str(index).zfill(5)+'.npz', rt_dict)
        return rt_dict
        # return np.array([chosen_whether_mov_index], dtype=np.long), np.array([chosen_num_moving_parts], dtype=np.long), \
        #        pc1, pc2, flow12, shape_seg_masks, motion_seg_masks, pc1, pc1

    def __len__(self):
        return len(self.shape_idxes) * self.n_samples

    def get_num_moving_parts_to_cnt(self):
        return self.num_mov_parts_to_cnt

    def reset_num_moving_parts_to_cnt(self):
        self.num_mov_parts_to_cnt = {}


if __name__ == '__main__':
    pass
