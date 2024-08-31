import math
import os
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


from torch.nn.modules.batchnorm import _BatchNorm
from vgtk.spconv import Gathering
from vgtk.spconv.functional import batched_index_select, acos_safe
from vgtk.functional import compute_rotation_matrix_from_quaternion, compute_rotation_matrix_from_ortho6d, so3_mean
import vgtk.so3conv as sgtk

# ------------------------------------- loss ------------------------------------
class CrossEntropyLoss(torch.nn.Module):
    def __init__(self):
        """For multi-class classification. """
        super(CrossEntropyLoss, self).__init__()
        self.metric = torch.nn.CrossEntropyLoss()

    def forward(self, pred, label):
        """The classification dim of pred is always at dim 1. label.shape == pred.shape[[0,2:]]. """
        _, pred_label = pred.max(1)

        pred_label = pred_label.reshape(-1)
        label_flattened = label.reshape(-1)

        acc = (pred_label == label_flattened).sum().float() / float(label_flattened.shape[0])
        return self.metric(pred, label), acc

class AttentionCrossEntropyLoss(torch.nn.Module):
    def __init__(self, loss_type, loss_margin):
        """CrossEntropyLoss on category and rotation. """
        super(AttentionCrossEntropyLoss, self).__init__()
        self.metric = CrossEntropyLoss()
        self.loss_type = loss_type
        self.loss_margin = loss_margin
        self.iter_counter = 0

    def forward(self, pred, label, wts, rlabel, pretrain_step=2000):
        """
        pred: unnormalized scores for category classification
        label: category label
        wts: (BxA) rotation class prediction
        rlabel: (B) rotation class label
        """
        ### category classification loss
        cls_loss, acc = self.metric(pred, label)

        ### rotation classification loss

        r_loss, racc = self.metric(wts, rlabel)

        m = self.loss_margin
        loss_type = self.loss_type

        if loss_type == 'schedule':
            cls_loss_wts = min(float(self.iter_counter) / pretrain_step, 1.0)
            loss = cls_loss_wts * cls_loss + (m + 1.0 - cls_loss_wts) * r_loss
        elif loss_type == 'default':
            loss = cls_loss + m * r_loss
        elif loss_type == 'no_reg':
            loss = cls_loss
        else:
            raise NotImplementedError(f"{loss_type} is not Implemented!")

        if self.training:
            self.iter_counter += 1

        ### total loss, cross entropy loss on categories, cross entropy loss on rotations, 
        ### accuracy of categories classification, accuracy of rotation classification
        return loss, cls_loss, r_loss, acc, racc

class AttPermuteCrossEntropyLoss(torch.nn.Module):
    def __init__(self, loss_type, loss_margin, device, anchor_ab_loss, cross_ab, cross_ab_T):
        """CrossEntropyLoss on category and BCEWithLogitsLoss on anchors. """
        super(AttPermuteCrossEntropyLoss, self).__init__()
        self.metric = CrossEntropyLoss()
        self.loss_type = loss_type
        self.loss_margin = loss_margin
        self.iter_counter = 0
        self.anchor_ab_loss = anchor_ab_loss
        self.cross_ab = cross_ab
        self.cross_ab_T = cross_ab_T
        if self.cross_ab:
            assert self.anchor_ab_loss, 'cross_ab is valid only if anchor_ab_loss is True'
            self.bn_classifier = nn.CrossEntropyLoss()
        else:
            positive_weight = 11 #11 = (12-1)/1, 11 = (720-60)/60  # 11/10/2022, previously 12
            self.bn_classifier = nn.BCEWithLogitsLoss(pos_weight=torch.ones([1], device=device)*positive_weight) # pos_weight has to be a tensor on the correct device

    def forward(self, pred, label, wts, rlabel, pretrain_step=2000, anchor_label=None):
        """
        pred: (BC) unnormalized scores for category classification
        label: (B) category label
        wts: (BRA, R for rotations, A for anchors) anchor alignment prediction
        rlabel: (B) rotation class label
        anchor_label: (BRA) anchor alignment label
        """
        ### category classification loss
        cls_loss, acc = self.metric(pred, label)
        
        ### rotation classification loss
        # binary loss on the anchors
        if self.cross_ab:
            if self.cross_ab_T:
                wts = wts.transpose(1,2)
            r_loss = self.bn_classifier(wts, anchor_label)  # b*12*12, b*12
        else:
            r_loss = self.bn_classifier(wts.reshape(-1), anchor_label.reshape(-1) )   #[-1], [-1], labels also need to be float

        # cross entropy loss on the rotations
        confidence = torch.sigmoid(wts)
        confidence_r = confidence.sum(-1)   # b, r
        cls_loss_r, racc = self.metric(confidence_r, rlabel)    # [b,r], [b], labels does not have the class dimension, therefore of ndim one less than prediction

        m = self.loss_margin
        loss_type = self.loss_type

        if loss_type == 'schedule':
            cls_loss_wts = min(float(self.iter_counter) / pretrain_step, 1.0)
            loss = cls_loss_wts * cls_loss + (m + 1.0 - cls_loss_wts) * r_loss
        elif loss_type == 'default':
            loss = cls_loss + m * r_loss
        elif loss_type == 'no_reg':
            loss = cls_loss
        else:
            raise NotImplementedError(f"{loss_type} is not Implemented!")

        if self.training:
            self.iter_counter += 1

        ### total loss, cross entropy loss on categories, binary classification loss on anchors, 
        ### accuracy of categories classification, accuracy of rotation classification
        return loss, cls_loss, r_loss, acc, racc


def batched_select_anchor(labels, y, rotation_mapping):
    '''
        (b, c, na_tgt, na_src) x (b, na_tgt)
            -> (b, na_src, c)
            -> (b, na, 3, 3)

    '''
    b, na = labels.shape
    preds_rs = labels.view(-1)[:,None]
    y_rs = y.transpose(1,3).contiguous()
    y_rs = y_rs.view(b*na,na,-1)
    # select into (nb, na, nc) features
    y_select = batched_index_select(y_rs, 1, preds_rs).view(b*na,-1)
    # (nb, na, 3, 3)
    pred_RAnchor = rotation_mapping(y_select).view(b,na,3,3).contiguous()
    return pred_RAnchor

class MultiTaskDetectionLoss(torch.nn.Module):
    def __init__(self, anchors, rot_ref_tgt, nr=4, w=10, threshold=1.0, s2_mode=False, r_cls_loss=False, topk=1, writer=None, logger=None):
        """Classification and regression loss on rotations. """
        super(MultiTaskDetectionLoss, self).__init__()
        self.classifier = CrossEntropyLoss()
        self.anchors = anchors
        self.nr = nr
        assert nr == 4 or nr == 6
        self.w = w
        self.threshold = threshold
        self.iter_counter = 0

        self.s2_mode = s2_mode
        self.r_cls_loss = r_cls_loss
        self.rot_ref_tgt = rot_ref_tgt
        self.topk = topk
        self.writer = writer
        self.logger = logger
        self.bn_classifier = nn.BCEWithLogitsLoss(pos_weight=torch.ones([1], device=anchors.device)*12) # pos_weight has to be a tensor on the correct device

    def forward_simple(self, wts, label, y, gt_R, gt_T, anchor_label):
        ''' setting for alignment regression:
                - label (nb,) or (nb,1)?:
                    label the targte anchor from the perspective of source anchor na
                - wts (nb, n_rotations, na) normalized confidence weights
                - y (nb, d_rotation) features
                - gt_R (nb, 3, 3)
                    relative rotation to regress from the perspective of source anchor na
                    gt_T = gt_R @ anchors[label]
                - gt_T (nb, 3, 3)
                    ground truth relative rotation: gt_T @ R_tgt = R_src
                - anchor_label (nb, n_rotations, na)
                    ground truth relative rotation: gt_T @ R_tgt = R_src
        '''
        b = wts.shape[0]
        nr = self.nr # 4 or 6
        na = wts.shape[1]
        rotation_mapping = compute_rotation_matrix_from_quaternion if nr == 4 else compute_rotation_matrix_from_ortho6d

        true_R = gt_T

        label = label.reshape(-1)
        ### rotation classification loss
        cls_loss = self.bn_classifier(wts.reshape(-1), anchor_label.reshape(-1) )   #[-1], [-1], labels also need to be float

        confidence = torch.sigmoid(wts)
        confidence_r = confidence.sum(-1)   # b, r
        cls_loss_r, r_acc = self.classifier(confidence_r, label)    # [b,r], [b], labels does not have the class dimension, therefore of ndim one less than prediction
        if self.r_cls_loss:
            cls_loss = cls_loss + cls_loss_r


        if self.logger is not None and self.iter_counter % 1000 == 0:
            confidence_softmax1 = F.softmax(wts.mean(-1), 1).detach()
            # confidence_softmax2 = F.softmax(confidence_r, 1).detach()
            # confidence_softmax3 = F.softmax(confidence.mean(-1), 1).detach()
            # self.logger.log('Loss', f'conf1 top 6: {torch.topk(confidence_softmax1, 6, 1)[0]}')
            # self.logger.log('Loss', f'conf2 top 10: {torch.topk(confidence_softmax2, 10, 1)[0]}')
            # self.logger.log('Loss', f'conf3 top 10: {torch.topk(confidence_softmax3, 10, 1)[0]}')
        if self.topk == 1:
            _, max_r = torch.max(confidence_r, 1)  # b
            anchor_pred = self.anchors[max_r]
            R_pred = rotation_mapping(y).view(b,3,3).contiguous()
            if self.rot_ref_tgt:
                pred_R = torch.matmul(anchor_pred, R_pred)
                gt_R_for_anchor_pred = torch.einsum("bji,bjk->bik", anchor_pred, gt_T)
            else:
                pred_R = torch.matmul(R_pred, anchor_pred)

                gt_R_for_anchor_pred = torch.einsum("bij,bkj->bik", gt_T, anchor_pred)
            
            # option 1: l2 loss for the prediction at each "tight" anchor pair
            l2_loss = torch.pow(gt_R_for_anchor_pred - R_pred,2).mean()
            loss = cls_loss + self.w * l2_loss
        else:
            confidence_softmax = F.softmax(wts.mean(-1), 1).detach()
            # if self.training:
            #     self.writer.add_histogram("wtsmean_sfm_tr", confidence_softmax, self.iter_counter)
            #     self.writer.add_histogram("cfsgsum_sfm_tr", F.softmax(confidence_r, 1).detach(), self.iter_counter)
            # else:
            #     self.writer.add_histogram("wtsmean_sfm_ev", confidence_softmax, self.iter_counter)
            #     self.writer.add_histogram("cfsgsum_sfm_ev", F.softmax(confidence_r, 1).detach(), self.iter_counter)
            
            top_conf, top_idx = torch.topk(confidence_softmax, self.topk, 1)    # b, topk

            top_mask = top_conf > 0.1
            top_mask_all = torch.ones_like(top_mask)

            top_mask_row = top_mask.sum(1)
            top_mask_more_than_one = top_mask_row > 1
            top_mask_row = top_mask_row.to(torch.bool)
            top_mask_zero = ~top_mask_row
            top_mask_ambiguous = top_mask_more_than_one | top_mask_zero
            top_mask_row = top_mask_row.unsqueeze(1).expand_as(top_mask)
            # top_mask_max[:,0] = True

            top_mask_final = torch.where(top_mask_row, top_mask, top_mask_all)
            top_conf = torch.where(top_mask_final, top_conf, torch.zeros_like(top_conf))
            top_conf = F.normalize(top_conf, p=1, dim=1)# b, topk

            anchor_pred = self.anchors[top_idx] # b, topk, 3, 3
            anchor_pred_top = anchor_pred[:, 0]
            y_top = y[:,self.topk]
            R_pred_top = rotation_mapping(y_top).view(b,3,3).contiguous()
            y = y[:, :self.topk]
            R_pred = rotation_mapping(y.reshape(-1, nr)).view(b, self.topk,3,3).contiguous()
            if self.rot_ref_tgt:
                pred_R = torch.einsum('btij, btjk->btik', anchor_pred, R_pred)
                gt_R_for_anchor_pred = torch.einsum("btji,bjk->btik", anchor_pred, gt_T)

                pred_R_top = torch.matmul(anchor_pred_top, R_pred_top)
                gt_R_for_anchor_pred_top = torch.einsum("bji,bjk->bik", anchor_pred_top, gt_T)
            else:
                pred_R = torch.einsum('btij, btjk->btik', R_pred, anchor_pred)
                gt_R_for_anchor_pred = torch.einsum("btij,btkj->btik", gt_T, anchor_pred)
                
                pred_R_top = torch.matmul(R_pred_top, anchor_pred_top)
                gt_R_for_anchor_pred_top = torch.einsum("bij,bkj->bik", gt_T, anchor_pred_top)

            # pred_R = so3_mean(pred_R, top_conf)
            pred_R = pred_R[:,0]    # bik

            pred_R = torch.where(top_mask_ambiguous.reshape(b,1,1).expand_as(pred_R), pred_R, pred_R_top)

            # l2_loss = torch.pow(true_R - pred_R,2).mean()
            l2_loss_ambiguous = torch.pow(gt_R_for_anchor_pred - R_pred,2).mean((2,3))
            # l2_loss_ambiguous = (l2_loss_ambiguous * top_conf).sum(1)
            l2_loss_ambiguous = l2_loss_ambiguous[:, 0]

            l2_loss_max = torch.pow(gt_R_for_anchor_pred_top - R_pred_top,2).mean((1,2))

            l2_loss = torch.where(top_mask_ambiguous, l2_loss_ambiguous, l2_loss_max).mean()

            if self.logger is not None and self.iter_counter % 1000 == 0:
                self.logger.log('Loss', f'top_mask_ambiguous: {top_mask_ambiguous}')
                self.logger.log('Loss', f'l2_loss_ambiguous: {l2_loss_ambiguous}')
                self.logger.log('Loss', f'l2_loss_max: {l2_loss_max}')

            loss = cls_loss + self.w * l2_loss

        if self.training:
            self.iter_counter += 1

        ### total loss, binary classification loss on anchors, l2 loss on residual rotation matrix, 
        ### accuracy of rotation classification, angular error of rotation prediction
        return loss, cls_loss, self.w * l2_loss, r_acc, mean_angular_error(pred_R, true_R)

    def forward(self, wts, label, y, gt_R, gt_T=None, anchor_label=None):
        ''' setting for alignment regression:
                - label (nb, na):
                    label the targte anchor from the perspective of source anchor na
                - wts (nb, na_tgt, na_src) normalized confidence weights
                - y (nb, nr, na_tgt, na_src) features
                - gt_R (nb, na, 3, 3)
                    relative rotation to regress from the perspective of source anchor na
                    Ra_tgti @ gt_R_i @ Ra_srci.T = gt_T for each i (Minghan: wrong,
                    should be Ra_srci @ gt_R_i @ Ra_tgti.T = gt_T for each i, see line with einsum)
                - gt_T (nb, 3, 3)
                    ground truth relative rotation: gt_T @ R_tgt = R_src

            setting for canonical regression:
                - label (nb)
                - wts (nb, na) normalized confidence weights
                - y (nb, nr, na) features to be mapped to 3x3 rotation matrices
                - gt_R (nb, na, 3, 3) relative rotation between gtR and each anchor
        '''

        if self.s2_mode:
            return self.forward_simple(wts, label, y, gt_R, gt_T, anchor_label)

        b = wts.shape[0]
        nr = self.nr # 4 or 6
        na = wts.shape[1]
        rotation_mapping = compute_rotation_matrix_from_quaternion if nr == 4 else compute_rotation_matrix_from_ortho6d

        true_R = gt_R[:,29] if gt_T is None else gt_T

        if na == 1:
            # single anchor regression problem
            target_R = true_R
            cls_loss = torch.zeros(1)
            r_acc = torch.zeros(1) + 1
            # Bx6 -> Bx3x3
            pred_R = rotation_mapping(y.view(b,nr))
            l2_loss = torch.pow(pred_R - target_R,2).mean()
            loss = self.w * l2_loss
        elif gt_T is not None and label.ndimension() == 2:
            # Alignment setting
            wts = wts.view(b,na,na)
            cls_loss, r_acc = self.classifier(wts, label)

            # first select the chosen target anchor (nb, na_src)
            confidence, preds = wts.max(1)

            # the followings are [nb, na, 3, 3] predictions of relative rotation
            select_RAnchor = batched_select_anchor(label, y, rotation_mapping)
            pred_RAnchor = batched_select_anchor(preds, y, rotation_mapping)

            # normalize the conrfidence
            confidence = confidence / (1e-6 + torch.sum(confidence,1,keepdim=True))

            # nb, na, 3, 3
            anchors_src = self.anchors[None].expand(b,-1,-1,-1).contiguous()
            pred_Rs = torch.einsum('baij, bajk, balk -> bail', \
                                   anchors_src, pred_RAnchor, self.anchors[preds])

            # pred_Rs_with_label = torch.einsum('baij, bajk, balk -> bail', \
            #                        anchors_src, select_RAnchor, self.anchors[label])

            ##############################################
            # gt_Rs = torch.einsum('baij, bajk, balk -> bail',\
            #                      anchors_src, gt_R, self.anchors[label])
            # gtrmean = so3_mean(gt_Rs)
            # print(torch.sum(gtrmean - true_R))
            # import ipdb; ipdb.set_trace()
            ####################################################

            # how the fuck do you average rotations? closed form under chordal l2 mean
            pred_R = so3_mean(pred_Rs, confidence)

            # option 1: l2 loss for the prediction at each "tight" anchor pair
            l2_loss = torch.pow(gt_R - select_RAnchor,2).mean()

            # option 2: l2 loss based on the relative prediction with gt label
            # l2_loss = torch.pow(true_R - pred_R_with_label,2).mean() # + torch.pow(gt_R - select_RAnchor,2).mean()

            # loss = self.w * l2_loss
            loss = cls_loss + self.w * l2_loss

        else:
            # single shape Canonical Regression setting
            wts = wts.view(b,-1)
            cls_loss, r_acc = self.classifier(wts, label)
            pred_RAnchor = rotation_mapping(y.transpose(1,2).contiguous().view(-1,nr)).view(b,-1,3,3)

            # option 1: only learn to regress the closest anchor
            #
            # pred_ra = batched_index_select(pred_RAnchor, 1, label.long().view(b,-1)).view(b,3,3)
            # target_R = batched_index_select(gt_R, 1, label.long().view(b,-1)).view(b,3,3)
            # l2_loss = torch.pow(pred_ra - target_R,2).mean()
            # loss = cls_loss + self.w * l2_loss

            # option 2: regress nearby anchors within an angular threshold
            gt_bias = angle_from_R(gt_R.view(-1,3,3)).view(b,-1)
            mask = (gt_bias < self.threshold)[:,:,None,None].float()
            l2_loss = torch.pow(gt_R * mask - pred_RAnchor * mask,2).sum()
            loss = cls_loss + self.w * l2_loss

            preds = torch.argmax(wts, 1)
            # The actual prediction is the classified anchor rotation @ regressed rotation
            pred_R = batched_index_select(pred_RAnchor, 1, preds.long().view(b,-1)).view(b,3,3)
            pred_R = torch.matmul(self.anchors[preds], pred_R)

        if self.training:
            self.iter_counter += 1

        ### total loss, cross entropy loss on rotations, l2 loss on residual rotation matrix, 
        ### accuracy of rotation classification, angular error of rotation prediction
        return loss, cls_loss, self.w * l2_loss, r_acc, mean_angular_error(pred_R, true_R)

def angle_from_R(R):
    return acos_safe(0.5 * (torch.einsum('bii->b',R) - 1))

def mean_angular_error(pred_R, gt_R):
    R_diff = torch.matmul(pred_R, gt_R.transpose(1,2).float())
    angles = angle_from_R(R_diff)
    return angles#.mean()

def pairwise_distance_matrix(x, y, eps=1e-6):
    M, N = x.size(0), y.size(0)
    x2 = torch.sum(x * x, dim=1, keepdim=True).repeat(1, N)
    y2 = torch.sum(y * y, dim=1, keepdim=True).repeat(1, M)
    dist2 = x2 + torch.t(y2) - 2.0 * torch.matmul(x, torch.t(y))
    dist2 = torch.clamp(dist2, min=eps)
    return torch.sqrt(dist2)


def batch_hard_negative_mining(dist_mat):
    M, N = dist_mat.size(0), dist_mat.size(1)
    assert M == N
    labels = torch.arange(N, device=dist_mat.device).view(N, 1).expand(N, N)
    is_neg = labels.ne(labels.t())
    dist_an, _ = torch.min(torch.reshape(dist_mat[is_neg], (N, -1)), 1, keepdim=False)
    return dist_an



class TripletBatchLoss(nn.Module):
    def __init__(self, opt, anchors=None, sigma=2e-1, \
                 interpolation='spherical', alpha=0.0,
                 beta=0.0, gamma=0.0, eta=0.0, use_innerp=False,
                 ):
        '''
            anchors: na x 3 x 3, default anchor rotations
            margin: float, for triplet loss margin value
            sigma: float, sigma for softmax function
            loss: str "none" | "soft" | "hard", for loss mode
            interpolation: str "spherical" | "linear"
        '''
        super(TripletBatchLoss, self).__init__()
        
        self.alpha = alpha
        self.beta = beta    # for permute loss
        self.gamma = gamma    # for anchor alignment loss
        self.eta = eta  # for normal classification loss
        self.sigma = sigma
        self.use_innerp = use_innerp

        if anchors is None:
            if opt.model.kanchor == 12:
                anchors = sgtk.get_anchorsV()
                anchors = torch.tensor(anchors).to(device=opt.device)
                if self.beta > 0:
                    trace_idx_ori, trace_idx_rot = sgtk.get_relativeVR_index(full=True)
                    self.register_buffer("trace_idx_ori", torch.tensor(trace_idx_ori, dtype=torch.long).to(device=opt.device), persistent=False)# 60*60 da
                    self.register_buffer("trace_idx_rot", torch.tensor(trace_idx_rot, dtype=torch.long).to(device=opt.device), persistent=False)# 60*60 db
                    # self.trace_idx_ori = torch.nn.Parameter(torch.tensor(trace_idx_ori, dtype=torch.long),
                    #                 requires_grad=False)   # 60*60 da

                    rot_T = self.trace_idx_rot.T
                    self.rot_k_rot_ik = torch.gather(self.trace_idx_rot, 1, rot_T)  # R_k^-1 R_i^-1 R_k
                    ### check that the indexing is correct
                    # R_kinv_iinv_k = torch.einsum('knm,ion,kop->kimp', anchors, anchors, anchors)
                    # R_kinv_iinv_k_byidx = anchors[self.rot_k_rot_ik]
                    # assert R_kinv_iinv_k.shape == R_kinv_iinv_k_byidx.shape, "{} {}".format(R_kinv_iinv_k.shape, R_kinv_iinv_k_byidx.shape)
                    # if not torch.allclose(R_kinv_iinv_k, R_kinv_iinv_k_byidx, 1e-3, 1e-3):
                    #     print('R_kinv_iinv_k', R_kinv_iinv_k)
                    #     print('R_kinv_iinv_k_byidx', R_kinv_iinv_k_byidx)
                    #     print('R_kinv_iinv_k - R_kinv_iinv_k_byidx', R_kinv_iinv_k - R_kinv_iinv_k_byidx)
                    #     raise ValueError('R_kinv_iinv_k and R_kinv_iinv_k_byidx not identical')
                    # else:
                    #     print('R_kinv_iinv_k == R_kinv_iinv_k_byidx')

                    self.trace_idx_rot = torch.gather(self.trace_idx_ori, 1, self.rot_k_rot_ik) # B_ori[k, B_rot[k, B_rot[i,k]]]
                    self.trace_idx_rot = self.trace_idx_rot.T
                elif self.gamma > 0:
                    trace_idx_ori, trace_idx_rot = sgtk.get_relativeV_index()   # 60*12, 60*12
                    self.register_buffer("trace_idx_ori", torch.tensor(trace_idx_ori, dtype=torch.long).to(device=opt.device), persistent=False)# 60*12 da
                    self.register_buffer("trace_idx_rot", torch.tensor(trace_idx_rot, dtype=torch.long).to(device=opt.device), persistent=False)# 60*12 db

                self.crossentropyloss = CrossEntropyLoss()
            else:
                anchors = sgtk.get_anchors(opt.model.kanchor)
                anchors = torch.tensor(anchors).to(device=opt.device)

        # anchors = sgtk.functinoal.get_anchors()
        self.register_buffer('anchors', anchors, persistent=False)

        self.device = opt.device
        self.loss = opt.train_loss.loss_type
        self.margin = opt.train_loss.margin
        self.interpolation = interpolation
        self.k_precision = 1
        
        # if opt.model.flag == 'attention':
        #     self.attention_params = {'attention_type': opt.train_loss.attention_loss_type,
        #                              'attention_margin': opt.train_loss.attention_margin,
        #                              'attention_pretrain_step' : opt.train_loss.attention_pretrain_step,
        #                             }
        
        self.iter_counter = 0

    def forward(self, src, tgt, T, equi_src=None, equi_tgt=None):
        # self._init_buffer(src.shape[0])
        self.gt_idx = torch.arange(src.shape[0], dtype=torch.int32).unsqueeze(1).expand(-1, self.k_precision).contiguous().int().to(self.device)
        if self.alpha > 0 and equi_src is not None and equi_tgt is not None:
            # assert hasattr(self, 'attention_params')
            # return self._forward_attention(src, tgt, T, attention_feats)
            return self._forward_equivariance(src, tgt, equi_src, equi_tgt, T)
        elif (self.beta > 0 or self.gamma > 0) and equi_src is not None and equi_tgt is not None:
            return self._forward_att_permute(src, tgt, equi_src, equi_tgt, T)
        elif self.eta > 0 and equi_src is not None and equi_tgt is not None:
            return self._forward_att_supervised_by_normal(src, tgt, equi_src, equi_tgt, T)
        else:
            return self._forward_invariance(src, tgt)

    def _forward_att_supervised_by_normal(self, src, tgt, equi_src, equi_tgt, T_label):
        """equi_src, equi_tgt: br; T_label: 1"""
        # Pts_src = T * Pts_tgt (rotation-wise) according to match_3dmatch.py
        inv_loss, acc, fp, cn = self._forward_invariance(src, tgt)
        inv_info = [inv_loss, acc, fp, cn]

        n_label_src, n_label_tgt = T_label
        
        cross_e1, accuracy_1 = self.crossentropyloss(equi_tgt, n_label_tgt)
        cross_e2, accuracy_2 = self.crossentropyloss(equi_src, n_label_src)
        cross_e = cross_e1 + cross_e2
        accuracy = (accuracy_1 + accuracy_2)/2
        equiv_info = [cross_e, accuracy]

        if self.eta > 0:
            total_loss = inv_loss + self.eta * cross_e
        else:
            raise ValueError('eta should be > 0')
        return total_loss, inv_info, equiv_info


    def _forward_att_permute(self, src, tgt, equi_src, equi_tgt, T_label):
        """equi_src, equi_tgt: br; T_label: 1"""
        # Pts_src = T * Pts_tgt (rotation-wise) according to match_3dmatch.py
        
        inv_loss, acc, fp, cn = self._forward_invariance(src, tgt)
        inv_info = [inv_loss, acc, fp, cn]

        if T_label.numel() == 1:
            trace_idx_rot_batch = self.trace_idx_rot[T_label.item()]    #60
            equi_tgt_permute = equi_tgt[:, trace_idx_rot_batch]
        else:
            assert T_label.shape[0] == src.shape[0], "T_label.shape {}, src.shape {}".format(T_label.shape, src.shape)
            trace_idx_rot_batch = self.trace_idx_rot[T_label, :]   # r(rotation)*r(anchor) -> b(rotation)*r(anchor)
            equi_tgt_permute = torch.gather(equi_tgt, 1, trace_idx_rot_batch)   # b*r

        # ### set rotations to one of the anchors in match_3dmatch.py, then use this part of code to check equivariance
        # if not torch.allclose(equi_tgt_permute, equi_src, 1e-3, 1e-3):
        #     print('equi_src', equi_src)
        #     print('equi_tgt_permute', equi_tgt_permute)
        #     print('equi_tgt', equi_tgt)
        #     print('equi_src - equi_tgt_permute', equi_src - equi_tgt_permute)
        #     raise ValueError('equi_tgt_permute and equi_src not identical')
        # else:
        #     print('equi_src == equi_tgt_permute')

        if self.use_innerp:
            equi_tgt_permute = F.normalize(equi_tgt_permute, p=2, dim=1)
            equi_src = F.normalize(equi_src, p=2, dim=1)
            cross_e = - (equi_tgt_permute * equi_src).sum(-1).mean(0)

            _, equi_src_max_r = equi_src.max(dim=1)         # b
            _, accuracy = self.crossentropyloss(equi_tgt_permute, equi_src_max_r)
        else:
            _, equi_tgt_max_r = equi_tgt_permute.max(dim=1) # b
            _, equi_src_max_r = equi_src.max(dim=1)         # b

            cross_e1, accuracy = self.crossentropyloss(equi_tgt_permute, equi_src_max_r)
            cross_e2, _ = self.crossentropyloss(equi_src, equi_tgt_max_r)
            cross_e = cross_e1 + cross_e2
        
        # accuracy = torch.sum(equi_tgt_max_r == equi_src_max_r).float() / float(equi_tgt_permute.shape[0])

        # print("\n equi_tgt.shape", equi_tgt.shape)
        # print("\n equi_tgt_permute.shape", equi_tgt_permute.shape)
        # print("\n T_label.shape", T_label.shape)
        # print("\n trace_idx_ori_batch.shape", trace_idx_ori_batch.shape)
        # print("\n self.trace_idx_ori.shape",  self.trace_idx_ori.shape)
        # print("\n equi_tgt_max_r.shape", equi_tgt_max_r.shape)
        # print("\n equi_src_max_r.shape", equi_src_max_r.shape)
        # print("\n accuracy", accuracy, "\n")
        equiv_info = [cross_e, accuracy]
        if self.beta > 0:
            total_loss = inv_loss + self.beta * cross_e
        elif self.gamma > 0:
            total_loss = inv_loss + self.gamma * cross_e
        else:
            raise ValueError('either beta or gamma should be > 0')
        return total_loss, inv_info, equiv_info

    def _forward_invariance(self, src, tgt):
        '''
            src, tgt: [nb, cdim]
        '''
        # L2 distance function
        dist_func = lambda a,b: (a-b)**2
        bdim = src.size(0)

        # furthest positive

        all_dist = pairwise_distance_matrix(src, tgt)
        furthest_positive = torch.diagonal(all_dist)
        closest_negative = batch_hard_negative_mining(all_dist)
        # soft mining (deprecated)
        # closest_negative = (all_dist.sum(1) - all_dist.diag()) / (bdim - 1)
        # top k hard mining (deprecated)
        # masked_dist = all_dist + 1e5 * self.mask_one
        # nval, _ = masked_dist.topk(dim=1, k=3, largest=False)
        # closest_negative = nval.mean()
        # hard mining
        # masked_dist = all_dist + 1e5 * self.mask_one
        # closest_negative, cnidx = masked_dist.min(dim=1)
        diff = furthest_positive - closest_negative
        if self.loss == 'hard':
            diff = F.relu(diff + self.margin)
        elif self.loss == 'soft':
            diff = F.softplus(diff, beta=self.margin)
        elif self.loss == 'contrastive':
            diff = furthest_positive + F.relu(self.margin - closest_negative)
        # evaluate accuracy
        _, idx = torch.topk(all_dist, k=self.k_precision, dim=1, largest=False)
        accuracy = torch.sum(idx.int() == self.gt_idx).float() / float(bdim)
        # gather info for debugging
        self.match_idx = idx
        self.all_dist = all_dist
        self.fpos = furthest_positive
        self.cneg = closest_negative

        return diff.mean(), accuracy, furthest_positive.mean(), closest_negative.mean()

    def _forward_equivariance(self, src, tgt, equi_src, equi_tgt, T):

        inv_loss, acc, fp, cn = self._forward_invariance(src, tgt)

        # equi feature: nb, nc, na
        # L2 distance function
        dist_func = lambda a,b: (a-b)**2
        bdim = src.size(0)

        # so3 interpolation
        # equi_srcR = self._interpolate(equi_src, T, sigma=self.sigma).view(bdim, -1)
        # equi_tgt = equi_tgt.view(bdim, -1)
        equi_tgt = self._interpolate(equi_tgt, T, sigma=self.sigma).view(bdim, -1)
        equi_srcR = equi_src.view(bdim, -1)


        
        # furthest positive
        all_dist = pairwise_distance_matrix(equi_srcR, equi_tgt)
        furthest_positive = torch.diagonal(all_dist)
        closest_negative = batch_hard_negative_mining(all_dist)
        
        diff = furthest_positive - closest_negative
        if self.loss == 'hard':
            diff = F.relu(diff + self.margin)
        elif self.loss == 'soft':
            diff = F.softplus(diff, beta=self.margin)
        elif self.loss == 'contrastive':
            diff = furthest_positive + F.relu(self.margin - closest_negative)
        # evaluate accuracy
        _, idx = torch.topk(all_dist, k=self.k_precision, dim=1, largest=False)
        accuracy = torch.sum(idx.int() == self.gt_idx).float() / float(bdim)
        
        inv_info = [inv_loss, acc, fp, cn]
        equi_loss = diff.mean()
        total_loss = inv_loss + self.alpha * equi_loss
        equi_info = [equi_loss, accuracy, furthest_positive.mean(), closest_negative.mean()]
        
        return total_loss, inv_info, equi_info

    def _forward_attention(self, src, tgt, T, feats):
        '''
            src, tgt: [nb, cdim]
            feats: (src_feat, tgt_feat) [nb, 1, na], normalized attention weights to be aligned
        '''
        # confidence divergence ?
        dist_func = lambda a,b: (a-b)**2

        src_wts = feats[0].squeeze().clamp(min=1e-5)
        tgt_wts = feats[1].squeeze().clamp(min=1e-5)

        inv_loss, acc, fpos, cneg = self._forward_invariance(src, tgt)

        # src_wtsR = self._interpolate(src_wts, T, sigma=self.sigma)
        # r_loss = dist_func(src_wtsR, tgt_wts).mean()

        loss_type = self.attention_params['attention_type']
        # m = self.attention_params['attention_margin']
        # pretrain_step = self.attention_params['attention_pretrain_step']

        #### DEPRECATED
        if src_wts.ndimension() == 3:
            src_wts = src_wts.mean(-1)
            tgt_wts = tgt_wts.mean(-1)

        entropy = -(src_wts * src_wts.log() + tgt_wts * tgt_wts.log())
        entropy_loss = 1e-2 * entropy.sum()

        if loss_type == 'no_reg':
            loss = inv_loss
        else:
            raise NotImplementedError(f"{loss_type} is not Implemented!")

        if self.training:
            self.iter_counter += 1

        return loss, inv_loss, entropy_loss, acc, fpos, cneg


    # knn interpolation of rotated feature
    def _interpolate(self, feature, T, knn=3, sigma=1e-1):
        '''
            :param:
                anchors: [na, 3, 3]
                feature: [nb, cdim, na]
                T: [nb, 4, 4] rigid transformations or [nb, 3, 3]
            :return:
                rotated_feature: [nb, cdim, na]
        '''
        bdim, cdim, adim = feature.shape

        R = T[:,:3,:3]
        # TOCHECK:
        # b, na, 3, 3
        r_anchors = torch.einsum('bij,njk->bnik', R.transpose(1,2), self.anchors)

        # b, 1, na, k
        influences, idx = self._rotation_distance(r_anchors, self.anchors, k=knn)
        influences = F.softmax(influences/sigma, 2)[:,None]

        # print(T)
        # print(influences[0,0,0])
        
        idx = idx.view(-1)
        feat = feature[:,:,idx].reshape(bdim, cdim, adim, knn)

        # b, cdim, na x b, na, k -> b, cdim, na, k
        # feat = sgtk.batched_index_select(feature, 2, idx.reshape(bdim, -1)).reshape(bdim, cdim, adim, knn)
        feat = (feat * influences).sum(-1)

        # spherical gaussian function: e^(lambda*(dot(p,v)-1))
        # see https://mynameismjp.wordpress.com/2016/10/09/sg-series-part-2-spherical-gaussians-101/
        # if self.interpolation == 'spherical':
        #     dists = torch.sum(anchors_tgt*tiled_anchors, dim=3) - 1.0
        #     val, idx = dists.topk(k=knn,dim=2, largest=True)
        # else:
        #     dists = torch.sum((anchors_tgt - tiled_anchors)**2, dim=3)
        #     val, idx = dists.topk(k=knn,dim=2, largest=False)
        return feat


    # b,n,3,3 x m,3,3 -> b,n,k
    def _rotation_distance(self, r0, r1, k=3):
        diff_r = torch.einsum('bnij, mjk->bnmik', r0, r1.transpose(1,2))
        traces = torch.einsum('bnmii->bnm', diff_r)
        return traces.topk(k=k, dim=2)
