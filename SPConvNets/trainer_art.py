from importlib import import_module
from SPConvNets.models.art_so3net_pn import build_model_from
from SPConvNets.datasets.LightDataset import LightDataset
from SPConvNets.datasets.RealDataset import RealDataset
#from SPConvNets.datasets.MotionDatasetPartial import MotionDataset
#from SPConvNets.datasets.MotionDataset import MotionDataset
from tqdm import tqdm
import torch
import vgtk
import vgtk.pc as pctk
import numpy as np
import os
import time
import torch.nn.functional as F
from sklearn.neighbors import KDTree
from SPConvNets.datasets.evaluation.retrieval import modelnet_retrieval_mAP
import vgtk.so3conv.functional as L
from thop import profile
from fvcore.nn import FlopCountAnalysis
from SPConvNets.models.art_metric import Art_Metric

class Trainer(vgtk.Trainer):
    def __init__(self, opt):
        """Trainer for modelnet40 classification. """
        self.attention_model = opt.model.flag.startswith('attention') and opt.debug_mode != 'knownatt'
        self.attention_loss = self.attention_model and opt.train_loss.cls_attention_loss
        self.att_permute_loss = opt.model.flag == 'permutation'
        if opt.group_test:
            self.rot_set = [None, 'so3'] # 'ico' 'z', 
            if opt.train_rot is None:
                ### Test the icosahedral equivariance when not using rotation augmentation in training
                self.rot_set.append('ico')
        super(Trainer, self).__init__(opt)

        self.summary.register(['Loss'])
        self.epoch_counter = 0
        self.iter_counter = 0
        if self.opt.group_test:
            # self.best_accs_ori = {None: 0, 'z': 0, 'so3': 0}
            # self.best_accs_aug = {None: 0, 'z': 0, 'so3': 0}
            # self.test_accs_ori = {None: [], 'z': [], 'so3': []}
            # self.test_accs_aug = {None: [], 'z': [], 'so3': []}
            self.best_accs_ori = dict()
            self.best_accs_aug = dict()
            self.test_accs_ori = dict()
            self.test_accs_aug = dict()
            for rot in self.rot_set:
                self.best_accs_ori[rot] = 0
                self.best_accs_aug[rot] = 0
                self.test_accs_ori[rot] = []
                self.test_accs_aug[rot] = []
        else:
            self.test_accs = 0
            self.best_acc = 0

        
        if self.opt.model.kanchor == 12:
            self.anchors = L.get_anchorsV()
            self.trace_idx_ori, self.trace_idx_rot = L.get_relativeV_index()
            self.trace_idx_ori = torch.tensor(self.trace_idx_ori).to(self.opt.device)
            self.trace_idx_rot = torch.tensor(self.trace_idx_rot).to(self.opt.device)
        else:
            self.anchors = L.get_anchors(self.opt.model.kanchor)

    def _setup_datasets(self):
        if self.opt.equi_settings.dataset_type == 'Real':
            D = RealDataset
        elif self.opt.equi_settings.dataset_type == 'Light':
            D = LightDataset
        else:
            raise NotImplementedError
        
        dataset = D(
            npoints=self.opt.model.input_num,
            split='train',
            partial=self.opt.equi_settings.partial,
            nmask=self.opt.equi_settings.nmasks,
            shape_type=self.opt.equi_settings.shape_type,
            args=self.opt,
        )
        #dataset = MotionDataset(
        #    npoints=1024,
        #    split='test',
        #    nmask=2,
        #   shape_type='oven',
        #    global_rot=1,
        #    args=self.opt
        #)
        self.dataset = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.opt.batch_size,
            shuffle=True,
            num_workers=self.opt.num_thread)
        self.dataset_iter = iter(self.dataset)

        dataset_test = D(
            npoints=self.opt.model.input_num,
            split="test",
            partial=self.opt.equi_settings.partial,
            nmask=self.opt.equi_settings.nmasks,
            shape_type=self.opt.equi_settings.shape_type,
            args=self.opt,
        )
        self.dataset_test = torch.utils.data.DataLoader(
            dataset_test,
            batch_size=self.opt.test_batch_size,
            shuffle=False,
            num_workers=self.opt.num_thread)

    def _setup_model(self):
        if self.opt.mode == 'train':
            param_outfile = os.path.join(self.root_dir, "params.json")
        else:
            param_outfile = None

        self.model = build_model_from(self.opt, param_outfile)

        pytorch_total_params = sum(p.numel() for p in self.model.parameters())
        pytorch_total_buffers = sum(p.numel() for p in self.model.buffers())
        self.logger.log("Training", "Total number of parameters: {}".format(pytorch_total_params))
        self.logger.log("Training", "Total number of buffers: {}".format(pytorch_total_buffers))

            
        input = torch.randn(1, 1024, 3).to(self.opt.device)
        input2 = torch.randn(1, 1024, 3).to(self.opt.device)
        macs, params = profile(self.model, inputs=(input, input2))
        print(
            "Batch size: 1 | params(M): %.2f | FLOPs(G) %.5f" % (params / (1000 ** 2), macs / (1000 ** 3))
        )
        self.profiled = 1 # 0

    def _setup_metric(self):
        self.metric = Art_Metric(
            self.opt.equi_settings.shape_type,
            self.opt.equi_settings.nmasks,
            self.opt.model.rigid_cd_w,
            self.opt.model.color_cd_w,
            self.opt.model.prob_threshold,
            self.opt.equi_settings.dataset_type == 'Real'
            )

    # For epoch-based training
    def epoch_step(self):
        for it, data in tqdm(enumerate(self.dataset)):
            if self.opt.debug_mode == 'check_equiv':
                self._check_equivariance(data)
            else:
                self._optimize(data)

    # For iter-based training
    def step(self):
        try:
            data = next(self.dataset_iter)
            if data['pc'].shape[0] < self.opt.batch_size:
                raise StopIteration
        except StopIteration:
            # New epoch
            torch.cuda.empty_cache()
            self.epoch_counter += 1
            #print("[DataLoader]: At Epoch %d!"%self.epoch_counter)
            self.dataset_iter = iter(self.dataset)
            data = next(self.dataset_iter)

        if self.opt.debug_mode == 'check_equiv':
            self._check_equivariance(data)
        else:
            self._optimize(data)
        self.iter_counter += 1

    def cos_sim(self, f1, f2):
        ### both bc(p)a
        f1_norm = torch.norm(f1, dim=1)
        f2_norm = torch.norm(f2, dim=1)
        cos_similarity = (f1 * f2).sum(1) / (f1_norm * f2_norm)
        return cos_similarity

    def _check_equivariance(self, data):
        self.model.eval()
        in_tensors = data['pc'].to(self.opt.device)
        in_label = data['label'].to(self.opt.device).reshape(-1)
        in_Rlabel = data['R_label'].to(self.opt.device) #if self.opt.debug_mode == 'knownatt' else None #!!!!
        in_R = data['R'].to(self.opt.device)

        feat_conv, x = self.model(in_tensors, in_Rlabel)
        pred, feat, x_feat = x
        n_anchors = feat.shape[-1]
        x_feat = x_feat.reshape(x_feat.shape[0], -1, n_anchors)

        in_tensors_ori = torch.matmul(in_tensors, in_R) # B*n*3, B*3*3
        feat_conv_ori, x_ori = self.model(in_tensors_ori, in_Rlabel)  # bn, bra, b[ca]
        pred_ori, feat_ori, x_feat_ori = x_ori
        n_anchors = feat_ori.shape[-1]
        x_feat_ori = x_feat_ori.reshape(x_feat_ori.shape[0], -1, n_anchors)

        trace_idx_ori = self.trace_idx_ori[in_Rlabel.flatten()] # ba
        trace_idx_ori_p = trace_idx_ori[:,None,None].expand_as(feat_conv_ori) #bcpa
        feat_conv_align = torch.gather(feat_conv, -1, trace_idx_ori_p)

        trace_idx_ori_global = trace_idx_ori[:,None].expand_as(x_feat_ori) #bca
        x_feat_align = torch.gather(x_feat, -1, trace_idx_ori_global)

        # self.logger.log('TestEquiv', f'feat_ori: {feat_ori.shape}, x_feat_ori: {x_feat_ori.shape}')
        # self.logger.log('TestEquiv', f'x_feat: {x_feat.shape}, x_feat_from_ori: {x_feat_from_ori.shape}')
        # self.logger.log('TestEquiv', f'in_Rlabel: {in_Rlabel}, in_R: {in_R}')

        cos_sim_before = self.cos_sim(feat_conv, feat_conv_ori)
        cos_sim_after = self.cos_sim(feat_conv_align, feat_conv_ori)

        self.logger.log('TestEquiv', f'per point cos before: {cos_sim_before}, after: {cos_sim_after}')

        cos_sim_before = self.cos_sim(x_feat, x_feat_ori)
        cos_sim_after = self.cos_sim(x_feat_align, x_feat_ori)
        self.logger.log('TestEquiv', f'global cos before: {cos_sim_before}, after: {cos_sim_after}')

    def _optimize(self, data):
        pc = data['pc'].to(self.opt.device)
        color = data['color'].to(self.opt.device)
        pose_label = data['pose_segs'].to(self.opt.device)
        joint_label = torch.cat([data['part_pv_point'], data['part_axis']], dim=-1).to(self.opt.device)
        segmentation_label = data['label'].to(self.opt.device)
        bdim = pc.shape[0]

        if self.profiled < 1:
            self.logger.log('Profile', f'in_tensors: {pc.shape}')
            flops = FlopCountAnalysis(self.model, (pc, color))
            self.logger.log('Profile', f'flops: {flops.total()/ (1000**3)}')
            self.logger.log('Profile', f'flops.by_module(): {flops.by_module()}')
            self.profiled +=1
    
        ret = self.model(pc, color)
        self.optimizer.zero_grad()
        loss = self.metric(ret, [pose_label, joint_label, segmentation_label], mode="train")

        if torch.isnan(loss):
            self.logger.log('Training', 'detect Nan in loss, skip!')
            self.optimizer.zero_grad()
            return
        
        self.loss = loss
        self.loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.model.parameters(), 32)
        self.optimizer.step()

        # Log training stats
        log_info = {
            'Loss': loss.item(),
        }
        self.summary.update(log_info)


    def _print_running_stats(self, step):
        stats = self.summary.get()
        
        mem_used_max_GB = torch.cuda.max_memory_allocated() / (1024*1024*1024)
        torch.cuda.reset_peak_memory_stats()
        mem_str = f', Mem: {mem_used_max_GB:.3f}GB'

        self.logger.log('Training', f'{step}: {stats}'+mem_str)
        # self.summary.reset(['Loss', 'Pos', 'Neg', 'Acc', 'InvAcc'])

    def test(self, dataset=None):
        self.model.eval()
        self.metric.eval()
        torch.cuda.reset_peak_memory_stats()
        best_model = False
        if dataset is None:
                dataset = self.dataset_test

        with torch.no_grad():
            csv, viz_file, acc_score = val(
                self.dataset_test,
                self.dataset,
                self.model,
                self.metric,
                self.opt.device,
                self.logger,
                )
            torch.cuda.empty_cache()
        if acc_score > self.best_acc or self.best_acc == 0:
            self.best_acc = acc_score
            best_model = True
            self._save_csv(csv)
            self._save_viz(viz_file)

        self.model.train()
        self.metric.train()
        return csv, viz_file, best_model
        
    def train_iter(self):
        for i in range(self.opt.num_iterations+1):
            self.timer.set_point('train_iter')
            self.lr_schedule.step()
            self.step()
            # print({'Time': self.timer.reset_point('train_iter')})
            self.summary.update({'Time': self.timer.reset_point('train_iter')})

            if i % self.opt.log_freq == 0:
                if hasattr(self, 'epoch_counter'):
                    step = f'Epoch {self.epoch_counter}, Iter {i}'
                else:
                    step = f'Iter {i}'
                self._print_running_stats(step)

            if i > 0 and i % self.opt.save_freq == 0:
                self._save_network(f'Iter{i}')
                csv, viz_file, best_model = self.test()
                if best_model:
                    self.logger.log('Testing', 'Best model Update.')
                    self._save_network('best')

def val(dataset_test, dataset_train, model, metric, device, logger):

    #Standardization Step
    #dataset_test.dataset.set_seed(0)
    logger.log('Testing','Standardizating training set.')
    for it, data in enumerate(tqdm(dataset_train, miniters=100, maxinterval=600)):
        pc = data['pc'].to(device)
        color = data['color'].to(device)
        pose_label = data['pose_segs'].to(device)
        joint_label = torch.cat([data['part_pv_point'], data['part_axis']], dim=-1).to(device)
        segmentation_label = data['label'].to(device)
        pred = model(pc, color)
        metric(pred, [pose_label, joint_label, segmentation_label], mode="standardization")
    R_acc, T_acc = metric.standardization_ransac()
    logger.log('Testing', 'RANSAC Rotation Completed with acc: ' + str(100 * R_acc))
    logger.log('Testing', 'RANSAC Translation Completed with acc: ' + str(100 * T_acc))
        
    #Test Step
    logger.log('Testing','Running test set.')
    torch.cuda.reset_peak_memory_stats()
    all_seg, all_joint, all_drct, all_trans, all_rotation, all_idx, all_time, all_viz = [], [], [], [], [], [], [], []
    for it, data in enumerate(tqdm(dataset_test, miniters=100, maxinterval=600)):
        pc = data['pc'].to(device)
        color = data['color'].to(device)
        pose_label = data['pose_segs'].to(device)
        joint_label = torch.cat([data['part_pv_point'], data['part_axis']], dim=-1).to(device)
        segmentation_label = data['label'].to(device)
        expand_label = data['expand'].to(device)
        center_label = data['center'].to(device)
        name = data['name']
        t_start = time.time()
        pred = model(pc, color)
        t_inference = time.time() - t_start
        segmentation_error, joint_error, drct_error, rotation_error, trans_error, viz_f = metric(
            pred, [pose_label, joint_label, segmentation_label, expand_label, center_label], mode="test")
        all_seg.append(segmentation_error)
        all_joint.append(joint_error)
        all_drct.append(drct_error)
        all_trans.append(trans_error)
        all_rotation.append(rotation_error)
        all_idx.append(data['idx'])
        all_time.append(t_inference)
        viz_f['idx'] = data['idx']
        all_viz.append(viz_f)
    mem_used_max_GB = torch.cuda.max_memory_allocated() / (1024*1024*1024)
    all_seg = torch.cat(all_seg, dim=0).detach().cpu()
    all_joint = torch.cat(all_joint, dim=0).detach().cpu()
    all_drct = torch.cat(all_drct, dim=0).detach().cpu()
    all_trans = torch.cat(all_trans, dim=0).detach().cpu()
    all_rotation = torch.cat(all_rotation, dim=0).detach().cpu()
    all_idx = torch.cat(all_idx, dim=0).detach().cpu()
    if all_idx.dim()==1:
        all_idx = all_idx.reshape(-1,1)
    all_time = torch.tensor(all_time)
    mean_seg, mean_joint, mean_drct, mean_trans, mean_rotation, mean_time = all_seg.mean(0), all_joint.mean(0), all_drct.mean(0), all_trans.mean(0), all_rotation.mean(0), all_time.mean()
    # 1 Joint, 2 Part
    if all_joint.shape[-1] == 1:
        joint_5degree = all_drct < 5
        joint_10degree = all_drct < 10
        joint_5cm = all_joint < 0.5
        joint_10cm = all_joint < 1
        part_5degree = torch.logical_and(all_rotation[:,0] < 5, all_rotation[:,1] < 5)
        part_10degree = torch.logical_and(all_rotation[:,0] < 10, all_rotation[:,1] < 10)
        part_5cm = torch.logical_and(all_trans[:,0] < 0.5, all_trans[:,1] < 0.5)
        part_10cm = torch.logical_and(all_trans[:,0] < 1, all_trans[:,1] < 1)
        seg_50 = torch.logical_and(all_seg[:,0] > 0.50, all_seg[:,1] > 0.50)
        seg_75 = torch.logical_and(all_seg[:,0] > 0.75, all_seg[:,1] > 0.75)
    # 2 Joint, 3 Part
    else:
        joint_5degree = torch.logical_and(all_drct[:,0] < 5, all_drct[:,1] < 5)
        joint_10degree = torch.logical_and(all_drct[:,0] < 10, all_drct[:,1] < 10)
        joint_5cm = torch.logical_and(all_joint[:,0] < 0.5, all_drct[:,1] < 0.5)
        joint_10cm = torch.logical_and(all_joint[:,0] < 1, all_drct[:,1] < 1)
        part_5degree = torch.logical_and(torch.logical_and(all_rotation[:,0] < 5, all_rotation[:,1] < 5), all_rotation[:,2] < 5)
        part_10degree = torch.logical_and(torch.logical_and(all_rotation[:,0] < 5, all_rotation[:,1] < 5), all_rotation[:,2] < 5)
        part_5cm= torch.logical_and(torch.logical_and(all_trans[:,0] < 0.5, all_trans[:,1] < 0.5), all_trans[:,2] < 0.5)
        part_10cm = torch.logical_and(torch.logical_and(all_trans[:,0] < 1, all_trans[:,1] < 1), all_trans[:,2] < 1)
        seg_50 = torch.logical_and(torch.logical_and(all_seg[:,0] > 0.50, all_seg[:,1] > 0.50), all_seg[:,2] > 0.50)
        seg_75 = torch.logical_and(torch.logical_and(all_seg[:,0] > 0.75, all_seg[:,1] > 0.75), all_seg[:,2] > 0.75)
    joint_5d5c = torch.logical_and(joint_5degree, joint_5cm).sum() / all_idx.shape[0]
    joint_5d10c = torch.logical_and(joint_5degree, joint_10cm).sum() / all_idx.shape[0]
    joint_10d5c = torch.logical_and(joint_10degree, joint_5cm).sum() / all_idx.shape[0]
    joint_10d10c = torch.logical_and(joint_10degree, joint_10cm).sum() / all_idx.shape[0]
    part_5d5c = torch.logical_and(part_5degree, part_5cm).sum() / all_idx.shape[0]
    part_5d10c = torch.logical_and(part_5degree, part_10cm).sum() / all_idx.shape[0]
    part_10d5c = torch.logical_and(part_10degree, part_5cm).sum() / all_idx.shape[0]
    part_10d10c = torch.logical_and(part_10degree, part_10cm).sum() / all_idx.shape[0]
    seg_50 = seg_50.sum() / all_idx.shape[0]
    seg_75 = seg_75.sum() / all_idx.shape[0]

    logger.log('Testing', 'Average Segmentation IoU: ' + str(100 * mean_seg))
    logger.log('Testing', 'Average Joint Position Error: ' + str(mean_joint))
    logger.log('Testing', 'Average Joint Direction Error: ' + str(mean_drct))
    logger.log('Testing', 'Average Part Rotation Error: ' + str(mean_rotation))
    logger.log('Testing', 'Average Part Translation Error: ' + str(mean_trans))
    logger.log('Testing', 'Part 5degree5cm: ' + str(100 * part_5d5c))
    logger.log('Testing', 'Part 5degree10cm: ' + str(100 * part_5d10c))
    logger.log('Testing', 'Part 10degree5cm: ' + str(100 * part_10d5c))
    logger.log('Testing', 'Part 10degree10cm: ' + str(100 * part_10d10c))
    logger.log('Testing', 'Joint 5degree5cm: ' + str(100 * joint_5d5c))
    logger.log('Testing', 'Joint 5degree10cm: ' + str(100 * joint_5d10c))
    logger.log('Testing', 'Joint 10degree5cm: ' + str(100 * joint_10d5c))
    logger.log('Testing', 'Joint 10degree10cm: ' + str(100 * joint_10d10c))
    logger.log('Testing', 'Segmentation 50: ' + str(100 * seg_50))
    logger.log('Testing', 'Segmentation 75: ' + str(100 * seg_75))
    logger.log('Testing', f'Mem: {mem_used_max_GB:.3f}GB')
    logger.log('Testing', 'FPS: ' + str(1/mean_time.item()))

    acc_score = part_10d10c + joint_10d10c + seg_50

    csv = torch.cat([all_idx, all_time.unsqueeze(-1), all_seg, all_joint, all_drct, all_rotation, all_trans], dim=-1)
    return csv, all_viz, acc_score