import sys
import os
#os.environ["PYOPENGL_PLATFORM"] = "osmesa"
#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import torch
sys.path.append(os.path.join(os.path.dirname(__file__),'vgtk') )

from SPConvNets.trainer_art import Trainer
from SPConvNets.options import opt


with torch.autograd.set_detect_anomaly(False):
    if __name__ == '__main__':
        opt.model.model = "art_so3net_pn"
        '''
        if opt.mode == 'train':
            # overriding training parameters here
            opt.batch_size = 12
            opt.test_batch_size = 24
        elif opt.mode == 'eval':
            opt.batch_size = 24
        '''
        trainer = Trainer(opt)
        if opt.mode == 'train':
            trainer.train()
        elif opt.mode == 'test':
            trainer.test() 
