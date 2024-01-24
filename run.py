import os
import cv2
import torch
import argparse
import numpy as np
import src.Geometry.tet32 as tet32
import src.Geometry.sampling as sampler
import src.IO.ply as ply
from src.Confs.VisHull import Load_Visual_Hull
from src.IO.dataset import Dataset
from pyhocon import ConfigFactory

class Runner:
    def __init__(self, conf_path, data_name, mode='train', is_continue=False, checkpoint = ''):
        self.device = torch.device('cuda')

        # Configuration
        self.conf_path = conf_path
        f = open(self.conf_path)
        conf_text = f.read()
        f.close()

        self.conf = ConfigFactory.parse_string(conf_text)
        self.base_exp_dir = self.conf['general.base_exp_dir']
        self.base_exp_dir  = self.base_exp_dir.replace('DATA_NAME', data_name)
        os.makedirs(self.base_exp_dir, exist_ok=True)
        os.makedirs(self.base_exp_dir + "/validations_fine", exist_ok=True)


    def train(self, data_name, verbose = True):
        ##### 2. Load initial sites
        if not hasattr(self, 'tet32'):
            ##### 2. Load initial sites
            visual_hull = [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]
            import src.Geometry.sampling as sampler
            res = 16
            sites = sampler.sample_Bbox(visual_hull[0:3], visual_hull[3:6], res, perturb_f =  (visual_hull[3] - visual_hull[0])*0.1)

            self.tet32 = tet32.Tet32(sites)
            self.tet32.save("data/bmvs_man/test.ply")       

            sites = np.asarray(self.tet32.vertices)      
            
            outside_flag = np.zeros(sites.shape[0], np.int32)
            outside_flag[sites[:,0] < visual_hull[0] + (visual_hull[2]-visual_hull[0])/(2*res)] = 1
            outside_flag[sites[:,1] < visual_hull[1] + (visual_hull[3]-visual_hull[1])/(2*res)] = 1
            outside_flag[sites[:,0] > visual_hull[2] - (visual_hull[2]-visual_hull[0])/(2*res)] = 1
            outside_flag[sites[:,1] > visual_hull[3] - (visual_hull[3]-visual_hull[1])/(2*res)] = 1
        #else:
        #    sites, _ = ply.load_ply(base_exp_dir + "/sites_init_" + data_name +"_32.ply")

        sites = torch.from_numpy(sites.astype(np.float32)).cuda()
        print (sites.shape)

        
        ##### 2. Initialize SDF field    
        if not hasattr(self, 'sdf'):
            norm_sites = torch.linalg.norm(sites, ord=2, axis=-1, keepdims=True)
            self.sdf = norm_sites[:,0] - 0.5 #norm_sites[:,0]**2 - 50.0**2    
            self.sdf = self.sdf.contiguous()
            self.sdf.requires_grad_(True)
        
        self.tet32.surface_from_sdf(self.sdf.detach().cpu().numpy().reshape(-1), "data/bmvs_man/test_tri.ply")
            




if __name__=='__main__':
    print("Code by Diego Thomas")

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='src/Confs/test.conf')
    parser.add_argument('--data_name', type=str, default='bmvs_stone')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--is_continue', default=False, action="store_true")
    parser.add_argument('--checkpoint', type=str, default='coarse_ckpt_010000.pth')
    parser.add_argument('--resolution', type=int, default=16)
    parser.add_argument('--output', type=str, default='')
    parser.add_argument('--nb_images', type=int, default=1)
    parser.add_argument('--gpu', type=int, default=0)
    
    args = parser.parse_args()

    ## Initialise CUDA device for torch computations
    torch.cuda.set_device(args.gpu)
    
    runner = Runner(args.conf, args.data_name, args.mode, args.is_continue, args.checkpoint)
    
    if args.mode == 'train':
        runner.train(args.data_name, False)