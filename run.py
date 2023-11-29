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

def init(base_exp_dir, data_name, sample_dim):
    visual_hull = Load_Visual_Hull(data_name, self.dataset)
    sites = sampler.sample_Bbox(visual_hull[0:3], visual_hull[3:6], sample_dim, perturb_f =  visual_hull[3]*0.005)     


def train(base_exp_dir, data_name):
    ##### 2. Load initial sites
    sites, _ = ply.load_ply(base_exp_dir + "/sites_init_" + data_name +"_32.ply")
    sites = torch.from_numpy(sites.astype(np.float32)).cuda()


if __name__=='__main__':
    print("Code by Diego Thomas")

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./NeuS/confs/test_fine.conf')
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