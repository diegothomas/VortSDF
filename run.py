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
from tqdm import tqdm
from timeit import default_timer as timer 

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

        self.dataset = Dataset(self.conf['dataset'], data_name)

        self.intrinsics = self.dataset.intrinsics_all.float().flatten().cuda().contiguous()

        self.extrinsics = []
        for idx in range(self.dataset.n_images):
            pose = torch.zeros((4,4))
            pose[:3,:3] = self.dataset.pose_all[idx, :3, :3].inverse()
            pose[:3,3] = -self.dataset.pose_all[idx, :3, 3]
            self.extrinsics.append(pose)
        self.extrinsics = torch.stack(self.extrinsics).cuda()
        self.extrinsics = self.extrinsics.float().flatten().cuda().contiguous()

        
        # Training parameters
        self.end_iter = self.conf.get_int('train.end_iter')


    def train(self, data_name, verbose = True):
        ##### 2. Load initial sites
        if not hasattr(self, 'tet32'):
            ##### 2. Load initial sites
            visual_hull = [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]
            import src.Geometry.sampling as sampler
            res = 16
            sites = sampler.sample_Bbox(visual_hull[0:3], visual_hull[3:6], res, perturb_f =  (visual_hull[3] - visual_hull[0])*0.1)
            
            #### Add cameras as sites 
            cam_sites = np.stack([self.dataset.pose_all[id, :3,3].cpu().numpy() for id in range(self.dataset.n_images)])
            sites = np.concatenate((sites, cam_sites))

            self.tet32 = tet32.Tet32(sites)
            self.tet32.save("Exp/bmvs_man/test.ply")       

            sites = np.asarray(self.tet32.vertices)  
            cam_ids = np.stack([np.where((sites == cam_sites[i,:]).all(axis = 1))[0] for i in range(cam_sites.shape[0])]).reshape(-1)
            
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
            self.sdf = norm_sites[:,0] - 0.5
            self.sdf = self.sdf.contiguous()
            self.sdf.requires_grad_(True)
        
        self.tet32.surface_from_sdf(self.sdf.detach().cpu().numpy().reshape(-1), "Exp/bmvs_man/test_tri.ply")
            
        self.Allocate_batch_data()
        
        self.s_max = 1000
        self.R = 50
        self.s_start = 0.1
        self.inv_s = 0.1
        image_perm = self.get_image_perm()
        for iter_step in tqdm(range(self.end_iter)):
            img_idx = image_perm[iter_step % len(image_perm)] 
            self.inv_s = min(self.s_max, iter_step/self.R + self.s_start)

            ## Generate rays
            data = self.dataset.gen_random_rays_zbuff_at(img_idx, 512, 0)  
            rays_o, rays_d_batch, true_rgb_batch, mask_batch = data[:, :3], data[:, 3: 6], data[:, 6: 9], data[:, 9: 10]

            rays_o = rays_o.reshape(-1, 3)
            rays_d = rays_d.reshape(-1, 3)
            true_rgb_batch = true_rgb_batch.reshape(-1, 3)
            mask_batch = mask_batch.reshape(-1, 1)

            rays_o = rays_o.contiguous()
            rays_d = rays_d.contiguous()
            true_rgb_batch = true_rgb_batch.contiguous()
            mask_batch = mask_batch.contiguous()
            
            ## sample points along the rays
            start = timer()
            self.offsets[:] = 0
            nb_samples = self.tet32.sample_rays_cuda(img_idx, rays_o, rays_d, self.sdf, 
                                                   self.in_z, self.in_sdf, self.in_ids, 
                                                   self.offsets, nb_samples = self.n_samples)
            if verbose:
                print('CVT_Sample time:', timer() - start)    



            print("DONE")
            input()


    def Allocate_batch_data(self):
        self.samples = torch.zeros([self.n_samples * self.batch_size, 2], dtype=torch.float32).cuda()
        self.samples = self.samples.contiguous()
        
        self.samples_loc = torch.zeros([self.n_samples * self.batch_size, 2], dtype=torch.float32).cuda()
        self.samples_loc = self.samples_loc.contiguous()
        
        self.samples_rays = torch.zeros([self.n_samples * self.batch_size, 2], dtype=torch.float32).cuda()
        self.samples_rays = self.samples_rays.contiguous()

        self.in_ids = -torch.ones([3*self.n_samples* self.batch_size], dtype=torch.int32).cuda()
        self.in_ids = self.in_ids.contiguous()
        
        self.in_z = torch.zeros([2*self.n_samples * self.batch_size], dtype=torch.float32).cuda()
        self.in_z = self.in_z.contiguous()
        
        self.in_sdf = torch.zeros([2*self.n_samples * self.batch_size], dtype=torch.float32).cuda()
        self.in_sdf = self.in_sdf.contiguous()
        
        self.out_ids = -torch.ones([self.n_samples* self.batch_size, 3], dtype=torch.int32).cuda()
        self.out_ids = self.out_ids.contiguous()
        
        self.out_z = torch.zeros([self.n_samples * self.batch_size], dtype=torch.float32).cuda()
        self.out_z = self.out_z.contiguous()
        
        self.out_sdf = torch.zeros([2*self.n_samples * self.batch_size], dtype=torch.float32).cuda()
        self.out_sdf = self.out_sdf.contiguous()
        
        self.offsets = torch.zeros([2*self.batch_size], dtype=torch.int32).cuda()
        self.offsets = self.offsets.contiguous()

    def get_image_perm(self):
        return torch.randperm(self.dataset.n_images)

if __name__=='__main__':
    print("Code by Diego Thomas")

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='src/Confs/test.conf')
    parser.add_argument('--data_name', type=str, default='bmvs_man')
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