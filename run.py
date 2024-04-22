import os
import cv2
import torch
import argparse
import math
import numpy as np
import src.Geometry.tet32 as tet32
import src.Geometry.sampling as sampler
import src.IO.ply as ply
from src.Confs.VisHull import Load_Visual_Hull
from src.IO.dataset import Dataset
from pyhocon import ConfigFactory
from tqdm import tqdm
from timeit import default_timer as timer 
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from src.Models.VortSDFRenderer import VortSDFRenderer, VortSDFRenderingFunction, VortSDFDirectRenderer
from src.Models.fields import ColorNetwork, SDFNetwork
from numpy import random


from torch.utils.cpp_extension import load
tet32_march_cuda = load(
    'tet32_march_cuda', ['src/Cuda/tet32_march_cuda.cpp', 'src/Cuda/tet32_march_cuda.cu'], verbose=True)

renderer_cuda = load(
    'renderer_cuda', ['src/Models/renderer.cpp', 'src/Models/renderer.cu'], verbose=True)

backprop_cuda = load(
    'backprop_cuda', ['src/Models/backprop.cpp', 'src/Models/backprop.cu'], verbose=True)

cvt_grad_cuda = load(
    'cvt_grad_cuda', ['src/Geometry/CVT_gradients.cpp', 'src/Geometry/CVT_gradients.cu'], verbose=True)

class Runner:
    def __init__(self, conf_path, data_name, mode='train', is_continue=False, checkpoint = '', position_encoding = True, double_net = True):
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
        self.n_samples = self.conf.get_int('model.cvt_renderer.n_samples')
        self.batch_size = self.conf.get_int('train.batch_size')
        self.learning_rate = self.conf.get_float('train.learning_rate')
        self.learning_rate_sdf = self.conf.get_float('train.learning_rate_sdf')
        self.learning_rate_feat = self.conf.get_float('train.learning_rate_feat')
        self.learning_rate_alpha = self.conf.get_float('train.learning_rate_alpha')
        lr=self.learning_rate_cvt = self.conf.get_float('train.learning_rate_cvt')

        self.dim_feats = self.conf.get_int('train.dim_feats')

        self.end_iter_loc = 2000
        self.s_w = 1.0e-2
        self.e_w = 1.0e-6
        self.tv_w = 5.0e-3
        """self.s_w = 1.0e-2
        self.e_w = 1.0e-3 
        self.tv_w = 1.0e-3"""
        self.tv_f = 1.0e-8
        self.f_w = 0.0

        self.report_freq = self.conf.get_int('train.report_freq')
        self.val_freq = self.conf.get_int('train.val_freq')
        self.CVT_freq = self.conf.get_int('train.CVT_freq')

        self.position_encoding = position_encoding
        
        self.double_net = double_net

        if self.double_net:
            self.vortSDF_renderer_coarse_net = VortSDFRenderer(**self.conf['model.cvt_renderer'])
        else:
            self.vortSDF_renderer_coarse = VortSDFDirectRenderer(**self.conf['model.cvt_renderer'])

        self.vortSDF_renderer_fine = VortSDFRenderer(**self.conf['model.cvt_renderer'])
        
        if self.double_net:
            self.color_coarse = ColorNetwork(**self.conf['model.color_geo_network']).to(self.device)

        self.color_network = ColorNetwork(**self.conf['model.color_network']).to(self.device)

        self.sdf_network = SDFNetwork(**self.conf['model.sdf_network']).to(self.device)
        
        params_to_train = []
        params_to_train += list(self.color_network.parameters())
        if self.double_net:
            params_to_train += list(self.color_coarse.parameters())
        params_to_train += list(self.sdf_network.parameters())
        self.optimizer = torch.optim.Adam(params_to_train, lr=self.learning_rate)
        
        posbase_pe = 5
        viewbase_pe= 4
        self.posfreq = torch.FloatTensor([(2**i) for i in range(posbase_pe)]).cuda()
        self.viewfreq = torch.FloatTensor([(2**i) for i in range(viewbase_pe)]).cuda()

        k_posbase_pe=5
        k_viewbase_pe= 4
        self.k_posfreq = torch.FloatTensor([(2**i) for i in range(k_posbase_pe)]).cuda()
        self.k_viewfreq = torch.FloatTensor([(2**i) for i in range(k_viewbase_pe)]).cuda()

        # Load checkpoint
        latest_model_name = None
        if is_continue:
            if checkpoint == '':
                model_list_raw = os.listdir(os.path.join(self.base_exp_dir, 'checkpoints'))
                model_list = []
                for model_name in model_list_raw:
                    if model_name[-3:] == 'pth' and int(model_name[5:-4]) <= self.end_iter:
                        model_list.append(model_name)
                model_list.sort()
                latest_model_name = model_list[-1]
            else:
                latest_model_name = checkpoint
            print(latest_model_name)

        if latest_model_name is not None:
            self.load_checkpoint(latest_model_name)
            start = timer()

    def Prep_CVT(self, data_name):
        print("Preparing CVT model")
        self.tet32 = tet32.Tet32(self.sites, id = 0)
        self.tet32.run(0.3) 
        self.tet32.load_cuda()
        self.sites = np.asarray(self.tet32.vertices)  
        
        self.cam_ids = np.stack([np.where((self.sites == self.cam_sites[i,:]).all(axis = 1))[0] for i in range(self.cam_sites.shape[0])]).reshape(-1)
        self.tet32.make_adjacencies(self.cam_ids)
        self.cam_ids = torch.from_numpy(self.cam_ids).int().cuda()
        
        self.Allocate_data()
            
        self.Allocate_batch_data()

    def train(self, data_name, K_NN = 24, verbose = True):
        ##### 2. Load initial sites
        if not hasattr(self, 'tet32'):
            ##### 2. Load initial sites
            #visual_hull = [-1.1, -1.1, -1.1, 1.1, 1.1, 1.1] #-> man
            #visual_hull = [-0.8, -1.0, -0.8, 0.7, 1.0, 0.8] #-> sculpture
            #visual_hull = [-1.0,-1.2,-1.0,0.9,0.4,1.0] # -> dog
            #visual_hull = [-1.2, -1.2, -1.2, 1.2, 1.2, 1.2] #-> stone
            #visual_hull = [-1.2, -1.5, -1.5, 1.2, 1.3, 1.2] #-> durian
            visual_hull = [-1.0, -1.0, -1.0, 1.2, 1.0, 0.6] #-> bear
            #visual_hull = [-1.2, -1.1, -1.2, 1.1, 1.1, 1.1] #-> clock
            
            import src.Geometry.sampling as sampler
            res = 16
            sites = sampler.sample_Bbox(visual_hull[0:3], visual_hull[3:6], res, perturb_f =  (visual_hull[3] - visual_hull[0])*0.02)
            #ext_sites1 = sampler.exterior_Bbox(visual_hull[0:3], visual_hull[3:6], 32, 10.0)
            #ext_sites2 = sampler.exterior_Bbox(visual_hull[0:3], visual_hull[3:6], 32, 11.0)
            #sites, _ = ply.load_ply("Data/bmvs_man/bmvs_man_colmap_aligned.ply")


            #### Add cameras as sites 
            cam_sites = np.stack([self.dataset.pose_all[id, :3,3].cpu().numpy() for id in range(self.dataset.n_images)])
            sites = np.concatenate((sites, cam_sites))
            #sites = np.concatenate((ext_sites1, ext_sites2, sites, cam_sites))

            self.tet32 = tet32.Tet32(sites, id = 0)
            #self.tet32.start() 
            #print("parallel process started")
            #self.tet32.join()
            self.tet32.run(0.3) 
            self.tet32.load_cuda()
            #self.tet32.save("Exp/bmvs_man/test.ply")    
            
            sites = np.asarray(self.tet32.vertices)  
            cam_ids = np.stack([np.where((sites == cam_sites[i,:]).all(axis = 1))[0] for i in range(cam_sites.shape[0])]).reshape(-1)
            self.tet32.make_adjacencies(cam_ids)

            cam_ids = torch.from_numpy(cam_ids).int().cuda()
            
            outside_flag = np.zeros(sites.shape[0], np.int32)
            outside_flag[sites[:,0] < visual_hull[0] + (visual_hull[3]-visual_hull[0])/(res)] = 1
            outside_flag[sites[:,1] < visual_hull[1] + (visual_hull[4]-visual_hull[1])/(res)] = 1
            outside_flag[sites[:,2] < visual_hull[2] + (visual_hull[5]-visual_hull[2])/(res)] = 1
            outside_flag[sites[:,0] > visual_hull[3] - (visual_hull[3]-visual_hull[0])/(res)] = 1
            outside_flag[sites[:,1] > visual_hull[4] - (visual_hull[4]-visual_hull[1])/(res)] = 1
            outside_flag[sites[:,2] > visual_hull[5] - (visual_hull[5]-visual_hull[2])/(res)] = 1
            
            ext_flag = np.zeros(sites.shape[0], np.int32)
            ext_flag[sites[:,0] < visual_hull[0] - 10.5] = 1
            ext_flag[sites[:,1] < visual_hull[1] - 10.5] = 1
            ext_flag[sites[:,2] < visual_hull[2] - 10.5] = 1
            ext_flag[sites[:,0] > visual_hull[3] + 10.5] = 1
            ext_flag[sites[:,1] > visual_hull[4] + 10.5] = 1
            ext_flag[sites[:,2] > visual_hull[5] + 10.5] = 1

        #else:
        #    sites, _ = ply.load_ply(base_exp_dir + "/sites_init_" + data_name +"_32.ply")

        self.sites = torch.from_numpy(sites.astype(np.float32)).cuda()
        self.sites = self.sites.contiguous()
        #self.sites.requires_grad_(True)
        
        print(self.sites.shape)

        delta_sites = torch.zeros(self.sites.shape).float().cuda()
        with torch.no_grad():  
            delta_sites[:] = self.sites[:]
        
        ##### 2. Initialize SDF field    
        if not hasattr(self, 'sdf'):
            with torch.no_grad():
                norm_sites = torch.linalg.norm(self.sites, ord=2, axis=-1, keepdims=True)
            self.sdf = norm_sites[:,0] - 0.5
            self.sdf = self.sdf.contiguous()
            self.sdf.requires_grad_(True)      

                    
        #self.tet32.surface_from_sdf(self.sdf.detach().cpu().numpy().reshape(-1), "Exp/bmvs_man/test_tri.ply")
        #self.tet32.marching_tets(self.sdf.detach(), "Exp/bmvs_man/test_MT.ply")
        #input()
        
        ##### 2. Initialize feature field    
        if not hasattr(self, 'fine_features'):
            self.fine_features = 0.5*torch.ones([self.sdf.shape[0], self.dim_feats]).cuda()       
            self.fine_features = self.fine_features.contiguous()
            self.fine_features.requires_grad_(True)

        ############# CVT optimization #############################
        ############# CVT optimization #############################
        ############# CVT optimization #############################
        self.tet32.CVT(outside_flag, cam_ids, self.sdf.detach(), self.fine_features.detach())
        self.tet32.run(0.3)
        self.tet32.load_cuda()
        self.tet32.save("Exp/bmvs_man/test_CVT.ply")  
        sites = np.asarray(self.tet32.vertices)  
        cam_ids = np.stack([np.where((sites == cam_sites[i,:]).all(axis = 1))[0] for i in range(cam_sites.shape[0])]).reshape(-1)
        self.tet32.make_adjacencies(cam_ids)

        self.tet32.make_multilvl_knn()

        cam_ids = torch.from_numpy(cam_ids).int().cuda()
        
        outside_flag = np.zeros(sites.shape[0], np.int32)
        outside_flag[sites[:,0] < visual_hull[0] + (visual_hull[3]-visual_hull[0])/(2*res)] = 1
        outside_flag[sites[:,1] < visual_hull[1] + (visual_hull[4]-visual_hull[1])/(2*res)] = 1
        outside_flag[sites[:,2] < visual_hull[2] + (visual_hull[5]-visual_hull[2])/(2*res)] = 1
        outside_flag[sites[:,0] > visual_hull[3] - (visual_hull[3]-visual_hull[0])/(2*res)] = 1
        outside_flag[sites[:,1] > visual_hull[4] - (visual_hull[4]-visual_hull[1])/(2*res)] = 1
        outside_flag[sites[:,2] > visual_hull[5] - (visual_hull[5]-visual_hull[2])/(2*res)] = 1

        ext_flag = np.zeros(sites.shape[0], np.int32)
        """ext_flag[sites[:,0] < visual_hull[0] - 10.5] = 1
        ext_flag[sites[:,1] < visual_hull[1] - 10.5] = 1
        ext_flag[sites[:,2] < visual_hull[2] - 10.5] = 1
        ext_flag[sites[:,0] > visual_hull[3] + 10.5] = 1
        ext_flag[sites[:,1] > visual_hull[4] + 10.5] = 1
        ext_flag[sites[:,2] > visual_hull[5] + 10.5] = 1"""

        self.sites = torch.from_numpy(sites.astype(np.float32)).cuda()
        self.sites = self.sites.contiguous()

        with torch.no_grad():
            norm_sites = torch.linalg.norm(self.sites, ord=2, axis=-1, keepdims=True)
            self.sdf[:] = norm_sites[:,0] - 0.5
            self.sdf[ext_flag[:] == 1] = -1000.0
            self.tet32.sdf_init = self.sdf.detach().clone()

        #self.tet32.move_sites(outside_flag, cam_ids, self.sdf.detach(), self.fine_features.detach())

        #self.tet32.clipped_cvt(self.sdf.detach(), self.fine_features.detach(), outside_flag, 
        #                       cam_ids, self.learning_rate_cvt, "Exp/bmvs_man/clipped_CVT.ply")
        #input()  
        
        self.Allocate_data()
            
        self.Allocate_batch_data()
        
        cvt_grad_cuda.diff_tensor(self.tet32.nb_tets, self.tet32.summits, self.sites, self.vol_tet32, self.weights_diff, self.weights_tot_diff)
                        
        if not hasattr(self, 'optimizer_sdf'):
            self.optimizer_sdf = torch.optim.Adam([self.sdf], lr=self.learning_rate_sdf)     # Beta ??   0.98, 0.995
            self.optimizer_feat = torch.optim.Adam([self.fine_features], lr=self.learning_rate_feat) 
            #self.optimizer_cvt = torch.optim.Adam([self.sites], lr=self.learning_rate_cvt) 

        if self.double_net:
            self.vortSDF_renderer_coarse_net.prepare_buffs(self.batch_size, self.n_samples, self.sites.shape[0])
        else:
            self.vortSDF_renderer_coarse.prepare_buffs(self.batch_size, self.n_samples, self.sites.shape[0])
        
        self.vortSDF_renderer_fine.prepare_buffs(self.batch_size, self.n_samples, self.sites.shape[0])

        grad_sdf = torch.zeros(self.sdf.shape).float().cuda()

        self.mask_background = torch.zeros(self.sdf.shape).float().cuda()
        
        """self.inv_s = 1000
        self.render_image(cam_ids, 0)
        input()"""
        
        acc_it = 1
        self.s_max = 10000
        self.R = 40
        self.s_start = 10.0
        self.inv_s = 0.1
        self.sigma = 0.12
        self.sigma_feat = 0.06
        step_size = 0.01
        self.w_g = 0.01
        #step_size = 0.003
        warm_up = 200
        self.loc_iter = 0
        image_perm = self.get_image_perm()
        num_rays = self.batch_size

        for iter_step in tqdm(range(self.end_iter)):
            if iter_step % acc_it == 0:
                img_idx = image_perm[iter_step % len(image_perm)].item() 
                self.inv_s = min(self.s_max, self.loc_iter/self.R + self.s_start)

            ## Generate rays
            if iter_step +1 < 2000:
                num_rays = 512
            elif iter_step+1 < 10000:
                num_rays = 2048
            elif iter_step+1 < 30000:
                num_rays = 4096
            #elif iter_step+1 < 70000:
            #    num_rays = 4096
            else:
                num_rays = self.batch_size

            data = self.dataset.gen_random_rays_zbuff_at(img_idx, num_rays, 0) 
            rays_o, rays_d, true_rgb, mask = data[:, :3], data[:, 3: 6], data[:, 6: 9], data[:, 9: 10]

            rays_o = rays_o.reshape(-1, 3)
            rays_d = rays_d.reshape(-1, 3)
            true_rgb = true_rgb.reshape(-1, 3)
            mask = mask.reshape(-1, 1)
            
            """if iter_step > 30000:
                rays_o = rays_o[mask[:,0] > 0.5]
                rays_d = rays_d[mask[:,0]  > 0.5]
                true_rgb = true_rgb[mask[:,0] > 0.5]
                mask = mask[mask[:,0] > 0.5]"""

            rays_o = rays_o.contiguous()
            rays_d = rays_d.contiguous()
            true_rgb = true_rgb.contiguous()
            mask = mask.contiguous()        

            if iter_step % acc_it == 0:
                self.activated[:] = 1
                with torch.no_grad():
                    self.sdf_smooth[:] = 0.0
                backprop_cuda.knn_smooth(self.sites.shape[0], 96, self.sigma, self.sigma_feat, 1, self.sites,  self.activated,
                                        self.grad_sdf_space, self.sdf, self.fine_features, self.tet32.knn_sites, self.sdf_smooth)
            
            ## sample points along the rays
            start = timer()
            self.offsets[:] = 0
            self.activated[:] = 0
            nb_samples = self.tet32.sample_rays_cuda(self.inv_s, img_idx, rays_d, self.sdf, cam_ids, self.in_weights, self.in_z, self.in_sdf, self.in_ids, self.offsets, self.activated, self.n_samples)    
            if verbose:
                print('CVT_Sample time:', timer() - start)   

            if nb_samples == 0:
                continue

            start = timer()
            self.activated_buff[:] = self.activated[:]
            backprop_cuda.activate_sites(rays_o.shape[0], self.sites.shape[0], 96, self.out_ids, self.offsets, self.tet32.knn_sites, self.activated_buff, self.activated)
            self.activated = self.activated + self.activated_buff
            if verbose:
                print('activate time:', timer() - start)   
                
            #if iter_step % 3 == 0:
            #    self.activated[:] = 1

            ############ Compute spatial SDF gradients
            if iter_step % acc_it == 0:
                start = timer()   
                self.grad_sdf_space[:] = 0.0
                self.grad_feat_space[:] = 0.0
                self.grad_mean_curve[:] = 0.0
                self.weights_grad[:] = 0.0
                self.grad_eik[:] = 0.0
                self.grad_norm_smooth[:] = 0.0
                self.eik_loss[:] = 0.0
                #self.activated[:] = 1

                cvt_grad_cuda.eikonal_grad(self.tet32.nb_tets, self.sites.shape[0], self.tet32.summits, self.sites, self.activated, self.sdf.detach(), self.sdf_smooth, self.fine_features.detach(), 
                                            self.grad_eik, self.grad_norm_smooth, self.grad_sdf_space, self.grad_feat_space, self.vol_tet32, self.weights_diff, self.weights_tot_diff, self.eik_loss)
                
                if math.isnan(self.grad_sdf_space.mean()):
                    print("self.grad_sdf_space", self.grad_sdf_space.mean())
                    input()

                self.norm_grad = torch.linalg.norm(self.grad_sdf_space, ord=2, axis=-1, keepdims=True).reshape(-1, 1)
                self.norm_grad[self.norm_grad == 0.0] = 1.0
                self.unormed_grad[:] = self.grad_sdf_space[:]
                self.grad_sdf_space = self.grad_sdf_space / self.norm_grad.expand(-1, 3)
                
                if math.isnan(self.grad_sdf_space.mean()):
                    print("self.grad_sdf_space unit", self.grad_sdf_space.mean())
                    input()

            
                self.grad_eik[outside_flag[:] == 1] = 0.0
                self.grad_eik[ext_flag[:] == 1] = 0.0   
                #self.grad_eik[:] = self.grad_eik[:] / self.sites.shape[0]
                self.grad_norm_smooth[outside_flag[:] == 1] = 0.0
                self.grad_norm_smooth[ext_flag[:] == 1] = 0.0   
                #self.grad_norm_smooth[:] = self.grad_norm_smooth[:] / self.sites.shape[0]
                self.grad_mean_curve[outside_flag[:] == 1.0] = 0.0
                self.grad_mean_curve[ext_flag[:] == 1] = 0.0   
                eik_loss = 0.0 #self.eik_loss.sum()

                ############## Concat multi-res features and normals ##############
                #with torch.no_grad():
                #    self.fine_features[:,8:] = 0.0
                self.grad_sdf_feat[:,:3] = self.grad_sdf_space[:,:3]
                self.grad_sdf_feat[:,3:] = 0.0
                cvt_grad_cuda.concat_feat(self.sites.shape[0], 96, 8, self.sites, self.activated, self.grad_sdf_feat, self.fine_features, self.tet32.knn_sites)

            if verbose:
                print('eikonal_grad time:', timer() - start)
            
            
            self.offsets[self.offsets[:,1] == -1] = 0                         
            start = timer()
            self.samples[:] = 0.0
            self.out_grads[:] = 0.0
            tet32_march_cuda.fill_samples(rays_o.shape[0], self.n_samples, rays_o, rays_d, self.sites, 
                                        self.in_z, self.in_sdf, self.fine_features, self.in_weights, self.grad_sdf_feat, self.in_ids, 
                                        self.out_z, self.out_sdf, self.out_feat, self.out_weights, self.out_grads, self.out_ids, 
                                        self.offsets, self.samples, self.samples_rays)
            if verbose:
                print('fill_samples time:', timer() - start)  

            #print(torch.linalg.norm(self.out_grads[:nb_samples,:3], ord=2, axis=-1, keepdims=True).min())
           

            if False: #(iter_step+1) > 10000: 
                #pts = self.samples[:nb_samples,:].cpu()
                norm = matplotlib.colors.Normalize(vmin=-0.3, vmax=0.3 , clip = False)
                sdf_rgb = plt.cm.jet(norm(self.out_sdf[:,0].cpu())).astype(np.float32)
                print("NB rays == ", rays_o.shape[0])
                print("img_idx == ", img_idx)
                print("nb_samples == ", nb_samples)
                """sdf_samp = torch.zeros((rays_o.shape[0]*256, 3)).cuda()
                for i in range(rays_o.shape[0]):
                    for j in range(256):
                        sdf_samp[i*256+j, :] = rays_o[i,:] + self.in_z[2*i*256+2*j]*rays_d[i,:]"""
                """colors_samples = torch.zeros_like(pts)
                for i in range(rays_o.shape[0]):                            
                    start = self.offsets[2*i]
                    end = self.offsets[2*i+1]
                    for j in range(start, start+end-1):
                        colors_samples[j,:] = true_rgb[i,:]"""
                ply.save_ply("Exp/bmvs_man/samples.ply", np.transpose(self.samples[:nb_samples,:].cpu()), col = 255*np.transpose(sdf_rgb))
                #ply.save_ply("TMP/meshes/samples_"+str(self.iter_step).zfill(5)+".ply", np.transpose(pts.cpu()), col = 255*np.transpose(sdf_rgb))
                #print("nb samples: ", nb_samples)
                exit()
                input()

            #samples = (self.samples[:nb_samples,:] + self.samples_loc[:nb_samples,:])/2.0
            #samples = self.samples[:nb_samples,:]
            #samples = samples.contiguous()
            self.samples[:] = (self.samples[:] + 1.1)/2.2

            ##### ##### ##### ##### ##### ##### 
            xyz_emb = (self.samples[:nb_samples,:].unsqueeze(-1) * self.posfreq).flatten(-2)
            xyz_emb = torch.cat([self.samples[:nb_samples,:], xyz_emb.sin(), xyz_emb.cos()], -1)

            """self.xyz_emb[:nb_samples, 33:48] = (self.samples[:nb_samples,:].unsqueeze(-1) * self.posfreq).flatten(-2)
            self.xyz_emb[:nb_samples, :3] = self.samples[:nb_samples,:]
            self.xyz_emb[:nb_samples, 3:18] = self.xyz_emb[:nb_samples, 33:48].sin()
            self.xyz_emb[:nb_samples, 18:33] = self.xyz_emb[:nb_samples, 33:48].cos()"""

            viewdirs_emb = (self.samples_rays[:nb_samples,:].unsqueeze(-1) * self.viewfreq).flatten(-2)
            viewdirs_emb = torch.cat([self.samples_rays[:nb_samples,:], viewdirs_emb.sin(), viewdirs_emb.cos()], -1)                       

            """self.viewdirs_emb[:nb_samples, 9:12] = (self.samples_rays[:nb_samples,:].unsqueeze(-1) * self.viewfreq).flatten(-2)
            self.viewdirs_emb[:nb_samples, :3] = self.samples_rays[:nb_samples,:]
            self.viewdirs_emb[:nb_samples, 3:6] = self.viewdirs_emb[:nb_samples, 9:12].sin()
            self.viewdirs_emb[:nb_samples, 6:9] = self.viewdirs_emb[:nb_samples, 9:12].cos()"""

            if self.double_net:
                coarse_feat = torch.cat([xyz_emb, viewdirs_emb, self.out_sdf[:nb_samples,:], self.out_grads[:nb_samples,:], self.out_feat[:nb_samples,:32]], -1)       
                #coarse_feat = torch.cat([xyz_emb, viewdirs_emb, self.geo_features[:nb_samples,:]], -1)       
                #coarse_feat = torch.cat([xyz_emb, viewdirs_emb, self.out_sdf[:nb_samples,:]], -1)       
                coarse_feat.requires_grad_(True)
                coarse_feat.retain_grad()

                """if self.coarse_feat.grad is not None:
                    self.coarse_feat.grad[:] = 0.0

                with torch.no_grad():
                    self.coarse_feat[:nb_samples,:33] = self.xyz_emb[:nb_samples,:33]
                    self.coarse_feat[:nb_samples, 33:42] = self.viewdirs_emb[:nb_samples,:9]
                    self.coarse_feat[:nb_samples, 42:44] = self.out_sdf[:nb_samples,:]
                    self.coarse_feat[:nb_samples, 44:56] = self.out_grads[:nb_samples,:]
                    self.coarse_feat[:nb_samples, 56:] = self.out_feat[:nb_samples,:32]"""

                #self.colors_feat = self.color_coarse.rgb(self.coarse_feat[:nb_samples,:])  
                colors_feat = self.color_coarse.rgb(coarse_feat)  
                self.colors = torch.sigmoid(colors_feat)
                self.colors = self.colors.contiguous()         
            
            # network interpolation
            """ if self.rgb_feat.grad is not None:
                self.rgb_feat.grad[:] = 0.0"""

            #with torch.no_grad():
            if self.double_net:
                if self.position_encoding:
                    """self.rgb_feat[:nb_samples,:33] = self.xyz_emb[:nb_samples,:33]
                    self.rgb_feat[:nb_samples, 33:42] = self.viewdirs_emb[:nb_samples,:9]
                    self.rgb_feat[:nb_samples, 42:45] = self.colors_feat[:nb_samples,:]
                    self.rgb_feat[:nb_samples, 45:47] = self.out_sdf[:nb_samples,:]
                    self.rgb_feat[:nb_samples, 47:59] = self.out_grads[:nb_samples,:]
                    self.rgb_feat[:nb_samples, 59:] = self.out_feat[:nb_samples,:32]"""
                    rgb_feat = torch.cat([xyz_emb, viewdirs_emb, colors_feat, self.out_sdf[:nb_samples,:], self.out_grads[:nb_samples,:], self.out_feat[:nb_samples,:32]], -1)
                else:
                    rgb_feat = torch.cat([viewdirs_emb, colors_feat.detach(), self.out_sdf[:nb_samples,:], self.out_grads[:nb_samples,:], self.out_feat[:nb_samples,:]], -1) #, self.out_weights[:nb_samples,:]
            else:
                if self.position_encoding:
                    rgb_feat = torch.cat([xyz_emb, viewdirs_emb, self.out_sdf[:nb_samples,:], self.out_grads[:nb_samples,:], self.out_feat[:nb_samples,:]], -1)
                else:
                    rgb_feat = torch.cat([viewdirs_emb, self.out_sdf[:nb_samples,:], self.out_grads[:nb_samples,:], self.out_feat[:nb_samples,:]], -1) #, self.out_weights[:nb_samples,:]
                
            rgb_feat.requires_grad_(True)
            rgb_feat.retain_grad()

            #self.colors_fine = torch.sigmoid(self.color_network.rgb(rgb_feat)) 
            if self.double_net:
                self.colors_fine = torch.sigmoid(self.color_network.rgb(rgb_feat) + colors_feat.detach()) #+ self.colors
            else:
                self.colors_fine = torch.sigmoid(self.color_network.rgb(rgb_feat)) #+ self.colors"""
            self.colors_fine = self.colors_fine.contiguous()

            ########################################
            ####### Render the image ###############
            ########################################
            mask = (mask > 0.5).float()

            start = timer()       
            if self.double_net:
                color_coarse_loss = VortSDFRenderingFunction.apply(self.vortSDF_renderer_coarse_net, rays_o.shape[0], self.inv_s, self.out_sdf, self.tet32.knn_sites, 
                                                                   self.out_weights, self.colors, true_rgb, mask, self.out_ids, self.offsets, self.out_grads, rays_d)
            #self.vortSDF_renderer_coarse.render_gpu(rays_o.shape[0], self.inv_s, self.out_sdf, self.tet32.knn_sites, self.out_weights, self.colors, true_rgb, mask, self.out_ids, self.offsets)
            
            color_fine_loss = VortSDFRenderingFunction.apply(self.vortSDF_renderer_fine, rays_o.shape[0], self.inv_s, self.out_sdf, self.tet32.knn_sites, 
                                                             self.out_weights, self.colors_fine, true_rgb, mask, self.out_ids, self.offsets, self.out_grads, rays_d)
            if verbose:
                print('RenderingFunction time:', timer() - start)

            # Total loss   
            mask_sum = mask.sum()
            
            self.optimizer.zero_grad()
            if self.double_net:
                loss = (color_fine_loss + color_coarse_loss) #/ (mask_sum + 1.0e-5)
            else:
                loss = color_fine_loss / (mask_sum + 1.0e-5)
            loss.backward()

            ########################################
            # Backprop feature gradients to gradients on sites
            # step optimize color features
            start = timer()   
            #sdf_features_grad = torch.cat([sdf_feat_entry.grad[:,3:6], sdf_feat_exit.grad[:,3:6]], -1).contiguous()
 
            if self.double_net:
                if self.position_encoding:
                    shift_p = xyz_emb.shape[1] + viewdirs_emb.shape[1]
                    self.fine_features_grad[:nb_samples,:] = rgb_feat.grad[:,shift_p+17:] + 0.5*coarse_feat.grad[:,shift_p+14:]
                    self.norm_features_grad[:nb_samples,:] = rgb_feat.grad[:,shift_p+5:shift_p+17] + 0.5*coarse_feat.grad[:,shift_p+2:shift_p+14]
                    self.sdf_features_grad[:nb_samples,:] = rgb_feat.grad[:,shift_p+3:shift_p+5] + 0.5*coarse_feat.grad[:,shift_p:shift_p+2]
                else:
                    fine_features_grad = rgb_feat.grad[:,36:]
                    norm_features_grad = rgb_feat.grad[:,30:36] + 0.5*self.coarse_feat.grad[:nb_samples,42:]
            else:
                if self.position_encoding:
                    fine_features_grad = rgb_feat.grad[:,48:] # 44 <- view pose encoding = 1 rgb_feat.grad[:,62:]  <- view pose encoding = 4
                    norm_features_grad = rgb_feat.grad[:,42:48]
                else:
                    fine_features_grad = rgb_feat.grad[:,33:]
                    norm_features_grad = rgb_feat.grad[:,27:33]
            #fine_features_grad = fine_features_grad.contiguous()
            #norm_features_grad = norm_features_grad.contiguous()

            if verbose:
                print('input grads time:', timer() - start)
             
            start = timer()
            self.grad_features[:] = 0.0
            self.counter[:] = 0.0
            backprop_cuda.backprop_feat(nb_samples, self.sites.shape[0], self.dim_feats, self.grad_features, self.counter, self.fine_features_grad, self.out_ids, self.out_weights)  
            if verbose:
                print('backprop_feat samples time:', timer() - start)
                
            start = timer()
            self.grad_norm_feat[:] = 0.0
            self.counter[:] = 0.0
            backprop_cuda.backprop_feat(nb_samples, self.sites.shape[0], 12, self.grad_norm_feat, self.counter, self.norm_features_grad, self.out_ids, self.out_weights)
            self.grad_norm[:,:] = self.grad_norm_feat[:,:3]
            #self.grad_features[:,8:] = 0.0

            if verbose:
                print('backprop_feat grads time:', timer() - start)

            ############ Backprop gradients to neighbors ############                   
            """start = timer()
            self.grad_norm[:] = 0.0
            backprop_cuda.backprop_multi(self.sites.shape[0], 96, 8, self.sites, self.activated, self.grad_norm, self.grad_norm_feat, self.grad_features, self.tet32.knn_sites)
            self.grad_features[:,8:] = 0.0
            if verbose:
                print('backprop_multi sites time:', timer() - start)"""
             
            start = timer()
            self.grad_sdf_norm[:] = 0.0
            #backprop_cuda.backprop_norm(self.tet32.nb_tets, self.tet32.summits, self.sites, self.vol_tet32, self.weights_diff, self.weights_tot_diff, self.grad_norm_feat, self.grad_sdf_norm, self.activated)
            backprop_cuda.backprop_unit_norm(self.tet32.nb_tets, self.tet32.summits, self.sites, self.norm_grad, self.unormed_grad, self.vol_tet32, self.weights_diff, self.weights_tot_diff, self.grad_norm, self.grad_sdf_norm, self.activated)
            self.grad_sdf_norm[outside_flag[:] == 1] = 0.0    
            self.grad_sdf_norm[ext_flag[:] == 1] = 0.0   
            if verbose:
                print('backprop_unit_norm sites time:', timer() - start)

            start = timer()
            self.grad_sdf_net[:] = 0.0
            backprop_cuda.backprop_sdf(nb_samples, self.grad_sdf_net, self.sdf_features_grad, self.out_ids, self.out_weights)    
            ##backprop_cuda.backprop_norm(nb_samples, self.grad_sdf_net, self.grad_norm_feat, self.weights_grad)  
            self.grad_sdf_net[outside_flag[:] == 1] = 0.0   
            self.grad_sdf_net[ext_flag[:] == 1] = 0.0   
            
            if verbose:
                print('backprop_sdf sites time:', timer() - start)
            
            self.optimizer.step()

            #self.activated[:] = 1
            
            ########################################
            ####### Regularization terms ###########
            ########################################

            if self.double_net:     
                #grad_sdf =  self.f_w*(self.grad_sdf_net + self.grad_sdf_norm) +\
                #    ((1.0 - lamda_c)*self.vortSDF_renderer_fine.grads_sdf + lamda_c*self.vortSDF_renderer_coarse_net.grads_sdf)
                #grad_sdf =  (self.vortSDF_renderer_fine.grads_sdf)
                self.grad_sdf = self.f_w*(self.grad_sdf_net + self.grad_sdf_norm) +\
                             (self.vortSDF_renderer_fine.grads_sdf) # + 0.5*self.vortSDF_renderer_coarse_net.grads_sdf)
            else:
                self.grad_sdf = self.vortSDF_renderer_fine.grads_sdf / (mask_sum + 1.0e-5) + self.f_w*self.grad_sdf_net

            self.grad_sdf[outside_flag[:] == 1] = 0.0   
            self.grad_sdf[ext_flag[:] == 1] = 0.0   

            """self.grad_sdf_smooth[:] = grad_sdf[:]
            self.counter_smooth[:] = 1.0
            backprop_cuda.smooth(self.tet32.edges.shape[0], self.sites.shape[0], self.sigma, 1, self.sites, grad_sdf, 
                                 self.fine_features, self.tet32.edges, self.grad_sdf_smooth, self.counter_smooth)"""
            
            #self.activated[:] = -1
            start = timer()   
            #self.activated[abs(self.grad_sdf) > 0.0] = 1
            #self.activated[grad_sdf == 0.0] = -1
            self.grad_sdf_smooth[:] = 0.0
            backprop_cuda.knn_smooth(self.sites.shape[0], 96, self.sigma, self.sigma_feat, 1, self.sites, self.activated,
                                        self.grad_sdf_space, self.grad_sdf, self.fine_features, self.tet32.knn_sites, self.grad_sdf_smooth)
            self.grad_sdf[:] = self.grad_sdf_smooth[:]
            self.grad_sdf[outside_flag[:] == 1.0] = 0.0   
            if verbose:
                print('knn_smooth time:', timer() - start)
                            
            #### SMOOTH FEATURE GRADIENT
            """self.grad_feat_smooth[:] = self.grad_features[:]
            self.counter_smooth[:] = 1.0
            backprop_cuda.smooth(self.tet32.edges.shape[0], self.sites.shape[0], self.sigma, self.dim_feats, self.sites, self.grad_features, 
                                 self.tet32.edges, self.grad_feat_smooth, self.counter_smooth)"""
            
            """self.grad_feat_smooth[:] = 0.0
            backprop_cuda.knn_smooth(self.sites.shape[0], 96, self.sigma, self.sigma_feat, self.dim_feats, self.sites, self.activated,
                                     self.grad_sdf_space, self.grad_features, self.fine_features, self.tet32.knn_sites, self.grad_feat_smooth) 
            self.grad_features[:] = self.grad_feat_smooth[:]
            self.grad_features[outside_flag[:] == 1.0] = 0.0"""  
            
                
            """start = timer()   
            self.grad_sdf_reg[:] = 0.0
            self.grad_feat_reg[:] = 0.0
            backprop_cuda.space_reg(rays_o.shape[0], rays_d, self.grad_sdf_space, self.out_weights, self.out_z, self.out_sdf, self.out_feat, self.out_ids, self.offsets, self.grad_sdf_reg, self.grad_feat_reg)
            
            if verbose:
                print('space_reg time:', timer() - start)"""
        
            if iter_step % acc_it == 0 and self.tv_w > 0.0: # and (iter_step % 3 == 0 or (iter_step+1) < 10000): # and ((iter_step+1) < 35000 or iter_step % 3 == 0):
                start = timer()   
                self.grad_sdf_smooth[:] = 0.0
                self.grad_feat_smooth[:] = 0.0
                self.weight_sdf_smooth[:] = 0.0
                #self.activated[:] = 1
                #if (iter_step+1) < 35000:
                #self.activated[grad_sdf == 0.0] = 0 #self.sigma
                backprop_cuda.smooth_sdf(self.tet32.edges.shape[0], self.sigma, self.sites, self.activated,
                                         self.sdf, self.fine_features, self.tet32.edges, self.grad_sdf_smooth, self.grad_feat_smooth, self.weight_sdf_smooth)
                self.weight_sdf_smooth[self.weight_sdf_smooth[:] == 0.0] = 1.0
                self.grad_sdf_smooth = self.grad_sdf_smooth / self.weight_sdf_smooth[:]
                #self.grad_sdf_smooth[:] = self.grad_sdf_smooth[:] / self.sites.shape[0]
                self.grad_feat_smooth = self.grad_feat_smooth / self.weight_sdf_smooth[:].reshape(-1,1)
                self.grad_feat_smooth[:,:] = self.grad_feat_smooth[:,:] / self.sigma
                self.grad_sdf_smooth[outside_flag[:] == 1] = 0.0   
                self.grad_sdf_smooth[ext_flag[:] == 1] = 0.0   
                if verbose:
                    print('smooth_sdf time:', timer() - start)
            #else:
            #    self.grad_feat_smooth[:] = 0.0
            
            ########################################
            ####### Optimize features ##############
            ########################################
            
            if iter_step % acc_it == 0:
                self.optimizer_feat.zero_grad()

            if self.fine_features.grad is None:
                self.fine_features.grad = self.grad_features  
            else:
                self.fine_features.grad = self.fine_features.grad + self.grad_features # + self.tv_f*self.tv_w*self.grad_feat_smooth #+ 1.0e+5*self.grad_feat_reg / (mask_sum + 1.0e-5)
            

            if iter_step % acc_it == acc_it-1:
                self.fine_features.grad = self.fine_features.grad + self.tv_f*self.tv_w*self.grad_feat_smooth
                self.optimizer_feat.step()

            if iter_step > 5000 and self.loc_iter < warm_up:
                self.update_learning_rate(self.loc_iter)
                self.loc_iter = self.loc_iter + 1
                continue

            ########################################
            ####### Optimize sdf values ############
            ########################################

            #grad_sdf = (0.5*self.voro_renderer_coarse.grads_sdf[:,0] + 1.0*self.voro_renderer_fine.grads_sdf[:,0]) / (mask_sum + 1.0e-5)
           
            norm_grad = torch.linalg.norm(self.grad_sdf_space, ord=2, axis=-1, keepdims=True).reshape(-1)

            """if (iter_step+1) > 5000:
                self.grad_norm_smooth[abs(self.sdf) > 3.0*self.sigma] = 0.0
                self.grad_eik[abs(self.sdf) > 3.0*self.sigma] = 0.0
                self.grad_sdf_smooth[abs(self.sdf) > 3.0*self.sigma] = 0.0"""
            
            #self.grad_norm_smooth = self.grad_norm_smooth / self.sites.shape[0]
            #self.grad_eik = self.grad_eik / self.sites.shape[0]
            #self.grad_sdf_smooth = self.grad_sdf_smooth / self.sigma
                
            if False: #iter_step % 3 != 0 or (iter_step+1) < 25000:    
                self.grad_norm_smooth[grad_sdf == 0.0] = 0.0
                self.grad_eik[grad_sdf == 0.0] = 0.0
                self.grad_sdf_smooth[grad_sdf == 0.0] = 0.0
                
            """if iter_step % 3 != 0: # or (iter_step+1) > 5000:# and (iter_step+1) > 5000: # or (iter_step+1) > 10000:   
                self.grad_norm_smooth[:] = 0.0
                self.grad_eik[:] = 0.0
                self.grad_sdf_smooth[:] = 0.0"""

                
            """if iter_step % 5 != 0 and (iter_step+1) > 15000: # or (iter_step+1) > 10000:   
                self.grad_norm_smooth[:] = 0.0
                self.grad_eik[:] = 0.0
                self.grad_sdf_smooth[:] = 0.0""" 


            if iter_step % acc_it == 0:
                self.optimizer_sdf.zero_grad() # 0.00001*self.grad_mean_curve +\ # self.e_w*self.grad_eik +\

            self.norm_grad = torch.linalg.norm(self.unormed_grad, ord=2, axis=-1, keepdims=True).reshape(-1)
            #self.grad_sdf[self.mask_background[:] == 1.0] = 0.0

            ###self.norm_grad * 
            if (iter_step+1) < 2000: 
                if self.sdf.grad is None:
                    self.sdf.grad = self.grad_sdf #+ (self.e_w*self.grad_eik+\
                                            #    self.s_w*self.grad_norm_smooth  +\
                                             #   self.tv_w*self.grad_sdf_smooth) 
                else:
                    self.sdf.grad = self.sdf.grad + self.grad_sdf #+ (self.e_w*self.grad_eik+\
                                                #self.s_w*self.grad_norm_smooth  +\
                                                #self.tv_w*self.grad_sdf_smooth) 
            else:
                if self.sdf.grad is None:
                    self.sdf.grad = self.grad_sdf #+ abs(grad_sdf)*(0.0000001*self.grad_eik+\
                                            #                0.0005*self.grad_norm_smooth+\
                                            #                0.0001*self.grad_sdf_smooth) 
                else:
                    self.sdf.grad = self.sdf.grad + self.grad_sdf #+ abs(grad_sdf)*(0.0000001*self.grad_eik+\
                                                            #0.0005*self.grad_norm_smooth+\
                                                           # 0.0001*self.grad_sdf_smooth) 
            if iter_step % acc_it == acc_it-1:
                if (iter_step+1) < 2000: 
                    self.sdf.grad = self.sdf.grad + (self.e_w*self.grad_eik+\
                                                self.s_w*self.grad_norm_smooth  +\
                                                self.tv_w*self.grad_sdf_smooth) 
                else:
                    self.grad_norm_smooth[self.sdf.grad[:] == 0.0] = 0.0
                    self.grad_eik[self.sdf.grad[:] == 0.0] = 0.0
                    self.grad_sdf_smooth[self.sdf.grad[:] == 0.0] = 0.0
                    """if not iter_step % 3 == 0:
                        self.grad_norm_smooth[:] = 0.0
                        self.grad_eik[:] = 0.0
                        self.grad_sdf_smooth[:] = 0.0"""
                    """self.grad_norm_smooth[self.activated[:] < 2] = 0.0
                    self.grad_eik[self.activated[:] < 2] = 0.0
                    self.grad_sdf_smooth[self.activated[:] < 2] = 0.0"""
                    #abs(self.sdf.grad)*

                    #w_photo = (self.grad_sdf_space * self.samples_rays[:nb_samples,:]).sum()
                    self.sdf.grad = self.sdf.grad + (self.e_w*self.grad_eik+\
                                                                    self.s_w*self.grad_norm_smooth+\
                                                                    self.tv_w*self.grad_sdf_smooth) #abs(self.sdf.grad)*
                self.optimizer_sdf.step()


            #with torch.no_grad():
            #    self.fine_features[:,0] = self.sdf[:]
            #    self.fine_features[self.activated[:] == 1, 1:4] = self.grad_sdf_space[self.activated[:] == 1,:]

            ########################################
            ##### Optimize sites positions #########
            ########################################
            if (iter_step+1) == 2000 or (iter_step+1) == 10000 or (iter_step+1) == 30000 or (iter_step+1) == 50000 or (iter_step+1) == 70000:# or (iter_step+1) == 45000:
                
                with torch.no_grad():
                    self.sdf[ext_flag[:] == 1] = 1000.0

                #if (iter_step+1) == 20000:
                #    self.batch_size = 10240

                #if self.sites.shape[0] > 500000:
                #    self.sdf, self.fine_features = self.tet32.upsample(self.sdf.detach().cpu().numpy(), self.fine_features.detach().cpu().numpy(), visual_hull, res, cam_sites, self.learning_rate_cvt, False, 0.0) #(iter_step+1) > 2000
                #else:
                self.sdf, self.fine_features, self.mask_background = self.tet32.upsample(self.sdf.detach().cpu().numpy(), self.fine_features.detach().cpu().numpy(), visual_hull, res, cam_sites, self.learning_rate_cvt, (iter_step+1) > 30000, self.sigma)
                self.sdf = self.sdf.contiguous()
                self.sdf.requires_grad_(True)
                self.fine_features = self.fine_features.contiguous()
                self.fine_features.requires_grad_(True)
                self.tet32.load_cuda()

                print(self.mask_background.max())
                print(self.mask_background.min())

                sites = np.asarray(self.tet32.vertices)  
                cam_ids = np.stack([np.where((sites == cam_sites[i,:]).all(axis = 1))[0] for i in range(cam_sites.shape[0])]).reshape(-1)
                self.tet32.make_adjacencies(cam_ids)

                self.tet32.make_multilvl_knn()

                cam_ids = torch.from_numpy(cam_ids).int().cuda()
                
                outside_flag = np.zeros(sites.shape[0], np.int32)
                outside_flag[sites[:,0] < visual_hull[0] + (visual_hull[3]-visual_hull[0])/(res)] = 1
                outside_flag[sites[:,1] < visual_hull[1] + (visual_hull[4]-visual_hull[1])/(res)] = 1
                outside_flag[sites[:,2] < visual_hull[2] + (visual_hull[5]-visual_hull[2])/(res)] = 1
                outside_flag[sites[:,0] > visual_hull[3] - (visual_hull[3]-visual_hull[0])/(res)] = 1
                outside_flag[sites[:,1] > visual_hull[4] - (visual_hull[4]-visual_hull[1])/(res)] = 1
                outside_flag[sites[:,2] > visual_hull[5] - (visual_hull[5]-visual_hull[2])/(res)] = 1
                
                ext_flag = np.zeros(sites.shape[0], np.int32)
                """ext_flag[sites[:,0] < visual_hull[0] - 10.5] = 1
                ext_flag[sites[:,1] < visual_hull[1] - 10.5] = 1
                ext_flag[sites[:,2] < visual_hull[2] - 10.5] = 1
                ext_flag[sites[:,0] > visual_hull[3] + 10.5] = 1
                ext_flag[sites[:,1] > visual_hull[4] + 10.5] = 1
                ext_flag[sites[:,2] > visual_hull[5] + 10.5] = 1"""

                with torch.no_grad():
                    self.sdf[ext_flag[:] == 1] = -1000.0
                
                self.sites = torch.from_numpy(sites.astype(np.float32)).cuda()
                self.sites = self.sites.contiguous()
                self.sites.requires_grad_(True)

                self.Allocate_data()
                
                cvt_grad_cuda.diff_tensor(self.tet32.nb_tets, self.tet32.summits, self.sites, self.vol_tet32, self.weights_diff, self.weights_tot_diff)
                                
                delta_sites = torch.zeros(self.sites.shape).float().cuda()
                with torch.no_grad():  
                    delta_sites[:] = self.sites[:]
                
                #with torch.no_grad():  
                #    self.fine_features[:] = 0.5

                """if (iter_step+1) == 50000:
                    self.conf['model.color_network']['rgbnet_width'] = 128
                    self.conf['model.color_network']['rgbnet_depth'] = 4
                    self.color_network = ColorNetwork(**self.conf['model.color_network']).to(self.device)
                    params_to_train = []
                    params_to_train += list(self.color_network.parameters())
                    self.optimizer = torch.optim.Adam(params_to_train, lr=self.learning_rate)"""
                    
                """if (iter_step+1) == 70000:
                    self.conf['model.color_network']['rgbnet_width'] = 64
                    self.conf['model.color_network']['rgbnet_depth'] = 2
                    self.color_network = ColorNetwork(**self.conf['model.color_network']).to(self.device)
                    params_to_train = []
                    params_to_train += list(self.color_network.parameters())
                    self.optimizer = torch.optim.Adam(params_to_train, lr=self.learning_rate)"""

                self.optimizer_sdf = torch.optim.Adam([self.sdf], lr=self.learning_rate_sdf)        
                self.optimizer_feat = torch.optim.Adam([self.fine_features], lr=self.learning_rate_feat) 
                #self.optimizer_cvt = torch.optim.Adam([self.sites], lr=self.learning_rate_cvt) 
                
                if self.double_net:
                    self.vortSDF_renderer_coarse_net.prepare_buffs(self.batch_size, self.n_samples, self.sites.shape[0])
                else:
                    self.vortSDF_renderer_coarse.prepare_buffs(self.batch_size, self.n_samples, self.sites.shape[0])
                self.vortSDF_renderer_fine.prepare_buffs(self.batch_size, self.n_samples, self.sites.shape[0])

                step_size = step_size / 1.5
                self.e_w = self.e_w / 10.0
                self.learning_rate_sdf = 1.0e-4
                self.tv_w = self.tv_w / 2.0
                self.learning_rate_cvt = self.learning_rate_cvt / 2.0

                self.sigma = self.sigma / 2.0

                if (iter_step+1) == 2000:
                    """self.s_w = 1.0e-4
                    self.e_w = 5.0e-6
                    self.tv_w = 1.0e-5"""
                    self.R = 10
                    self.s_w = 1.0e-3
                    self.e_w = 1.0e-4 #5.0e-3
                    self.tv_w = 0.0#5.0e-3 #1.0e-1
                    self.s_start = 50.0
                    self.learning_rate = 1e-3
                    self.learning_rate_sdf = 1.0e-3
                    self.learning_rate_feat = 1.0e-3
                    self.end_iter_loc = 8000
                    self.learning_rate_alpha = 1.0e-8
                    self.vortSDF_renderer_fine.mask_reg = 1.0e-1
                    #verbose = True

                if (iter_step+1) == 10000:
                    self.R = 30
                    self.s_start = 100.0
                    self.s_max = 300
                    """self.s_w = 1.0e-3
                    self.e_w = 5.0e-4
                    self.tv_w = 1.0e-5"""
                    self.s_w = 1.0e-3 #1e-6
                    self.e_w = 1.0e-6#1.0e-8 #5.0e-3
                    self.tv_w = 5.0e-4 #1.0e-8 #1.0e-1
                    self.tv_f = 1.0e-8
                    self.f_w = 1.0 #1.0
                    self.learning_rate = 5e-4
                    self.learning_rate_sdf = 5.0e-4 #3.0e-4 #5
                    self.learning_rate_feat = 5.0e-4
                    self.end_iter_loc = 20000
                    self.learning_rate_alpha = 1.0e-4
                    self.vortSDF_renderer_fine.mask_reg = 1.0e-1
                    #acc_it = 10
                    
                if (iter_step+1) == 30000:
                    warm_up = 200
                    self.R = 40
                    self.s_start = 200.0
                    self.s_max = 600
                    #self.sigma = 0.02
                    """self.s_w = 5.0e-3
                    self.e_w = 1.0e-3
                    self.tv_w = 1.0e-3"""
                    self.s_w = 1.0e-4 #5.0e-4
                    self.e_w = 1.0e-6 #1.0e-9 #1.0e-7 #5.0e-3
                    self.tv_w = 5.0e-5 #1.0e-8 #1.0e-1
                    self.w_g = 0.0
                    #acc_it = 10

                    self.tv_f = 0.0
                    self.f_w = 1.0
                    self.learning_rate = 5e-4
                    self.learning_rate_sdf = 5e-4
                    self.learning_rate_feat = 5e-4 #1.0e-2
                    self.end_iter_loc = 20000
                    self.vortSDF_renderer_fine.mask_reg = 1.0e-2
                    self.learning_rate_alpha = 1.0e-2

                if (iter_step+1) == 50000:
                    #warm_up = 2000
                    self.R = 40
                    self.s_start = 400.0
                    self.s_max = 1600
                    #self.sigma = 0.01
                    """self.s_w = 2.0e-4 #2.0e-6
                    self.e_w = 1.0e-5 #1.0e-7 #5.0e-3
                    self.tv_w = 1.0e-4 #1.0e-8 #1.0e-1"""
                    self.s_w = 1.0e-4 #2.0e-6
                    self.e_w = 1.0e-7 #1.0e-9 #1.0e-7 #5.0e-3
                    self.tv_w = 5.0e-5 #1.0e-8 #1.0e-1
                    self.tv_f = 0.0 #1.0e-4
                    self.f_w = 1.0
                    self.end_iter_loc = 20000
                    self.learning_rate = 1e-4
                    self.learning_rate_sdf = 1.0e-4
                    self.learning_rate_feat = 1.0e-4
                    self.vortSDF_renderer_fine.mask_reg = 1.0e-2
                    self.learning_rate_alpha = 1.0e-4
                    
                if (iter_step+1) == 70000:
                    self.R = 40
                    self.s_start = 600.0 #400
                    self.s_max = 4000
                    #self.sigma = 0.01
                    #self.sigma_feat = 0.02
                    """self.s_w = 1.0e-2
                    self.e_w = 1.0e-4
                    self.tv_w = 1.0e-2"""
                    self.s_w = 1.0e-4 #5.0e-4
                    self.e_w = 1.0e-9
                    self.tv_w = 5.0e-6 #1.0e-4 #1.0e-3
                    self.tv_f = 0.0 #1.0e-3
                    self.end_iter_loc = 20000
                    self.learning_rate = 5e-5
                    self.learning_rate_sdf = 5.0e-5
                    self.learning_rate_feat = 5.0e-5
                    self.vortSDF_renderer_fine.mask_reg = 1.0e-3
                    self.learning_rate_alpha = 1.0e-8
                    self.val_freq = 2000
                    #verbose = True
                    
                if (iter_step+1) == 36000:
                    self.R = 25
                    self.s_w = 1.0e-8
                    self.e_w = 1.0e-10
                    self.tv_w = 1.0e-7
                    self.end_iter_loc = 10000
                    self.learning_rate = 1e-4
                    self.learning_rate_sdf = 1.0e-5
                    self.learning_rate_feat = 1.0e-4
                    self.vortSDF_renderer_fine.mask_reg = 0.0

                print("SIGMA => ", self.sigma)
                #with torch.no_grad():
                #    self.sdf[:] = self.sdf[:] + self.sigma
                
                self.loc_iter = 0
                
                #self.save_checkpoint()                    
                torch.cuda.empty_cache()

                #verbose = True
                #self.tet32.save("Exp/bmvs_man/test_up.ply") 
                self.tet32.save_multi_lvl("Exp/bmvs_man/multi_lvl")    
                #self.render_image(cam_ids, img_idx)
                self.tet32.surface_from_sdf(self.sdf.detach().cpu().numpy().reshape(-1), "Exp/bmvs_man/test_tri_up_{}.ply".format(iter_step), self.dataset.scale_mats_np[0][:3, 3][None], self.dataset.scale_mats_np[0][0, 0])                          
                torch.cuda.empty_cache()

            if iter_step % self.report_freq == 0:
                print('iter:{:8>d} loss = {}, scale={}, grad={}, lr={}'.format(iter_step, loss, self.inv_s, abs(self.grad_sdf[abs(self.grad_sdf) > 0.0]).mean(), self.optimizer.param_groups[0]['lr']))
                print('iter:{:8>d} s_w = {}, e_w = {}, tv_w = {}, nb_samples={}, lr={}'.format(iter_step, self.s_w, self.e_w, self.tv_w, nb_samples, self.optimizer_sdf.param_groups[0]['lr']))
                #print('iter:{:8>d} loss CVT = {} lr={}'.format(iter_step, loss_cvt, self.optimizer_cvt.param_groups[0]['lr']))
                

            if iter_step % self.val_freq == 0:
                #self.inv_s = 1000     
                #self.render_image(cam_ids, img_idx)
                with torch.no_grad():
                    self.sdf[ext_flag[:] == 1] = 1000.0
                self.tet32.surface_from_sdf(self.sdf.detach().cpu().numpy().reshape(-1), "Exp/bmvs_man/meshes/test_tri_{}.ply".format(iter_step), self.dataset.scale_mats_np[0][:3, 3][None], self.dataset.scale_mats_np[0][0, 0])
                #if iter_step % 3 == 0:
                """self.activated[:] = 1
                with torch.no_grad():
                    self.sdf_smooth[:] = 0.0
                backprop_cuda.knn_smooth(self.sites.shape[0], 96, self.sigma, self.sigma_feat, 1, self.sites,  self.activated,
                                        self.grad_sdf_space, self.sdf, self.fine_features, self.tet32.knn_sites, self.sdf_smooth)
                """
                self.tet32.surface_from_sdf(self.sdf_smooth.detach().cpu().numpy().reshape(-1), "Exp/bmvs_man/test_tri_smooth.ply", self.dataset.scale_mats_np[0][:3, 3][None], self.dataset.scale_mats_np[0][0, 0])                               
                with torch.no_grad():
                    self.sdf[ext_flag[:] == 1] = -1000.0
                #self.tet32.marching_tets(self.sdf.detach(), "Exp/bmvs_man/test_MT.ply")
                #if iter_step > 1000:
                #    self.tet32.save("Exp/bmvs_man/test.ply")    
                torch.cuda.empty_cache()

            #if iter_step == 1000:                
            #    self.tv_w = 0.0
            #if iter_step == 12000:  
            #    self.learning_rate_sdf = 0.1*self.learning_rate_sdf
            #    self.tv_w = 0.0
            #    self.s_w = 1.0e-6
            #    #self.learning_rate_sdf = 1.0e-5
                
            """if iter_step == 23000 or iter_step == 33000: # self.loc_iter == round(0.9*self.end_iter_loc):          
                self.s_w = 10.0*self.s_w 
                #self.learning_rate_sdf = 0.1*self.learning_rate_sdf     
                #with torch.no_grad():
                #    self.sdf[:] = self.sdf[:] * 4.0"""
                
            """if self.loc_iter == round(0.8*self.end_iter_loc): # iter_step == 22000:
                #self.learning_rate_sdf = 0.1*self.learning_rate_sdf
                #self.tv_w = 1.0e-4
                self.s_w = 0.1*self.s_w
                self.e_w = 0.1*self.e_w
                self.tv_w = 0.1*self.e_w"""

            """if self.loc_iter == round(0.2*self.end_iter_loc):
                self.f_w = 1.0
                #self.s_w = 0.1*self.s_w
                #self.s_w = 0.5*self.s_w"""

            """if self.loc_iter == round(0.2*self.end_iter_loc): #iter_step == 10000:# or iter_step == 13000 or iter_step == 20000:   
                sdf_smooth_lapl = self.tet32.smooth_sdf(self.sdf.detach().cpu().numpy().reshape(-1))
                with torch.no_grad():
                    self.sdf[:] = sdf_smooth_lapl[:]"""

            self.update_learning_rate(self.loc_iter)
            self.loc_iter = self.loc_iter + 1

        #self.render_image(cam_ids, 0)
        self.save_checkpoint()  
        self.tet32.surface_from_sdf(self.sdf.detach().cpu().numpy().reshape(-1), "Exp/bmvs_man/final_tri.ply", self.dataset.scale_mats_np[0][:3, 3][None], self.dataset.scale_mats_np[0][0, 0])
           
        #self.tet32.clipped_cvt(self.sdf.detach(), self.fine_features.detach(), outside_flag, 
        #                       cam_ids, self.learning_rate_cvt, "Exp/bmvs_man/clipped_CVT.ply")

    @torch.no_grad()
    def render_image(self, cam_ids, img_idx = 0, iter_step = 0, resolution_level = 1):
        ## Generate rays
        data = self.dataset.gen_rays_RGB_at(img_idx)  
        rays_o, rays_d, true_rgb, mask = data[:, :, :3], data[:, :, 3: 6], data[:, :, 6: 9], data[:, :, 9: 10]

        rays_o = rays_o.reshape(-1, 3)
        rays_d = rays_d.reshape(-1, 3)
        true_rgb = true_rgb.reshape(-1, 3)
        mask = mask.reshape(-1, 1)

        rays_o = rays_o.contiguous()
        rays_d = rays_d.contiguous()
        true_rgb = true_rgb.contiguous()
        mask = mask.contiguous()
        
        img = torch.zeros([3*(self.dataset.H // resolution_level) * (self.dataset.W // resolution_level)], dtype = torch.float32).cuda()
        img = img.contiguous()
        img[:] = 0
        
        img_coarse = torch.zeros([3*(self.dataset.H // resolution_level) * (self.dataset.W // resolution_level)], dtype = torch.float32).cuda()
        img_coarse = img_coarse.contiguous()
        img_coarse[:] = 0
        
        
        img_mask = torch.zeros([(self.dataset.H // resolution_level) * (self.dataset.W // resolution_level)], dtype = torch.float32).cuda()
        img_mask = img_mask.contiguous()
        img_mask[:] = 0
        
        """self.grad_sdf_space[:] = 0.0
        self.grad_feat_space[:] = 0.0
        self.weights_grad_space[:] = 0.0
        self.activated[:] = 1
        cvt_grad_cuda.knn_sdf_space_grad(self.sites.shape[0], self.tet32.KNN, self.tet32.knn_sites, self.sites, self.activated,
                                            self.sdf, self.fine_features, self.grad_sdf_space, self.grad_feat_space, self.weights_grad_space)
        """

        self.grad_sdf_space[:] = 0.0
        self.grad_feat_space[:] = 0.0
        self.grad_mean_curve[:] = 0.0
        self.weights_grad[:] = 0.0
        self.grad_eik[:] = 0.0
        self.grad_norm_smooth[:] = 0.0
        self.eik_loss[:] = 0.0
        self.activated[:] = 1
        cvt_grad_cuda.eikonal_grad(self.tet32.nb_tets, self.sites.shape[0], self.tet32.summits, self.sites, self.activated, self.sdf.detach(), self.sdf_smooth, self.fine_features.detach(), 
                                    self.grad_eik, self.grad_norm_smooth, self.grad_sdf_space, self.grad_feat_space, self.weights_grad, self.eik_loss)
        
        norm_grads = torch.linalg.norm(self.grad_sdf_space, ord=2, axis=-1, keepdims=True).reshape(-1, 1)
        self.grad_sdf_space = self.grad_sdf_space / norm_grads.expand(-1, 3)

        colors_out = torch.zeros([self.batch_size*3]).to(torch.device('cuda')).contiguous()
        colors_out_coarse = torch.zeros([self.batch_size*3]).to(torch.device('cuda')).contiguous() 
        mask_out = torch.zeros([self.batch_size]).to(torch.device('cuda')).contiguous()
        it = 0
        for rays_o_batch, rays_d_batch in zip(rays_o.split(self.batch_size), rays_d.split(self.batch_size)):
            rays_o_batch = rays_o_batch.contiguous()
            rays_d_batch = rays_d_batch.contiguous()

            ## sample points along the rays
            start = timer()
            self.offsets[:] = 0
            nb_samples = self.tet32.sample_rays_cuda(0.01, self.inv_s, self.sigma, img_idx, rays_d_batch, self.sdf, self.fine_features, cam_ids, self.in_weights, self.in_z, self.in_sdf, self.in_feat, self.in_ids, self.offsets, self.activated, self.n_samples)    
                
            start = timer()
            self.samples[:] = 0.0
            tet32_march_cuda.fill_samples(rays_o_batch.shape[0], self.n_samples, rays_o_batch, rays_d_batch, self.tet32.sites, 
                                            self.in_z, self.in_sdf, self.in_feat, self.in_weights, self.grad_sdf_space, self.in_ids, 
                                            self.out_z, self.out_sdf, self.out_feat, self.out_weights, self.out_grads, self.out_ids, 
                                            self.offsets, self.samples, self.samples_loc, self.samples_rays)
            
            
            #samples = (self.samples[:nb_samples,:] + self.samples_loc[:nb_samples,:])/2.0
            samples = self.samples[:nb_samples,:]
            samples = samples.contiguous()
            #samples = (samples + 1.1)/2.2

            """samp_entry = self.samples_loc[:nb_samples,:] + self.out_z[:nb_samples,0].reshape(-1,1).expand(-1, 3)*self.samples_rays[:nb_samples,:]
            samp_exit = self.samples_loc[:nb_samples,:] + self.out_z[:nb_samples,1].reshape(-1,1).expand(-1, 3)*self.samples_rays[:nb_samples,:]
            self.out_sdf[:nb_samples,0] = self.sdf_network.sdf(torch.cat([samp_entry, self.out_sdf[:nb_samples,2:5], self.out_weights[:nb_samples,:3]], -1))[:,0]
            self.out_sdf[:nb_samples,1] = self.sdf_network.sdf(torch.cat([samp_exit, self.out_sdf[:nb_samples,5:8], self.out_weights[:nb_samples,3:6]], -1))[:,0]"""

            #fine_features = (self.out_feat[:nb_samples,:6] + self.out_feat[:nb_samples,:-6])/2.0
            #fine_features = self.out_feat[:nb_samples,:6]

            #self.colors = self.out_feat[:nb_samples,36:39] 
            self.colors = self.out_feat[:nb_samples,:3]
            self.colors = self.colors.contiguous()
            
            ##### ##### ##### ##### ##### ##### 
            xyz_emb = (samples.unsqueeze(-1) * self.posfreq).flatten(-2)
            xyz_emb = torch.cat([samples, xyz_emb.sin(), xyz_emb.cos()], -1)

            viewdirs_emb = (self.samples_rays[:nb_samples,:].unsqueeze(-1) * self.viewfreq).flatten(-2)
            viewdirs_emb = torch.cat([self.samples_rays[:nb_samples,:], viewdirs_emb.sin(), viewdirs_emb.cos()], -1)
           
            
            if self.double_net:
                self.geo_features[:] = 0.0
                backprop_cuda.geo_feat(nb_samples, 96, self.sigma, samples, self.sites, self.grad_sdf_space, self.sdf.detach(), 
                                self.geo_features, self.tet32.summits, self.tet32.knn_sites, self.out_ids)          
                coarse_feat = torch.cat([xyz_emb, viewdirs_emb, self.out_sdf[:nb_samples,:], self.out_feat[:nb_samples,:]], -1)      
                #coarse_feat = torch.cat([xyz_emb, viewdirs_emb, self.geo_features[:nb_samples,:]], -1)      
                #geo_feat = torch.cat([xyz_emb, viewdirs_emb, self.out_sdf[:nb_samples,:]], -1)
                colors_feat = self.color_coarse.rgb(coarse_feat)   
                self.colors = torch.sigmoid(colors_feat)
                self.colors = self.colors.contiguous()

            if self.double_net:
                if self.position_encoding:
                    rgb_feat = torch.cat([xyz_emb, viewdirs_emb, colors_feat.detach(), self.out_grads[:nb_samples,:], self.out_feat[:nb_samples,:]], -1)
                else:
                    rgb_feat = torch.cat([viewdirs_emb, colors_feat.detach(), self.out_grads[:nb_samples,:], self.out_feat[:nb_samples,:]], -1) #, self.out_weights[:nb_samples,:]
            else:
                if self.position_encoding:
                    rgb_feat = torch.cat([xyz_emb, viewdirs_emb, self.out_grads[:nb_samples,:], self.out_feat[:nb_samples,:]], -1)
                else:
                    rgb_feat = torch.cat([viewdirs_emb, self.out_grads[:nb_samples,:], self.out_feat[:nb_samples,:]], -1) #, self.out_weights[:nb_samples,:]
            
            #rgb_feat = torch.cat([xyz_emb, viewdirs_emb, self.out_sdf[:nb_samples, :], self.out_feat[:nb_samples,:]], -1) #, self.out_weights[:nb_samples,:]
            
            #rgb_feat = torch.cat([xyz_emb, viewdirs_emb, fine_features[:nb_samples]], -1)
            #rgb_feat = torch.cat([xyz_emb, viewdirs_emb, self.out_feat[:nb_samples]], -1)

            #self.colors_fine = torch.sigmoid(self.color_network.rgb(rgb_feat) + colors_feat.detach())
            if self.double_net:
                self.colors_fine = torch.sigmoid(self.color_network.rgb(rgb_feat) + colors_feat.detach()) #+ self.colors
            else:
                self.colors_fine = torch.sigmoid(self.color_network.rgb(rgb_feat)) #+ self.colors
            self.colors_fine = self.colors_fine.contiguous()

            """ TEST 
            norm = matplotlib.colors.Normalize(vmin=-0.3, vmax=0.3 , clip = False)
            sdf_samp = (self.out_sdf[:nb_samples,0] + self.out_sdf[:nb_samples,1]) / 2.0
            self.colors_fine = plt.cm.jet(norm(sdf_samp.cpu())).astype(np.float32)[:,:3]
            self.colors_fine = torch.from_numpy(self.colors_fine).float().cuda()           
            self.colors_fine = self.colors_fine.contiguous()"""

            ########################################
            ####### Render the image ###############
            ########################################
            norm_map = (1.0+(self.out_grads[:nb_samples,:3] + self.out_grads[:nb_samples,3:6])/2.0)/2.0
            renderer_cuda.render_no_grad(rays_o_batch.shape[0], self.inv_s, self.out_sdf, self.colors_fine, self.offsets, colors_out, mask_out)
            renderer_cuda.render_no_grad(rays_o_batch.shape[0], self.inv_s, self.out_sdf, norm_map, self.offsets, colors_out_coarse, mask_out)

            start = 3*it*self.batch_size
            end = min(3*(it+1)*self.batch_size, 3*(self.dataset.H // resolution_level) * (self.dataset.W // resolution_level))
            img[start:end] = colors_out[:(end-start)]
            img_coarse[start:end] = colors_out_coarse[:(end-start)]

            start = it*self.batch_size
            end = min((it+1)*self.batch_size, (self.dataset.H // resolution_level) * (self.dataset.W // resolution_level))           
            img_mask[start:end] = mask_out[:(end-start)]
            
            it = it + 1

            
        if False: 
            #pts = self.samples[:nb_samples,:].cpu()
            norm = matplotlib.colors.Normalize(vmin=-0.3, vmax=0.3 , clip = False)
            sdf_rgb = plt.cm.jet(norm(self.out_sdf[:,0].cpu())).astype(np.float32)
            print("NB rays == ", rays_o.shape[0])
            print("img_idx == ", img_idx)
            print("nb_samples == ", nb_samples)
            ply.save_ply("Exp/bmvs_man/samples.ply", np.transpose(self.samples[:nb_samples,:].cpu()), col = 255*np.transpose(sdf_rgb))
            
        mask = img_mask.reshape(-1,1)

        img = img.reshape(self.dataset.H // resolution_level, self.dataset.W // resolution_level, 3)
        img = img.cpu().numpy()
        cv2.imwrite('Exp/synt.png', 255*img[:,:])
        cv2.imwrite('Exp/Rendering/img_{}.png'.format(img_idx), 255*img[:,:])
        
        img_coarse = img_coarse.reshape(self.dataset.H // resolution_level, self.dataset.W // resolution_level, 3)
        img_coarse = img_coarse.cpu().numpy()
        cv2.imwrite('Exp/synt_coarse.png', 255*img_coarse[:,:])
        
        GTimg = true_rgb.reshape(self.dataset.H // resolution_level, self.dataset.W // resolution_level, 3).cpu().numpy()
        cv2.imwrite('Exp/GT.png', 255*GTimg[:,:])

        mask = mask.reshape(self.dataset.H // resolution_level, self.dataset.W // resolution_level).cpu().numpy()
        cv2.imwrite('Exp/Mask.png', 255*mask[:])
        print("rendering done")
        
    def Allocate_data(self, K_NN = 24):        
        self.activated_buff = torch.zeros(self.sites.shape[0], dtype=torch.int32).cuda().contiguous()     
        self.activated = torch.zeros(self.sites.shape[0], dtype=torch.int32).cuda().contiguous()      

        self.sdf_smooth = torch.zeros([self.sdf.shape[0]]).float().cuda().contiguous()  
        self.counter = torch.zeros([self.sdf.shape[0]]).float().cuda().contiguous()   

        self.weight_sdf_smooth = torch.zeros([self.sdf.shape[0]]).float().cuda().contiguous()  
        
        self.grad_features = torch.zeros([self.sdf.shape[0], self.dim_feats]).float().cuda().contiguous()  
        self.grad_norm_feat = torch.zeros([self.sdf.shape[0], 12]).float().cuda().contiguous()  
        self.grad_norm = torch.zeros([self.sites.shape[0], 3]).float().cuda().contiguous()
        self.grad_feat_smooth = torch.zeros([self.sdf.shape[0], self.dim_feats]).float().cuda().contiguous()

        self.grad_sdf_smooth = torch.zeros([self.sdf.shape[0]]).float().cuda().contiguous()  

        self.grad_sdf = torch.zeros([self.sdf.shape[0]]).float().cuda().contiguous()  
        self.grad_sdf_net = torch.zeros([self.sdf.shape[0]]).float().cuda().contiguous()          
        self.grad_sdf_norm = torch.zeros([self.sdf.shape[0]]).float().cuda().contiguous()                

        self.grad_sdf_feat = torch.zeros([self.sites.shape[0], 4*3]).float().cuda().contiguous()
        self.grad_sdf_space = torch.zeros([self.sites.shape[0], 3]).float().cuda().contiguous()
        self.unormed_grad = torch.zeros([self.sites.shape[0], 3]).float().cuda().contiguous()
        self.grad_feat_space = torch.zeros([self.sites.shape[0], 3, self.dim_feats]).float().cuda().contiguous()
        self.grad_mean_curve = torch.zeros([self.sites.shape[0]]).float().cuda().contiguous()
        self.weights_grad_space = torch.zeros([3*self.tet32.KNN*self.sites.shape[0]]).float().cuda().contiguous()
        self.weights_grad = torch.zeros([self.sites.shape[0], 1]).float().cuda().contiguous()
        self.eik_loss = torch.zeros([self.sites.shape[0], 1]).float().cuda().contiguous()

        self.grad_eik = torch.zeros([self.sites.shape[0]]).float().cuda().contiguous() 
        self.grad_norm_smooth = torch.zeros([self.sites.shape[0]]).float().cuda().contiguous() 

        self.vol_tet32 = torch.zeros([self.tet32.nb_tets]).float().cuda().contiguous() 
        self.weights_diff = torch.zeros([12*self.tet32.nb_tets]).float().cuda().contiguous() 
        self.weights_tot_diff = torch.zeros([self.sites.shape[0]]).float().cuda().contiguous() 
        
    def Allocate_batch_data(self, K_NN = 24):
        self.samples = torch.zeros([self.n_samples * self.batch_size, 3], dtype=torch.float32).cuda()
        self.samples = self.samples.contiguous()
        
        self.colors = torch.zeros([self.n_samples * self.batch_size, 3], dtype=torch.float32).cuda()
        self.colors = self.colors.contiguous()
        
        self.colors_fine = torch.zeros([self.n_samples * self.batch_size, 3], dtype=torch.float32).cuda()
        self.colors_fine = self.colors_fine.contiguous()
        
        self.samples_rays = torch.zeros([self.n_samples * self.batch_size, 3], dtype=torch.float32).cuda()
        self.samples_rays = self.samples_rays.contiguous()
        
        self.in_weights = torch.zeros([6*self.n_samples * self.batch_size], dtype=torch.float32).cuda()
        self.in_weights = self.in_weights.contiguous()

        self.in_ids = -torch.ones([6*self.n_samples* self.batch_size], dtype=torch.int32).cuda()
        self.in_ids = self.in_ids.contiguous()
        
        self.in_z = torch.zeros([2*self.n_samples * self.batch_size], dtype=torch.float32).cuda()
        self.in_z = self.in_z.contiguous()
        
        self.in_sdf = torch.zeros([2*self.n_samples * self.batch_size], dtype=torch.float32).cuda()
        self.in_sdf = self.in_sdf.contiguous()        
        
        self.out_ids = -torch.ones([self.n_samples* self.batch_size, 6], dtype=torch.int32).cuda()
        self.out_ids = self.out_ids.contiguous()
        
        self.out_z = torch.zeros([self.n_samples * self.batch_size,2], dtype=torch.float32).cuda()
        self.out_z = self.out_z.contiguous()
        
        self.out_sdf = torch.zeros([self.n_samples * self.batch_size, 2], dtype=torch.float32).cuda()
        self.out_sdf = self.out_sdf.contiguous()
        
        self.out_feat = torch.zeros([self.n_samples * self.batch_size, self.dim_feats], dtype=torch.float32).cuda()
        self.out_feat = self.out_feat.contiguous()
        
        self.out_weights = torch.zeros([self.n_samples * self.batch_size, 7], dtype=torch.float32).cuda()
        self.out_weights = self.out_weights.contiguous()
        
        self.out_grads = torch.zeros([self.n_samples * self.batch_size, 12], dtype=torch.float32).cuda().contiguous()

        self.offsets = torch.zeros([self.batch_size, 2], dtype=torch.int32).cuda()
        self.offsets = self.offsets.contiguous()

        """self.xyz_emb = torch.zeros([self.n_samples * self.batch_size, 48]).float().cuda().contiguous() 
        self.viewdirs_emb = torch.zeros([self.n_samples * self.batch_size, 12]).float().cuda().contiguous()
        self.coarse_feat = torch.zeros([self.n_samples * self.batch_size, 88]).float().cuda().contiguous()       
        self.coarse_feat.requires_grad_(True)

        self.rgb_feat = torch.zeros([self.n_samples * self.batch_size, 91]).float().cuda().contiguous()       
        self.rgb_feat.requires_grad_(True)

        self.colors_feat = torch.zeros([self.n_samples * self.batch_size, 3]).float().cuda().contiguous()       
        self.colors_feat.requires_grad_(True)"""

        self.fine_features_grad = torch.zeros([self.n_samples * self.batch_size, 32]).float().cuda().contiguous()   
        self.norm_features_grad = torch.zeros([self.n_samples * self.batch_size, 12]).float().cuda().contiguous()  
        self.sdf_features_grad = torch.zeros([self.n_samples * self.batch_size, 2]).float().cuda().contiguous()  
       
        

    def get_image_perm(self):
        return torch.randperm(self.dataset.n_images)
    
    def update_learning_rate(self, it = 0):
        alpha = self.learning_rate_alpha
        progress = it / self.end_iter_loc
        learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha

        for g in self.optimizer.param_groups:
            g['lr'] = self.learning_rate * learning_factor

        for g in self.optimizer_feat.param_groups:
            g['lr'] = self.learning_rate_feat * learning_factor
            
        for g in self.optimizer_sdf.param_groups:
            g['lr'] = self.learning_rate_sdf * learning_factor
            
        #for g in self.optimizer_cvt.param_groups:
        #    g['lr'] = self.learning_rate_cvt * learning_factor

    @torch.no_grad()
    def save_checkpoint(self):
        checkpoint = {
            'color_geo_network': self.color_coarse.state_dict(),
            'color_fine_network': self.color_network.state_dict(),
        }

        self.iter_step = 0

        os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)
        torch.save(checkpoint, os.path.join(self.base_exp_dir, 'checkpoints', 'ckpt_{:0>6d}.pth'.format(self.iter_step)))
        np.save(os.path.join(self.base_exp_dir, 'checkpoints', 'sdf_{:0>6d}.npy'.format(self.iter_step)), self.sdf.detach().cpu().numpy())
        np.save(os.path.join(self.base_exp_dir, 'checkpoints', 'features_{:0>6d}.npy'.format(self.iter_step)), self.fine_features.detach().cpu().numpy())
        np.save(os.path.join(self.base_exp_dir, 'checkpoints', 'sites_{:0>6d}.npy'.format(self.iter_step)), self.sites.cpu().numpy())
   
    def load_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name), map_location=self.device)
        self.color_coarse.load_state_dict(checkpoint['color_geo_network'])
        self.color_network.load_state_dict(checkpoint['color_fine_network'])

        print(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name))

        print(os.path.join(self.base_exp_dir, 'checkpoints', 'sites_{:0>6d}.npy'.format(self.iter_curr)))
        self.sites = np.load(os.path.join(self.base_exp_dir, 'checkpoints', 'sites_{:0>6d}.npy'.format(self.iter_step)))
        self.sites = torch.from_numpy(self.sites.astype(np.float32)).cuda()
        self.sdf = np.load(os.path.join(self.base_exp_dir, 'checkpoints', 'sdf_{:0>6d}.npy'.format(self.iter_step)))
        self.sdf = torch.from_numpy(self.sdf.astype(np.float32)).cuda()
        self.sdf.requires_grad_(True)
        
        self.fine_features = np.load(os.path.join(self.base_exp_dir, 'checkpoints', 'features_{:0>6d}.npy'.format(self.iter_step)))
        self.fine_features = torch.from_numpy(self.fine_features.astype(np.float32)).cuda()
        self.fine_features.requires_grad_(True)
                
        self.optimizer_sdf = torch.optim.Adam([self.sdf], lr=self.learning_rate_sdf)        
        self.optimizer_feat= torch.optim.Adam([self.fine_features], lr=self.learning_rate_feat)
        
        self.optimizer_sdf.load_state_dict(checkpoint['optimizer_sdf'])
        self.optimizer_feat.load_state_dict(checkpoint['optimizer_feat'])
     

if __name__=='__main__':
    print("Code by Diego Thomas")

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='src/Confs/test.conf')
    parser.add_argument('--data_name', type=str, default='bmvs_man')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--is_continue', default=False, action="store_true")
    parser.add_argument('--checkpoint', type=str, default='ckpt_000000.pth')
    parser.add_argument('--resolution', type=int, default=16)
    parser.add_argument('--output', type=str, default='')
    parser.add_argument('--nb_images', type=int, default=1)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--position_encoding', default=False, action="store_true")
    parser.add_argument('--double_net', default=False, action="store_true")
    
    args = parser.parse_args()

    ## Initialise CUDA device for torch computations
    torch.cuda.set_device(args.gpu)
    
    runner = Runner(args.conf, args.data_name, args.mode, args.is_continue, args.checkpoint, args.position_encoding, args.double_net)
    
    if args.mode == 'train':
        runner.train(args.data_name, 24, False)
        #for id_im in tqdm(range(args.nb_images)):
        #    runner.render_image(runner.cam_ids, img_idx = id_im)
    """elif args.mode == 'render_images':
        #runner.prep_CVT(args.data_name, [-0.6,-0.6,-0.6,0.6,0.6,0.6])
        #for id_im in tqdm(range(args.nb_images)):
        #    runner.render_image(runner.cam_ids, img_idx = id_im)
        #    #runner.validate_image_hybrid(1000.0, idx=id_im)"""