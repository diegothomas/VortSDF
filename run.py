import os
import cv2
import torch
import argparse
import math
import numpy as np
import tet32 as tet32
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
from glob import glob
import scipy.spatial
import igl
import scipy as sp


from torch.utils.cpp_extension import load
tet32_march_cuda = load('tet32_march_cuda', ['src/Cuda/tet32_march_cuda.cpp', 'src/Cuda/tet32_march_cuda.cu'], verbose=True)

renderer_cuda = load('renderer_cuda', ['src/Models/renderer.cpp', 'src/Models/renderer.cu'], verbose=True)

backprop_cuda = load('backprop_cuda', ['src/Models/backprop.cpp', 'src/Models/backprop.cu'], verbose=True)

cvt_grad_cuda = load('cvt_grad_cuda', ['src/Geometry/CVT_gradients.cpp', 'src/Geometry/CVT_gradients.cu'], verbose=True)

laplacian = load(name='laplacian', sources=['src/Geometry/laplacian.cpp', 'src/Geometry/laplacian.cu'], extra_include_paths=['C:/Users/thomas/Documents/Projects/lib/eigen-3.4.0', 'C:/Users/thomas/Documents/Projects/lib/libigl/include'], verbose=True)

up_iters = [2000, 10000, 20000, 30000, 40000]
#up_iters = [2000, 5000, 10000, 15000, 20000]

class Runner:
    def __init__(self, conf_path, data_name, mode='train', is_continue=False, checkpoint = '', position_encoding = True, double_net = True):
        self.device = torch.device('cuda')

        self.data_name = data_name
        print(self.data_name)

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

        
        self.data_dir = self.conf['dataset'].get_string('data_dir')
        self.data_dir = self.data_dir.replace('DATA_NAME', data_name)
        f = open(self.data_dir+ "/bbox.conf")
        conf_text = f.read()
        f.close()
        self.conf_bbox = ConfigFactory.parse_string(conf_text)
        self.visual_hull = self.conf_bbox.get_list('data_info.visual_hull')
        print(self.visual_hull)
        
        # Training parameters
        self.end_iter = self.conf.get_int('train.end_iter')
        self.n_samples = self.conf.get_int('model.cvt_renderer.n_samples')
        self.batch_size = self.conf.get_int('train.batch_size')
        self.learning_rate_list = self.conf.get_list('train.learning_rate')
        self.learning_rate_sdf_list = self.conf.get_list('train.learning_rate_sdf')
        self.learning_rate_feat_list = self.conf.get_list('train.learning_rate_feat')
        self.learning_rate_alpha_list = self.conf.get_list('train.learning_rate_alpha')
        self.learning_rate_cvt_list = self.conf.get_list('train.learning_rate_cvt')

        
        self.mask_w_list = self.conf.get_list('train.mask_weight')
        self.s_w_list = self.conf.get_list('train.smooth_weight')
        self.e_w_list = self.conf.get_list('train.eik_weight')
        self.tv_w_list= self.conf.get_list('train.tv_weight')
        self.tv_f_list = self.conf.get_list('train.tv_f_weight')

        self.res = self.conf.get_int('train.res')        
        self.dim_feats = self.conf.get_int('train.dim_feats')
        self.knn = self.conf.get_int('train.knn')
        self.hlvl = self.conf.get_int('train.hlvl')

        self.learning_rate = self.learning_rate_list[0]
        self.learning_rate_sdf = self.learning_rate_sdf_list[0]
        self.learning_rate_feat = self.learning_rate_feat_list[0]
        self.learning_rate_alpha = self.learning_rate_alpha_list[0]
        self.learning_rate_cvt =  self.learning_rate_cvt_list[0]

        self.iter_step = 0
        self.end_iter_loc = 2000
        self.s_w = self.s_w_list[0]
        self.e_w =  self.e_w_list[0]
        self.tv_w = self.tv_w_list[0]
        self.tv_f = self.tv_f_list[0]
        self.f_w = 0.0#1.0e0

        self.report_freq = self.conf.get_int('train.report_freq')
        self.val_freq = self.conf.get_int('train.val_freq')
        self.CVT_freq = self.conf.get_int('train.CVT_freq')

        self.position_encoding = position_encoding
        
        self.double_net = double_net

        if self.double_net:
            self.vortSDF_renderer_coarse_net = VortSDFRenderer(**self.conf['model.cvt_renderer'])
            self.vortSDF_renderer_coarse_net.mask_reg = self.mask_w_list[0]
        else:
            self.vortSDF_renderer_coarse = VortSDFDirectRenderer(**self.conf['model.cvt_renderer'])

        self.vortSDF_renderer_fine = VortSDFRenderer(**self.conf['model.cvt_renderer'])
        self.vortSDF_renderer_fine.mask_reg = self.mask_w_list[0]
        
        if self.double_net:
            self.color_coarse = ColorNetwork(**self.conf['model.color_geo_network']).to(self.device)

        self.color_network = ColorNetwork(**self.conf['model.color_network']).to(self.device)
        
        params_to_train = []
        params_to_train += list(self.color_network.parameters())
        if self.double_net:
            params_to_train += list(self.color_coarse.parameters())
        self.optimizer = torch.optim.Adam(params_to_train, lr=self.learning_rate)
        
        posbase_pe = 5
        viewbase_pe= 1
        self.posfreq = torch.FloatTensor([(2**i) for i in range(posbase_pe)]).cuda()
        self.viewfreq = torch.FloatTensor([(2**i) for i in range(viewbase_pe)]).cuda()

        k_posbase_pe=5
        k_viewbase_pe= 1
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
                print(model_list)
            else:
                latest_model_name = checkpoint
            print(latest_model_name)

        if latest_model_name is not None:
            self.load_checkpoint(latest_model_name)
            start = timer()

    def prep_CVT(self):
        print("Preparing CVT model")
        sites = self.tet32.sites
        #self.tet32 = tet32.Tet32(self.tet32.sites, id = 0)
        self.tet32.run(2.0*self.sigma_start) 
        self.tet32.mask_background = abs(self.sdf) > 8.0*self.sigma_start
        self.tet32.mask_background = self.tet32.mask_background.int().cuda()
        self.tet32.load_cuda(2.0*self.sigma_start)

        #reorganize sdf
        KDtree = scipy.spatial.KDTree(self.tet32.vertices)
        
        prev_SDF = np.copy(self.sdf.detach().cpu())
        prev_Feat = np.copy(self.fine_features.detach().cpu())
        new_SDF = np.copy(self.sdf.detach().cpu())
        new_Feat = np.copy(self.fine_features.detach().cpu())
        
        _, idx = KDtree.query(sites, k=1)
        new_SDF[idx[:]] = prev_SDF[:] 
        new_Feat[idx[:]] = prev_Feat[:] 
        new_SDF = torch.from_numpy(new_SDF).float().cuda()
        new_Feat = torch.from_numpy(new_Feat).float().cuda()
        with torch.no_grad():
            self.sdf[:] = new_SDF[:]
            self.fine_features[:] = new_Feat[:]

        sites = np.asarray(self.tet32.vertices)  

        self.cam_sites = np.stack([self.dataset.pose_all[id, :3,3].cpu().numpy() for id in range(self.dataset.n_images)])
            
        self.cam_ids = np.stack([np.where((sites == self.cam_sites[i,:]).all(axis = 1))[0] for i in range(self.cam_sites.shape[0])]).reshape(-1)
        self.tet32.make_adjacencies(self.cam_ids)
        self.tet32.make_multilvl_knn()

        self.cam_ids = torch.from_numpy(self.cam_ids).int().cuda()
        #self.tet32.sites = torch.from_numpy(self.tet32.sites.astype(np.float32)).cuda()
        #self.tet32.sites = self.tet32.sites.contiguous()
        
        print("start data allocation")
        self.Allocate_data()
            
        #input()
        print("start batch allocation")
        self.Allocate_batch_data()

        print("end allocation")
        #input()

        cvt_grad_cuda.diff_tensor(self.tet32.nb_tets, self.tet32.summits, self.tet32.sites, self.vol_tet32, self.weights_diff, self.weights_tot_diff)
        
        self.tet32.surface_from_sdf(self.sdf.detach().cpu().numpy().reshape(-1), "Exp/{}/meshes/test_tri_raw.ply".format(self.data_name), self.dataset.scale_mats_np[0][:3, 3][None], self.dataset.scale_mats_np[0][:3, :3]) 
             

    def train(self, data_name, K_NN = 24, verbose = True, is_continue = False):
        ##### 2. Load initial sites
        res = self.res
        if not hasattr(self, 'tet32'):
            ##### 2. Load initial sites
            #visual_hull = [-1.1, -1.1, -1.1, 1.1, 1.1, 1.1] #-> man
            #visual_hull = [-0.8, -1.0, -0.8, 0.7, 1.0, 0.8] #-> sculpture
            #visual_hull = [-1.0,-1.2,-1.0,0.9,0.4,1.0] # -> dog
            #visual_hull = [-1.2, -1.2, -1.2, 1.2, 1.2, 1.2] #-> stone
            #visual_hull = [-1.2, -1.5, -1.5, 1.2, 1.3, 1.2] #-> durian
            #self.visual_hull = [-1.0, -1.0, -1.0, 1.2, 1.0, 0.6] #-> bear
            #visual_hull = [-1.2, -1.1, -1.2, 1.1, 1.1, 1.1] #-> clock
            
            import src.Geometry.sampling as sampler
            sites = sampler.sample_Bbox(self.visual_hull[0:3], self.visual_hull[3:6], res, perturb_f =  (self.visual_hull[3] - self.visual_hull[0])*0.02)
            #ext_sites1 = sampler.exterior_Bbox(visual_hull[0:3], visual_hull[3:6], 32, 10.0)
            #ext_sites2 = sampler.exterior_Bbox(visual_hull[0:3], visual_hull[3:6], 32, 11.0)
            #sites, _ = ply.load_ply("Data/bmvs_man/bmvs_man_colmap_aligned.ply")

            #### Add cameras as sites 
            cam_sites = np.stack([self.dataset.pose_all[id, :3,3].cpu().numpy() for id in range(self.dataset.n_images)])
            sites = np.concatenate((sites, cam_sites))
            #sites = np.concatenate((ext_sites1, ext_sites2, sites, cam_sites))

            self.tet32 = tet32.Tet32(sites, 0, self.knn, self.hlvl)
            #self.tet32.start() 
            #print("parallel process started")
            #self.tet32.join()
            self.tet32.run(0.3) 
            self.tet32.mask_background = torch.zeros(sites.shape[0]).int().cuda().contiguous()  
            self.tet32.load_cuda()
            #self.tet32.save("Exp/bmvs_man/test.ply")    
            
            sites = np.asarray(self.tet32.vertices)  
            cam_ids = np.stack([np.where((sites == cam_sites[i,:]).all(axis = 1))[0] for i in range(cam_sites.shape[0])]).reshape(-1)
            self.tet32.make_adjacencies(cam_ids)

            cam_ids = torch.from_numpy(cam_ids).int().cuda()
            
            outside_flag = np.zeros(sites.shape[0], np.int32)
            outside_flag[sites[:,0] < self.visual_hull[0] + (self.visual_hull[3]-self.visual_hull[0])/(res)] = 1
            outside_flag[sites[:,1] < self.visual_hull[1] + (self.visual_hull[4]-self.visual_hull[1])/(res)] = 1
            outside_flag[sites[:,2] < self.visual_hull[2] + (self.visual_hull[5]-self.visual_hull[2])/(res)] = 1
            outside_flag[sites[:,0] > self.visual_hull[3] - (self.visual_hull[3]-self.visual_hull[0])/(res)] = 1
            outside_flag[sites[:,1] > self.visual_hull[4] - (self.visual_hull[4]-self.visual_hull[1])/(res)] = 1
            outside_flag[sites[:,2] > self.visual_hull[5] - (self.visual_hull[5]-self.visual_hull[2])/(res)] = 1
        else:
            cam_sites = self.cam_sites

        """if not is_continue:
            self.tet32.sites = torch.from_numpy(sites.astype(np.float32)).cuda()
            self.tet32.sites = self.tet32.sites.contiguous()"""
        
        print(self.tet32.sites.shape)

        ##### 2. Initialize SDF field    
        if not hasattr(self, 'sdf'):
            centers = torch.zeros(self.tet32.sites.shape).float().cuda()
            centers[:,0] = (self.visual_hull[3]+self.visual_hull[0])/2.0
            centers[:,1] = (self.visual_hull[4]+self.visual_hull[1])/2.0
            centers[:,2] = (self.visual_hull[5]+self.visual_hull[2])/2.0
            """with torch.no_grad():
                norm_sites = torch.linalg.norm(self.tet32.sites - centers, ord=2, axis=-1, keepdims=True)
            self.sdf = norm_sites[:,0] - 0.2
            self.sdf = self.sdf.contiguous()
            self.sdf.requires_grad_(True) """
            self.sdf = 1000.0*torch.ones(self.tet32.sites.shape[0]).float().cuda()     
            self.sdf = self.sdf.contiguous()
            self.sdf.requires_grad_(True)  

        ##### 2. Initialize feature field    
        if not hasattr(self, 'fine_features'):
            self.fine_features = 0.5*torch.ones([self.sdf.shape[0], self.dim_feats]).cuda()       
            self.fine_features = self.fine_features.contiguous()
            self.fine_features.requires_grad_(True)

        ############# CVT optimization #############################
        ############# CVT optimization #############################
        ############# CVT optimization #############################
        if not is_continue:
            self.tet32.CVT(outside_flag, cam_ids.long(), self.sdf.detach(), self.fine_features.detach(), lr = self.learning_rate_cvt)

            self.tet32.run(0.3)
            self.tet32.load_cuda()
            #self.tet32.save("Exp/bmvs_man/test_CVT.ply")  
            sites = np.asarray(self.tet32.vertices)  
            cam_ids = np.stack([np.where((sites == cam_sites[i,:]).all(axis = 1))[0] for i in range(cam_sites.shape[0])]).reshape(-1)
            self.tet32.make_adjacencies(cam_ids)

            self.tet32.make_multilvl_knn()

            cam_ids = torch.from_numpy(cam_ids).int().cuda()
            self.cam_ids = cam_ids
            
            #self.tet32.sites = torch.from_numpy(sites.astype(np.float32)).cuda()
            #self.tet32.sites = self.tet32.sites.contiguous()

            with torch.no_grad():
                norm_sites = torch.linalg.norm(self.tet32.sites.float() - centers, ord=2, axis=-1, keepdims=True)
                self.sdf[:] = norm_sites[:,0] - 0.2
                self.tet32.sdf_init = self.sdf.detach().clone()

            #self.tet32.move_sites(outside_flag, cam_ids, self.sdf.detach(), self.fine_features.detach())

            #self.tet32.clipped_cvt(self.sdf.detach(), self.fine_features.detach(), outside_flag, 
            #                       cam_ids, self.learning_rate_cvt, "Exp/bmvs_man/clipped_CVT.ply")
            #input()  

            self.Allocate_data()
            
            self.Allocate_batch_data()
                
        else:
            cam_ids = self.cam_ids
        
        
        #self.tet32.save_multi_lvl("Exp/{}/multi_lvl".format(self.data_name))    
        

        sites = self.tet32.sites.cpu().numpy()
        outside_flag = np.zeros(self.tet32.sites.shape[0], np.int32)
        outside_flag[sites[:,0] < self.visual_hull[0] + (self.visual_hull[3]-self.visual_hull[0])/(res)] = 1
        outside_flag[sites[:,1] < self.visual_hull[1] + (self.visual_hull[4]-self.visual_hull[1])/(res)] = 1
        outside_flag[sites[:,2] < self.visual_hull[2] + (self.visual_hull[5]-self.visual_hull[2])/(res)] = 1
        outside_flag[sites[:,0] > self.visual_hull[3] - (self.visual_hull[3]-self.visual_hull[0])/(res)] = 1
        outside_flag[sites[:,1] > self.visual_hull[4] - (self.visual_hull[4]-self.visual_hull[1])/(res)] = 1
        outside_flag[sites[:,2] > self.visual_hull[5] - (self.visual_hull[5]-self.visual_hull[2])/(res)] = 1
        
        cvt_grad_cuda.diff_tensor(self.tet32.nb_tets, self.tet32.summits, self.tet32.sites, self.vol_tet32, self.weights_diff, self.weights_tot_diff)

        M_vals, L_vals, L_nonZeros, L_outer, L_sizes = laplacian.MakeLaplacian(self.tet32.sites.shape[0], self.tet32.nb_tets, self.tet32.sites.cpu(), self.tet32.summits.cpu(), self.tet32.valid_tets.cpu())
            

        L_nnZ = L_sizes[0]
        L_outerSize = L_sizes[1]
        L_cols = L_sizes[2]
        
        M_vals = M_vals.float().cuda()
        L_vals = L_vals.float().cuda()
        L_nonZeros = L_nonZeros.cuda()
        L_outer = L_outer.cuda()        

        #l = igl.cotmatrix(self.tet32.sites.cpu().numpy(), self.tet32.tetras)
        m = igl.massmatrix(self.tet32.sites.cpu().numpy(), self.tet32.tetras, igl.MASSMATRIX_TYPE_BARYCENTRIC)
        m_d = m.diagonal()
        print(m_d[m_d > 0.0].min())
        #for i in range(self.tet32.sites.shape[0]):
        #    print(m_d[i])
        
        M_vals = torch.from_numpy(np.asarray(m_d)).float().cuda().reshape(-1, 1)
        #M_vals = torch.ones(self.tet32.sites.shape[0]).float().cuda()

        """
        minv = sp.sparse.diags(1 / m.diagonal())
        print(l.shape)
        print(minv.shape)
        with torch.no_grad():
            hn = -minv.dot(l.dot(self.sdf.detach().cpu().numpy()))
        print(hn.shape)
        exit()"""

                        
        if not hasattr(self, 'optimizer_sdf'):
            self.optimizer_sdf = torch.optim.Adam([self.sdf], lr=self.learning_rate_sdf) #, betas=(0.9, 0.98))     # Beta ??   0.98, 0.995 => 0.9
            self.optimizer_feat = torch.optim.Adam([self.fine_features], lr=self.learning_rate_feat) #, betas=(0.9, 0.98)) 

        if self.double_net:
            self.vortSDF_renderer_coarse_net.prepare_buffs(self.batch_size, self.n_samples, self.tet32.sites.shape[0])
        else:
            self.vortSDF_renderer_coarse.prepare_buffs(self.batch_size, self.n_samples, self.tet32.sites.shape[0])
        
        self.vortSDF_renderer_fine.prepare_buffs(self.batch_size, self.n_samples, self.tet32.sites.shape[0])

        grad_sdf = torch.zeros(self.sdf.shape).float().cuda()

        if not is_continue:
            self.mask_background = torch.zeros(self.sdf.shape).float().cuda()
        
        acc_it = 1
        step_size = 0.01
        if not is_continue:
            self.s_max = 50
            self.R = 40
            self.s_start = 10.0
            self.inv_s = 0.1
            self.sigma_start = min((self.visual_hull[3]-self.visual_hull[0]), (self.visual_hull[4]-self.visual_hull[1]), (self.visual_hull[5]-self.visual_hull[2]))/res #0.1 #
            self.sigma_max = min((self.visual_hull[3]-self.visual_hull[0]), (self.visual_hull[4]-self.visual_hull[1]), (self.visual_hull[5]-self.visual_hull[2]))/res
            self.w_g = 1.0

        lamda_c = 1.0          #####################################
        #weight__fine = 1.0
            
        self.sigma_feat = 0.06
        self.loc_iter = 0
        image_perm = self.get_image_perm()
        num_rays = self.batch_size

        full_reg = 3 #self.end_iter #3 #self.end_iter       #####################################
        self.activated[:] = 1
        with torch.no_grad():
            self.sdf_smooth[:] = self.sdf[:]

        if not is_continue:
            self.dataset.gen_all_rays(3)
            warm_up = 0
        else:
            warm_up = 500
        
        print("Start optimization")
        #input()

        for iter_step in tqdm(range(self.end_iter)):
            if iter_step <= self.iter_step:
                continue
            self.iter_step = iter_step

            img_idx = image_perm[iter_step % len(image_perm)].item() 

            #self.inv_s = min(self.s_max, self.loc_iter/self.R + self.s_start)
            #self.inv_s = self.s_start + (self.s_max-self.s_start)*(1.0 - math.cos(0.5*math.pi*self.loc_iter/self.end_iter_loc))
            self.inv_s = self.s_start + (self.s_max-self.s_start)*(self.loc_iter/self.end_iter_loc)
            self.sigma = self.sigma_start #1.0e-5 + self.sigma_start*math.sin(((self.loc_iter)/(self.end_iter_loc))*0.5*math.pi) #+ (self.sigma_max-self.sigma_start)*(self.loc_iter/self.end_iter_loc)
            fact_w = 1.0 - (0.8)*(self.loc_iter/self.end_iter_loc)

            ## Generate rays
            lvl = 1
            """if iter_step +1 < up_iters[0]:
                lvl = 5
                num_rays = 512
            elif iter_step+1 < up_iters[1]:
                lvl = 3
                num_rays = 2048
            elif iter_step+1 < up_iters[2]:
                lvl = 2
                num_rays = 4096
            elif iter_step+1 < up_iters[4]:
                num_rays = 4096
            else:
                num_rays = self.batch_size"""
            num_rays = self.batch_size

            #data = self.dataset.gen_random_rays_zbuff_at(img_idx, num_rays, 0) 
            #data = self.dataset.gen_random_rays_smooth_at(img_idx, num_rays, 1) 
            if True: #iter_step % 2 == 0 or iter_step+1 <= up_iters[4]:
                data = self.dataset.get_random_rays(num_rays)
            else:
                data = self.dataset.get_random_rays_masked(num_rays)

            rays_o, rays_d, true_rgb, mask, img_ids = data[:, :3], data[:, 3: 6], data[:, 6: 9], data[:, 9: 10], data[:, 10: 11]

            rays_o = rays_o.reshape(-1, 3)
            rays_d = rays_d.reshape(-1, 3)
            true_rgb = true_rgb.reshape(-1, 3)
            mask = mask.reshape(-1, 1)
            img_ids = img_ids.reshape(-1, 1)
            
            """if iter_step > 30000:
                rays_o = rays_o[mask[:,0] > 0.5]
                rays_d = rays_d[mask[:,0]  > 0.5]
                true_rgb = true_rgb[mask[:,0] > 0.5]
                mask = mask[mask[:,0] > 0.5]"""

            rays_o = rays_o.contiguous()
            rays_d = rays_d.contiguous()
            true_rgb = true_rgb.contiguous()
            mask = mask.contiguous()        
            img_ids = img_ids.contiguous().int()    

            if True: #not (iter_step > 5000 and self.loc_iter > 0 and self.loc_iter < warm_up): 
                self.activated[:] = 2
                with torch.no_grad():
                    self.sdf_smooth[:] = self.sdf[:]
                backprop_cuda.knn_smooth(self.tet32.sites.shape[0], self.knn, self.hlvl, self.sigma, self.sigma_feat, 1, self.tet32.sites, self.activated,
                                        self.grad_sdf_space, self.sdf.detach(), self.fine_features.detach(), self.tet32.knn_sites, self.sdf_smooth)
                
                self.sdf_smooth_2[:] = self.sdf_smooth[:]
                backprop_cuda.knn_smooth(self.tet32.sites.shape[0], self.knn, self.hlvl, self.sigma, self.sigma_feat, 1, self.tet32.sites, self.activated,
                                        self.grad_sdf_space, self.sdf_smooth, self.fine_features.detach(), self.tet32.knn_sites, self.sdf_smooth_2)
                
                """with torch.no_grad():
                    self.sdf_smooth[:] = self.sdf[:]
                laplacian.MeanCurve(self.sdf_smooth, self.sdf, 1, self.activated, M_vals, L_vals, L_outer, L_nonZeros, L_nnZ, L_outerSize, L_cols)
                self.sdf_smooth_2[:] = self.sdf_smooth[:]
                laplacian.MeanCurve(self.sdf_smooth_2, self.sdf_smooth, 1, self.activated, M_vals, L_vals, L_outer, L_nonZeros, L_nnZ, L_outerSize, L_cols)"""
                    
                
            ## sample points along the rays
            """start = timer()
            self.offsets[:] = 0
            self.activated[:] = 0
            nb_samples = self.tet32.sample_rays_cuda(self.inv_s, img_ids, rays_d, self.sdf_smooth, cam_ids, self.in_weights, self.in_z, self.in_sdf, self.in_ids, self.offsets, self.activated, self.n_samples)    
            if verbose:
                print('CVT_Sample time:', timer() - start)   """
            
            start = timer()
            self.offsets[:] = 0
            self.activated[:] = 0
            if False: #iter_step <= up_iters[4]:
                nb_samples = tet32_march_cuda.tet32_march_count(self.inv_s, rays_o.shape[0], rays_d, self.tet32.sites, self.sdf, self.tet32.summits, self.tet32.neighbors, img_ids, 
                                               cam_ids, self.tet32.offsets_cam, self.tet32.cam_tets, self.activated, self.offsets)                
            else:
                nb_samples = tet32_march_cuda.tet32_march_count(self.inv_s, rays_o.shape[0], rays_d, self.tet32.sites.float(), self.sdf_smooth, self.tet32.summits, self.tet32.neighbors, self.tet32.valid_tets, img_ids, 
                                               cam_ids, self.tet32.offsets_cam, self.tet32.cam_tets, self.activated, self.offsets)
            
            if verbose:
                print('tet32_march_count time:', timer() - start) 

            if nb_samples == 0:
                continue

            if nb_samples >= num_rays*self.n_samples:
                print("Too many samples!!!")
                input()
                 

            start = timer()
            self.activated_buff[:] = self.activated[:]
            backprop_cuda.activate_sites(rays_o.shape[0], self.tet32.sites.shape[0], self.knn*self.hlvl, self.out_ids, self.offsets, self.tet32.knn_sites, self.activated_buff, self.activated)
            self.activated = self.activated + self.activated_buff
            if verbose:
                print('activate time:', timer() - start)   
                
            if iter_step % full_reg == 0:
                self.activated[:] = 1

            ############ Compute spatial SDF gradients
            start = timer()   
            self.grad_sdf_space[:] = 0.0
            self.weights_grad[:] = 0.0
            self.grad_eik[:] = 0.0
            self.grad_norm_smooth[:] = 0.0
            self.eik_loss[:] = 0.0
            #self.activated[:] = 1
            #self.activated[self.cam_ids] = -1

            cvt_grad_cuda.eikonal_grad(self.tet32.nb_tets, self.tet32.sites.shape[0], self.tet32.summits, self.tet32.valid_tets, self.tet32.sites.float(), self.activated, self.sdf_smooth, self.sdf_smooth_2, self.fine_features.detach(), 
                                        self.grad_eik, self.grad_norm_smooth, self.grad_sdf_space, self.vol_tet32, self.weights_diff, self.weights_tot_diff, self.eik_loss)
            
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
            
            self.grad_norm_smooth[outside_flag[:] == 1] = 0.0
            eik_loss = self.eik_loss.mean()

            if verbose:
                print('eikonal_grad time:', timer() - start)
            
            start = timer()
            self.samples[:] = 0.0
            self.out_grads[:] = 0.0
            self.out_feat[:] = 0.0
            self.out_sdf[:] = 0.0
            self.out_weights[:] = 0.0
            self.out_ids[:] = 0
            if False: #iter_step <= up_iters[4]:
                tet32_march_cuda.tet32_march_offset(self.inv_s, rays_o.shape[0], rays_d, self.tet32.sites, self.sdf, self.tet32.summits, self.tet32.neighbors, img_ids, 
                                                    cam_ids, self.tet32.offsets_cam, self.tet32.cam_tets, self.grad_sdf_space, self.fine_features.detach(),  
                                                    self.out_weights, self.out_z, self.out_sdf, self.out_ids, self.out_grads, self.out_feat, self.samples_rays, self.samples, 
                                                    self.offsets)
            else:
                tet32_march_cuda.tet32_march_offset(self.inv_s, rays_o.shape[0], rays_d, self.tet32.sites.float(), self.sdf_smooth, self.tet32.summits, self.tet32.neighbors, self.tet32.valid_tets, img_ids, 
                                                    cam_ids, self.tet32.offsets_cam, self.tet32.cam_tets, self.grad_sdf_space, self.fine_features.detach(),  
                                                    self.out_weights, self.out_z, self.out_sdf, self.out_ids, self.out_grads, self.out_feat, self.samples_rays, self.samples_reff, self.samples, 
                                                    self.offsets)
                      
            if verbose:
                print('tet32_march_offset time:', timer() - start) 

            """self.offsets[self.offsets[:,1] == -1] = 0                         
            start = timer()
            self.samples[:] = 0.0
            self.out_grads[:] = 0.0
            tet32_march_cuda.fill_samples(rays_o.shape[0], self.n_samples, rays_o, rays_d, self.tet32.sites, 
                                        self.in_z, self.in_sdf, self.fine_features.detach(), self.in_weights, self.grad_sdf_space, self.in_ids, 
                                        self.out_z, self.out_sdf, self.out_feat, self.out_weights, self.out_grads, self.out_ids, 
                                        self.offsets, self.samples, self.samples_rays)
            if verbose:
                print('fill_samples time:', timer() - start)  """

            """print("self.dim_feats", self.dim_feats)
            print("nb_samples", nb_samples)
            print("self.fine_features unit", self.fine_features.mean())
            print("self.out_feat unit", self.out_feat.mean())
            print("self.out_sdf unit", self.in_sdf.mean())
            print("self.out_sdf unit", self.out_sdf.mean())
            input()"""
            #print(torch.linalg.norm(self.out_grads[:nb_samples,:3], ord=2, axis=-1, keepdims=True).min())
           
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
                #coarse_feat = torch.cat([xyz_emb, viewdirs_emb, self.out_sdf[:nb_samples,:2], self.out_grads[:nb_samples,:3]], -1)  
                #coarse_feat = torch.cat([xyz_emb, viewdirs_emb, self.out_sdf[:nb_samples,:2], self.out_grads[:nb_samples,:3], self.out_feat[:nb_samples,:8]], -1) 
                coarse_feat = torch.cat([xyz_emb, viewdirs_emb, self.out_sdf[:nb_samples,:2], self.out_grads[:nb_samples,:3], self.out_feat[:nb_samples,:8]], -1) 
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

            norm_ref = torch.linalg.norm(self.samples_reff, ord=2, axis=-1, keepdims=True).reshape(-1, 1)
            norm_ref[norm_ref == 0.0] = 1.0
            self.samples_reff = self.samples_reff / norm_ref.expand(-1, 3)

            xyz_emb_fine = (self.samples[:nb_samples,:].unsqueeze(-1) * self.k_posfreq).flatten(-2)
            xyz_emb_fine = torch.cat([self.samples[:nb_samples,:], xyz_emb_fine.sin(), xyz_emb_fine.cos()], -1)

            viewdirs_emb_fine = (self.samples_reff[:nb_samples,:].unsqueeze(-1) * self.k_viewfreq).flatten(-2)
            viewdirs_emb_fine = torch.cat([self.samples_reff[:nb_samples,:], viewdirs_emb_fine.sin(), viewdirs_emb_fine.cos()], -1)  

            #with torch.no_grad():
            if self.double_net:
                if self.position_encoding:
                    #rgb_feat = torch.cat([xyz_emb, viewdirs_emb_fine, colors_feat.detach(), self.out_sdf[:nb_samples,:2], self.out_grads[:nb_samples,:3], self.out_feat[:nb_samples,:self.dim_feats]], -1)
                    #rgb_feat = torch.cat([xyz_emb, viewdirs_emb_fine, colors_feat.detach(), self.out_sdf[:nb_samples,:2], self.out_grads[:nb_samples,:3], self.out_feat[:nb_samples,8:self.dim_feats]], -1)
                    rgb_feat = torch.cat([xyz_emb_fine, viewdirs_emb_fine, colors_feat.detach(), self.out_sdf[:nb_samples,:2], self.out_grads[:nb_samples,:3], self.out_feat[:nb_samples,8:self.dim_feats]], -1)
                else:
                    rgb_feat = torch.cat([viewdirs_emb, colors_feat.detach(), self.out_sdf[:nb_samples,:2], self.out_grads[:nb_samples,:3], self.out_feat[:nb_samples,:]], -1) #, self.out_weights[:nb_samples,:]
            else:
                if self.position_encoding:
                    rgb_feat = torch.cat([xyz_emb, viewdirs_emb, self.out_sdf[:nb_samples,:2], self.out_grads[:nb_samples,:3], self.out_feat[:nb_samples,:self.dim_feats]], -1)
                else:
                    rgb_feat = torch.cat([viewdirs_emb, self.out_sdf[:nb_samples,:2], self.out_grads[:nb_samples,:], self.out_feat[:nb_samples,:]], -1) #, self.out_weights[:nb_samples,:]
                
            rgb_feat.requires_grad_(True)
            rgb_feat.retain_grad()

            #self.colors_fine = torch.sigmoid(self.color_network.rgb(rgb_feat)) 
            if self.double_net:
                self.colors_fine = torch.sigmoid(self.color_network.rgb(rgb_feat) + colors_feat.detach()) #+ self.colors
            else:
                self.colors_fine = torch.sigmoid(self.color_network.rgb(rgb_feat)) #+ self.colors"""
            self.colors_fine = self.colors_fine.contiguous()

            
            if False: #(iter_step+1) > 10000: 
                colors_out = torch.zeros((rays_o.shape[0], 3)).cuda()
                mask_out = torch.zeros((rays_o.shape[0])).cuda()
                #renderer_cuda.render_no_grad(rays_o.shape[0], self.inv_s, self.out_sdf, self.colors_fine, self.offsets, colors_out, mask_out)
                if self.double_net:
                    renderer_cuda.render_no_grad(rays_o.shape[0], self.inv_s, self.out_sdf, self.colors, self.offsets, colors_out, mask_out)
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
                colors_samples = torch.zeros_like(self.samples[:nb_samples,:])
                true_rgb = true_rgb.cpu()
                colors_out = colors_out.cpu()
                for i in range(rays_o.shape[0]):                            
                    start = self.offsets[i,0]
                    end = self.offsets[i,1]
                    for j in range(start, start+end):
                        #colors_samples[j,:] = colors_out[i,:]
                        #colors_samples[j,:] = abs(colors_out[i,:] - true_rgb[i,:])
                        colors_samples[j,:] = true_rgb[i,:]
                ply.save_ply("Exp/samples.ply", np.transpose(2.2*self.samples[:nb_samples,:].cpu() - 1.1), col = 255*np.transpose(colors_samples.cpu().numpy()))
                #ply.save_ply("Exp/samples.ply", np.transpose(self.samples[:nb_samples,:].cpu()), col = 255*np.transpose(sdf_rgb))
                #ply.save_ply("TMP/meshes/samples_"+str(self.iter_step).zfill(5)+".ply", np.transpose(pts.cpu()), col = 255*np.transpose(sdf_rgb))
                #print("nb samples: ", nb_samples)
                input()
                exit()
                input()

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

            #self.dataset.weight_rays[rays_ids] = color_fine_loss + 1.0e-6

            # Total loss   
            mask_sum = mask.sum()
            
            self.optimizer.zero_grad()
            if self.double_net:
                loss = (color_fine_loss.sum() + lamda_c*color_coarse_loss.sum()) #/ (mask_sum + 1.0e-5)
            else:
                loss = color_fine_loss.sum() # / (mask_sum + 1.0e-5)
            loss.backward()

            ########################################
            # Backprop feature gradients to gradients on sites
            # step optimize color features
            start = timer()   
            #sdf_features_grad = torch.cat([sdf_feat_entry.grad[:,3:6], sdf_feat_exit.grad[:,3:6]], -1).contiguous()
 
            if self.double_net:
                if self.position_encoding:
                    shift_p = xyz_emb.shape[1] + viewdirs_emb.shape[1]
                    shift_p_f = xyz_emb_fine.shape[1] + viewdirs_emb_fine.shape[1]
                    #shift_p_f = 3 + viewdirs_emb_fine.shape[1]
                    #self.fine_features_grad[:nb_samples,:] = rgb_feat.grad[:,shift_p_f+8:]
                    self.fine_features_grad[:nb_samples,8:] = rgb_feat.grad[:,shift_p_f+8:]
                    self.fine_features_grad[:nb_samples,:8] = coarse_feat.grad[:,shift_p+5:] 
                    #self.fine_features_grad[:nb_samples,:8] = coarse_feat.grad[:,shift_p+3:] 
                    self.norm_features_grad[:nb_samples,:] = rgb_feat.grad[:,shift_p_f+5:shift_p_f+8] + lamda_c*coarse_feat.grad[:,shift_p + 2:shift_p+5]  #math.sin(0.5*math.pi*self.loc_iter/self.end_iter_loc)*rgb_feat.grad[:,shift_p+5:shift_p+8] +\
                                                                #(1.0 - math.sin(0.5*math.pi*self.loc_iter/self.end_iter_loc))*coarse_feat.grad[:,shift_p+2:shift_p+5]
                    #self.sdf_features_grad[:nb_samples,:] = rgb_feat.grad[:,shift_p_f+3:shift_p_f+5] + lamda_c*coarse_feat.grad[:,shift_p:shift_p+2]
                else:
                    fine_features_grad = rgb_feat.grad[:,36:]
                    norm_features_grad = rgb_feat.grad[:,30:36] + 0.5*self.coarse_feat.grad[:nb_samples,42:]
            else:
                if self.position_encoding:
                    shift_p = xyz_emb.shape[1] + viewdirs_emb.shape[1]
                    self.fine_features_grad[:nb_samples,:] = rgb_feat.grad[:,shift_p+5:]  # 44 <- view pose encoding = 1 rgb_feat.grad[:,62:]  <- view pose encoding = 4
                    self.norm_features_grad[:nb_samples,:] = rgb_feat.grad[:,shift_p+2:shift_p+5] 
                    self.sdf_features_grad[:nb_samples,:] = rgb_feat.grad[:,shift_p:shift_p+2]
                else:
                    fine_features_grad = rgb_feat.grad[:,33:]
                    norm_features_grad = rgb_feat.grad[:,27:33]
            #fine_features_grad = fine_features_grad.contiguous()
            #norm_features_grad = norm_features_grad.contiguous()

            """if iter_step % 100 == 0:
                print("coarse => ", coarse_feat.grad[:,shift_p+3:].mean())
                print("fine => ", rgb_feat.grad[:,shift_p_f+6:].mean())
                print("nrm => ", self.norm_features_grad[:nb_samples,:].mean())"""

            if verbose:
                print('input grads time:', timer() - start)
             
            start = timer()
            self.grad_features[:] = 0.0
            self.counter[:] = 0.0
            backprop_cuda.backprop_feat(nb_samples, self.tet32.sites.shape[0], self.dim_feats, self.out_sdf, self.grad_features, self.counter, self.fine_features_grad, self.out_ids, self.out_weights)  
            if verbose:
                print('backprop_feat samples time:', timer() - start)

            #print(self.grad_features.mean())
                
            if self.f_w > 0.0: #not (iter_step > 5000 and self.loc_iter > 0 and self.loc_iter < warm_up): 
                start = timer()
                self.grad_norm_feat[:] = 0.0
                self.counter[:] = 0.0
                backprop_cuda.backprop_grad(nb_samples, self.tet32.sites.shape[0], self.out_sdf, self.grad_norm_feat, self.norm_features_grad, self.out_ids, self.out_weights)
                self.grad_norm[:,:] = self.grad_norm_feat[:,:3]
                #self.grad_features[:,8:] = 0.0

                if verbose:
                    print('backprop_feat grads time:', timer() - start)

                ############ Backprop gradients to neighbors ############                   
                """start = timer()
                self.grad_norm[:] = 0.0
                backprop_cuda.backprop_multi(self.tet32.sites.shape[0], 96, 8, self.tet32.sites, self.activated, self.grad_norm, self.grad_norm_feat, self.grad_features, self.tet32.knn_sites)
                self.grad_features[:,8:] = 0.0
                if verbose:
                    print('backprop_multi sites time:', timer() - start)"""
                
                
                start = timer()
                self.grad_sdf_norm[:] = 0.0
                #backprop_cuda.backprop_norm(self.tet32.nb_tets, self.tet32.summits, self.tet32.sites, self.vol_tet32, self.weights_diff, self.weights_tot_diff, self.grad_norm_feat, self.grad_sdf_norm, self.activated)
                backprop_cuda.backprop_unit_norm(self.tet32.nb_tets, self.tet32.summits, self.tet32.sites.float(), self.norm_grad, self.unormed_grad, self.vol_tet32, self.weights_diff, self.weights_tot_diff, self.grad_norm, self.grad_sdf_norm, self.activated)
                self.grad_sdf_norm[outside_flag[:] == 1] = 0.0    
                if verbose:
                    print('backprop_unit_norm sites time:', timer() - start)

                """start = timer()
                self.grad_sdf_net[:] = 0.0
                backprop_cuda.backprop_sdf(nb_samples, self.out_sdf, self.grad_sdf_net, self.sdf_features_grad, self.out_ids, self.out_weights)    
                ##backprop_cuda.backprop_norm(nb_samples, self.grad_sdf_net, self.grad_norm_feat, self.weights_grad)  
                self.grad_sdf_net[outside_flag[:] == 1] = 0.0   
                
                if verbose:
                    print('backprop_sdf sites time:', timer() - start)"""
            
            self.optimizer.step()

            #self.activated[:] = 1
            
            ########################################
            ####### Regularization terms ###########
            ########################################

            if self.double_net:     
                #grad_sdf =  self.f_w*(self.grad_sdf_net + self.grad_sdf_norm) +\
                #    ((1.0 - lamda_c)*self.vortSDF_renderer_fine.grads_sdf + lamda_c*self.vortSDF_renderer_coarse_net.grads_sdf)
                #grad_sdf =  (self.vortSDF_renderer_fine.grads_sdf)
                self.grad_sdf = self.vortSDF_renderer_fine.grads_sdf + lamda_c*self.vortSDF_renderer_coarse_net.grads_sdf #+ self.f_w*self.grad_sdf_norm #(self.grad_sdf_net + self.grad_sdf_norm) # +\
                             #(math.sin(0.5*math.pi*self.loc_iter/self.end_iter_loc)*self.vortSDF_renderer_fine.grads_sdf +\
                             #   (1.0 - math.sin(0.5*math.pi*self.loc_iter/self.end_iter_loc))*self.vortSDF_renderer_coarse_net.grads_sdf)
            else:
                self.grad_sdf = self.vortSDF_renderer_fine.grads_sdf + self.f_w*(self.grad_sdf_net + self.grad_sdf_norm)  #/ (mask_sum + 1.0e-5) + self.f_w*self.grad_sdf_net

            self.grad_sdf[outside_flag[:] == 1] = 0.0   

              

            
            if False: #iter_step > up_iters[4]:
                start = timer()   
                self.grad_sdf_smooth[:] = 0.0
                backprop_cuda.knn_smooth(self.tet32.sites.shape[0], self.knn, self.hlvl, self.sigma, self.sigma_feat, 1, self.tet32.sites, self.activated,
                                            self.grad_sdf_space, self.grad_sdf, self.fine_features, self.tet32.knn_sites, self.grad_sdf_smooth)
                self.grad_sdf[:] = self.grad_sdf_smooth[:]
                self.grad_sdf[outside_flag[:] == 1.0] = 0.0   
                if verbose:
                    print('knn_smooth time:', timer() - start)


                            
            #### SMOOTH FEATURE GRADIENT
            """self.grad_feat_smooth[:] = self.grad_features[:]
            self.counter_smooth[:] = 1.0
            backprop_cuda.smooth(self.tet32.edges.shape[0], self.tet32.sites.shape[0], self.sigma, self.dim_feats, self.tet32.sites, self.grad_features, 
                                 self.tet32.edges, self.grad_feat_smooth, self.counter_smooth)"""
            
            """self.grad_feat_smooth[:] = 0.0
            backprop_cuda.knn_smooth(self.tet32.sites.shape[0], 96, self.sigma, self.sigma_feat, self.dim_feats, self.tet32.sites, self.activated,
                                     self.grad_sdf_space, self.grad_features, self.fine_features, self.tet32.knn_sites, self.grad_feat_smooth) 
            self.grad_features[:] = self.grad_feat_smooth[:]
            self.grad_features[outside_flag[:] == 1.0] = 0.0"""  
            
                
            """start = timer()   
            self.grad_sdf_reg[:] = 0.0
            self.grad_feat_reg[:] = 0.0
            backprop_cuda.space_reg(rays_o.shape[0], rays_d, self.grad_sdf_space, self.out_weights, self.out_z, self.out_sdf, self.out_feat, self.out_ids, self.offsets, self.grad_sdf_reg, self.grad_feat_reg)
            
            if verbose:
                print('space_reg time:', timer() - start)"""
        
            if iter_step % full_reg == 0: #iter_step > up_iters[4]: #iter_step % full_reg == 0: # and (iter_step % 3 == 0 or (iter_step+1) < 10000): # and ((iter_step+1) < 35000 or iter_step % 3 == 0):
                start = timer()   
                self.grad_sdf_smooth[:] = 0.0
                self.grad_feat_smooth[:] = 0.0
                self.weight_sdf_smooth[:] = 0.0
                self.activated[:] = 2
                #if (iter_step+1) < 35000:
                #self.activated[grad_sdf == 0.0] = 0 #self.sigma

                #backprop_cuda.smooth_sdf(self.tet32.edges.shape[0], self.sigma, self.tet32.sites.float(), self.activated,
                #                         self.sdf, self.fine_features, self.tet32.edges, self.grad_sdf_smooth, self.grad_feat_smooth, self.weight_sdf_smooth)
                
                """with torch.no_grad():
                    self.sdf_smooth_2[:] = self.sdf[:]
                backprop_cuda.knn_smooth(self.tet32.sites.shape[0], self.knn, self.hlvl, self.sigma, self.sigma_feat, 1, self.tet32.sites, self.activated,
                                        self.grad_sdf_space, self.sdf, self.fine_features.detach(), self.tet32.knn_sites, self.sdf_smooth_2)"""
                
                self.grad_sdf_L2[:] = 0.0
                backprop_cuda.sdf_smooth(self.tet32.sites.shape[0], 1, self.activated, self.sdf, self.sdf_smooth, self.grad_sdf_L2)
                
                """start = timer()   
                self.grad_sdf_smooth[:] = 0.0
                self.activated[:] = 2
                backprop_cuda.knn_smooth(self.tet32.sites.shape[0], self.knn, self.hlvl, self.sigma, self.sigma_feat, 1, self.tet32.sites, self.activated,
                                            self.grad_sdf_space, self.grad_sdf_L2, self.fine_features, self.tet32.knn_sites, self.grad_sdf_smooth)
                self.grad_sdf_L2[:] = self.grad_sdf_L2[:] - self.grad_sdf_smooth[:]
                if verbose:
                    print('knn_smooth time:', timer() - start)"""

                """"""""""""""""""""""""""""""
                """"""""""""""""""""""""""""""
                """ Smooth normals """
                
                self.div_2[:] = 0.0
                #self.div_norm[:] = 0.0
                #laplacian.TVNorm(self.div_2, self.div_norm, self.unormed_grad, self.activated, M_vals, L_vals, L_outer, L_nonZeros, L_nnZ, L_outerSize, L_cols)
                #self.div_norm[self.div_norm[:] == 0.0] = 1.0 
                #self.div_2 = self.div_2 / self.div_norm.expand(-1, 3)
                #self.div_2 = self.div_2 * M_vals.expand(-1, 3)
                #self.div_2[M_vals[:] == 0.0, :] = 0.0
                
                backprop_cuda.smooth_sdf(self.tet32.edges.shape[0], self.sigma, 3, self.tet32.sites.float(), self.activated,
                                         self.unormed_grad, self.fine_features, self.tet32.edges, self.div_2, self.grad_feat_smooth, self.weight_sdf_smooth)
                
                self.grad_norm_sdf_L2[:] = 0.0
                cvt_grad_cuda.backprop_norm_grad(self.tet32.nb_tets, self.tet32.summits, self.tet32.sites.float(), self.activated, self.grad_norm_sdf_L2, self.div_2, self.vol_tet32, self.weights_diff, self.weights_tot_diff)
                self.grad_norm_sdf_L2[outside_flag[:] == 1] = 0.0 

                """
                
                self.grad_norm_L2[:] = 0.0
                #backprop_cuda.smooth_sdf(self.tet32.edges.shape[0], self.sigma, self.tet32.sites.float(), self.activated,
                #                         self.grad_sdf_space, self.fine_features, self.tet32.edges, grad_norm_L2, self.grad_feat_smooth, self.weight_sdf_smooth)

                self.grad_norm_sdf_L2[:] = 0.0
                #cvt_grad_cuda.backprop_norm_grad(self.tet32.nb_tets, self.tet32.summits, self.tet32.sites, self.activated, self.grad_norm_L2, self.grad_norm_sdf_L2, self.vol_tet32, self.weights_diff, self.weights_tot_diff)
                cvt_grad_cuda.backprop_unit_norm(self.tet32.nb_tets, self.tet32.summits, self.tet32.sites.float(), self.norm_grad, self.unormed_grad, self.vol_tet32, self.weights_diff, self.weights_tot_diff, self.grad_norm, self.grad_sdf_norm, self.activated)
               

                self.grad_norm_sdf_L2[outside_flag[:] == 1] = 0.0   """
                """"""""""""""""""""""""""""""
                """"""""""""""""""""""""""""""
                
                """"""""""""""""""""""""""""""
                """"""""""""""""""""""""""""""
                """ Smooth features """
                
                self.grad_feat_smooth[:] = 0.0
                backprop_cuda.smooth_sdf(self.tet32.edges.shape[0], self.sigma, self.dim_feats, self.tet32.sites.float(), self.activated,
                                         self.fine_features, self.fine_features, self.tet32.edges, self.grad_feat_smooth, self.grad_feat_smooth, self.weight_sdf_smooth)
                self.grad_feat_smooth[outside_flag[:] == 1,:] = 0.0 
                
                """self.feat_smooth[:] = 0.0
                backprop_cuda.knn_smooth(self.tet32.sites.shape[0], self.knn, self.hlvl, self.sigma, self.sigma_feat, self.dim_feats, self.tet32.sites, self.activated,
                                            self.grad_sdf_space, self.fine_features, self.fine_features, self.tet32.knn_sites, self.feat_smooth)
                
                self.grad_feat_smooth[:] = 0.0
                backprop_cuda.sdf_smooth(self.tet32.sites.shape[0], self.dim_feats, self.activated, self.fine_features, self.feat_smooth, self.grad_feat_smooth)

                self.grad_feat_smooth[outside_flag[:] == 1,:] = 0.0   """
                """"""""""""""""""""""""""""""
                """"""""""""""""""""""""""""""

                #self.weight_sdf_smooth[self.weight_sdf_smooth[:] == 0.0] = 1.0
                #self.grad_sdf_smooth = self.grad_sdf_smooth / self.weight_sdf_smooth[:]
                #self.grad_sdf_smooth[:] = self.grad_sdf_smooth[:] / self.tet32.sites.shape[0]
                #self.grad_feat_smooth = self.grad_feat_smooth / self.weight_sdf_smooth[:].reshape(-1,1)
                #self.grad_feat_smooth[:,:] = self.grad_feat_smooth[:,:] / self.sigma
                self.grad_sdf_L2[outside_flag[:] == 1] = 0.0   
                if verbose:
                    print('smooth_sdf time:', timer() - start)
            else:
                self.grad_norm_sdf_L2[:] = 0.0
                self.grad_feat_smooth[:] = 0.0
                self.grad_sdf_L2[:] = 0.0
            
            ########################################
            ####### Optimize features ##############
            ########################################
            
            lbda = 0.5 #0.5 #0.6 # 0.5
            mu = -0.53   
            
            """if iter_step % full_reg == 0:
                self.div_feat[:] = 0
                laplacian.MeanCurve(self.div_feat, self.fine_features, self.dim_feats, self.activated, M_vals, L_vals, L_outer, L_nonZeros, L_nnZ, L_outerSize, L_cols)
                self.grad_feat_smooth[:] = 0.0
                backprop_cuda.sdf_smooth(self.tet32.sites.shape[0], self.dim_feats, self.activated, self.fine_features, self.div_feat, self.grad_feat_smooth)"""

            if not iter_step % full_reg == 0:
                self.grad_feat_smooth[:] = 0.0
                #self.grad_feat_smooth[self.grad_sdf[:] == 0.0] = 0.0
            
            self.optimizer_feat.zero_grad()

            """if iter_step % 2 == 0:
                self.fine_features.grad = self.grad_features + self.tv_f*self.grad_feat_smooth #+ lbda*self.tv_f*self.div_feat[:] #+ self.tv_f*self.grad_feat_smooth #
            else:
                self.fine_features.grad = self.grad_features + self.tv_f*self.grad_feat_smooth #+ mu*self.tv_f*self.div_feat[:] #+ self.tv_f*self.grad_feat_smooth #"""
            self.fine_features.grad = self.grad_features + self.tv_f*self.grad_feat_smooth 
            

            self.optimizer_feat.step()


            if iter_step > 5000 and self.loc_iter < warm_up:
                self.update_learning_rate(self.loc_iter)
                self.loc_iter = self.loc_iter + 1
                continue

            ########################################
            ####### Optimize sdf values ############
            ########################################
            self.optimizer_sdf.zero_grad() 

            #self.norm_grad = torch.linalg.norm(self.unormed_grad, ord=2, axis=-1, keepdims=True).reshape(-1)           
                
            """self.div_norm[:] = 0.0
            laplacian.MeanCurve(self.div_norm, self.unormed_grad, 3, self.activated, L_vals, L_outer, L_nonZeros, L_nnZ, L_outerSize, L_cols)
            self.grad_norm_Lapl[:] = 0.0
            cvt_grad_cuda.backprop_norm_grad(self.tet32.nb_tets, self.tet32.summits, self.tet32.sites, self.activated, self.div_norm, self.grad_norm_Lapl, self.vol_tet32, self.weights_diff, self.weights_tot_diff)"""

            """self.div[:] = 0.0
            if False: #iter_step > 2000:
                laplacian.MeanCurve(self.div, self.sdf, 1, self.activated, M_vals, L_vals, L_outer, L_nonZeros, L_nnZ, L_outerSize, L_cols)
            else:
                laplacian.MeanCurve(self.div, self.sdf_smooth, 1, self.activated, M_vals, L_vals, L_outer, L_nonZeros, L_nnZ, L_outerSize, L_cols)
            self.div_2[:] = 0.0
            #laplacian.MeanCurve(self.div_2, self.div, 1, self.activated, L_vals, L_outer, L_nonZeros, L_nnZ, L_outerSize, L_cols)
            self.div[outside_flag[:] == 1] = 0.0
            #self.div[self.cam_ids[:] ] = 0.0
            self.div_2[outside_flag[:] == 1] = 0.0
            self.div_2[self.cam_ids[:]] = 0.0
            self.div[self.mask_background[:] == 1] = 0.0
            self.div_2[self.mask_background[:] == 1] = 0.0
            self.grad_eik[self.mask_background[:] == 1] = 0.0"""            
            
            #self.grad_norm_smooth[self.mask_background[:] == 1] = 0.0


            self.div[:] = 0.0
            if iter_step % full_reg == 0:
                #self.grad_eik[:] = 0.0
                backprop_cuda.smooth_sdf(self.tet32.edges.shape[0], self.sigma, 1, self.tet32.sites.float(), self.activated,
                                         self.sdf_smooth, self.fine_features, self.tet32.edges, self.div, self.grad_feat_smooth, self.weight_sdf_smooth)
                #backprop_cuda.sdf_smooth(self.tet32.sites.shape[0], 1, self.activated, self.sdf_smooth, self.sdf_smooth_2, self.div)
                #laplacian.MeanCurve(self.div, self.sdf_smooth, 1, self.activated, M_vals, L_vals, L_outer, L_nonZeros, L_nnZ, L_outerSize, L_cols)
                #self.div = self.div * M_vals
                #self.div[M_vals[:] == 0.0] = 0.0
                self.div[outside_flag[:] == 1] = 0.0
                #self.div[self.mask_background[:] == 1] = 0.0
            
            if not iter_step % full_reg == 0:
                self.grad_norm_smooth[:] = 0.0
                self.grad_eik[:] = 0.0
                self.grad_sdf_L2[:] = 0.0
                self.div[:] = 0.0
                self.grad_norm_sdf_L2[:] = 0.0
            """else:
                self.grad_norm_smooth[self.visible[:] == 0.0] = 0.0
                self.grad_eik[self.visible[:] == 0.0] = 0.0
                self.div[self.visible[:] == 0.0] = 0.0
                self.grad_sdf_L2[self.visible[:] == 0.0] = 0.0
                self.grad_norm_sdf_L2[self.visible[:] == 0.0] = 0.0"""

            """ if iter_step % 2 == 0:
                self.grad_sdf = self.grad_sdf +  ((self.s_w*self.grad_norm_smooth + lbda*self.tv_w*self.div) + self.e_w*self.grad_eik + 1.0e1*self.tv_w*self.grad_sdf_L2)
            else:
                self.grad_sdf = self.grad_sdf + ((self.s_w*self.grad_norm_smooth + mu*self.tv_w*self.div) + self.e_w*self.grad_eik + 1.0e1*self.tv_w*self.grad_sdf_L2) #   self.tv_w*self.grad_sdf_smooth #"""

            #self.grad_sdf = self.grad_sdf + fact_w*(1.0e1*self.s_w*self.grad_norm_smooth + 1.0e2*self.tv_w*self.div + 1.0e2*self.tv_w*self.grad_eik) #+ self.tv_w*self.grad_sdf_L2  #+ 1.0e7*self.tv_w*self.grad_sdf_L2 + self.e_w*self.grad_eik
            #self.grad_sdf = self.grad_sdf + self.visible*(1.0e0*self.s_w*self.grad_norm_sdf_L2 + 1.0e2*self.s_w*self.grad_norm_smooth + 1.0e-1*self.s_w*self.grad_eik) # + 1.0e-3*self.tv_w*self.div
            #self.grad_sdf = self.grad_sdf + fact_w*(1.0e2*self.s_w*(self.grad_norm_sdf_L2) + 1.0e-2*self.tv_w*self.div + 1.0e0*self.s_w*self.grad_eik) 
            self.grad_sdf = self.grad_sdf + (self.tv_w*(1.0e0*self.div) + 1.0e0*self.s_w*self.grad_norm_smooth)  #+ 1.0e0*self.tv_w*self.div + 1.0e1*self.e_w*self.grad_eik  + 1.0e2*self.grad_norm_sdf_L2 + self.e_w*self.grad_eik
            #self.grad_sdf[self.mask_background[:] == 1] = 0.0
            
            #self.visible[abs(self.grad_sdf[:]) > 0.0] = self.visible[abs(self.grad_sdf[:]) > 0.0] + 1
            self.visible[:] = self.visible[:] + abs(self.grad_sdf[:])

            if iter_step % 300 == 0:
                print("self.div_2 => ", self.div_2[self.visible[:] > 0].mean())
                print("self.grad_norm_sdf_L2 => ", self.grad_norm_sdf_L2[self.visible[:] > 0].max())
                print("self.grad_norm_smooth => ", self.grad_norm_smooth[self.visible[:] > 0].max())
                print("self.div => ", self.div[self.visible[:] > 0].mean())
                print("self.grad_eik => ", self.grad_eik[self.visible[:] > 0].max())
                print("self.grad_sdf_norm => ", self.grad_sdf_norm[self.visible[:] > 0].mean())

            if True: #iter_step > up_iters[4]:
                start = timer()   
                self.grad_sdf_smooth[:] = 0.0
                backprop_cuda.knn_smooth(self.tet32.sites.shape[0], self.knn, self.hlvl, self.sigma, self.sigma_feat, 1, self.tet32.sites, self.activated,
                                            self.grad_sdf_space, self.grad_sdf, self.fine_features, self.tet32.knn_sites, self.grad_sdf_smooth)
                self.grad_sdf[:] = self.grad_sdf_smooth[:]
                self.grad_sdf[outside_flag[:] == 1.0] = 0.0                   
                #self.grad_sdf[self.mask_background[:] == 1] = 0.0
                if verbose:
                    print('knn_smooth time:', timer() - start)

                """laplacian.MeanCurve(self.grad_sdf_smooth, self.grad_sdf, 1, self.activated, M_vals, L_vals, L_outer, L_nonZeros, L_nnZ, L_outerSize, L_cols)
                self.grad_sdf[:] = self.grad_sdf_smooth[:]
                self.grad_sdf[outside_flag[:] == 1] = 0.0"""

            if False: #iter_step > 2000:
                self.sdf.grad = self.grad_sdf
            else:
                #self.sdf.grad = self.grad_sdf + self.s_w*(0.01*self.grad_norm_smooth + self.grad_sdf_L2)
                """self.div[:] = 0.0
                laplacian.MeanCurve(self.div, self.sdf, 1, self.activated, L_vals, L_outer, L_nonZeros, L_nnZ, L_outerSize, L_cols)
                self.div[self.cam_ids[:] ] = 0.0
                self.div[outside_flag[:] == 1] = 0.0
                self.div[self.mask_background[:] == 1] = 0.0"""
                self.sdf.grad = self.grad_sdf + 1.0e0*self.e_w*self.grad_sdf_L2 #+ self.e_w*self.grad_sdf_L2 #+ 0.001*self.s_w*self.grad_norm_smooth #self.s_w*self.grad_sdf_L2 + sign*self.div
                #if iter_step % 2 == 0:
                #    self.sdf.grad = self.grad_sdf + self.e_w*self.grad_eik + self.grad_sdf_smooth #lbda*self.tv_w*self.div #+ self.s_w*self.grad_norm_smooth  #+ 1.0e7*self.tv_w*self.grad_sdf_L2
                #else:
                #    self.sdf.grad = self.grad_sdf + self.e_w*self.grad_eik + self.grad_sdf_smooth #mu*self.tv_w*self.div #+ self.s_w*self.grad_norm_smooth   #+ 1.0e7*self.tv_w*self.grad_sdf_L2

            #self.sdf.grad[self.mask_background[:] == 1] = 0.0


            if False: # iter_step > 2000:
                with torch.no_grad():
                    if iter_step % 2 == 0:
                        self.sdf[:] = self.sdf[:] + 0.0001*lbda*self.div[:]
                    else:
                        self.sdf[:] = self.sdf[:] + 0.0001*mu*self.div[:]
            else:
                self.optimizer_sdf.step()

            """with torch.no_grad():
                self.div[:] = 0.0
                laplacian.MeanCurve(self.div, self.sdf, 1, self.activated, M_vals, L_vals, L_outer, L_nonZeros, L_nnZ, L_outerSize, L_cols)
                self.div[outside_flag[:] == 1] = 0.0
                self.div[self.mask_background[:] == 1] = 0.0
                if iter_step % 2 == 0:
                    self.sdf[:] = self.sdf[:] + 0.001*lbda*self.div[:]
                else:
                    self.sdf[:] = self.sdf[:] + 0.001*mu*self.div[:]"""
            
            if iter_step % full_reg == 0:
                self.visible[:] = 0


            ########################################
            ##### Optimize sites positions #########
            ########################################
            if (iter_step+1) in up_iters: #== 2000 or (iter_step+1) == 10000 or (iter_step+1) == 30000 or (iter_step+1) == 50000 or (iter_step+1) == 70000:# or (iter_step+1) == 45000: 

                if (iter_step+1) == up_iters[1]:
                    self.batch_size = 8192      
                    self.Allocate_batch_data()  
                  
                #for id_im in tqdm(range(runner.dataset.n_images)):
                #    runner.render_image(img_idx = id_im)

                self.optimizer_sdf = torch.optim.Adam([self.sdf], lr=self.learning_rate_sdf, betas=(0.9, 0.98))     
                for Taub_it in range(0):
                    self.optimizer_sdf.zero_grad()      
                    self.activated[:] = 2
                    backprop_cuda.knn_smooth(self.tet32.sites.shape[0], self.knn, self.hlvl, self.sigma, self.sigma_feat, 1, self.tet32.sites, self.activated,
                                                self.grad_sdf_space, self.sdf.detach(), self.fine_features, self.tet32.knn_sites, self.sdf_smooth_2)
                    
                    self.div[:] = 0.0
                    laplacian.MeanCurve(self.div, self.sdf_smooth, 1, self.activated, M_vals, L_vals, L_outer, L_nonZeros, L_nnZ, L_outerSize, L_cols)
                    self.div[outside_flag[:] == 1] = 0.0
                    
                    self.grad_sdf[:] = 0.0
                    if Taub_it % 2 == 0:
                        self.grad_sdf = lbda*self.div 
                    else:
                        self.grad_sdf = mu*self.div 
                        
                    start = timer()   
                    self.grad_sdf_smooth[:] = 0.0
                    backprop_cuda.knn_smooth(self.tet32.sites.shape[0], self.knn, self.hlvl, self.sigma, self.sigma_feat, 1, self.tet32.sites, self.activated,
                                                self.grad_sdf_space, self.grad_sdf, self.fine_features, self.tet32.knn_sites, self.grad_sdf_smooth)
                    self.sdf.grad = self.grad_sdf_smooth
                    if verbose:
                        print('knn_smooth time:', timer() - start)

                    self.optimizer_sdf.step()

                """with torch.no_grad():
                    self.sdf_smooth[self.mask_background[:]*(self.sdf_smooth[:] > 0.0) == 1] = 1.0
                    self.sdf_smooth[self.mask_background[:]*(self.sdf_smooth[:] < 0.0) == 1] = -1.0
                    valid_sites = self.dataset.clean_pc(self.tet32.sites.float())
                    self.sdf_smooth[valid_sites[:] == 0] = 1.0"""
                
                self.tet32.surface_from_sdf(self.sdf_smooth.cpu().numpy().reshape(-1), "Exp/{}/meshes/test_tri_up_{}.ply".format(self.data_name, (iter_step+1)), self.dataset.scale_mats_np[0][:3, 3][None], self.dataset.scale_mats_np[0][:3, :3])
                                
                
                #if (iter_step+1) == 20000:
                #    self.batch_size = 10240

                if (iter_step+1) == up_iters[4]:
                    self.sdf, self.fine_features, self.mask_background = self.tet32.upsample(self.sdf_smooth.detach().cpu().numpy(), self.sdf_smooth.detach().cpu().numpy(), self.fine_features.detach().cpu().numpy(), 
                                                                                         self.visual_hull, res, cam_sites, cam_ids, self.learning_rate_cvt, 300, (iter_step+1) <= up_iters[2], self.sigma_max)
                elif (iter_step+1) == up_iters[3]:
                    self.sdf, self.fine_features, self.mask_background = self.tet32.upsample(self.sdf_smooth.detach().cpu().numpy(), self.sdf_smooth.detach().cpu().numpy(), self.fine_features.detach().cpu().numpy(), 
                                                                                         self.visual_hull, res, cam_sites, cam_ids, self.learning_rate_cvt, 500, (iter_step+1) <= up_iters[2], self.sigma_max)
                else:
                    self.sdf, self.fine_features, self.mask_background = self.tet32.upsample(self.sdf_smooth.detach().cpu().numpy(), self.sdf_smooth.detach().cpu().numpy(), self.fine_features.detach().cpu().numpy(), 
                                                                                         self.visual_hull, res, cam_sites, cam_ids, self.learning_rate_cvt, 1000, (iter_step+1) <= up_iters[2], self.sigma_max)

                self.sdf = self.sdf.contiguous()
                self.sdf.requires_grad_(True)
                self.fine_features = self.fine_features.contiguous()
                self.fine_features.requires_grad_(True)
                self.tet32.load_cuda(self.sigma_max)

                print(self.mask_background.max())
                print(self.mask_background.min())

                sites = np.asarray(self.tet32.vertices)  
                cam_ids = np.stack([np.where((sites == cam_sites[i,:]).all(axis = 1))[0] for i in range(cam_sites.shape[0])]).reshape(-1)
                self.tet32.make_adjacencies(cam_ids)

                self.tet32.make_multilvl_knn()

                cam_ids = torch.from_numpy(cam_ids).int().cuda()
                self.cam_ids = cam_ids
                
                outside_flag = np.zeros(sites.shape[0], np.int32)
                outside_flag[sites[:,0] < self.visual_hull[0] + 1.5*(self.visual_hull[3]-self.visual_hull[0])/(res)] = 1
                outside_flag[sites[:,1] < self.visual_hull[1] + 1.5*(self.visual_hull[4]-self.visual_hull[1])/(res)] = 1
                outside_flag[sites[:,2] < self.visual_hull[2] + 1.5*(self.visual_hull[5]-self.visual_hull[2])/(res)] = 1
                outside_flag[sites[:,0] > self.visual_hull[3] - 1.5*(self.visual_hull[3]-self.visual_hull[0])/(res)] = 1
                outside_flag[sites[:,1] > self.visual_hull[4] - 1.5*(self.visual_hull[4]-self.visual_hull[1])/(res)] = 1
                outside_flag[sites[:,2] > self.visual_hull[5] - 1.5*(self.visual_hull[5]-self.visual_hull[2])/(res)] = 1
                
                self.Allocate_data()
                self.activated[:] = 1
                
                cvt_grad_cuda.diff_tensor(self.tet32.nb_tets, self.tet32.summits, self.tet32.sites, self.vol_tet32, self.weights_diff, self.weights_tot_diff)

                M_vals, L_vals, L_nonZeros, L_outer, L_sizes = laplacian.MakeLaplacian(self.tet32.sites.shape[0], self.tet32.nb_tets, self.tet32.sites.cpu(), self.tet32.summits.cpu(), self.tet32.valid_tets.cpu())

                L_nnZ = L_sizes[0]
                L_outerSize = L_sizes[1]
                L_cols = L_sizes[2]
                
                M_vals = M_vals.float().cuda()
                L_vals = L_vals.float().cuda()
                L_nonZeros = L_nonZeros.cuda()
                L_outer = L_outer.cuda()        

                m = igl.massmatrix(self.tet32.sites.cpu().numpy(), self.tet32.tetras, igl.MASSMATRIX_TYPE_BARYCENTRIC)
                m_d = m.diagonal()
                print(m_d[m_d > 0.0].min())
                
                M_vals = torch.from_numpy(np.asarray(m_d)).float().cuda().reshape(-1, 1)
                #M_vals = torch.ones(self.tet32.sites.shape[0]).float().cuda()

                #mask_L = torch.zeros(self.tet32.sites.shape[0]).int().cuda()
                #mask_L[self.cam_ids[:]] = 1
                #laplacian.MaskLaplacian(mask_L, L_vals, L_outer, L_nonZeros, L_nnZ, L_outerSize, L_cols)
                                
                delta_sites = torch.zeros(self.tet32.sites.shape).float().cuda()
                with torch.no_grad():  
                    delta_sites[:] = self.tet32.sites[:]

                with torch.no_grad():
                    self.sdf_smooth[:] = self.sdf[:]

                self.optimizer_sdf = torch.optim.Adam([self.sdf], lr=self.learning_rate_sdf, betas=(0.9, 0.98))        

                step_size = step_size / 1.5
                self.e_w = self.e_w / 10.0
                self.learning_rate_sdf = 1.0e-4
                self.tv_w = self.tv_w / 2.0
                self.learning_rate_cvt = self.learning_rate_cvt / 1.5 #2.0

                self.sigma_start = self.sigma_start/2.0
                """if (iter_step+1) == up_iters[3]:
                    self.sigma_start = self.sigma_start/1.8
                    #self.sigma_start = (min((self.visual_hull[3]-self.visual_hull[0]), (self.visual_hull[4]-self.visual_hull[1]), (self.visual_hull[5]-self.visual_hull[2]))/res)/16.0
                elif (iter_step+1) == up_iters[4]:
                    self.sigma_start = self.sigma_start/1.8
                else :
                    self.sigma_start = self.sigma_start/2.0"""
                
                self.sigma_max = self.sigma_max/2.0
                self.sigma = self.sigma_start #self.sigma / 2.0
                self.w_g = 1.0
                #full_reg = self.end_iter
                centers = torch.zeros(self.tet32.sites.shape).float().cuda()
                centers[:,0] = (self.visual_hull[3]+self.visual_hull[0])/2.0
                centers[:,1] = (self.visual_hull[4]+self.visual_hull[1])/2.0
                centers[:,2] = (self.visual_hull[5]+self.visual_hull[2])/2.0

                #self.tet32.surface_from_sdf(self.sdf_smooth.cpu().numpy().reshape(-1), "Exp/{}/meshes/test_tri_up_{}.ply".format(self.data_name, (iter_step+1))) #, self.dataset.scale_mats_np[0][:3, 3][None], self.dataset.scale_mats_np[0][:3, :3])
                                
                self.activated[:] = 2
                for _ in range(200):           
                    self.optimizer_sdf.zero_grad()      
                    self.activated[:] = 2
                    with torch.no_grad():
                        self.sdf_smooth_2[:] = self.sdf[:]
                    #laplacian.MeanCurve(self.sdf_smooth_2, self.sdf, 1, self.activated, M_vals, L_vals, L_outer, L_nonZeros, L_nnZ, L_outerSize, L_cols)
                    backprop_cuda.knn_smooth(self.tet32.sites.shape[0], self.knn, self.hlvl, self.sigma, self.sigma_feat, 1, self.tet32.sites, self.activated,
                                                self.grad_sdf_space, self.sdf.detach(), self.fine_features, self.tet32.knn_sites, self.sdf_smooth_2)
                    
                    self.grad_sdf_L2[:] = 0.0
                    backprop_cuda.sdf_smooth(self.tet32.sites.shape[0], 1, self.activated, self.sdf_smooth_2, self.sdf_smooth, self.grad_sdf_L2)

                    
                    start = timer()   
                    self.grad_sdf_smooth[:] = 0.0
                    #laplacian.MeanCurve(self.grad_sdf_smooth, self.grad_sdf_L2, 1, self.activated, M_vals, L_vals, L_outer, L_nonZeros, L_nnZ, L_outerSize, L_cols)
                    backprop_cuda.knn_smooth(self.tet32.sites.shape[0], self.knn, self.hlvl, self.sigma, self.sigma_feat, 1, self.tet32.sites, self.activated,
                                                self.grad_sdf_space, self.grad_sdf_L2, self.fine_features, self.tet32.knn_sites, self.grad_sdf_smooth)
                    
                    self.grad_sdf_L2[:] = 0.0
                    backprop_cuda.sdf_smooth(self.tet32.sites.shape[0], 1, self.activated, self.sdf, self.sdf_smooth, self.grad_sdf_L2)
                    #print(self.grad_sdf_smooth.mean())
                    #print(self.grad_sdf_L2.mean())
                    #print(abs(self.sdf - self.sdf_smooth).mean())

                    self.sdf.grad = self.grad_sdf_smooth + self.grad_sdf_L2
                    if verbose:
                        print('knn_smooth time:', timer() - start)

                    self.optimizer_sdf.step()

                #input()

                print("diff_tensor", self.weights_diff.mean())
                print("weights_tot_diff", self.weights_tot_diff.mean())
                
                self.activated[:] = 2
                self.sdf_smooth[:] = 0.0
                #laplacian.MeanCurve(self.sdf_smooth, self.sdf, 1, self.activated, M_vals, L_vals, L_outer, L_nonZeros, L_nnZ, L_outerSize, L_cols)

                #self.sdf_smooth_2[:] = 0.0
                #laplacian.MeanCurve(self.sdf_smooth_2, self.sdf_smooth, 1, self.activated, M_vals, L_vals, L_outer, L_nonZeros, L_nnZ, L_outerSize, L_cols)
                backprop_cuda.knn_smooth(self.tet32.sites.shape[0], self.knn, self.hlvl, self.sigma, self.sigma_feat, 1, self.tet32.sites,  self.activated,
                                        self.grad_sdf_space, self.sdf, self.fine_features, self.tet32.knn_sites, self.sdf_smooth)
                                
                #self.tet32.surface_from_sdf(self.sdf.detach().cpu().numpy().reshape(-1), "Exp/{}/meshes/test_tri_up_{}.ply".format(self.data_name, (iter_step+1))) #, self.dataset.scale_mats_np[0][:3, 3][None], self.dataset.scale_mats_np[0][:3, :3])
                                
                if self.double_net:
                    self.vortSDF_renderer_coarse_net.prepare_buffs(self.batch_size, self.n_samples, self.tet32.sites.shape[0])
                else:
                    self.vortSDF_renderer_coarse.prepare_buffs(self.batch_size, self.n_samples, self.tet32.sites.shape[0])
                self.vortSDF_renderer_fine.prepare_buffs(self.batch_size, self.n_samples, self.tet32.sites.shape[0])

                

                if (iter_step+1) == up_iters[0]:
                    warm_up = 500
                    self.s_start = 10 #30.0/(12.0*self.sigma) 
                    self.s_max = 100 #30.0/(4.0*self.sigma) 

                    self.learning_rate = self.learning_rate_list[1]
                    self.learning_rate_sdf = self.learning_rate_sdf_list[1]
                    self.learning_rate_feat = self.learning_rate_feat_list[1]
                    self.learning_rate_alpha = self.learning_rate_alpha_list[1]
                    self.learning_rate_cvt =  self.learning_rate_cvt_list[1]
                    self.s_w = self.s_w_list[1]
                    self.e_w =  self.e_w_list[1]
                    self.tv_w = self.tv_w_list[1]
                    self.tv_f = self.tv_f_list[1]
                    self.f_w = 0.0 #1.0

                    self.end_iter_loc = up_iters[1] - up_iters[0]
                    self.vortSDF_renderer_fine.mask_reg = self.mask_w_list[1]
                    if self.double_net:
                        self.vortSDF_renderer_coarse_net.mask_reg = self.mask_w_list[1]
                    #verbose = True
                    

                if (iter_step+1) == up_iters[1]:
                    warm_up = 2000
                    self.s_start = 30 #200 #30/(10.0*self.sigma) #50.0
                    self.s_max = 400 #600 #60/(5.0*self.sigma) #200
                    #self.s_start = 60.0/(10.0*self.sigma) 
                    #self.s_max = 60.0/(3.0*self.sigma) 

                    self.learning_rate = self.learning_rate_list[2]
                    self.learning_rate_sdf = self.learning_rate_sdf_list[2]
                    self.learning_rate_feat = self.learning_rate_feat_list[2]
                    self.learning_rate_alpha = self.learning_rate_alpha_list[2]
                    self.learning_rate_cvt =  self.learning_rate_cvt_list[2]
                    self.s_w = self.s_w_list[2]
                    self.e_w =  self.e_w_list[2]
                    self.tv_w = self.tv_w_list[2]
                    self.tv_f = self.tv_f_list[2]
                    self.f_w = 0.0 #1.0
                    #weight__fine = 1.0e1

                    self.end_iter_loc = up_iters[2] - up_iters[1]
                    self.vortSDF_renderer_fine.mask_reg = self.mask_w_list[2]
                    if self.double_net:
                        self.vortSDF_renderer_coarse_net.mask_reg = self.mask_w_list[2]
                    #self.dataset.gen_all_rays(3)
                    
                    
                if (iter_step+1) == up_iters[2]:
                    warm_up = 2000
                    self.s_start = 200 #400 #30/(10.0*self.sigma) #50.0
                    self.s_max = 1000 #2000 #60/(5.0*self.sigma) #200
                    #self.s_start = 60.0/(8.0*self.sigma) 
                    #self.s_max = 60.0/(2.0*self.sigma) 

                    self.learning_rate = self.learning_rate_list[3]
                    self.learning_rate_sdf = self.learning_rate_sdf_list[3]
                    self.learning_rate_feat = self.learning_rate_feat_list[3]
                    self.learning_rate_alpha = self.learning_rate_alpha_list[3]
                    self.learning_rate_cvt =  self.learning_rate_cvt_list[3]
                    self.s_w = self.s_w_list[3]
                    self.e_w =  self.e_w_list[3]
                    self.tv_w = self.tv_w_list[3]
                    self.tv_f = self.tv_f_list[3]
                    self.f_w = 0.0 #1.0
                    #weight__fine = 1.0e2

                    self.end_iter_loc = up_iters[3] - up_iters[2]
                    self.vortSDF_renderer_fine.mask_reg = self.mask_w_list[3]
                    if self.double_net:
                        self.vortSDF_renderer_coarse_net.mask_reg = self.mask_w_list[3]
                    #lamda_c = 0.2
                    #full_reg = 6
                    self.dataset.gen_all_rays(2)
                    

                if (iter_step+1) == up_iters[3]:
                    warm_up = 2000 #0.02*(up_iters[4] - up_iters[3])
                    self.s_start = 400 #1500# 30/(10.0*self.sigma) #50.0
                    self.s_max = 2000 #4000# 60/(5.0*self.sigma) #200
                    #self.s_start = 60.0/(8.0*self.sigma) 
                    #self.s_max = 60.0/(2.0*self.sigma) 

                    self.learning_rate = self.learning_rate_list[4]
                    self.learning_rate_sdf = self.learning_rate_sdf_list[4]
                    self.learning_rate_feat = self.learning_rate_feat_list[4]
                    self.learning_rate_alpha = self.learning_rate_alpha_list[4]
                    self.learning_rate_cvt =  self.learning_rate_cvt_list[4]
                    self.s_w = self.s_w_list[4]
                    self.e_w =  self.e_w_list[4]
                    self.tv_w = self.tv_w_list[4]
                    self.tv_f = self.tv_f_list[4]
                    self.f_w = 0.0 #1.0e0
                    #weight__fine = 1.0e2

                    self.end_iter_loc = up_iters[4] - up_iters[3]
                    self.vortSDF_renderer_fine.mask_reg = self.mask_w_list[4]
                    if self.double_net:
                        self.vortSDF_renderer_coarse_net.mask_reg = self.mask_w_list[4]
                    lamda_c = 0.5
                    full_reg = 6
                    self.dataset.gen_all_rays(1)
                    
                if (iter_step+1) == up_iters[4]:
                    warm_up = 2000 #0.02*(self.end_iter - up_iters[4])
                    self.s_start = 600 #3000#30/(10.0*self.sigma) #50.0
                    self.s_max = 4000 #8000#60/(5.0*self.sigma) #200
                    #self.s_start = 60.0/(8.0*self.sigma) 
                    #self.s_max = 60.0/(2.0*self.sigma) 

                    self.learning_rate = self.learning_rate_list[5]
                    self.learning_rate_sdf = self.learning_rate_sdf_list[5]
                    self.learning_rate_feat = self.learning_rate_feat_list[5]
                    self.learning_rate_alpha = self.learning_rate_alpha_list[5]
                    self.learning_rate_cvt =  self.learning_rate_cvt_list[5]
                    self.s_w = self.s_w_list[5]
                    self.e_w =  self.e_w_list[5]
                    self.tv_w = self.tv_w_list[5]
                    self.tv_f = self.tv_f_list[5]
                    #self.f_w = 1.0
                    #weight__fine = 1.0e2

                    self.end_iter_loc = self.end_iter - up_iters[4]
                    self.vortSDF_renderer_fine.mask_reg = self.mask_w_list[5]
                    if self.double_net:
                        self.vortSDF_renderer_coarse_net.mask_reg = self.mask_w_list[5]
                    lamda_c = 0.0
                    full_reg = 12
                    self.dataset.gen_all_rays(1)
                    #self.dataset.gen_all_rays_masked(1)


                if False: #(iter_step+1) == up_iters[2]:
                    """self.fine_features = 0.5*torch.ones([self.sdf.shape[0], self.dim_feats]).cuda()       
                    self.fine_features = self.fine_features.contiguous()
                    self.fine_features.requires_grad_(True)"""

                    if self.double_net:
                        self.color_coarse = ColorNetwork(73, 128, 4, 3.0).to(self.device)
                        #self.color_coarse = ColorNetwork(**self.conf['model.color_geo_network']).to(self.device)

                    self.color_network = ColorNetwork(76, 128, 4, 3.0).to(self.device)
                    #self.color_network = ColorNetwork(**self.conf['model.color_network']).to(self.device)
                    
                    params_to_train = []
                    params_to_train += list(self.color_network.parameters())
                    if self.double_net:
                        params_to_train += list(self.color_coarse.parameters())
                    self.optimizer = torch.optim.Adam(params_to_train, lr=self.learning_rate)
                    
                if False: #(iter_step+1) == up_iters[3]:
                    """self.fine_features = 0.5*torch.ones([self.sdf.shape[0], self.dim_feats]).cuda()       
                    self.fine_features = self.fine_features.contiguous()
                    self.fine_features.requires_grad_(True)"""

                    if self.double_net:
                        self.color_coarse = ColorNetwork(55, 192, 4, 3.0).to(self.device)
                        #self.color_coarse = ColorNetwork(**self.conf['model.color_geo_network']).to(self.device)

                    self.color_network = ColorNetwork(58, 192, 4, 3.0).to(self.device)
                    #self.color_network = ColorNetwork(**self.conf['model.color_network']).to(self.device)
                    
                    params_to_train = []
                    params_to_train += list(self.color_network.parameters())
                    if self.double_net:
                        params_to_train += list(self.color_coarse.parameters())
                    self.optimizer = torch.optim.Adam(params_to_train, lr=self.learning_rate)

                self.optimizer_sdf = torch.optim.Adam([self.sdf], lr=self.learning_rate_sdf, betas=(0.9, 0.98))        
                self.optimizer_feat = torch.optim.Adam([self.fine_features], lr=self.learning_rate_feat, betas=(0.9, 0.98))  



                print("SIGMA => ", self.sigma)
                #with torch.no_grad():
                #    self.sdf[:] = self.sdf[:] + self.sigma
                
                self.loc_iter = 0
                
                self.save_checkpoint()                    
                torch.cuda.empty_cache()

                #verbose = True
                #self.tet32.save("Exp/bmvs_man/test_up.ply") 
                #self.tet32.save_multi_lvl("Exp/{}/multi_lvl".format(self.data_name))    
                #self.render_image(cam_ids, img_idx)
                #self.tet32.surface_from_sdf(self.sdf.detach().cpu().numpy().reshape(-1), "Exp/bmvs_man/test_tri_up_{}.ply".format(iter_step), self.dataset.scale_mats_np[0][:3, 3][None], self.dataset.scale_mats_np[0][0, 0])                          
                #torch.cuda.empty_cache()

            if (iter_step+1) % self.report_freq == 0:
                lapl_loss = self.div# ((lbda + mu)*self.div - lbda*mu*self.div_2)
                print('\niter:{:8>d} loss = {}, loss_coarse = {}, scale={}, grad={}, grad_feat = {}, grad_nrm = {}, grad_tv_feat = {} lr={}'.format(iter_step, color_fine_loss.sum(), color_coarse_loss.sum(), self.inv_s, abs(self.grad_sdf[abs(self.grad_sdf) > 0.0]).mean(), abs(self.grad_features[abs(self.grad_features) > 0.0]).mean(), abs(self.grad_norm_smooth[abs(self.grad_norm_smooth) > 0.0]).mean(), abs(self.grad_feat_smooth[abs(self.grad_feat_smooth) > 0.0]).mean(), self.optimizer.param_groups[0]['lr']))
                print('iter:{:8>d} learning_rate_alpha:{} grad_lapl={}, grad_sdf_L2 = {}'.format(iter_step, self.learning_rate_alpha, abs(lapl_loss[abs(lapl_loss) > 0.0]).mean(), abs(self.grad_sdf_L2[abs(self.grad_sdf_L2) > 0.0]).mean()))
                print('iter:{:8>d} s_w = {}, e_w = {}, tv_w = {}, nb_samples={}, sigma={}, w_g={}, full_reg={}, lr={}'.format(iter_step, self.s_w, self.e_w, self.tv_w, nb_samples, self.sigma, self.w_g, full_reg, self.optimizer_sdf.param_groups[0]['lr']))
                print('iter:{:8>d} eik_loss = {}'.format(iter_step, eik_loss))
                

            #if verbose and iter_step % self.val_freq == 0:
            if (iter_step+1) % self.val_freq == 0:
                #self.inv_s = 1000     
                #self.render_image(img_idx)

                #self.tet32.make_clipped_CVT(self.sdf.detach(), self.grad_sdf_space, self.visual_hull, "Exp/{}/meshes/test_CVT_{}.obj".format(self.data_name, iter_step), self.dataset.scale_mats_np[0][:3, 3][None], self.dataset.scale_mats_np[0][0, 0])

                self.tet32.surface_from_sdf(self.sdf_smooth.detach().cpu().numpy().reshape(-1), "Exp/{}/meshes/test_tri_{}.ply".format(self.data_name, iter_step), self.dataset.scale_mats_np[0][:3, 3][None], self.dataset.scale_mats_np[0][:3, :3])
                #if iter_step % 3 == 0:
                
                """self.div[:] = 0.0
                laplacian.MeanCurve(self.div, self.sdf, 1, self.activated, M_vals, L_vals, L_outer, L_nonZeros, L_nnZ, L_outerSize, L_cols)
                #self.div_2[:] = 0.0
                #laplacian.MeanCurve(self.div_2, self.div, 1, self.activated, M_vals, L_vals, L_outer, L_nonZeros, L_nnZ, L_outerSize, L_cols)
                #self.div[self.cam_ids[:] ] = 0.0
                self.div[outside_flag[:] == 1] = 0.0
                #self.div[self.mask_background[:] == 1] = 0.0
                with torch.no_grad():
                    self.sdf_smooth[:] = self.div[:] # self.sdf[:] + 1.0e-5*self.div #((lbda + mu)*self.div - lbda*mu*self.div_2)
                    #self.sdf_smooth[:] = self.sdf[:] + 0.001*((lbda + mu)*self.div - lbda*mu*self.div_2)"""
                self.tet32.surface_from_sdf(self.sdf.detach().cpu().numpy().reshape(-1), "Exp/{}/meshes/test_tri_raw.ply".format(self.data_name), self.dataset.scale_mats_np[0][:3, 3][None], self.dataset.scale_mats_np[0][:3, :3])  

                #self.tet32.surface_from_sdf(self.sdf_smooth_2.cpu().numpy().reshape(-1), "Exp/{}/meshes/test_tri_smooth.ply".format(self.data_name), self.dataset.scale_mats_np[0][:3, 3][None], self.dataset.scale_mats_np[0][:3, :3])                               
                
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

            if self.loc_iter == round(0.7*self.end_iter_loc):
                self.s_w = 5e-3
                #self.e_w =  1e-3
                self.tv_w = 1e-2
                #self.tv_f = 5e-3
                """self.s_start = 4*self.s_start
                self.s_max = 4*self.s_max
                with torch.no_grad():
                    self.sdf[:] = self.sdf[:] * 4.096"""

            """if iter_step > up_iters[3] and self.loc_iter == round(0.7*self.end_iter_loc): 
                self.s_w = 5e-4
                #self.e_w =  1e-3
                #self.tv_w = 1e-4"""

                
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
            
            if self.loc_iter % 100 == 0:
                torch.cuda.empty_cache()

            self.update_learning_rate(self.loc_iter)
            self.loc_iter = self.loc_iter + 1

            


        ############# Final CVT optimization ######################
        if False:
            self.sdf, self.fine_features, self.mask_background = self.tet32.upsample(self.sdf.detach().cpu().numpy(), 
                                                                                     self.fine_features.detach().cpu().numpy(), 
                                                                                     self.visual_hull, res, cam_sites, self.learning_rate_cvt, 
                                                                                     False, self.sigma)
            self.sdf = self.sdf.contiguous()
            self.fine_features = self.fine_features.contiguous()
            self.tet32.load_cuda()

            sites = np.asarray(self.tet32.vertices)  
            cam_ids = np.stack([np.where((sites == cam_sites[i,:]).all(axis = 1))[0] for i in range(cam_sites.shape[0])]).reshape(-1)
            self.tet32.make_adjacencies(cam_ids)

            self.tet32.make_multilvl_knn()

            cam_ids = torch.from_numpy(cam_ids).int().cuda()
            self.cam_ids = cam_ids
            
            outside_flag = np.zeros(sites.shape[0], np.int32)
            outside_flag[sites[:,0] < self.visual_hull[0] + (self.visual_hull[3]-self.visual_hull[0])/(res)] = 1
            outside_flag[sites[:,1] < self.visual_hull[1] + (self.visual_hull[4]-self.visual_hull[1])/(res)] = 1
            outside_flag[sites[:,2] < self.visual_hull[2] + (self.visual_hull[5]-self.visual_hull[2])/(res)] = 1
            outside_flag[sites[:,0] > self.visual_hull[3] - (self.visual_hull[3]-self.visual_hull[0])/(res)] = 1
            outside_flag[sites[:,1] > self.visual_hull[4] - (self.visual_hull[4]-self.visual_hull[1])/(res)] = 1
            outside_flag[sites[:,2] > self.visual_hull[5] - (self.visual_hull[5]-self.visual_hull[2])/(res)] = 1
            
            self.tet32.sites = torch.from_numpy(sites.astype(np.float32)).cuda()
            self.tet32.sites = self.tet32.sites.contiguous()
            self.tet32.sites.requires_grad_(True)

            self.Allocate_data()
            
            cvt_grad_cuda.diff_tensor(self.tet32.nb_tets, self.tet32.summits, self.tet32.sites, self.vol_tet32, self.weights_diff, self.weights_tot_diff)
            self.grad_sdf_space[:] = 0.0
            self.weights_grad[:] = 0.0
            self.grad_eik[:] = 0.0
            self.grad_norm_smooth[:] = 0.0
            self.eik_loss[:] = 0.0

            cvt_grad_cuda.eikonal_grad(self.tet32.nb_tets, self.tet32.sites.shape[0], self.tet32.summits, self.tet32.sites, self.activated, self.sdf.detach(), self.sdf.detach(), self.fine_features.detach(), 
                                        self.grad_eik, self.grad_norm_smooth, self.grad_sdf_space, self.vol_tet32, self.weights_diff, self.weights_tot_diff, self.eik_loss)
            
            self.norm_grad = torch.linalg.norm(self.grad_sdf_space, ord=2, axis=-1, keepdims=True).reshape(-1, 1)
            self.norm_grad[self.norm_grad == 0.0] = 1.0
            self.unormed_grad[:] = self.grad_sdf_space[:]
            self.grad_sdf_space = self.grad_sdf_space / self.norm_grad.expand(-1, 3)
            
        
        for id_im in tqdm(range(runner.dataset.n_images)):
            runner.render_image(img_idx = id_im)

        #self.render_image(cam_ids, 0)
        self.save_checkpoint()  
        self.tet32.surface_from_sdf(self.sdf.detach().cpu().numpy().reshape(-1), os.path.join(self.base_exp_dir, 'final_MT.ply'), self.dataset.scale_mats_np[0][:3, 3][None], self.dataset.scale_mats_np[0][:3, :3])        
        #self.tet32.make_clipped_CVT(self.sdf.detach(), 2*self.sigma, self.grad_sdf_space, self.visual_hull,  "Exp/{}/final_CVT.obj".format(self.data_name), self.dataset.scale_mats_np[0][:3, 3][None], self.dataset.scale_mats_np[0][0, 0])
        
        self.activated[:] = 2
        with torch.no_grad():
            self.sdf_smooth[:] = 0.0
        backprop_cuda.knn_smooth(self.tet32.sites.shape[0], self.knn, self.hlvl, self.sigma, self.sigma_feat, 1, self.tet32.sites,  self.activated,
                                        self.grad_sdf_space, self.sdf, self.fine_features, self.tet32.knn_sites, self.sdf_smooth)
            
        self.grad_sdf_space[:] = 0.0
        cvt_grad_cuda.eikonal_grad(self.tet32.nb_tets, self.tet32.sites.shape[0], self.tet32.summits, self.tet32.valid_tets, self.tet32.sites.float(), self.activated, self.sdf_smooth, self.sdf_smooth_2, self.fine_features.detach(), 
                                            self.grad_eik, self.grad_norm_smooth, self.grad_sdf_space, self.vol_tet32, self.weights_diff, self.weights_tot_diff, self.eik_loss)
        self.norm_grad = torch.linalg.norm(self.grad_sdf_space, ord=2, axis=-1, keepdims=True).reshape(-1, 1)
        self.norm_grad[self.norm_grad == 0.0] = 1.0
        self.unormed_grad[:] = self.grad_sdf_space[:]
        self.grad_sdf_space = self.grad_sdf_space / self.norm_grad.expand(-1, 3)
    
        self.tet32.surface_from_sdf(self.sdf_smooth.detach().cpu().numpy().reshape(-1), os.path.join(self.base_exp_dir, 'final_MT_smooth.ply'), self.dataset.scale_mats_np[0][:3, 3][None], self.dataset.scale_mats_np[0][:3, :3])
        #self.tet32.make_clipped_CVT(self.sdf_smooth, 2*self.sigma, self.grad_sdf_space, self.visual_hull, "Exp/{}/final_CVT_smooth.obj".format(self.data_name), self.dataset.scale_mats_np[0][:3, 3][None], self.dataset.scale_mats_np[0][0, 0])
        
        
        #self.tet32.clipped_cvt(self.sdf.detach(), self.fine_features.detach(), outside_flag, 
        #                       cam_ids, self.learning_rate_cvt, "Exp/bmvs_man/clipped_CVT.ply")

    @torch.no_grad()
    def render_image(self, img_idx = 0, iter_step = 0, resolution_level = 1):
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
        
        img_nrm = torch.zeros([3*(self.dataset.H // resolution_level) * (self.dataset.W // resolution_level)], dtype = torch.float32).cuda()
        img_nrm = img_nrm.contiguous()
        img_nrm[:] = 0
        
        img_mask = torch.zeros([(self.dataset.H // resolution_level) * (self.dataset.W // resolution_level)], dtype = torch.float32).cuda()
        img_mask = img_mask.contiguous()
        img_mask[:] = 0
        
        self.grad_sdf_space[:] = 0.0
        self.weights_grad[:] = 0.0
        self.grad_eik[:] = 0.0
        self.grad_norm_smooth[:] = 0.0
        self.eik_loss[:] = 0.0
        self.activated[:] = 2
        
        with torch.no_grad():
            self.sdf_smooth[:] = 0.0
        backprop_cuda.knn_smooth(self.tet32.sites.shape[0], self.knn, self.hlvl, self.sigma, self.sigma_feat, 1, self.tet32.sites,  self.activated,
                                self.grad_sdf_space, self.sdf, self.fine_features, self.tet32.knn_sites, self.sdf_smooth)

        cvt_grad_cuda.eikonal_grad(self.tet32.nb_tets, self.tet32.sites.shape[0], self.tet32.summits, self.tet32.valid_tets, self.tet32.sites.float(), self.activated, self.sdf_smooth, self.sdf_smooth, self.fine_features.detach(), 
                                            self.grad_eik, self.grad_norm_smooth, self.grad_sdf_space, self.vol_tet32, self.weights_diff, self.weights_tot_diff, self.eik_loss)
        
        norm_grads = torch.linalg.norm(self.grad_sdf_space, ord=2, axis=-1, keepdims=True).reshape(-1, 1)
        norm_grads[norm_grads[:] == 0.0] = 1.0
        self.grad_sdf_space = self.grad_sdf_space / norm_grads.expand(-1, 3)

        colors_out = torch.zeros([self.batch_size*3]).to(torch.device('cuda')).contiguous()
        colors_out_coarse = torch.zeros([self.batch_size*3]).to(torch.device('cuda')).contiguous()
        nrm_out = torch.zeros([self.batch_size*3]).to(torch.device('cuda')).contiguous()  
        mask_out = torch.zeros([self.batch_size]).to(torch.device('cuda')).contiguous()
        it = 0
        for rays_o_batch, rays_d_batch in zip(rays_o.split(self.batch_size), rays_d.split(self.batch_size)):
            rays_o_batch = rays_o_batch.contiguous()
            rays_d_batch = rays_d_batch.contiguous()

            ## sample points along the rays
            start = timer()
            self.offsets[:] = 0
            self.activated[:] = 1
            img_ids = img_idx*torch.ones((rays_o_batch.shape[0], 1)).cuda().int()
            nb_samples = tet32_march_cuda.tet32_march_count(self.inv_s, rays_o_batch.shape[0], rays_d_batch, self.tet32.sites.float(), self.sdf_smooth, self.tet32.summits, self.tet32.neighbors, self.tet32.valid_tets, img_ids, 
                                               self.cam_ids, self.tet32.offsets_cam, self.tet32.cam_tets, self.activated, self.offsets)
            #print(nb_samples)

            #nb_samples = self.tet32.sample_rays_cuda(self.inv_s, img_ids, rays_d_batch, self.sdf_smooth, self.cam_ids, self.in_weights, self.in_z, self.in_sdf, self.in_ids, self.offsets, self.activated, self.n_samples)    
            #nb_samples = self.tet32.sample_rays_cuda(0.01, self.inv_s, self.sigma, img_idx, rays_d_batch, self.sdf, self.fine_features, cam_ids, self.in_weights, self.in_z, self.in_sdf, self.in_feat, self.in_ids, self.offsets, self.activated, self.n_samples)    
                
            start = timer()
            #self.offsets[self.offsets[:,1] == -1] = 0                         
            start = timer()
            self.samples[:] = 0.0
            self.out_grads[:] = 0.0
            tet32_march_cuda.tet32_march_offset(self.inv_s, rays_o_batch.shape[0], rays_d_batch, self.tet32.sites.float(), self.sdf_smooth, self.tet32.summits, self.tet32.neighbors, self.tet32.valid_tets, img_ids, 
                                                self.cam_ids, self.tet32.offsets_cam, self.tet32.cam_tets, self.grad_sdf_space, self.fine_features.detach(),  
                                                self.out_weights, self.out_z, self.out_sdf, self.out_ids, self.out_grads, self.out_feat, self.samples_rays, self.samples_reff, self.samples, 
                                                self.offsets)
            """tet32_march_cuda.fill_samples(rays_o_batch.shape[0], self.n_samples, rays_o_batch, rays_d_batch, self.tet32.sites, 
                                        self.in_z, self.in_sdf, self.fine_features, self.in_weights, self.grad_sdf_space, self.in_ids, 
                                        self.out_z, self.out_sdf, self.out_feat, self.out_weights, self.out_grads, self.out_ids, 
                                        self.offsets, self.samples, self.samples_rays)"""
                        
            #samples = (self.samples[:nb_samples,:] + self.samples_loc[:nb_samples,:])/2.0
            samples = (self.samples[:nb_samples] + 1.1)/2.2
            samples = samples.contiguous()
            #samples = (samples + 1.1)/2.2
            
            ##### ##### ##### ##### ##### ##### 
            xyz_emb = (samples.unsqueeze(-1) * self.posfreq).flatten(-2)
            xyz_emb = torch.cat([samples, xyz_emb.sin(), xyz_emb.cos()], -1)

            viewdirs_emb = (self.samples_rays[:nb_samples,:].unsqueeze(-1) * self.viewfreq).flatten(-2)
            viewdirs_emb = torch.cat([self.samples_rays[:nb_samples,:], viewdirs_emb.sin(), viewdirs_emb.cos()], -1)
           
            
            if self.double_net:
                coarse_feat = torch.cat([xyz_emb, viewdirs_emb, self.out_sdf[:nb_samples,:2], self.out_grads[:nb_samples,:3], self.out_feat[:nb_samples,:8]], -1)     
                #coarse_feat = torch.cat([xyz_emb, viewdirs_emb, self.out_sdf[:nb_samples,:2], self.out_grads[:nb_samples,:3]], -1)      
                colors_feat = self.color_coarse.rgb(coarse_feat)   
                self.colors = torch.sigmoid(colors_feat)
                self.colors = self.colors.contiguous()

            
            norm_ref = torch.linalg.norm(self.samples_reff, ord=2, axis=-1, keepdims=True).reshape(-1, 1)
            norm_ref[norm_ref == 0.0] = 1.0
            self.samples_reff = self.samples_reff / norm_ref.expand(-1, 3)

            xyz_emb_fine = (self.samples[:nb_samples,:].unsqueeze(-1) * self.k_posfreq).flatten(-2)
            xyz_emb_fine = torch.cat([self.samples[:nb_samples,:], xyz_emb_fine.sin(), xyz_emb_fine.cos()], -1)

            viewdirs_emb_fine = (self.samples_reff[:nb_samples,:].unsqueeze(-1) * self.k_viewfreq).flatten(-2)
            viewdirs_emb_fine = torch.cat([self.samples_reff[:nb_samples,:], viewdirs_emb_fine.sin(), viewdirs_emb_fine.cos()], -1)  

            if self.double_net:
                if self.position_encoding:
                    rgb_feat = torch.cat([xyz_emb_fine, viewdirs_emb_fine, colors_feat, self.out_sdf[:nb_samples,:2], self.out_grads[:nb_samples,:3], self.out_feat[:nb_samples,8:self.dim_feats]], -1)
                else:
                    rgb_feat = torch.cat([viewdirs_emb, colors_feat.detach(), self.out_grads[:nb_samples,:3], self.out_feat[:nb_samples,:]], -1) #, self.out_weights[:nb_samples,:]
            else:
                if self.position_encoding:
                    rgb_feat = torch.cat([xyz_emb, viewdirs_emb, self.out_sdf[:nb_samples,:2], self.out_grads[:nb_samples,:3], self.out_feat[:nb_samples,:self.dim_feats]], -1)
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

            if nb_samples == 0 or self.colors_fine.shape[0] == 0:
                start = 3*it*self.batch_size
                end = min(3*(it+1)*self.batch_size, 3*(self.dataset.H // resolution_level) * (self.dataset.W // resolution_level))
                #print(start)
                #print(end)
                """img[start:end,:] = 0.0
                if self.double_net:
                    img_coarse[start:end, :] = 0.0
                img_nrm[start:end, :] = 0.0

                start = it*self.batch_size
                end = min((it+1)*self.batch_size, (self.dataset.H // resolution_level) * (self.dataset.W // resolution_level))           
                img_mask[start:end,:] = 1.0"""           
                
                it = it + 1
                continue

            """print(" Col min", self.colors_fine.min())
            print("mean", self.colors_fine.mean())
            print("min", self.out_grads[:nb_samples,:3].min())
            print("norm mean", self.out_grads[:nb_samples,:3].mean())"""

            ########################################
            ####### Render the image ###############
            ########################################
            norm_map = (1.0+self.out_grads[:nb_samples,:3])/2.0 #(1.0+(self.out_grads[:nb_samples,:3] + self.out_grads[:nb_samples,3:6])/2.0)/2.0
            renderer_cuda.render_no_grad(rays_o_batch.shape[0], self.inv_s, self.out_sdf, self.colors_fine, self.offsets, colors_out, mask_out)
            if self.double_net:
                #self.colors[:] = 0.5
                renderer_cuda.render_no_grad(rays_o_batch.shape[0], self.inv_s, self.out_sdf, self.colors, self.offsets, colors_out_coarse, mask_out)
            renderer_cuda.render_no_grad(rays_o_batch.shape[0], self.inv_s, self.out_sdf, norm_map, self.offsets, nrm_out, mask_out)

            start = 3*it*self.batch_size
            end = min(3*(it+1)*self.batch_size, 3*(self.dataset.H // resolution_level) * (self.dataset.W // resolution_level))
            img[start:end] = colors_out[:(end-start)]
            if self.double_net:
                img_coarse[start:end] = colors_out_coarse[:(end-start)]
            img_nrm[start:end] = nrm_out[:(end-start)]

            start = it*self.batch_size
            end = min((it+1)*self.batch_size, (self.dataset.H // resolution_level) * (self.dataset.W // resolution_level))           
            img_mask[start:end] = mask_out[:(end-start)]            
            
            it = it + 1

        mask = img_mask.reshape(-1,1)

        img = img.reshape(self.dataset.H // resolution_level, self.dataset.W // resolution_level, 3)
        img = img.cpu().numpy()
        cv2.imwrite('Exp/{}/validations_fine/cam-{}.png'.format(self.data_name, img_idx+1), 255*img[:,:])
        
        if self.double_net:
            img_coarse = img_coarse.reshape(self.dataset.H // resolution_level, self.dataset.W // resolution_level, 3)
            img_coarse = img_coarse.cpu().numpy()
            cv2.imwrite('Exp/{}/validations_coarse/img_coarse{}.png'.format(self.data_name, img_idx+1), 255*img_coarse[:,:])
        
        #cv2.imwrite('Exp/img_diff.png', 10*255*abs(img_coarse[:,:]-img[:,:]))        

        img_nrm = img_nrm.reshape(self.dataset.H // resolution_level, self.dataset.W // resolution_level, 3)
        img_nrm = img_nrm.cpu().numpy()
        cv2.imwrite('Exp/{}/validations_nrm/img_nrm{}.png'.format(self.data_name, img_idx+1), 255*img_nrm[:,:])

        print("rendering done image {}".format(img_idx))
        
    def Allocate_data(self, K_NN = 24):        
        self.activated_buff = torch.zeros(self.tet32.sites.shape[0], dtype=torch.int32).cuda().contiguous()     
        self.activated = torch.zeros(self.tet32.sites.shape[0], dtype=torch.int32).cuda().contiguous()      
        self.visible = torch.zeros(self.tet32.sites.shape[0], dtype=torch.float32).cuda().contiguous()      

        self.sdf_smooth = torch.zeros([self.sdf.shape[0]]).float().cuda().contiguous()  
        self.sdf_smooth_2 = torch.zeros([self.sdf.shape[0]]).float().cuda().contiguous()  
        self.counter = torch.zeros([self.sdf.shape[0]]).float().cuda().contiguous()   

        self.weight_sdf_smooth = torch.zeros([self.sdf.shape[0]]).float().cuda().contiguous()  

        self.feat_smooth = torch.zeros([self.sdf.shape[0], self.dim_feats]).float().cuda().contiguous()  
        
        self.grad_features = torch.zeros([self.sdf.shape[0], self.dim_feats]).float().cuda().contiguous()  
        self.grad_norm_feat = torch.zeros([self.sdf.shape[0], 12]).float().cuda().contiguous()  
        self.grad_norm = torch.zeros([self.tet32.sites.shape[0], 3]).float().cuda().contiguous()
        self.grad_feat_smooth = torch.zeros([self.sdf.shape[0], self.dim_feats]).float().cuda().contiguous()

        self.grad_sdf_smooth = torch.zeros([self.sdf.shape[0]]).float().cuda().contiguous()  

        self.grad_sdf = torch.zeros([self.sdf.shape[0]]).float().cuda().contiguous()  
        self.grad_sdf_net = torch.zeros([self.sdf.shape[0]]).float().cuda().contiguous()          
        self.grad_sdf_norm = torch.zeros([self.sdf.shape[0]]).float().cuda().contiguous()         
        self.grad_sdf_L2 = torch.zeros([self.sdf.shape[0]]).float().cuda().contiguous()           

        self.div = torch.zeros(self.sdf.shape[0]).float().cuda().contiguous() 
        self.div_2 = torch.zeros(self.sdf.shape[0], 3).float().cuda().contiguous() 
        self.div_feat = torch.zeros([self.sdf.shape[0], self.dim_feats]).float().cuda().contiguous() 
        #self.div_norm[:] = 0.0
        #self.grad_norm_Lapl = torch.zeros([self.sdf.shape[0]]).float().cuda().contiguous()      

        
        self.norm_smooth = torch.zeros([self.tet32.sites.shape[0], 3]).float().cuda().contiguous()   
        self.grad_norm_L2 = torch.zeros([self.tet32.sites.shape[0], 3]).float().cuda().contiguous()   
        self.grad_norm_sdf_L2 = torch.zeros(self.sdf.shape[0]).float().cuda().contiguous()    

        
        #print("sdf related")
        #input()

        self.grad_sdf_space = torch.zeros([self.tet32.sites.shape[0], 3]).float().cuda().contiguous()
        self.unormed_grad = torch.zeros([self.tet32.sites.shape[0], 3]).float().cuda().contiguous()
        #self.grad_feat_space = torch.zeros([self.tet32.sites.shape[0], 3, self.dim_feats]).float().cuda().contiguous()
        self.weights_grad = torch.zeros([self.tet32.sites.shape[0], 1]).float().cuda().contiguous()
        self.eik_loss = torch.zeros([self.tet32.sites.shape[0], 1]).float().cuda().contiguous()

        self.grad_eik = torch.zeros([self.tet32.sites.shape[0]]).float().cuda().contiguous() 
        self.grad_norm_smooth = torch.zeros([self.tet32.sites.shape[0]]).float().cuda().contiguous() 
        
        self.div_norm = torch.zeros([self.tet32.sites.shape[0], 3]).float().cuda().contiguous() 

        #print("sites related")
        #input()

        self.vol_tet32 = torch.zeros([self.tet32.nb_tets]).float().cuda().contiguous() 
        self.weights_diff = torch.zeros([12*self.tet32.nb_tets]).float().cuda().contiguous() 
        self.weights_tot_diff = torch.zeros([self.tet32.sites.shape[0]]).float().cuda().contiguous() 
        #print("tet related")
        #input()
        
    def Allocate_batch_data(self, K_NN = 24):
        self.samples = torch.zeros([self.n_samples * self.batch_size, 3], dtype=torch.float32).cuda()
        self.samples = self.samples.contiguous()
        
        self.colors = torch.zeros([self.n_samples * self.batch_size, 3], dtype=torch.float32).cuda()
        self.colors = self.colors.contiguous()
        
        self.colors_fine = torch.zeros([self.n_samples * self.batch_size, 3], dtype=torch.float32).cuda()
        self.colors_fine = self.colors_fine.contiguous()
        
        self.samples_rays = torch.zeros([self.n_samples * self.batch_size, 3], dtype=torch.float32).cuda()
        self.samples_rays = self.samples_rays.contiguous()
        
        self.samples_reff = torch.zeros([self.n_samples * self.batch_size, 3], dtype=torch.float32).cuda()
        self.samples_reff = self.samples_reff.contiguous()
        
        """self.in_weights = torch.zeros([6*self.n_samples * self.batch_size], dtype=torch.float32).cuda()
        self.in_weights = self.in_weights.contiguous()

        self.in_ids = -torch.ones([6*self.n_samples* self.batch_size], dtype=torch.int32).cuda()
        self.in_ids = self.in_ids.contiguous()
        
        self.in_z = torch.zeros([2*self.n_samples * self.batch_size], dtype=torch.float32).cuda()
        self.in_z = self.in_z.contiguous()
        
        self.in_sdf = torch.zeros([2*self.n_samples * self.batch_size], dtype=torch.float32).cuda()
        self.in_sdf = self.in_sdf.contiguous()"""        
        
        self.out_ids = -torch.ones([self.n_samples* self.batch_size, 6], dtype=torch.int32).cuda()
        self.out_ids = self.out_ids.contiguous()
        
        self.out_z = torch.zeros([self.n_samples * self.batch_size,2], dtype=torch.float32).cuda()
        self.out_z = self.out_z.contiguous()
        
        self.out_sdf = torch.zeros([self.n_samples * self.batch_size, 4], dtype=torch.float32).cuda()
        self.out_sdf = self.out_sdf.contiguous()
        
        self.out_feat = torch.zeros([self.n_samples * self.batch_size, self.dim_feats], dtype=torch.float32).cuda()
        self.out_feat = self.out_feat.contiguous()
        
        self.out_weights = torch.zeros([self.n_samples * self.batch_size, 7], dtype=torch.float32).cuda()
        self.out_weights = self.out_weights.contiguous()
        
        self.out_grads = torch.zeros([self.n_samples * self.batch_size, 3], dtype=torch.float32).cuda().contiguous()

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

        self.fine_features_grad = torch.zeros([self.n_samples * self.batch_size, self.dim_feats]).float().cuda().contiguous()   
        self.norm_features_grad = torch.zeros([self.n_samples * self.batch_size, 3]).float().cuda().contiguous()  
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
        if self.double_net:
            checkpoint = {
                'color_geo_network': self.color_coarse.state_dict(),
                'color_fine_network': self.color_network.state_dict(),
            }
        else :
            checkpoint = {
                'color_fine_network': self.color_network.state_dict(),
            }

        os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)
        torch.save(checkpoint, os.path.join(self.base_exp_dir, 'checkpoints', 'ckpt_{:0>6d}.pth'.format(self.iter_step)))
        np.save(os.path.join(self.base_exp_dir, 'checkpoints', 'sdf_{:0>6d}.npy'.format(self.iter_step)), self.sdf.detach().cpu().numpy())
        np.save(os.path.join(self.base_exp_dir, 'checkpoints', 'features_{:0>6d}.npy'.format(self.iter_step)), self.fine_features.detach().cpu().numpy())
        np.save(os.path.join(self.base_exp_dir, 'checkpoints', 'sites_{:0>6d}.npy'.format(self.iter_step)), self.tet32.sites.cpu().numpy())
        for lvl_curr in range(self.tet32.lvl):
            np.save(os.path.join(self.base_exp_dir, 'checkpoints', 'sites_{:0>6d}_lvl{}.npy'.format(self.iter_step, lvl_curr)), self.tet32.lvl_sites[lvl_curr])

    def load_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name), map_location=self.device)
        self.color_coarse.load_state_dict(checkpoint['color_geo_network'])
        self.color_network.load_state_dict(checkpoint['color_fine_network'])

        print(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name))
        self.iter_step = int(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name).split("_")[-1].split(".")[0])
        print(self.iter_step)

        sites = np.load(os.path.join(self.base_exp_dir, 'checkpoints', 'sites_{:0>6d}.npy'.format(self.iter_step)))
        #self.tet32.sites = torch.from_numpy(self.tet32.sites.astype(np.float32)).cuda()

        sites_list = sorted(glob(os.path.join(os.path.join(self.base_exp_dir, 'checkpoints', 'sites_{:0>6d}_lvl*'.format(self.iter_step)))))
        print(sites_list)
        
        self.tet32 = tet32.Tet32(sites, 0, self.knn, self.hlvl)
        self.tet32.lvl = len(sites_list)-1

        self.tet32.lvl_sites = []
        for lvl_curr in range(self.tet32.lvl+1):
            self.tet32.lvl_sites.append(np.load(sites_list[lvl_curr]))

        self.sdf = np.load(os.path.join(self.base_exp_dir, 'checkpoints', 'sdf_{:0>6d}.npy'.format(self.iter_step)))
        self.sdf = torch.from_numpy(self.sdf.astype(np.float32)).cuda()
        self.sdf.requires_grad_(True)

        self.inv_s = 2000.0
        
        self.fine_features = np.load(os.path.join(self.base_exp_dir, 'checkpoints', 'features_{:0>6d}.npy'.format(self.iter_step)))
        self.fine_features = torch.from_numpy(self.fine_features.astype(np.float32)).cuda()
        self.fine_features.requires_grad_(True)
                
        self.optimizer_sdf = torch.optim.Adam([self.sdf], lr=self.learning_rate_sdf)        
        self.optimizer_feat= torch.optim.Adam([self.fine_features], lr=self.learning_rate_feat)
        
        #self.optimizer_sdf.load_state_dict(checkpoint['optimizer_sdf'])
        #self.optimizer_feat.load_state_dict(checkpoint['optimizer_feat'])
        #exit()

        self.sigma = 0.1
        self.sigma_start = 0.002
        self.sigma_max = 0.0015
        
        if (self.iter_step+1) >= up_iters[0]:
            self.sigma = self.sigma/2.0
            
        if (self.iter_step+1) >= up_iters[1]:
            self.sigma = self.sigma/2.0
            
        if (self.iter_step+1) >= up_iters[2]:
            self.sigma = self.sigma/2.0
            
        if (self.iter_step+1) >= up_iters[3]:
            self.sigma = self.sigma/2.0
            
        if (self.iter_step+1) >= up_iters[4]:
            self.sigma = self.sigma/2.0

        if (self.iter_step+1) == up_iters[0]:
            self.s_start = 50 #30/(10.0*self.sigma) #50.0
            self.s_max = 200 #60/(5.0*self.sigma) #200

            self.learning_rate = self.learning_rate_list[1]
            self.learning_rate_sdf = self.learning_rate_sdf_list[1]
            self.learning_rate_feat = self.learning_rate_feat_list[1]
            self.learning_rate_alpha = self.learning_rate_alpha_list[1]
            self.learning_rate_cvt =  self.learning_rate_cvt_list[1]
            self.s_w = self.s_w_list[1]
            self.e_w =  self.e_w_list[1]
            self.tv_w = self.tv_w_list[1]
            self.tv_f = self.tv_f_list[1]

            self.end_iter_loc = up_iters[1] - up_iters[0]
            self.vortSDF_renderer_fine.mask_reg = self.mask_w_list[1]
            if self.double_net:
                self.vortSDF_renderer_coarse_net.mask_reg = self.mask_w_list[1]
            #verbose = True
            

        if (self.iter_step+1) == up_iters[1]:
            self.s_start = 200 #30/(10.0*self.sigma) #50.0
            self.s_max = 800 #60/(5.0*self.sigma) #200

            self.learning_rate = self.learning_rate_list[2]
            self.learning_rate_sdf = self.learning_rate_sdf_list[2]
            self.learning_rate_feat = self.learning_rate_feat_list[2]
            self.learning_rate_alpha = self.learning_rate_alpha_list[2]
            self.learning_rate_cvt =  self.learning_rate_cvt_list[2]
            self.s_w = self.s_w_list[2]
            self.e_w =  self.e_w_list[2]
            self.tv_w = self.tv_w_list[2]
            self.tv_f = self.tv_f_list[2]

            self.end_iter_loc = up_iters[2] - up_iters[1]
            self.vortSDF_renderer_fine.mask_reg = self.mask_w_list[2]
            if self.double_net:
                self.vortSDF_renderer_coarse_net.mask_reg = self.mask_w_list[2]
            self.sigma_start = 0.03
            self.sigma_max = 0.015
            
            
        if (self.iter_step+1) == up_iters[2]:
            warm_up = 500
            self.s_start = 60/(10.0*self.sigma) #50.0
            self.s_max = 60/(3.0*self.sigma) #200

            self.learning_rate = self.learning_rate_list[3]
            self.learning_rate_sdf = self.learning_rate_sdf_list[3]
            self.learning_rate_feat = self.learning_rate_feat_list[3]
            self.learning_rate_alpha = self.learning_rate_alpha_list[3]
            self.learning_rate_cvt =  self.learning_rate_cvt_list[3]
            self.s_w = self.s_w_list[3]
            self.e_w =  self.e_w_list[3]
            self.tv_w = self.tv_w_list[3]
            self.tv_f = self.tv_f_list[3]

            self.end_iter_loc = up_iters[3] - up_iters[2]
            self.vortSDF_renderer_fine.mask_reg = self.mask_w_list[3]
            if self.double_net:
                self.vortSDF_renderer_coarse_net.mask_reg = self.mask_w_list[3]
            lamda_c = 0.5
            self.sigma_start = 0.015
            self.sigma_max = 0.008
            self.dataset.gen_all_rays(1)
            

        if (self.iter_step+1) == up_iters[3]:
            warm_up = 500
            self.s_start = 60/(10.0*self.sigma) #50.0
            self.s_max = 60/(3.0*self.sigma) #200


            self.learning_rate = self.learning_rate_list[4]
            self.learning_rate_sdf = self.learning_rate_sdf_list[4]
            self.learning_rate_feat = self.learning_rate_feat_list[4]
            self.learning_rate_alpha = self.learning_rate_alpha_list[4]
            self.learning_rate_cvt =  self.learning_rate_cvt_list[4]
            self.s_w = self.s_w_list[4]
            self.e_w =  self.e_w_list[4]
            self.tv_w = self.tv_w_list[4]
            self.tv_f = self.tv_f_list[4]

            self.end_iter_loc = up_iters[4] - up_iters[3]
            self.vortSDF_renderer_fine.mask_reg = self.mask_w_list[4]
            if self.double_net:
                self.vortSDF_renderer_coarse_net.mask_reg = self.mask_w_list[4]
            lamda_c = 0.5
            #full_reg = 3
            self.sigma_start = 0.007
            self.sigma_max = 0.004
            self.dataset.gen_all_rays(1)
            
        if (self.iter_step+1) == up_iters[4]:
            self.s_start = 1000#30/(10.0*self.sigma) #50.0
            self.s_max = 4000#60/(5.0*self.sigma) #200

            self.learning_rate = self.learning_rate_list[5]
            self.learning_rate_sdf = self.learning_rate_sdf_list[5]
            self.learning_rate_feat = self.learning_rate_feat_list[5]
            self.learning_rate_alpha = self.learning_rate_alpha_list[5]
            self.learning_rate_cvt =  self.learning_rate_cvt_list[5]
            self.s_w = self.s_w_list[5]
            self.e_w =  self.e_w_list[5]
            self.tv_w = self.tv_w_list[5]
            self.tv_f = self.tv_f_list[5]
            self.f_w = 0.0 #1.0#1.0e0

            self.end_iter_loc = self.end_iter - up_iters[4]
            self.vortSDF_renderer_fine.mask_reg = self.mask_w_list[5]
            if self.double_net:
                self.vortSDF_renderer_coarse_net.mask_reg = self.mask_w_list[5]
            lamda_c = 0.2
            self.sigma_start = 0.004
            self.sigma_max = 0.002
            self.dataset.gen_all_rays(1)
            #self.dataset.gen_all_rays_masked(1)
       
        
        self.w_g = 1.0
        self.mask_background = abs(self.sdf) > 12.0*self.sigma_start
        self.sigma_feat = 0.06
        self.inv_s = 60/(5.0*self.sigma)
        print(self.sigma)
        print(self.inv_s)
     

if __name__=='__main__':
    print("Code by Diego Thomas")

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='src/Confs/test.conf')
    parser.add_argument('--data_name', type=str, default='bmvs_man')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--is_continue', default=False, action="store_true")
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--resolution', type=int, default=16)
    parser.add_argument('--output', type=str, default='')
    parser.add_argument('--nb_images', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--position_encoding', default=False, action="store_true")
    parser.add_argument('--double_net', default=False, action="store_true")
    
    args = parser.parse_args()

    ## Initialise CUDA device for torch computations
    torch.cuda.set_device(args.gpu)
    
    runner = Runner(args.conf, args.data_name, args.mode, args.is_continue, args.checkpoint, args.position_encoding, args.double_net)
    
    if args.mode == 'train':
        if args.is_continue:
            runner.prep_CVT()
        runner.train(args.data_name, 24, False, args.is_continue)
        #for id_im in tqdm(range(args.nb_images)):
        #    runner.render_image(runner.cam_ids, img_idx = id_im)
    elif args.mode == 'render_images':
        runner.prep_CVT()
        if args.nb_images == 0:
            nb_im = runner.dataset.n_images
        else:
            nb_im = args.nb_images
        for id_im in tqdm(range(nb_im)):
            runner.render_image(img_idx = id_im)
        #    #runner.validate_image_hybrid(1000.0, idx=id_im)"""