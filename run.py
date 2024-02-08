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
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from src.Models.VortSDFRenderer import VortSDFRenderer, VortSDFRenderingFunction, VortSDFDirectRenderer
from src.Models.fields import ColorNetwork

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
        self.n_samples = self.conf.get_int('model.cvt_renderer.n_samples')
        self.batch_size = self.conf.get_int('train.batch_size')
        self.learning_rate = self.conf.get_float('train.learning_rate')
        self.learning_rate_sdf = self.conf.get_float('train.learning_rate_sdf')
        self.learning_rate_feat = self.conf.get_float('train.learning_rate_feat')
        self.learning_rate_alpha = self.conf.get_float('train.learning_rate_alpha')
        lr=self.learning_rate_cvt = self.conf.get_float('train.learning_rate_cvt')

        self.report_freq = self.conf.get_int('train.report_freq')
        self.val_freq = self.conf.get_int('train.val_freq')
        self.CVT_freq = self.conf.get_int('train.CVT_freq')

        self.vortSDF_renderer_coarse = VortSDFDirectRenderer(**self.conf['model.cvt_renderer'])

        self.vortSDF_renderer_coarse_net = VortSDFRenderer(**self.conf['model.cvt_renderer'])

        self.vortSDF_renderer_fine = VortSDFRenderer(**self.conf['model.cvt_renderer'])
        
        self.color_coarse = ColorNetwork(**self.conf['model.color_geo_network']).to(self.device)

        self.color_network = ColorNetwork(**self.conf['model.color_network']).to(self.device)
        
        params_to_train = []
        params_to_train += list(self.color_network.parameters())
        self.optimizer = torch.optim.Adam(params_to_train, lr=self.learning_rate)
        
        posbase_pe=5
        viewbase_pe= 1#4
        self.posfreq = torch.FloatTensor([(2**i) for i in range(posbase_pe)]).cuda()
        self.viewfreq = torch.FloatTensor([(2**i) for i in range(viewbase_pe)]).cuda()

        k_posbase_pe=5
        k_viewbase_pe= 1
        self.k_posfreq = torch.FloatTensor([(2**i) for i in range(k_posbase_pe)]).cuda()
        self.k_viewfreq = torch.FloatTensor([(2**i) for i in range(k_viewbase_pe)]).cuda()


    def train(self, data_name, K_NN = 24, verbose = True):
        ##### 2. Load initial sites
        if not hasattr(self, 'tet32'):
            ##### 2. Load initial sites
            visual_hull = [-1.1, -1.1, -1.1, 1.1, 1.1, 1.1]
            import src.Geometry.sampling as sampler
            res = 32
            sites = sampler.sample_Bbox(visual_hull[0:3], visual_hull[3:6], res, perturb_f =  (visual_hull[3] - visual_hull[0])*0.05)
            
            #### Add cameras as sites 
            cam_sites = np.stack([self.dataset.pose_all[id, :3,3].cpu().numpy() for id in range(self.dataset.n_images)])
            sites = np.concatenate((sites, cam_sites))

            self.tet32 = tet32.Tet32(sites, id = 0)
            #self.tet32.start() 
            #print("parallel process started")
            #self.tet32.join()
            self.tet32.run() 
            self.tet32.load_cuda()
            self.tet32.save("Exp/bmvs_man/test.ply")    
            
            sites = np.asarray(self.tet32.vertices)  
            cam_ids = np.stack([np.where((sites == cam_sites[i,:]).all(axis = 1))[0] for i in range(cam_sites.shape[0])]).reshape(-1)
            self.tet32.make_adjacencies(cam_ids)

            cam_ids = torch.from_numpy(cam_ids).int().cuda()
            
            outside_flag = np.zeros(sites.shape[0], np.int32)
            outside_flag[sites[:,0] < visual_hull[0] + (visual_hull[3]-visual_hull[0])/(2*res)] = 1
            outside_flag[sites[:,1] < visual_hull[1] + (visual_hull[4]-visual_hull[1])/(2*res)] = 1
            outside_flag[sites[:,2] < visual_hull[2] + (visual_hull[5]-visual_hull[2])/(2*res)] = 1
            outside_flag[sites[:,0] > visual_hull[3] - (visual_hull[3]-visual_hull[0])/(2*res)] = 1
            outside_flag[sites[:,1] > visual_hull[4] - (visual_hull[4]-visual_hull[1])/(2*res)] = 1
            outside_flag[sites[:,2] > visual_hull[5] - (visual_hull[5]-visual_hull[2])/(2*res)] = 1
        #else:
        #    sites, _ = ply.load_ply(base_exp_dir + "/sites_init_" + data_name +"_32.ply")

        self.sites = torch.from_numpy(sites.astype(np.float32)).cuda()
        self.sites = self.sites.contiguous()
        self.sites.requires_grad_(True)
        
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
                    
        self.tet32.surface_from_sdf(self.sdf.detach().cpu().numpy().reshape(-1), "Exp/bmvs_man/test_tri.ply")
        #self.tet32.marching_tets(self.sdf.detach(), "Exp/bmvs_man/test_MT.ply")
        #input()
        
        ##### 2. Initialize feature field    
        if not hasattr(self, 'fine_features'):
            self.fine_features = 0.5*torch.ones([self.sdf.shape[0], 6]).cuda()       
            self.fine_features = self.fine_features.contiguous()
            self.fine_features.requires_grad_(True)
        
        self.Allocate_data()
            
        self.Allocate_batch_data()

        
        if not hasattr(self, 'optimizer_sdf'):
            self.optimizer_sdf = torch.optim.Adam([self.sdf], lr=self.learning_rate_sdf)        
            self.optimizer_feat = torch.optim.Adam([self.fine_features], lr=self.learning_rate_feat) 
            self.optimizer_cvt = torch.optim.Adam([self.sites], lr=self.learning_rate_cvt) 

        self.vortSDF_renderer_coarse.prepare_buffs(self.batch_size, self.n_samples, self.sites.shape[0])
        self.vortSDF_renderer_coarse_net.prepare_buffs(self.batch_size, self.n_samples, self.sites.shape[0])
        self.vortSDF_renderer_fine.prepare_buffs(self.batch_size, self.n_samples, self.sites.shape[0])
        
        #self.inv_s = 1000
        #self.render_image(cam_ids, 0)
        #input()
        
        self.s_max = 1000
        self.R = 40
        self.s_start = 10.0
        self.inv_s = 0.1
        self.sigma = 0.03
        image_perm = self.get_image_perm()
        num_rays = self.batch_size
        for iter_step in tqdm(range(self.end_iter)):
            img_idx = image_perm[iter_step % len(image_perm)].item() 
            self.inv_s = min(self.s_max, iter_step/self.R + self.s_start)

            ## Generate rays
            data = self.dataset.gen_random_rays_zbuff_at(img_idx, num_rays, 0)  
            rays_o, rays_d, true_rgb, mask = data[:, :3], data[:, 3: 6], data[:, 6: 9], data[:, 9: 10]

            rays_o = rays_o.reshape(-1, 3)
            rays_d = rays_d.reshape(-1, 3)
            true_rgb = true_rgb.reshape(-1, 3)
            mask = mask.reshape(-1, 1)

            rays_o = rays_o.contiguous()
            rays_d = rays_d.contiguous()
            true_rgb = true_rgb.contiguous()
            mask = mask.contiguous()
            
            ## sample points along the rays
            start = timer()
            self.offsets[:] = 0
            nb_samples = self.tet32.sample_rays_cuda(img_idx, rays_d, self.sdf, self.fine_features, cam_ids, self.in_weights, self.in_z, self.in_sdf, self.in_feat, self.in_ids, self.offsets, self.n_samples)    
            if verbose:
                print('CVT_Sample time:', timer() - start)   

            """if self.offsets[self.offsets[:,1] == -1].sum() != 0:
                print("no good topologie", self.offsets[self.offsets[:,1] == -1, 1].sum())
                #self.tet32.start() 
                #input()"""
            
            self.offsets[self.offsets[:,1] == -1] = 0     
                
            start = timer()
            self.samples[:] = 0.0
            tet32_march_cuda.fill_samples(rays_o.shape[0], self.n_samples, rays_o, rays_d, self.sites, 
                                        self.in_z, self.in_sdf, self.in_feat, self.in_weights, self.in_ids, 
                                        self.out_z, self.out_sdf, self.out_feat, self.out_weights, self.out_ids, 
                                        self.offsets, self.samples, self.samples_loc, self.samples_rays)
            if verbose:
                print('fill_samples time:', timer() - start)   


            if False: 
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
                input()

            #samples = (self.samples[:nb_samples,:] + self.samples_loc[:nb_samples,:])/2.0
            samples = self.samples[:nb_samples,:]
            samples = samples.contiguous()

            ##### ##### ##### ##### ##### ##### ##### 
            # Build fine features
            #fine_features = (self.out_feat[:nb_samples,:6] + self.out_feat[:nb_samples,:-6])/2.0#self.fine_features[self.out_ids[:nb_samples, 1]]
            fine_features = self.out_feat[:nb_samples,:6]
            self.colors = fine_features[:,:3] 
            self.colors = self.colors.contiguous()

            
            ##### ##### ##### ##### ##### ##### 
            xyz_emb = (samples.unsqueeze(-1) * self.posfreq).flatten(-2)
            xyz_emb = torch.cat([samples, xyz_emb.sin(), xyz_emb.cos()], -1)

            viewdirs_emb = (self.samples_rays[:nb_samples,:].unsqueeze(-1) * self.viewfreq).flatten(-2)
            viewdirs_emb = torch.cat([self.samples_rays[:nb_samples,:], viewdirs_emb.sin(), viewdirs_emb.cos()], -1)

            # Get sdf values
            #sdf_prev = self.sdf_network.rgb(torch.cat([xyz_emb, self.out_feat[:nb_samples,3:6]], -1))
            #sdf_next = self.sdf_network.rgb(torch.cat([xyz_emb, self.out_feat[:nb_samples,:-3]], -1))

                        
            """geo_feat = torch.cat([xyz_emb, viewdirs_emb, self.out_sdf[:nb_samples,:]], -1)
            colors_feat = self.color_coarse.rgb(geo_feat)   
            self.colors = torch.sigmoid(colors_feat)
            self.colors = self.colors.contiguous()
            rgb_feat = torch.cat([xyz_emb, viewdirs_emb, colors_feat.detach(), fine_features], -1)"""
            
            # network interpolation
            rgb_feat = torch.cat([xyz_emb, viewdirs_emb, self.out_sdf[:nb_samples,:-6], self.out_weights[:nb_samples,:6], self.out_feat[:nb_samples,:]], -1)

            # linear interpolation
            #rgb_feat = torch.cat([xyz_emb, viewdirs_emb, fine_features], -1)

            rgb_feat.requires_grad_(True)
            rgb_feat.retain_grad()

            #self.colors_fine = torch.sigmoid(self.color_network.rgb(rgb_feat) + colors_feat.detach()) 
            self.colors_fine = torch.sigmoid(self.color_network.rgb(rgb_feat)) #+ self.colors
            self.colors_fine = self.colors_fine.contiguous()


            ########################################
            ####### Render the image ###############
            ########################################
            mask = (mask > 0.5).float()

            start = timer()       
            #self.vortSDF_renderer_coarse.render_gpu(rays_o.shape[0], self.inv_s, self.out_sdf, self.tet32.knn_sites, self.out_weights, self.colors, true_rgb, mask, self.out_ids, self.offsets)
            #color_coarse_loss = VortSDFRenderingFunction.apply(self.vortSDF_renderer_coarse_net, rays_o.shape[0], self.inv_s, self.out_sdf, self.tet32.knn_sites, self.out_weights, self.colors, true_rgb, mask, self.out_ids, self.offsets)
            
            color_fine_loss = VortSDFRenderingFunction.apply(self.vortSDF_renderer_fine, rays_o.shape[0], self.inv_s, self.out_sdf, self.tet32.knn_sites, self.out_weights, self.colors_fine, true_rgb, mask, self.out_ids, self.offsets)
            if verbose:
                print('RenderingFunction time:', timer() - start)

            # Total loss   
            mask_sum = mask.sum()
            
            self.optimizer.zero_grad()
            #loss = (color_fine_loss + color_coarse_loss) / (mask_sum + 1.0e-5)
            loss = color_fine_loss / (mask_sum + 1.0e-5)
            loss.backward()

            ########################################
            # Backprop feature gradients to gradients on sites
            # step optimize color features
            start = timer()   
            fine_features_grad = rgb_feat.grad[:,:-36]
            fine_features_grad = fine_features_grad.contiguous()
            self.grad_features[:] = 0.0
            backprop_cuda.backprop_feat(nb_samples, self.grad_features, fine_features_grad, self.out_ids, self.out_weights)
            self.grad_features[:, :3] = self.grad_features[:, :3] #+ 0.5*self.vortSDF_renderer_coarse.grads_color[:,:] / (mask_sum + 1.0e-5)
            #self.grad_features[:, :3] = self.vortSDF_renderer_coarse.grads_color[:,:] / (mask_sum + 1.0e-5)
            self.grad_features[outside_flag[:] == 1.0] = 0.0   
            if verbose:
                print('backprop_feat time:', timer() - start)
            
            self.optimizer.step()
            
            ########################################
            ####### Regularization terms ###########
            ########################################
            
            ############ Compute spatial SDF gradients
            start = timer()   
            cvt_grad_cuda.sdf_space_grad(self.sites.shape[0], K_NN, self.tet32.knn_sites, self.sites, self.sdf, self.fine_features, self.weights_grad, self.grad_sdf_space, self.grad_feat_space)
            if verbose:
                print('smooth_sdf time:', timer() - start)
                
            """start = timer()   
            self.grad_sdf_reg[:] = 0.0
            self.grad_feat_reg[:] = 0.0
            backprop_cuda.space_reg(rays_o.shape[0], rays_d, self.grad_sdf_space, self.out_weights, self.out_z, self.out_sdf, self.out_feat, self.out_ids, self.offsets, self.grad_sdf_reg, self.grad_feat_reg)
            
            if verbose:
                print('space_reg time:', timer() - start)"""
        
            start = timer()   
            self.grad_sdf_smooth[:] = 0.0
            self.grad_feat_smooth[:] = 0.0
            self.weight_sdf_smooth[:] = 0.0
            backprop_cuda.smooth_sdf(self.tet32.edges.shape[0], self.sigma, self.sites, self.sdf, self.fine_features, self.tet32.edges, self.grad_sdf_smooth, self.grad_feat_smooth, self.weight_sdf_smooth)
            self.weight_sdf_smooth[self.weight_sdf_smooth[:] == 0.0] = 1.0
            self.grad_sdf_smooth = self.grad_sdf_smooth / self.weight_sdf_smooth
            self.grad_feat_smooth = self.grad_feat_smooth / self.weight_sdf_smooth.reshape(-1,1)
            if verbose:
                print('smooth_sdf time:', timer() - start)
            
            ########################################
            ####### Optimize features ##############
            ########################################
            
            self.optimizer_feat.zero_grad()
            self.fine_features.grad = self.grad_features # + 0.00001*self.grad_feat_smooth #+ 1.0e+5*self.grad_feat_reg / (mask_sum + 1.0e-5)
            self.optimizer_feat.step()

            ########################################
            ####### Optimize sdf values ############
            ########################################

            #grad_sdf = (0.5*self.voro_renderer_coarse.grads_sdf[:,0] + 1.0*self.voro_renderer_fine.grads_sdf[:,0]) / (mask_sum + 1.0e-5)
            grad_sdf = self.vortSDF_renderer_fine.grads_sdf / (mask_sum + 1.0e-5)
            grad_sdf[outside_flag[:] == 1.0] = 0.0   

            self.optimizer_sdf.zero_grad()
            self.sdf.grad = grad_sdf + 0.001*self.grad_sdf_smooth #+ 1.0e-3*self.grad_sdf_reg / (mask_sum + 1.0e-5) #
            self.optimizer_sdf.step()

            ########################################
            ##### Optimize sites positions #########
            ########################################
            loss_cvt = 0.0
            if False: #iter_step > 10 and iter_step % self.CVT_freq == 0: # and iter_step < 5000 :
                thetas = torch.from_numpy(np.random.rand(self.sites.shape[0])).float().cuda()
                phis = torch.from_numpy(np.random.rand(self.sites.shape[0])).float().cuda()
                gammas = torch.from_numpy(np.random.rand(self.sites.shape[0])).float().cuda()

                ############ Compute spatial SDF gradients
                cvt_grad_cuda.sdf_space_grad(self.sites.shape[0], K_NN, self.tet32.knn_sites, self.sites, self.sdf, self.fine_features, self.weights_grad, self.grad_sdf_space, self.grad_feat_space)
                
                ############ Compute approximated CVT gradients at sites
                start = timer()       
                self.grad_sites[:] = 0.0
                loss_cvt = cvt_grad_cuda.cvt_grad(self.sites.shape[0], K_NN, thetas, phis, gammas, self.tet32.knn_sites, self.sites, self.grad_sites)
                self.grad_sites = self.grad_sites / self.sites.shape[0]
                
                self.grad_sites_sdf[:] = 0.0
                cvt_grad_cuda.sdf_grad(self.sites.shape[0], K_NN, self.tet32.knn_sites, self.sites, self.sdf, self.grad_sites_sdf)
                
                self.mask_grad[:,:] = 1.0
                self.mask_grad[(torch.linalg.norm(self.grad_sites_sdf, ord=2, axis=-1, keepdims=True) > 0.0).reshape(-1),:] = 1.0e-3

                if verbose:
                    print('cvt_grad time:', timer() - start)

                self.grad_sites[outside_flag == 1.0, :] = 0.0
                self.grad_sites[cam_ids, :] = 0.0
                self.grad_sites_sdf[outside_flag == 1.0] = 0.0
                self.optimizer_cvt.zero_grad()
                self.sites.grad = self.grad_sites*self.mask_grad + 1.0*self.grad_sites_sdf
                self.optimizer_cvt.step()

                with torch.no_grad():
                    self.sites[cam_ids] = delta_sites[cam_ids]
                    delta_sites[:] = self.sites[:] - delta_sites[:] 
                    if iter_step % self.report_freq == 0:
                        print(delta_sites.mean())

                    self.tet32.sites[:] = self.sites[:]

                with torch.no_grad():
                    self.sdf[:] = self.sdf[:] + (delta_sites*self.grad_sdf_space).sum(dim = 1)[:] #  + self.sdf_diff
                    for i in range(6):
                        self.fine_features[:, i] = self.fine_features[:, i] + (delta_sites*self.grad_feat_space[:, :, i]).sum(dim = 1)[:] # self.feat_diff[:]

                """with torch.no_grad():
                    if iter_step % 100 == 0 and iter_step < 3000:
                        self.tet32 = tet32.Tet32(self.sites.detach().cpu().numpy(), id = 0)
                        #self.tet32.start() 
                        #print("parallel process started")
                        #self.tet32.join()
                        self.tet32.run() 
                        self.tet32.load_cuda()
                        
                        sites = np.asarray(self.tet32.vertices)  
                        cam_ids = np.stack([np.where((sites == cam_sites[i,:]).all(axis = 1))[0] for i in range(cam_sites.shape[0])]).reshape(-1)
                        self.tet32.make_adjacencies(cam_ids)

                        cam_ids = torch.from_numpy(cam_ids).int().cuda()
                        
                        outside_flag[:] = 0
                        outside_flag[sites[:,0] < visual_hull[0] + (visual_hull[3]-visual_hull[0])/(2*res)] = 1
                        outside_flag[sites[:,1] < visual_hull[1] + (visual_hull[4]-visual_hull[1])/(2*res)] = 1
                        outside_flag[sites[:,2] < visual_hull[2] + (visual_hull[5]-visual_hull[2])/(2*res)] = 1
                        outside_flag[sites[:,0] > visual_hull[3] - (visual_hull[3]-visual_hull[0])/(2*res)] = 1
                        outside_flag[sites[:,1] > visual_hull[4] - (visual_hull[4]-visual_hull[1])/(2*res)] = 1
                        outside_flag[sites[:,2] > visual_hull[5] - (visual_hull[5]-visual_hull[2])/(2*res)] = 1

                        sites = torch.from_numpy(sites.astype(np.float32)).cuda()
                        self.sites[:] = sites[:]"""

                
                """with torch.no_grad():
                    norm_sites = torch.linalg.norm(self.sites, ord=2, axis=-1, keepdims=True)
                    self.sdf[:] = norm_sites[:,0] - 0.5"""

                if iter_step % (10*self.CVT_freq) == 0:
                     self.tet32.make_knn()

                with torch.no_grad():
                    delta_sites[:] = self.sites[:]

            if False: #(iter_step+1) % 10000 == 0 and iter_step < 20000:
                self.sdf, self.fine_features = self.tet32.upsample(self.sdf.detach().cpu().numpy(), self.fine_features.detach().cpu().numpy())
                self.sdf = self.sdf.contiguous()
                self.sdf.requires_grad_(True)
                self.fine_features = self.fine_features.contiguous()
                self.fine_features.requires_grad_(True)
                self.tet32.load_cuda()

                sites = np.asarray(self.tet32.vertices)  
                cam_ids = np.stack([np.where((sites == cam_sites[i,:]).all(axis = 1))[0] for i in range(cam_sites.shape[0])]).reshape(-1)
                self.tet32.make_adjacencies(cam_ids)

                cam_ids = torch.from_numpy(cam_ids).int().cuda()
                
                outside_flag = np.zeros(sites.shape[0], np.int32)
                outside_flag[sites[:,0] < visual_hull[0] + (visual_hull[3]-visual_hull[0])/(2*res)] = 1
                outside_flag[sites[:,1] < visual_hull[1] + (visual_hull[4]-visual_hull[1])/(2*res)] = 1
                outside_flag[sites[:,2] < visual_hull[2] + (visual_hull[5]-visual_hull[2])/(2*res)] = 1
                outside_flag[sites[:,0] > visual_hull[3] - (visual_hull[3]-visual_hull[0])/(2*res)] = 1
                outside_flag[sites[:,1] > visual_hull[4] - (visual_hull[4]-visual_hull[1])/(2*res)] = 1
                outside_flag[sites[:,2] > visual_hull[5] - (visual_hull[5]-visual_hull[2])/(2*res)] = 1

                
                self.sites = torch.from_numpy(sites.astype(np.float32)).cuda()
                self.sites = self.sites.contiguous()
                self.sites.requires_grad_(True)

                self.Allocate_data()
                
                delta_sites = torch.zeros(self.sites.shape).float().cuda()
                with torch.no_grad():  
                    delta_sites[:] = self.sites[:]
                
                self.optimizer_sdf = torch.optim.Adam([self.sdf], lr=self.learning_rate_sdf)        
                self.optimizer_feat = torch.optim.Adam([self.fine_features], lr=self.learning_rate_feat) 
                self.optimizer_cvt = torch.optim.Adam([self.sites], lr=self.learning_rate_cvt) 
                
                self.vortSDF_renderer_coarse.prepare_buffs(self.batch_size, self.n_samples, self.sites.shape[0])
                self.vortSDF_renderer_fine.prepare_buffs(self.batch_size, self.n_samples, self.sites.shape[0])

                self.sigma = self.sigma /2.0
                #verbose = True
                self.tet32.save("Exp/bmvs_man/test_up.ply")    
            
            if iter_step % self.report_freq == 0:
                print('iter:{:8>d} loss = {}, scale={}, lr={}'.format(iter_step, loss, self.inv_s, self.optimizer_sdf.param_groups[0]['lr']))
                print('iter:{:8>d} loss CVT = {} lr={}'.format(iter_step, loss_cvt, self.optimizer_cvt.param_groups[0]['lr']))

            if iter_step % self.val_freq == 0:
                self.render_image(cam_ids, img_idx)
                self.tet32.surface_from_sdf(self.sdf.detach().cpu().numpy().reshape(-1), "Exp/bmvs_man/test_tri.ply")
                #self.tet32.marching_tets(self.sdf.detach(), "Exp/bmvs_man/test_MT.ply")
                #if iter_step > 1000:
                #    self.tet32.save("Exp/bmvs_man/test.ply")                         
                torch.cuda.empty_cache()

            self.update_learning_rate(iter_step)

        self.render_image(cam_ids, 0)

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
            nb_samples = self.tet32.sample_rays_cuda(img_idx, rays_d_batch, self.sdf, self.fine_features, cam_ids, self.in_weights, self.in_z, self.in_sdf, self.in_feat, self.in_ids, self.offsets, self.n_samples)    
                
            start = timer()
            self.samples[:] = 0.0
            tet32_march_cuda.fill_samples(rays_o_batch.shape[0], self.n_samples, rays_o_batch, rays_d_batch, self.tet32.sites, 
                                            self.in_z, self.in_sdf, self.in_feat, self.in_weights, self.in_ids, 
                                            self.out_z, self.out_sdf, self.out_feat, self.out_weights, self.out_ids, 
                                            self.offsets, self.samples, self.samples_loc, self.samples_rays)
            
            #samples = (self.samples[:nb_samples,:] + self.samples_loc[:nb_samples,:])/2.0
            samples = self.samples[:nb_samples,:]
            samples = samples.contiguous()

            #fine_features = (self.out_feat[:nb_samples,:6] + self.out_feat[:nb_samples,:-6])/2.0
            fine_features = self.out_feat[:nb_samples,:6]
            
            self.colors = fine_features[:nb_samples,:3] 
            self.colors = self.colors.contiguous()
            
            ##### ##### ##### ##### ##### ##### 
            xyz_emb = (samples.unsqueeze(-1) * self.posfreq).flatten(-2)
            xyz_emb = torch.cat([samples, xyz_emb.sin(), xyz_emb.cos()], -1)

            viewdirs_emb = (self.samples_rays[:nb_samples,:].unsqueeze(-1) * self.viewfreq).flatten(-2)
            viewdirs_emb = torch.cat([self.samples_rays[:nb_samples,:], viewdirs_emb.sin(), viewdirs_emb.cos()], -1)

            
            """geo_feat = torch.cat([xyz_emb, viewdirs_emb, self.out_sdf[:nb_samples,:]], -1)
            colors_feat = self.color_coarse.rgb(geo_feat)   
            self.colors = torch.sigmoid(colors_feat)
            self.colors = self.colors.contiguous()
            rgb_feat = torch.cat([xyz_emb, viewdirs_emb, colors_feat.detach(), self.out_feat[:nb_samples]], -1)"""

            rgb_feat = torch.cat([xyz_emb, viewdirs_emb, self.out_sdf[:nb_samples, :-6], self.out_weights[:nb_samples,:6], self.out_feat[:nb_samples,:]], -1)
            #rgb_feat = torch.cat([xyz_emb, viewdirs_emb, fine_features[:nb_samples]], -1)
            #rgb_feat = torch.cat([xyz_emb, viewdirs_emb, self.out_feat[:nb_samples]], -1)
            rgb_feat.requires_grad_(True)
            rgb_feat.retain_grad()

            #self.colors_fine = torch.sigmoid(self.color_network.rgb(rgb_feat) + colors_feat.detach())
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
            renderer_cuda.render_no_grad(rays_o_batch.shape[0], self.inv_s, self.out_sdf, self.colors_fine, self.offsets, colors_out, mask_out)
            renderer_cuda.render_no_grad(rays_o_batch.shape[0], self.inv_s, self.out_sdf, self.colors, self.offsets, colors_out_coarse, mask_out)

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
        
        img_coarse = img_coarse.reshape(self.dataset.H // resolution_level, self.dataset.W // resolution_level, 3)
        img_coarse = img_coarse.cpu().numpy()
        cv2.imwrite('Exp/synt_coarse.png', 255*img_coarse[:,:])
        
        GTimg = true_rgb.reshape(self.dataset.H // resolution_level, self.dataset.W // resolution_level, 3).cpu().numpy()
        cv2.imwrite('Exp/GT.png', 255*GTimg[:,:])

        mask = mask.reshape(self.dataset.H // resolution_level, self.dataset.W // resolution_level).cpu().numpy()
        cv2.imwrite('Exp/Mask.png', 255*mask[:])
        print("rendering done")
        
    def Allocate_data(self, K_NN = 24):        
        self.grad_sites = torch.zeros(self.sites.shape).cuda()       
        self.grad_sites = self.grad_sites.contiguous()

        self.grad_sites_sdf = torch.zeros(self.sites.shape).cuda()       
        self.grad_sites_sdf = self.grad_sites_sdf.contiguous()

        self.mask_grad = torch.zeros(self.sites.shape).cuda()       
        self.mask_grad = self.mask_grad.contiguous()

        self.grad_sdf_smooth = torch.zeros([self.sdf.shape[0]]).cuda()       
        self.grad_sdf_smooth = self.grad_sdf_smooth.contiguous()
        
        self.weight_sdf_smooth = torch.zeros([self.sdf.shape[0]]).cuda()       
        self.weight_sdf_smooth = self.weight_sdf_smooth.contiguous()
        
        self.grad_features = torch.zeros([self.sdf.shape[0], 6]).cuda()       
        self.grad_features = self.grad_features.contiguous()
        
        self.grad_feat_smooth = torch.zeros([self.sdf.shape[0], 6]).cuda()       
        self.grad_feat_smooth = self.grad_feat_smooth.contiguous()

        self.grad_sdf_space = torch.zeros([self.sites.shape[0], 3]).float().cuda().contiguous()
        self.grad_feat_space = torch.zeros([self.sites.shape[0], 3, 6]).float().cuda().contiguous()
        self.weights_grad = torch.zeros([3*K_NN*self.sites.shape[0]]).float().cuda().contiguous()
        
        self.grad_feat_reg = torch.zeros([self.sdf.shape[0], 6]).cuda()       
        self.grad_feat_reg = self.grad_feat_reg.contiguous()
        
        self.grad_sdf_reg = torch.zeros([self.sdf.shape[0]]).cuda()       
        self.grad_sdf_reg = self.grad_sdf_reg.contiguous()
        
    def Allocate_batch_data(self, K_NN = 24):
        self.samples = torch.zeros([self.n_samples * self.batch_size, 3], dtype=torch.float32).cuda()
        self.samples = self.samples.contiguous()
        
        self.colors_fine = torch.zeros([self.n_samples * self.batch_size, 3], dtype=torch.float32).cuda()
        self.colors_fine = self.colors_fine.contiguous()
        
        self.samples_loc = torch.zeros([self.n_samples * self.batch_size, 3], dtype=torch.float32).cuda()
        self.samples_loc = self.samples_loc.contiguous()
        
        self.samples_rays = torch.zeros([self.n_samples * self.batch_size, 3], dtype=torch.float32).cuda()
        self.samples_rays = self.samples_rays.contiguous()
        
        self.in_weights = torch.zeros([6*self.n_samples * self.batch_size], dtype=torch.float32).cuda()
        #self.in_weights = torch.zeros([6*self.n_samples * self.batch_size], dtype=torch.float32).cuda()
        self.in_weights = self.in_weights.contiguous()

        self.in_ids = -torch.ones([6*self.n_samples* self.batch_size], dtype=torch.int32).cuda()
        self.in_ids = self.in_ids.contiguous()
        
        self.in_z = torch.zeros([2*self.n_samples * self.batch_size], dtype=torch.float32).cuda()
        self.in_z = self.in_z.contiguous()
        
        self.in_sdf = torch.zeros([8*self.n_samples * self.batch_size], dtype=torch.float32).cuda()
        #self.in_sdf = torch.zeros([2*self.n_samples * self.batch_size], dtype=torch.float32).cuda()
        self.in_sdf = self.in_sdf.contiguous()
        
        self.in_feat = torch.zeros([36*self.n_samples * self.batch_size], dtype=torch.float32).cuda()
        #self.in_feat = torch.zeros([12*self.n_samples * self.batch_size], dtype=torch.float32).cuda()
        self.in_feat = self.in_feat.contiguous()
        
        self.out_ids = -torch.ones([self.n_samples* self.batch_size, 6], dtype=torch.int32).cuda()
        self.out_ids = self.out_ids.contiguous()
        
        self.out_z = torch.zeros([self.n_samples * self.batch_size,2], dtype=torch.float32).cuda()
        self.out_z = self.out_z.contiguous()
        
        self.out_sdf = torch.zeros([self.n_samples * self.batch_size, 8], dtype=torch.float32).cuda()
        #self.out_sdf = torch.zeros([self.n_samples * self.batch_size, 2], dtype=torch.float32).cuda()
        self.out_sdf = self.out_sdf.contiguous()
        
        self.out_feat = torch.zeros([self.n_samples * self.batch_size, 36], dtype=torch.float32).cuda()
        #self.out_feat = torch.zeros([self.n_samples * self.batch_size, 12], dtype=torch.float32).cuda()
        self.out_feat = self.out_feat.contiguous()
        
        #self.out_weights = torch.zeros([7*self.n_samples * self.batch_size], dtype=torch.float32).cuda()
        self.out_weights = torch.zeros([self.n_samples * self.batch_size, 7], dtype=torch.float32).cuda()
        self.out_weights = self.out_weights.contiguous()
        
        self.offsets = torch.zeros([self.batch_size, 2], dtype=torch.int32).cuda()
        self.offsets = self.offsets.contiguous()

    def get_image_perm(self):
        return torch.randperm(self.dataset.n_images)
    
    def update_learning_rate(self, it = 0):
        alpha = self.learning_rate_alpha
        progress = it / self.end_iter
        learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha

        for g in self.optimizer.param_groups:
            g['lr'] = self.learning_rate * learning_factor

        for g in self.optimizer_feat.param_groups:
            g['lr'] = self.learning_rate_feat * learning_factor
            
        for g in self.optimizer_sdf.param_groups:
            g['lr'] = self.learning_rate_sdf * learning_factor
            
        for g in self.optimizer_cvt.param_groups:
            g['lr'] = self.learning_rate_cvt * learning_factor

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
        runner.train(args.data_name, 24, False)