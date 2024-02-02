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

        self.report_freq = self.conf.get_float('train.report_freq')
        self.val_freq = self.conf.get_float('train.val_freq')

        self.vortSDF_renderer_coarse = VortSDFDirectRenderer(**self.conf['model.cvt_renderer'])

        self.vortSDF_renderer_fine = VortSDFRenderer(**self.conf['model.cvt_renderer'])
        
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


    def train(self, data_name, verbose = True):
        ##### 2. Load initial sites
        if not hasattr(self, 'tet32'):
            ##### 2. Load initial sites
            visual_hull = [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]
            import src.Geometry.sampling as sampler
            res = 32
            sites = sampler.sample_Bbox(visual_hull[0:3], visual_hull[3:6], res, perturb_f =  (visual_hull[3] - visual_hull[0])*0.01)
            
            #### Add cameras as sites 
            cam_sites = np.stack([self.dataset.pose_all[id, :3,3].cpu().numpy() for id in range(self.dataset.n_images)])
            sites = np.concatenate((sites, cam_sites))

            self.tet32 = tet32.Tet32(sites)
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

        sites = torch.from_numpy(sites.astype(np.float32)).cuda()
        print (sites.shape)

        
        ##### 2. Initialize SDF field    
        if not hasattr(self, 'sdf'):
            norm_sites = torch.linalg.norm(sites, ord=2, axis=-1, keepdims=True)
            self.sdf = norm_sites[:,0] - 0.5
            self.sdf = self.sdf.contiguous()
            self.sdf.requires_grad_(True)
        
        self.tet32.surface_from_sdf(self.sdf.detach().cpu().numpy().reshape(-1), "Exp/bmvs_man/test_tri.ply")

        
        ##### 2. Initialize feature field    
        if not hasattr(self, 'fine_features'):
            self.fine_features = 0.5*torch.ones([self.sdf.shape[0], 6]).cuda()       
            self.fine_features = self.fine_features.contiguous()
            self.fine_features.requires_grad_(True)
            self.grad_features = torch.zeros([self.sdf.shape[0], 6]).cuda()       
            self.grad_features = self.grad_features.contiguous()
            
        self.Allocate_batch_data()

        
        if not hasattr(self, 'optimizer_sdf'):
            self.optimizer_sdf = torch.optim.Adam([self.sdf], lr=self.learning_rate_sdf)        
            self.optimizer_feat= torch.optim.Adam([self.fine_features], lr=self.learning_rate_feat) 

        self.vortSDF_renderer_coarse.prepare_buffs(self.batch_size, self.n_samples, sites.shape[0])
        self.vortSDF_renderer_fine.prepare_buffs(self.batch_size, self.n_samples, sites.shape[0])
        
        self.s_max = 1000
        self.R = 50
        self.s_start = 10.0
        self.inv_s = 0.1
        image_perm = self.get_image_perm()
        num_rays = 512
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
            nb_samples = self.tet32.sample_rays_cuda(img_idx, rays_d, self.sdf, self.fine_features, cam_ids, self.in_weights, self.in_z, self.in_sdf, self.in_feat, self.in_ids, self.offsets)    
            if verbose:
                print('CVT_Sample time:', timer() - start)    
                
            start = timer()
            self.samples[:] = 0.0
            tet32_march_cuda.fill_samples(rays_o.shape[0], self.n_samples, rays_o, rays_d, self.tet32.sites, 
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

            samples = self.samples[:nb_samples,:]
            samples = samples.contiguous()

            ##### ##### ##### ##### ##### ##### ##### 
            # Build fine features
            fine_features = self.out_feat[:nb_samples] #self.fine_features[self.out_ids[:nb_samples, 1]]
            self.colors = fine_features[:,:3] 
            self.colors = self.colors.contiguous()

            
            ##### ##### ##### ##### ##### ##### 
            xyz_emb = (samples.unsqueeze(-1) * self.posfreq).flatten(-2)
            xyz_emb = torch.cat([samples, xyz_emb.sin(), xyz_emb.cos()], -1)

            viewdirs_emb = (self.samples_rays[:nb_samples,:].unsqueeze(-1) * self.viewfreq).flatten(-2)
            viewdirs_emb = torch.cat([self.samples_rays[:nb_samples,:], viewdirs_emb.sin(), viewdirs_emb.cos()], -1)

            rgb_feat = torch.cat([xyz_emb, viewdirs_emb, fine_features], -1)
            rgb_feat.requires_grad_(True)
            rgb_feat.retain_grad()

            self.colors_fine = torch.sigmoid(self.color_network.rgb(rgb_feat)) 
            self.colors_fine = self.colors_fine.contiguous()

            ########################################
            ####### Render the image ###############
            ########################################
            start = timer()       
            self.vortSDF_renderer_coarse.render_gpu(rays_o.shape[0], self.inv_s, self.out_sdf, self.tet32.knn_sites, self.out_weights, self.colors, true_rgb, mask, self.out_ids, self.offsets)
            
            color_fine_loss = VortSDFRenderingFunction.apply(self.vortSDF_renderer_fine, rays_o.shape[0], self.inv_s, self.out_sdf, self.tet32.knn_sites, self.out_weights, self.colors_fine, true_rgb, mask, self.out_ids, self.offsets)
            if verbose:
                print('RenderingFunction time:', timer() - start)

            # Total loss   
            mask = (mask > 0.5).float()
            mask_sum = mask.sum()
            
            self.optimizer.zero_grad()
            loss = color_fine_loss / (mask_sum + 1.0e-5)
            loss.backward()

            ########################################
            # Backprop feature gradients to gradients on sites
            # step optimize color features
            self.grad_features[:] = 0.0
            #cvt_lib.Back_Prop_Feat(self.grad_feat_vol.data_ptr(), self.fine_rgb_feat.grad[:, :6].data_ptr(), self.fine_features_id.data_ptr(), self.z_facts.data_ptr(), self.fine_features_w.data_ptr(), nb_samples)
            self.grad_features[:, :3] = self.vortSDF_renderer_coarse.grads_color[:,:] / (mask_sum + 1.0e-5)
            self.grad_features[outside_flag[:] == 1.0] = 0.0   

            
            self.optimizer.step()
            
            self.optimizer_feat.zero_grad()
            self.fine_features.grad = self.grad_features
            self.optimizer_feat.step()

            ########################################
            ####### Optimize sdf values ############
            ########################################

            #grad_sdf = (0.5*self.voro_renderer_coarse.grads_sdf[:,0] + 1.0*self.voro_renderer_fine.grads_sdf[:,0]) / (mask_sum + 1.0e-5)
            grad_sdf = self.vortSDF_renderer_fine.grads_sdf / (mask_sum + 1.0e-5)
            grad_sdf[outside_flag[:] == 1.0] = 0.0   

            self.optimizer_sdf.zero_grad()
            self.sdf.grad = grad_sdf 
            self.optimizer_sdf.step()

            
            if iter_step % self.report_freq == 0:
                print('iter:{:8>d} loss = {} lr={}'.format(iter_step, loss, self.optimizer_sdf.param_groups[0]['lr']))

            if iter_step % self.val_freq == 0:
                self.render_image(cam_ids, img_idx)
                self.tet32.surface_from_sdf(self.sdf.detach().cpu().numpy().reshape(-1), "Exp/bmvs_man/test_tri.ply")

            self.update_learning_rate(iter_step)


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
        
        img_mask = torch.zeros([(self.dataset.H // resolution_level) * (self.dataset.W // resolution_level)], dtype = torch.float32).cuda()
        img_mask = img_mask.contiguous()
        img_mask[:] = 0
        
        colors_out = torch.zeros([self.batch_size*3]).to(torch.device('cuda')).contiguous()
        mask_out = torch.zeros([self.batch_size]).to(torch.device('cuda')).contiguous()
        it = 0
        for rays_o_batch, rays_d_batch in zip(rays_o.split(self.batch_size), rays_d.split(self.batch_size)):
            rays_o_batch = rays_o_batch.contiguous()
            rays_d_batch = rays_d_batch.contiguous()

            ## sample points along the rays
            start = timer()
            self.offsets[:] = 0
            nb_samples = self.tet32.sample_rays_cuda(img_idx, rays_d_batch, self.sdf, self.fine_features, cam_ids, self.in_weights, self.in_z, self.in_sdf, self.in_feat, self.in_ids, self.offsets)    
                
            start = timer()
            self.samples[:] = 0.0
            tet32_march_cuda.fill_samples(rays_o_batch.shape[0], self.n_samples, rays_o_batch, rays_d_batch, self.tet32.sites, 
                                            self.in_z, self.in_sdf, self.in_feat, self.in_weights, self.in_ids, 
                                            self.out_z, self.out_sdf, self.out_feat, self.out_weights, self.out_ids, 
                                            self.offsets, self.samples, self.samples_loc, self.samples_rays)
            
            samples = self.samples[:nb_samples,:]
            samples = samples.contiguous()
            
            #self.colors = self.out_feat[:nb_samples,:3] 
            #self.colors = self.colors_fine.contiguous()
            
            ##### ##### ##### ##### ##### ##### 
            xyz_emb = (samples.unsqueeze(-1) * self.posfreq).flatten(-2)
            xyz_emb = torch.cat([samples, xyz_emb.sin(), xyz_emb.cos()], -1)

            viewdirs_emb = (self.samples_rays[:nb_samples,:].unsqueeze(-1) * self.viewfreq).flatten(-2)
            viewdirs_emb = torch.cat([self.samples_rays[:nb_samples,:], viewdirs_emb.sin(), viewdirs_emb.cos()], -1)

            rgb_feat = torch.cat([xyz_emb, viewdirs_emb, self.out_feat[:nb_samples]], -1)
            rgb_feat.requires_grad_(True)
            rgb_feat.retain_grad()

            self.colors_fine = torch.sigmoid(self.color_network.rgb(rgb_feat)) 
            self.colors_fine = self.colors_fine.contiguous()

            ########################################
            ####### Render the image ###############
            ########################################
            renderer_cuda.render_no_grad(rays_o_batch.shape[0], self.inv_s, self.out_sdf, self.colors_fine, self.offsets, colors_out, mask_out)
            #renderer_cuda.render_no_grad(rays_o.shape[0], self.inv_s, self.out_sdf, self.colors, self.offsets, colors_out, mask_out)

            start = 3*it*self.batch_size
            end = min(3*(it+1)*self.batch_size, 3*(self.dataset.H // resolution_level) * (self.dataset.W // resolution_level))
            img[start:end] = colors_out[:(end-start)]

            start = it*self.batch_size
            end = min((it+1)*self.batch_size, (self.dataset.H // resolution_level) * (self.dataset.W // resolution_level))           
            img_mask[start:end] = mask_out[:(end-start)]
            
            it = it + 1

            
        mask = img_mask.reshape(-1,1)

        img = img.reshape(self.dataset.H // resolution_level, self.dataset.W // resolution_level, 3)
        img = img.cpu().numpy()
        cv2.imwrite('Exp/synt.png', 255*img[:,:])
        
        GTimg = true_rgb.reshape(self.dataset.H // resolution_level, self.dataset.W // resolution_level, 3).cpu().numpy()
        cv2.imwrite('Exp/GT.png', 255*GTimg[:,:])

        mask = mask.reshape(self.dataset.H // resolution_level, self.dataset.W // resolution_level).cpu().numpy()
        cv2.imwrite('Exp/Mask.png', 255*mask[:])
        
        
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
        self.in_weights = self.in_weights.contiguous()

        self.in_ids = -torch.ones([6*self.n_samples* self.batch_size], dtype=torch.int32).cuda()
        self.in_ids = self.in_ids.contiguous()
        
        self.in_z = torch.zeros([2*self.n_samples * self.batch_size], dtype=torch.float32).cuda()
        self.in_z = self.in_z.contiguous()
        
        self.in_sdf = torch.zeros([2*self.n_samples * self.batch_size], dtype=torch.float32).cuda()
        self.in_sdf = self.in_sdf.contiguous()
        
        self.in_feat = torch.zeros([6*self.n_samples * self.batch_size], dtype=torch.float32).cuda()
        self.in_feat = self.in_feat.contiguous()
        
        self.out_ids = -torch.ones([self.n_samples* self.batch_size, 6], dtype=torch.int32).cuda()
        self.out_ids = self.out_ids.contiguous()
        
        self.out_z = torch.zeros([self.n_samples * self.batch_size], dtype=torch.float32).cuda()
        self.out_z = self.out_z.contiguous()
        
        self.out_sdf = torch.zeros([self.n_samples * self.batch_size, 2], dtype=torch.float32).cuda()
        self.out_sdf = self.out_sdf.contiguous()
        
        self.out_feat = torch.zeros([self.n_samples * self.batch_size, 6], dtype=torch.float32).cuda()
        self.out_feat = self.out_feat.contiguous()
        
        self.out_weights = torch.zeros([6*self.n_samples * self.batch_size], dtype=torch.float32).cuda()
        self.out_weights = self.out_weights.contiguous()
        
        self.offsets = torch.zeros([2*self.batch_size], dtype=torch.int32).cuda()
        self.offsets = self.offsets.contiguous()

    def get_image_perm(self):
        return torch.randperm(self.dataset.n_images)
    
    def update_learning_rate(self, it = 0):
        alpha = self.learning_rate_alpha
        progress = it / self.end_iter
        learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha

        #for g in self.optimizer.param_groups:
        #    g['lr'] = self.learning_rate * learning_factor

        for g in self.optimizer_feat.param_groups:
            g['lr'] = self.learning_rate_feat * learning_factor
            
        for g in self.optimizer_sdf.param_groups:
            g['lr'] = self.learning_rate_sdf * learning_factor

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