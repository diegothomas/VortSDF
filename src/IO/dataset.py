import torch
import torch.nn.functional as F
import cv2 as cv
import numpy as np
import os
from glob import glob
from icecream import ic
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
from src.Confs.VisHull import Load_Visual_Hull
import cv2
from skimage.measure import block_reduce
import imageio
import xml.etree.cElementTree as ET


def load_Rt_from(filename):
    lines = open(filename).read().splitlines()
    lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
    pose = np.asarray(lines).astype(np.float32).squeeze()

    return pose

# This function is borrowed from IDR: https://github.com/lioryariv/idr
def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]
    
    pose_inv = np.eye(4, dtype=np.float32)
    pose_inv[:3, :3] = R
    pose_inv[:3, 3] = np.matmul(-(t[:3] / t[3])[:, 0], R)

    return intrinsics, pose, pose_inv


class Dataset:
    def __init__(self, conf , data_name = '', data_type = 'BMVS', res = 0):
        super(Dataset, self).__init__()
        print('Load data: Begin')
        self.device = torch.device('cuda')
        self.conf = conf

        self.data_dir = conf.get_string('data_dir')
        self.data_dir = self.data_dir.replace('DATA_NAME', data_name)
        self.data_type = conf.get_string('data_type')
        print(self.data_dir)
        print(self.data_type)
        if self.data_type == 'BMVS':
            self.render_cameras_name = conf.get_string('render_cameras_name')
            self.object_cameras_name = conf.get_string('object_cameras_name')

            self.camera_outside_sphere = conf.get_bool('camera_outside_sphere', default=True)
            self.scale_mat_scale = conf.get_float('scale_mat_scale', default=1.1)

            camera_dict = np.load(os.path.join(self.data_dir, self.render_cameras_name))
            self.camera_dict = camera_dict
            self.images_lis = sorted(glob(os.path.join(self.data_dir, 'image/*.png')))
            self.n_images = len(self.images_lis)
            self.images_np = np.stack([cv.imread(im_name) for im_name in self.images_lis]) / 255.0
            self.masks_lis = sorted(glob(os.path.join(self.data_dir, 'mask/*.png')))
            self.masks_np = np.stack([cv.imread(im_name) for im_name in self.masks_lis]) / 255.0
            for i in range(self.n_images):
                self.images_np[i][self.masks_np[i][:,:,0] == 0.0,:] = [0.5,0.5,0.5]

            # world_mat is a projection matrix from world to image
            self.world_mats_np = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

            self.scale_mats_np = []

            # scale_mat: used for coordinate normalization, we assume the scene to render is inside a unit sphere at origin.
            self.scale_mats_np = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

            self.intrinsics_all = []
            self.pose_all = []
            self.pose_all_inv = []
            self.z_buff = [] 

            for scale_mat, world_mat in zip(self.scale_mats_np, self.world_mats_np):
                P = world_mat @ scale_mat
                P = P[:3, :4]
                intrinsics, pose, pose_inv = load_K_Rt_from_P(None, P)
                self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
                self.pose_all.append(torch.from_numpy(pose).float())
                self.pose_all_inv.append(torch.from_numpy(pose_inv).float())
                
            #object_bbox_min = np.array([-1.01, -1.01, -1.01, 1.0])
            #object_bbox_max = np.array([ 1.01,  1.01,  1.01, 1.0])
            object_bbox_min = np.array([-0.51, -0.51, -0.51, 1.0])
            object_bbox_max = np.array([ 0.51,  0.51,  0.51, 1.0])
            # Object scale mat: region of interest to **extract mesh**
            object_scale_mat = np.load(os.path.join(self.data_dir, self.object_cameras_name))['scale_mat_0']
            object_bbox_min = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_min[:, None]
            object_bbox_max = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_max[:, None]
            self.object_bbox_min = object_bbox_min[:3, 0]
            self.object_bbox_max = object_bbox_max[:3, 0]

        elif self.data_type == 'TANKS_AND_TEMPLES':
            self.render_cameras_name = self.data_dir + 'intrinsics.txt'
            self.object_cameras_name = self.data_dir + 'test_traj.txt'
            self.images_lis = sorted(glob(os.path.join(self.data_dir, 'rgb/*')))
            self.n_images = len(self.images_lis)
            self.images_np = np.stack([cv.imread(im_name) for im_name in self.images_lis]) / 255.0
            print(self.images_np[0].max())
            self.masks_np = np.stack([np.ones_like(self.images_np[i]) for i in range(self.n_images)])
            for i in range(self.n_images):
                mark = self.images_np[i][:,:,0] * self.images_np[i][:,:,1] * self.images_np[i][:,:,2]                
                print(self.images_np[i].shape)
                if data_name == "Ignatius" or data_name == "Fountain":
                    self.masks_np[i][np.linalg.norm(self.images_np[i], ord=2, axis=-1, keepdims=True)[:,:,0] == 0,:] = 0
                else:
                    self.masks_np[i][mark == 1,:] = 0
                    
            for i in range(self.n_images):
                self.images_np[i][self.masks_np[i][:,:,0] == 0.0,:] = [1.0,1.0,1.0]
            
            self.intrinsics_all = []
            self.pose_all = []
            self.pose_all_inv = []
            self.z_buff = [] 

            self.pose_lis = sorted(glob(os.path.join(self.data_dir, 'pose/*.txt')))
            self.pose_all = [torch.from_numpy(load_Rt_from(pose_name)).float() for pose_name in self.pose_lis]
            self.pose_all_inv = [torch.from_numpy(load_Rt_from(pose_name)).float() for pose_name in self.pose_lis]

            self.intrinsics_all = [torch.from_numpy(load_Rt_from(glob(os.path.join(self.data_dir, 'intrinsics.txt'))[0])).float() for _ in self.pose_lis]
            
            lines = open(glob(os.path.join(self.data_dir, 'bbox.txt'))[0]).read().splitlines()
            lines = [[x[0], x[1], x[2], x[3], x[4], x[5]] for x in (x.split(" ") for x in lines)]
            self.object_bbox_min = np.asarray(lines[0][0:3]).astype(np.float32).squeeze()
            self.object_bbox_max = np.asarray(lines[0][3:6]).astype(np.float32).squeeze()
            print(self.object_bbox_min)
            print(self.object_bbox_max)

            # world_mat is a projection matrix from world to image
            self.world_mats_np = [np.array([0.0, 0.0, 0.0]).astype(np.float32) for idx in range(self.n_images)]

            self.scale_mats_np = [np.eye(4,4)] #[load_Rt_from(self.data_dir + 'Barn_trans.txt')] #
            print(load_Rt_from(self.data_dir + 'trans.txt'))

        elif self.data_type == 'DTU':
            normalize = True
            mask = True
            white_bg = True

            self.images_lis = sorted(glob(os.path.join(self.data_dir, 'image', '*png')))
            if len(self.images_lis) == 0:
                self.images_lis = sorted(glob(os.path.join(self.data_dir, 'image', '*jpg')))
            if len(self.images_lis) == 0:
                self.images_lis = sorted(glob(os.path.join(self.data_dir, 'rgb', '*png')))

            self.masks_lis = sorted(glob(os.path.join(self.data_dir, 'mask', '*png')))
            if len(self.masks_lis) == 0:
                self.masks_lis = sorted(glob(os.path.join(self.data_dir, 'mask', '*jpg')))

            self.render_cameras_name = 'cameras_sphere.npz' if normalize else 'cameras_large.npz'
            self.scale_mats_np = conf.get_float('scale_mat_scale', default=1.1)
            camera_dict = np.load(os.path.join(self.data_dir, self.render_cameras_name))
            self.world_mats_np = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(len(self.images_lis))]
            if normalize:
                self.scale_mats_np  = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(len(self.images_lis))]
            else:
                self.scale_mats_np  = None


            self.intrinsics_all = []
            self.pose_all = []
            self.pose_all_inv = []
            all_imgs = []
            all_masks = []
            for i, (world_mat, im_name) in enumerate(zip(self.world_mats_np, self.images_lis)):
                if normalize:
                    P = world_mat @ self.scale_mats_np[i]
                else:
                    P = world_mat
                P = P[:3, :4]
                intrinsics, pose, pose_inv = load_K_Rt_from_P(None, P)
                self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
                self.pose_all.append(torch.from_numpy(pose).float())
                self.pose_all_inv.append(torch.from_numpy(pose_inv).float())
                if len(self.masks_lis) > 0:
                    mask_ = (cv.imread(self.masks_lis[i]) / 255.).astype(np.float32)
                    if mask_.ndim == 3:
                        all_masks.append(mask_[...,:3])
                    else:
                        all_masks.append(mask_[...,None])
                all_imgs.append((cv.imread(im_name) / 255.).astype(np.float32))

                
            self.images_np = np.stack(all_imgs, 0)
            #self.pose_all = np.stack(self.all_poses, 0)
            #self.pose_all_inv = np.stack(self.pose_all_inv, 0)
            H, W = self.images_np[0].shape[:2]
            K = self.intrinsics_all[0]
            focal = self.intrinsics_all[0][0,0]
            print("Date original shape: ", H, W)
            self.masks_np = np.stack(all_masks, 0)
            if mask:
                assert len(self.masks_lis) > 0
                bg = 1. if white_bg else 0.
                self.images_np = self.images_np * self.masks_np + bg * (1 - self.masks_np)

                
            self.n_images = len(self.images_lis)

            self.z_buff = [] 
                
            object_bbox_min = np.array([-1.0, -1.0, -1.0, 1.0])
            object_bbox_max = np.array([ 1.0,  1.0,  1.0, 1.0])
            #object_scale_mat = np.load(os.path.join(self.data_dir, self.object_cameras_name))['scale_mat_0']
            #object_bbox_min = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_min[:, None]
            #object_bbox_max = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_max[:, None]
            self.object_bbox_min = object_bbox_min[:3]
            self.object_bbox_max = object_bbox_max[:3]

        elif self.data_type == 'KINOVIS':
            reso_level = 1
            mask = True
            white_bg = False
            NCAMS = 68
            rgb_paths = [self.data_dir + f"/ImagesUndistorted/cam-{i+1}.png" for i in range(NCAMS)]
            mask_paths = [self.data_dir + f"/Masks/cam-{i+1}.png" for i in range(NCAMS)]
            
            root = ET.parse(self.data_dir + "/calibration_undistorted.xml").getroot()
            matrices = []
            for c in root.findall("Camera"):
                K = np.array([float(f) for f in c.find("K").text.split(' ')]).reshape((3,3))
                R = np.array([float(f) for f in c.find("R").text.split(' ')]).reshape((3,3))
                T = np.array([float(f) for f in c.find("T").text.split(' ')]).reshape((3,1))
                pose = np.eye(4)
                pose[:3, :3] = R.T # From camera to world rotation
                pose[:3, 3] = (-R.T @ T)[:, 0] # Camera center in world pose
                matrices.append((K, pose))

            all_intrinsics = []
            all_poses = []
            all_imgs = []
            all_masks = []
            all_min_x = []
            all_max_x = []
            all_min_y = []
            all_max_y = []
            for i in range(NCAMS):
                print(f"Loading image {i}...")
                intrinsic_curr = np.array(matrices[i][0])
                intrinsic_curr[:2, :] /= reso_level
                all_intrinsics.append(torch.from_numpy(intrinsic_curr).float())
                all_poses.append(torch.from_numpy(matrices[i][1]).float())
                if len(mask_paths) > 0:
                    mask_ = (cv.imread(mask_paths[i]) / 255.).astype(np.float32)
                    if mask_.ndim == 3:
                        all_masks.append(mask_[...,:3])
                    else:
                        all_masks.append(mask_[...,None])
                        
                    tx = torch.linspace(0, mask_.shape[1] - 1, mask_.shape[1])
                    ty = torch.linspace(0, mask_.shape[0] - 1, mask_.shape[0])
                    p_x, p_y = torch.meshgrid(tx, ty)

                    mask_pp = mask_[(p_y.long(), p_x.long())]
                    p_front_x = p_x[mask_pp[:,:,0] == 1]
                    p_front_y = p_y[mask_pp[:,:,0] == 1]

                    min_x = int(p_front_x.min().numpy()) 
                    max_x = int(p_front_x.max().numpy()) 
                    min_y = int(p_front_y.min().numpy()) 
                    max_y = int(p_front_y.max().numpy()) 

                    all_min_x.append(min_x)
                    all_max_x.append(max_x)
                    all_min_y.append(min_y)
                    all_max_y.append(max_y)

                all_imgs.append((cv.imread(rgb_paths[i]) / 255.).astype(np.float32))
                print(cv.imread(rgb_paths[i]).shape)
            imgs = np.stack(all_imgs, 0)
            #poses = np.stack(all_poses, 0)
            H, W = imgs[0].shape[:2]
            focal = all_intrinsics[0][0,0]
            #all_intrinsics = torch.from_numpy(np.array(all_intrinsics))
            print("Date original shape: ", H, W)
            masks = np.stack(all_masks, 0)
            if mask:
                assert len(mask_paths) > 0
                bg = 1. if white_bg else 0.
                imgs = imgs * masks + bg * (1 - masks)
            if reso_level > 1:
                H, W = int(H / reso_level), int(W / reso_level)
                imgs =  F.interpolate(torch.from_numpy(imgs).permute(0,3,1,2), size=(H, W)).permute(0,2,3,1).numpy()
                if masks is not None:
                    masks =  F.interpolate(torch.from_numpy(masks).permute(0,3,1,2), size=(H, W)).permute(0,2,3,1).numpy()
                #all_intrinsics[:, :2, :] /= reso_level
                focal /= reso_level

            self.intrinsics_all = all_intrinsics
            self.pose_all = all_poses
            self.pose_all_inv = all_poses
            self.images_np = imgs
            self.masks_np = masks
            self.n_images = NCAMS
            self.all_min_x = np.stack(all_min_x, 0)
            self.all_max_x = np.stack(all_max_x, 0)
            self.all_min_y = np.stack(all_min_y, 0)
            self.all_max_y = np.stack(all_max_y, 0)
            
            for i in range(self.n_images):
                self.images_np[i][self.masks_np[i][:,:,0] == 0.0,:] = [1.0,1.0,1.0]
                
            # world_mat is a projection matrix from world to image
            trans_file = sorted(glob(os.path.join(self.data_dir, 'transform_*.txt')))
            print(trans_file)
            self.world_mats_np = [np.array([0.0, 0.0, 0.0]).astype(np.float32) for idx in range(self.n_images)]
            self.scale_mats_np = [load_Rt_from(trans_file[0])] #

            GTtrans = load_Rt_from(trans_file[0]) #
            rot_inv = GTtrans[:3,:3] 
            t_inv = -np.dot(GTtrans[:3,3].T, rot_inv)
            GTtrans[:3,3] = t_inv[:]
            self.scale_mats_np = [GTtrans]
        else:
            print("unknown datatype")
            exit()

        self.images_smooth_np_5 = block_reduce(self.images_np, block_size=(1,5,5,1), func=np.mean)
        self.masks_smooth_np_5 = block_reduce(self.masks_np, block_size=(1,5,5,1), func=np.mean)
        
        self.images_smooth_np_3 = block_reduce(self.images_np, block_size=(1,3,3,1), func=np.mean)
        self.masks_smooth_np_3 = block_reduce(self.masks_np, block_size=(1,3,3,1), func=np.mean)
        
        self.images_smooth_np_2 = block_reduce(self.images_np, block_size=(1,2,2,1), func=np.mean)
        self.masks_smooth_np_2 = block_reduce(self.masks_np, block_size=(1,2,2,1), func=np.mean)
        """for i in range(self.n_images):
            cv2.imwrite('Exp/img.png', 255*self.images_smooth_np[i,:,:,:])
            print(self.images_smooth_np.shape)
            print(self.images_np.shape)
            input()"""

        self.images = torch.from_numpy(self.images_np.astype(np.float32)).cpu()  # [n_images, H, W, 3]
        self.masks  = torch.from_numpy(self.masks_np.astype(np.float32)).cpu()   # [n_images, H, W, 3]
        self.images_smooth_5 = torch.from_numpy(self.images_smooth_np_5.astype(np.float32)).cpu()  # [n_images, H, W, 3]
        self.masks_smooth_5  = torch.from_numpy(self.masks_smooth_np_5.astype(np.float32)).cpu()   # [n_images, H, W, 3]
        self.images_smooth_3 = torch.from_numpy(self.images_smooth_np_3.astype(np.float32)).cpu()  # [n_images, H, W, 3]
        self.masks_smooth_3  = torch.from_numpy(self.masks_smooth_np_3.astype(np.float32)).cpu()   # [n_images, H, W, 3]
        self.images_smooth_2 = torch.from_numpy(self.images_smooth_np_2.astype(np.float32)).cpu()  # [n_images, H, W, 3]
        self.masks_smooth_2  = torch.from_numpy(self.masks_smooth_np_2.astype(np.float32)).cpu()   # [n_images, H, W, 3]
        self.intrinsics_all = torch.stack(self.intrinsics_all).to(self.device)   # [n_images, 4, 4]
        self.intrinsics_all_inv = torch.inverse(self.intrinsics_all)  # [n_images, 4, 4]
        self.focal = self.intrinsics_all[0][0, 0]
        self.pose_all = torch.stack(self.pose_all).to(self.device)  # [n_images, 4, 4]
        self.pose_all_inv = torch.stack(self.pose_all_inv).to(self.device)  # [n_images, 4, 4]
        self.H, self.W = self.images.shape[1], self.images.shape[2]
        self.H_smooth_2, self.W_smooth_2 = self.images_smooth_2.shape[1], self.images_smooth_2.shape[2]
        self.H_smooth_3, self.W_smooth_3 = self.images_smooth_3.shape[1], self.images_smooth_3.shape[2]
        self.H_smooth_5, self.W_smooth_5 = self.images_smooth_5.shape[1], self.images_smooth_5.shape[2]
        print(self.H, self.W)
        self.image_pixels = self.H * self.W

        print('Load data: End')

    def save_dataset(self, data_name):
        self.render_cameras_name = self.data_dir + 'intrinsics.txt'
        self.object_cameras_name = self.data_dir + 'test_traj.txt'
        intr = self.intrinsics_all[0].cpu().numpy()

        with open(self.render_cameras_name, 'w') as f:
            for i in range(4):
                f.write(f"{intr[i,0]} {intr[i,1]} {intr[i,2]} {intr[i,3]}\n")
                
        object_bbox = Load_Visual_Hull(data_name, self)
        with open(self.data_dir + 'bbox.txt', 'w') as f:
            f.write(f"{object_bbox[0]} {object_bbox[3]} {object_bbox[1]} {object_bbox[4]} {object_bbox[2]} {object_bbox[5]}\n")

        for idx in range(self.n_images):
            extr = self.pose_all[idx].cpu().numpy()
            with open(self.data_dir + '/pose/' + str(idx).zfill(3)+'.txt', 'w') as f:
                for i in range(4):
                    f.write(f"{extr[i,0]} {extr[i,1]} {extr[i,2]} {extr[i,3]}\n")

            img = self.images[idx] * self.masks[idx]
            img[self.masks[idx] == 0] = 1.0
            img = img.cpu().numpy()
            cv2.imwrite(self.data_dir + '/rgb/'+str(idx).zfill(3)+'.png', 255*img[:,:])

        exit()


    def cluster_cameras(self):
        self.clusters = torch.zeros((self.n_images,4))

        for im_id in range(self.n_images):
            dist_vector = torch.stack([self.pose_all[im_id, :3,3] - self.pose_all[id, :3,3] for id in range(self.n_images)])
            dist_vector = torch.linalg.norm(dist_vector, ord=2, axis=-1, keepdims=True)[:,0]         
            _, indices = torch.topk(dist_vector, 5, dim=0, largest = False)
            self.clusters[im_id, :] = indices[1:]
        return self.clusters
    
    def gen_rays_proj(self, img_idx, samples):

        z_buff_s = [] 
        color_s = [] 
        mask_s = [] 
        rays_o_s = [] 
        rays_v_s = [] 
        pixels_s = [] 
        img_ids = [] 
        for id in range(4):        
            id_curr = self.clusters[img_idx, id].long()
            rays_o = self.pose_all[id_curr, None, :3, 3].expand(samples.shape)
            p = samples-rays_o
            rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)
            
            p = torch.transpose(p,0,1)
            p = torch.matmul(self.pose_all_inv[id_curr,:3,:3], p[:,:]).squeeze()
            p = torch.matmul(self.intrinsics_all[id_curr, :3,:3], p[:,:]).squeeze()
            z = torch.stack([p[2,:], p[2,:]], dim=0)
            pixels = (p[:2,:]/z[:]).cpu().detach().numpy()
            pixels = pixels.astype(np.int32)
        
            rays_o = rays_o[pixels[0,:] >= 0,:]
            rays_v = rays_v[pixels[0,:] >= 0,:]
            pixels = pixels[:,pixels[0,:] >= 0]

            rays_o = rays_o[pixels[1,:] >= 0,:]
            rays_v = rays_v[pixels[1,:] >= 0,:]
            pixels = pixels[:,pixels[1,:] >= 0]

            rays_o = rays_o[pixels[0,:] <= self.W-1,:]
            rays_v = rays_v[pixels[0,:] <= self.W-1,:]
            pixels = pixels[:,pixels[0,:] <= self.W-1]

            rays_o = rays_o[pixels[1,:] <= self.H-1,:]
            rays_v = rays_v[pixels[1,:] <= self.H-1,:]
            pixels = pixels[:,pixels[1,:] <=  self.H-1]


            """pixels[:,pixels[0,:] < 0] = 0
            pixels[:,pixels[1,:] < 0] = 0
            pixels[:,pixels[0,:] > self.W-1] = 0
            pixels[:,pixels[1,:] > self.H-1] = 0"""

            pixels = torch.from_numpy(pixels).long()

            z_buff = self.z_buff[id_curr][pixels[1,:], pixels[0,:]].cuda() 
            norm_pix = np.linalg.norm(pixels) 
            z_buff[norm_pix == 0] = 0

            color = (self.images[id_curr][(pixels[1,:], pixels[0,:])]).cuda()    # batch_size, 3
            mask = (self.masks[id_curr][(pixels[1,:], pixels[0,:])]).cuda() 

            z_buff_s.append(z_buff)
            color_s.append(color)
            mask_s.append(mask)
            rays_o_s.append(rays_o)
            rays_v_s.append(rays_v)
            pixels_s.append(torch.transpose(pixels,0,1))
            img_ids.append(torch.from_numpy(np.array([id_curr, pixels.shape[1]])).long())
        
        z_buff_s = torch.concat(z_buff_s).to(self.device) 
        color_s = torch.concat(color_s).to(self.device) 
        mask_s = torch.concat(mask_s).to(self.device) 
        rays_o_s = torch.concat(rays_o_s).to(self.device) 
        rays_v_s = torch.concat(rays_v_s).to(self.device) 
        #pixels_s = torch.concat(pixels_s).to(self.device) 
        img_ids = torch.concat(img_ids).to(self.device) 

        return z_buff_s, pixels_s, img_ids, torch.cat([rays_o_s, rays_v_s, color_s, mask_s], dim=-1)
    
        #return z_buff.transpose(0, 1), torch.cat([rays_o.transpose(0, 1), rays_v.transpose(0, 1), color.transpose(0, 1), mask[:, :, :1].transpose(0, 1)], dim=-1)
    

    def gen_rays_re_proj(self, img_idx, samples, step):

        z_buff_s = [] 
        color_s = [] 
        mask_s = [] 
        rays_o_s = [] 
        rays_v_s = [] 
        pixels_s = [] 
        img_ids = [] 
        for id in range(4):   
            samples = samples + torch.rand(samples.shape).cuda()*step - step/2.0     
            id_curr = img_idx #self.clusters[img_idx, id].long()
            
            rays_o = self.pose_all[id_curr, None, :3, 3].expand(samples.shape)
            p = samples-rays_o
            rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)
            
            p = torch.transpose(p,0,1)
            p = torch.matmul(self.pose_all_inv[id_curr,:3,:3], p[:,:]).squeeze()
            p = torch.matmul(self.intrinsics_all[id_curr, :3,:3], p[:,:]).squeeze()
            z = torch.stack([p[2,:], p[2,:]], dim=0)
            pixels = (p[:2,:]/z[:]).cpu().detach().numpy()
            pixels = pixels.astype(np.int32)
        
            rays_o = rays_o[pixels[0,:] >= 0,:]
            rays_v = rays_v[pixels[0,:] >= 0,:]
            pixels = pixels[:,pixels[0,:] >= 0]

            rays_o = rays_o[pixels[1,:] >= 0,:]
            rays_v = rays_v[pixels[1,:] >= 0,:]
            pixels = pixels[:,pixels[1,:] >= 0]

            rays_o = rays_o[pixels[0,:] <= self.W-1,:]
            rays_v = rays_v[pixels[0,:] <= self.W-1,:]
            pixels = pixels[:,pixels[0,:] <= self.W-1]

            rays_o = rays_o[pixels[1,:] <= self.H-1,:]
            rays_v = rays_v[pixels[1,:] <= self.H-1,:]
            pixels = pixels[:,pixels[1,:] <=  self.H-1]


            """pixels[:,pixels[0,:] < 0] = 0
            pixels[:,pixels[1,:] < 0] = 0
            pixels[:,pixels[0,:] > self.W-1] = 0
            pixels[:,pixels[1,:] > self.H-1] = 0"""

            pixels = torch.from_numpy(pixels).long()

            z_buff = self.z_buff[id_curr][pixels[1,:], pixels[0,:]].cuda() 
            norm_pix = np.linalg.norm(pixels) 
            z_buff[norm_pix == 0] = 0

            color = (self.images[id_curr][(pixels[1,:], pixels[0,:])]).cuda()    # batch_size, 3
            mask = (self.masks[id_curr][(pixels[1,:], pixels[0,:])]).cuda() 

            z_buff_s.append(z_buff)
            color_s.append(color)
            mask_s.append(mask)
            rays_o_s.append(rays_o)
            rays_v_s.append(rays_v)
            pixels_s.append(torch.transpose(pixels,0,1))
            img_ids.append(torch.from_numpy(np.array([id_curr, pixels.shape[1]])).long())
        
        z_buff_s = torch.concat(z_buff_s).to(self.device) 
        color_s = torch.concat(color_s).to(self.device) 
        mask_s = torch.concat(mask_s).to(self.device) 
        rays_o_s = torch.concat(rays_o_s).to(self.device) 
        rays_v_s = torch.concat(rays_v_s).to(self.device) 
        #pixels_s = torch.concat(pixels_s).to(self.device) 
        img_ids = torch.concat(img_ids).to(self.device) 

        return z_buff_s, pixels_s, img_ids, torch.cat([rays_o_s, rays_v_s, color_s, mask_s], dim=-1)
    
        #return z_buff.transpose(0, 1), torch.cat([rays_o.transpose(0, 1), rays_v.transpose(0, 1), color.transpose(0, 1), mask[:, :, :1].transpose(0, 1)], dim=-1)
    

    def gen_rays_from_mesh(self, img_idx, vertices, normals, step):

        #print(normals.shape)
        #print(vertices.shape)
        vertices = vertices + torch.rand(vertices.shape).cuda()*step - step/2.0  
        rays_o = self.pose_all[img_idx, None, :3, 3].expand(vertices.shape)
        p = vertices-rays_o
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)

        #print(rays_o.shape)
        #print(rays_v.shape)
        """visibility = torch.mul(normals, rays_v)
        visibility = torch.sum(visibility, 1)
        p = p[visibility < 0, :]
        rays_v = rays_v[visibility < 0, :]
        rays_o = rays_o[visibility < 0, :]"""

        p = torch.transpose(p,0,1)
        p = torch.matmul(self.pose_all_inv[img_idx,:3,:3], p[:,:]).squeeze()
        p = torch.matmul(self.intrinsics_all[img_idx, :3,:3], p[:,:]).squeeze()
        z = torch.stack([p[2,:], p[2,:]], dim=0)
        pixels = (p[:2,:]/z[:]).cpu().detach().numpy()
        pixels = pixels.astype(np.int32)
    
        rays_o = rays_o[pixels[0,:] >= 0,:]
        rays_v = rays_v[pixels[0,:] >= 0,:]
        pixels = pixels[:,pixels[0,:] >= 0]

        rays_o = rays_o[pixels[1,:] >= 0,:]
        rays_v = rays_v[pixels[1,:] >= 0,:]
        pixels = pixels[:,pixels[1,:] >= 0]

        rays_o = rays_o[pixels[0,:] <= self.W-1,:]
        rays_v = rays_v[pixels[0,:] <= self.W-1,:]
        pixels = pixels[:,pixels[0,:] <= self.W-1]

        rays_o = rays_o[pixels[1,:] <= self.H-1,:]
        rays_v = rays_v[pixels[1,:] <= self.H-1,:]
        pixels = pixels[:,pixels[1,:] <=  self.H-1]

        pixels = torch.from_numpy(pixels).long()

        z_buff = self.z_buff[img_idx][pixels[1,:], pixels[0,:]].cuda() 
        norm_pix = np.linalg.norm(pixels) 
        z_buff[norm_pix == 0] = 0

        color = (self.images[img_idx][(pixels[1,:], pixels[0,:])]).cuda()    # batch_size, 3
        mask = (self.masks[img_idx][(pixels[1,:], pixels[0,:])]).cuda() 

        return z_buff, pixels, img_idx, torch.cat([rays_o, rays_v, color, mask], dim=-1)
  

    def gen_rays_at(self, img_idx, resolution_level=1):
        """
        Generate rays at world space from one camera.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        pixels_x_f = pixels_x.float()# + torch.rand(pixels_x.shape)-0.5
        pixels_y_f = pixels_y.float()# + torch.rand(pixels_y.shape)-0.5
        p = torch.stack([pixels_x_f, pixels_y_f, torch.ones_like(pixels_y_f)], dim=-1).to(self.device)   # W, H, 3
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        rays_v = torch.matmul(self.pose_all[img_idx, None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = self.pose_all[img_idx, None, None, :3, 3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)
        
    def gen_rays_z_buff_at(self, img_idx, resolution_level=1):
        """
        Generate rays at world space from one camera.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        #print(pixels_x)
        z_buff = self.z_buff[img_idx][(pixels_y.long(), pixels_x.long())]
        #mask = self.masks[img_idx][(pixels_y.int(), pixels_x.int())]
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).to(self.device)   # W, H, 3
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        rays_v = torch.matmul(self.pose_all[img_idx, None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = self.pose_all[img_idx, None, None, :3, 3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1), z_buff.transpose(0, 1)
        #return rays_o, rays_v, z_buff#, mask
        
    def gen_rays_RGB_at(self, img_idx, resolution_level=1):
        """
        Generate rays at world space from one camera.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - l, self.W // l)
        ty = torch.linspace(0, self.H - l, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        color = (self.images[img_idx][(pixels_y.long(), pixels_x.long())]).cuda()    # batch_size, 3
        mask = (self.masks[img_idx][(pixels_y.long(), pixels_x.long())]).cuda()      # batch_size, 3
        pixels_x_f = pixels_x.float() 
        pixels_y_f = pixels_y.float() 
        p = torch.stack([pixels_x_f, pixels_y_f, torch.ones_like(pixels_y_f)], dim=-1).to(self.device)   # W, H, 3
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        rays_v = torch.matmul(self.pose_all[img_idx, None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = self.pose_all[img_idx, None, None, :3, 3].expand(rays_v.shape)  # W, H, 3
        return torch.cat([rays_o.transpose(0, 1), rays_v.transpose(0, 1), color.transpose(0, 1), mask[:, :, :1].transpose(0, 1)], dim=-1)
    
    def gen_rays_z_id_RGB_at(self, img_idx, resolution_level=1, shift = 0):
        """
        Generate rays at world space from one camera.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - l, self.W // l) + shift
        ty = torch.linspace(0, self.H - l, self.H // l) + shift
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        color = (self.images[img_idx][(pixels_y.long(), pixels_x.long())]).cuda()    # batch_size, 3
        mask = (self.masks[img_idx][(pixels_y.long(), pixels_x.long())]).cuda()      # batch_size, 3
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).to(self.device)   # W, H, 3
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        rays_v = torch.matmul(self.pose_all[img_idx, None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = self.pose_all[img_idx, None, None, :3, 3].expand(rays_v.shape)  # W, H, 3
        return pixels_y.long().transpose(0, 1), pixels_x.long().transpose(0, 1), torch.cat([rays_o.transpose(0, 1), rays_v.transpose(0, 1), color.transpose(0, 1), mask[:, :, :1].transpose(0, 1)], dim=-1)
    

    def gen_random_rays_at(self, img_idx, batch_size):
        """
        Generate random rays at world space from one camera.
        """
        pixels_x = torch.randint(low=0, high=self.W, size=[batch_size])
        pixels_y = torch.randint(low=0, high=self.H, size=[batch_size])
        #print((pixels_y, pixels_x))
        color = self.images[img_idx][(pixels_y, pixels_x)]    # batch_size, 3
        mask = self.masks[img_idx][(pixels_y, pixels_x)]      # batch_size, 3
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float().to(self.device)   # batch_size, 3
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, :3, :3], p[:, :, None]).squeeze() # batch_size, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)    # batch_size, 3
        rays_v = torch.matmul(self.pose_all[img_idx, None, :3, :3], rays_v[:, :, None]).squeeze()  # batch_size, 3
        rays_o = self.pose_all[img_idx, None, :3, 3].expand(rays_v.shape) # batch_size, 3
        return torch.cat([rays_o.cpu(), rays_v.cpu(), color, mask[:, :1]], dim=-1).cuda()
    
    def gen_random_rays_id_at(self, img_idx, batch_size):
        """
        Generate random rays at world space from one camera.
        """
        pixels_x = torch.randint(low=0, high=self.W, size=[batch_size])
        pixels_y = torch.randint(low=0, high=self.H, size=[batch_size])
        #print((pixels_y, pixels_x))
        color = self.images[img_idx][(pixels_y, pixels_x)]    # batch_size, 3
        mask = self.masks[img_idx][(pixels_y, pixels_x)]      # batch_size, 3
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float().to(self.device)   # batch_size, 3
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, :3, :3], p[:, :, None]).squeeze() # batch_size, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)    # batch_size, 3
        rays_v = torch.matmul(self.pose_all[img_idx, None, :3, :3], rays_v[:, :, None]).squeeze()  # batch_size, 3
        rays_o = self.pose_all[img_idx, None, :3, 3].expand(rays_v.shape) # batch_size, 3
        return pixels_y, pixels_x, torch.cat([rays_o.cpu(), rays_v.cpu(), color, mask[:, :1]], dim=-1).cuda()
        
    def gen_random_rays_zbuff_at(self, img_idx, batch_size, patch_size = 1, shift = 2):
        """
        Generate random rays at world space from one camera.
        """
        """tx = torch.linspace(0, self.W - 1, self.W)
        ty = torch.linspace(0, self.H - 1, self.H)
        p_x, p_y = torch.meshgrid(tx, ty)

        mask = self.masks[img_idx][(p_y.long(), p_x.long())]
        p_front_x = p_x[mask[:,:,0] == 1]
        p_front_y = p_y[mask[:,:,0] == 1]
        p_back_x = p_x[mask[:,:,0] == 0]
        p_back_y = p_y[mask[:,:,0] == 0]

        nb_valid = int(mask[:,:,0].sum().numpy()) #sum(sum(mask[:,:,0]))[0]
        nb_back = int((mask[:,:,0] == 0).sum().numpy()) #sum(sum(mask[:,:,0]))[0]

        ids = torch.randint(low=0, high=nb_valid, size=[int(0.9*batch_size)])
        pixels_x = p_front_x[ids].long()
        pixels_y = p_front_y[ids].long()

        ids = torch.randint(low=0, high=nb_back, size=[int(0.1*batch_size)])
        pixels_x = torch.concat((pixels_x, p_back_x[ids].long()))
        pixels_y = torch.concat((pixels_y, p_back_y[ids].long()))"""

        pixels_x = torch.randint(low=0, high=self.W, size=[batch_size])
        pixels_y = torch.randint(low=0, high=self.H, size=[batch_size])
        if patch_size > 0:
            tmp_x = pixels_x
            pixels_x = torch.concat((pixels_x-shift, pixels_x))
            pixels_x = torch.concat((pixels_x, tmp_x))
            pixels_x = torch.concat((pixels_x, tmp_x))
            pixels_x = torch.concat((pixels_x, tmp_x+shift))
            
            tmp_y = pixels_y
            pixels_y = torch.concat((pixels_y, pixels_y-shift))
            pixels_y = torch.concat((pixels_y, tmp_y ))
            pixels_y = torch.concat((pixels_y, tmp_y + shift))
            pixels_y = torch.concat((pixels_y, tmp_y))

        pixels_x[pixels_x < 0] = 0
        pixels_x[pixels_x > self.W-1] = self.W-1
        pixels_y[pixels_y < 0] = 0
        pixels_y[pixels_y > self.H-1] = self.H-1

        color = self.images[img_idx][(pixels_y, pixels_x)]    # batch_size, 3
        mask = self.masks[img_idx][(pixels_y, pixels_x)]      # batch_size, 3

        pixels_x_f = pixels_x.float() + torch.rand(pixels_x.shape)-0.5
        pixels_y_f = pixels_y.float() + torch.rand(pixels_y.shape)-0.5
        p = torch.stack([pixels_x_f, pixels_y_f, torch.ones_like(pixels_y_f)], dim=-1).float().to(self.device)   # batch_size, 3
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, :3, :3], p[:, :, None]).squeeze() # batch_size, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)    # batch_size, 3
        rays_v = torch.matmul(self.pose_all[img_idx, None, :3, :3], rays_v[:, :, None]).squeeze()  # batch_size, 3
        rays_o = self.pose_all[img_idx, None, :3, 3].expand(rays_v.shape) # batch_size, 3
        return torch.cat([rays_o.cpu(), rays_v.cpu(), color, mask[:, :1]], dim=-1).cuda()     # batch_size, 10
    
    def gen_random_rays_smooth_at(self, img_idx, batch_size, lvl = 2):
        """
        Generate random rays at world space from one camera.
        """

        if self.data_type == 'KINOVIS':
            if lvl == 5.0:
                #pixels_x = torch.randint(low=0, high=self.W_smooth_5, size=[batch_size])
                #pixels_y = torch.randint(low=0, high=self.H_smooth_5, size=[batch_size])
                pixels_x = torch.randint(low=self.all_min_x[img_idx]//5, high=self.all_max_x[img_idx]//5, size=[batch_size])
                pixels_y = torch.randint(low=self.all_min_y[img_idx]//5, high=self.all_max_y[img_idx]//5, size=[batch_size])

                pixels_x[pixels_x < 0] = 0
                pixels_x[pixels_x > self.W_smooth_5-1] = self.W_smooth_5-1
                pixels_y[pixels_y < 0] = 0
                pixels_y[pixels_y > self.H_smooth_5-1] = self.H_smooth_5-1
                color = self.images_smooth_5[img_idx][(pixels_y, pixels_x)]    # batch_size, 3
                mask = self.masks_smooth_5[img_idx][(pixels_y, pixels_x)]      # batch_size, 3
            elif lvl == 3.0:
                #pixels_x = torch.randint(low=0, high=self.W_smooth_3, size=[batch_size])
                #pixels_y = torch.randint(low=0, high=self.H_smooth_3, size=[batch_size])
                pixels_x = torch.randint(low=self.all_min_x[img_idx]//3, high=self.all_max_x[img_idx]//3, size=[batch_size])
                pixels_y = torch.randint(low=self.all_min_y[img_idx]//3, high=self.all_max_y[img_idx]//3, size=[batch_size])

                pixels_x[pixels_x < 0] = 0
                pixels_x[pixels_x > self.W_smooth_3-1] = self.W_smooth_3-1
                pixels_y[pixels_y < 0] = 0
                pixels_y[pixels_y > self.H_smooth_3-1] = self.H_smooth_3-1
                color = self.images_smooth_3[img_idx][(pixels_y, pixels_x)]    # batch_size, 3
                mask = self.masks_smooth_3[img_idx][(pixels_y, pixels_x)]      # batch_size, 3
            elif lvl == 2.0:
                #pixels_x = torch.randint(low=0, high=self.W_smooth_2, size=[batch_size])
                #pixels_y = torch.randint(low=0, high=self.H_smooth_2, size=[batch_size])
                pixels_x = torch.randint(low=self.all_min_x[img_idx]//2, high=self.all_max_x[img_idx]//2, size=[batch_size])
                pixels_y = torch.randint(low=self.all_min_y[img_idx]//2, high=self.all_max_y[img_idx]//2, size=[batch_size])

                pixels_x[pixels_x < 0] = 0
                pixels_x[pixels_x > self.W_smooth_2-1] = self.W_smooth_2-1
                pixels_y[pixels_y < 0] = 0
                pixels_y[pixels_y > self.H_smooth_2-1] = self.H_smooth_2-1
                color = self.images_smooth_2[img_idx][(pixels_y, pixels_x)]    # batch_size, 3
                mask = self.masks_smooth_2[img_idx][(pixels_y, pixels_x)]      # batch_size, 3
            else:
                pixels_x = torch.randint(low=self.all_min_x[img_idx], high=self.all_max_x[img_idx], size=[batch_size])
                pixels_y = torch.randint(low=self.all_min_y[img_idx], high=self.all_max_y[img_idx], size=[batch_size])
                color = self.images[img_idx][(pixels_y, pixels_x)]    # batch_size, 3
                mask = self.masks[img_idx][(pixels_y, pixels_x)]      # batch_size, 3
        else:
            if lvl == 5:
                pixels_x = torch.randint(low=0, high=self.W_smooth_5, size=[batch_size])
                pixels_y = torch.randint(low=0, high=self.H_smooth_5, size=[batch_size])

                pixels_x[pixels_x < 0] = 0
                pixels_x[pixels_x > self.W_smooth_5-1] = self.W_smooth_5-1
                pixels_y[pixels_y < 0] = 0
                pixels_y[pixels_y > self.H_smooth_5-1] = self.H_smooth_5-1
                color = self.images_smooth_5[img_idx][(pixels_y, pixels_x)]    # batch_size, 3
                mask = self.masks_smooth_5[img_idx][(pixels_y, pixels_x)]      # batch_size, 3
            elif lvl == 3:
                pixels_x = torch.randint(low=0, high=self.W_smooth_3, size=[batch_size])
                pixels_y = torch.randint(low=0, high=self.H_smooth_3, size=[batch_size])

                pixels_x[pixels_x < 0] = 0
                pixels_x[pixels_x > self.W_smooth_3-1] = self.W_smooth_3-1
                pixels_y[pixels_y < 0] = 0
                pixels_y[pixels_y > self.H_smooth_3-1] = self.H_smooth_3-1
                color = self.images_smooth_3[img_idx][(pixels_y, pixels_x)]    # batch_size, 3
                mask = self.masks_smooth_3[img_idx][(pixels_y, pixels_x)]      # batch_size, 3
            elif lvl == 2:
                pixels_x = torch.randint(low=0, high=self.W_smooth_2, size=[batch_size])
                pixels_y = torch.randint(low=0, high=self.H_smooth_2, size=[batch_size])

                pixels_x[pixels_x < 0] = 0
                pixels_x[pixels_x > self.W_smooth_2-1] = self.W_smooth_2-1
                pixels_y[pixels_y < 0] = 0
                pixels_y[pixels_y > self.H_smooth_2-1] = self.H_smooth_2-1
                color = self.images_smooth_2[img_idx][(pixels_y, pixels_x)]    # batch_size, 3
                mask = self.masks_smooth_2[img_idx][(pixels_y, pixels_x)]      # batch_size, 3
            else:
                pixels_x = torch.randint(low=0, high=self.W-1, size=[batch_size])
                pixels_y = torch.randint(low=0, high=self.H-1, size=[batch_size])
                color = self.images[img_idx][(pixels_y, pixels_x)]    # batch_size, 3
                mask = self.masks[img_idx][(pixels_y, pixels_x)]      # batch_size, 3"""

        pixels_x_f = pixels_x.float()*lvl# + torch.rand(pixels_x.shape)-0.5
        pixels_y_f = pixels_y.float()*lvl# + torch.rand(pixels_y.shape)-0.5
        p = torch.stack([pixels_x_f, pixels_y_f, torch.ones_like(pixels_y_f)], dim=-1).float().to(self.device)   # batch_size, 3
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, :3, :3], p[:, :, None]).squeeze() # batch_size, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)    # batch_size, 3
        rays_v = torch.matmul(self.pose_all[img_idx, None, :3, :3], rays_v[:, :, None]).squeeze()  # batch_size, 3
        rays_o = self.pose_all[img_idx, None, :3, 3].expand(rays_v.shape) # batch_size, 3
        return torch.cat([rays_o.cpu(), rays_v.cpu(), color, mask[:, :1]], dim=-1).cuda()     # batch_size, 10
    
    def gen_random_rays_zbuff_topk_at(self, conf_map, img_idx, batch_size):
        """
        Generate random rays at world space from one camera.
        """
        conf_map = conf_map.reshape(-1)
        _, indices = torch.topk(conf_map, batch_size, dim=0)
        indices = indices.cpu()
        #pixels_x = indices // self.H #torch.randint(low=0, high=self.W, size=[batch_size])
        #pixels_y = indices - pixels_x*self.H # torch.randint(low=0, high=self.H, size=[batch_size])
        pixels_y = indices // self.W #torch.randint(low=0, high=self.W, size=[batch_size])
        pixels_x = indices - pixels_y*self.W # torch.randint(low=0, high=self.H, size=[batch_size])
        color = self.images[img_idx][(pixels_y, pixels_x)]    # batch_size, 3
        mask = self.masks[img_idx][(pixels_y, pixels_x)]      # batch_size, 3
        z_buff = self.z_buff[img_idx][(pixels_y, pixels_x)]      # batch_size, 9
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float().to(self.device)   # batch_size, 3
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, :3, :3], p[:, :, None]).squeeze() # batch_size, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)    # batch_size, 3
        rays_v = torch.matmul(self.pose_all[img_idx, None, :3, :3], rays_v[:, :, None]).squeeze()  # batch_size, 3
        rays_o = self.pose_all[img_idx, None, :3, 3].expand(rays_v.shape) # batch_size, 3
        return z_buff, pixels_y, pixels_x, torch.cat([rays_o.cpu(), rays_v.cpu(), color, mask[:, :1]], dim=-1).cuda()     # batch_size, 10
 
    def gen_random_patch_zbuff_at(self, img_idx, patch_size, batch_size):
        """
        Generate random rays at world space from one camera.
        """
        pix_x = torch.randint(low=0, high=self.W-patch_size, size=[batch_size])
        pix_y = torch.randint(low=0, high=self.H-patch_size, size=[batch_size])
        p_x =  (torch.arange(0,patch_size).int()).expand(patch_size, patch_size)
        p_y = p_x.transpose(1,0)
        p_x = p_x.reshape(1, patch_size,patch_size).expand(batch_size, patch_size, patch_size)
        p_y = p_y.reshape(1, patch_size,patch_size).expand(batch_size, patch_size, patch_size)
        pixels_x = p_x + (pix_x.reshape(-1,1,1)).expand(batch_size, patch_size, patch_size)
        pixels_y = p_y + (pix_y.reshape(-1,1,1)).expand(batch_size, patch_size, patch_size)
        #pixels_y = (torch.arange(0,patch_size).int()).expand(batch_size, patch_size) + (pix_y.reshape(-1,1)).expand(batch_size, patch_size)
        pixels_x = pixels_x.reshape(-1)
        pixels_y = pixels_y.reshape(-1)
        color = self.images[img_idx][(pixels_y, pixels_x)]    # batch_size, 3
        mask = self.masks[img_idx][(pixels_y, pixels_x)]      # batch_size, 3
        z_buff = self.z_buff[img_idx][(pixels_y, pixels_x)]      # batch_size, 9
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float().to(self.device)   # batch_size, 3
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, :3, :3], p[:, :, None]).squeeze() # batch_size, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)    # batch_size, 3
        rays_v = torch.matmul(self.pose_all[img_idx, None, :3, :3], rays_v[:, :, None]).squeeze()  # batch_size, 3
        rays_o = self.pose_all[img_idx, None, :3, 3].expand(rays_v.shape) # batch_size, 3
        return z_buff, torch.cat([rays_o.cpu(), rays_v.cpu(), color, mask[:, :1]], dim=-1).cuda()     # batch_size, 10

    def gen_rays_between(self, idx_0, idx_1, ratio, resolution_level=1):
        """
        Interpolate pose between two cameras.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)  # W, H, 3
        p = torch.matmul(self.intrinsics_all_inv[0, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        trans = self.pose_all[idx_0, :3, 3] * (1.0 - ratio) + self.pose_all[idx_1, :3, 3] * ratio
        pose_0 = self.pose_all[idx_0].detach().cpu().numpy()
        pose_1 = self.pose_all[idx_1].detach().cpu().numpy()
        pose_0 = np.linalg.inv(pose_0)
        pose_1 = np.linalg.inv(pose_1)
        rot_0 = pose_0[:3, :3]
        rot_1 = pose_1[:3, :3]
        rots = Rot.from_matrix(np.stack([rot_0, rot_1]))
        key_times = [0, 1]
        slerp = Slerp(key_times, rots)
        rot = slerp(ratio)
        pose = np.diag([1.0, 1.0, 1.0, 1.0])
        pose = pose.astype(np.float32)
        pose[:3, :3] = rot.as_matrix()
        pose[:3, 3] = ((1.0 - ratio) * pose_0 + ratio * pose_1)[:3, 3]
        pose = np.linalg.inv(pose)
        rot = torch.from_numpy(pose[:3, :3]).cuda()
        trans = torch.from_numpy(pose[:3, 3]).cuda()
        rays_v = torch.matmul(rot[None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = trans[None, None, :3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def gen_rays_from3Dpoints(self, img_idx, p, nmles, fact = 1):
        #print(p.shape)
        # make rays
        rays_o = self.pose_all[img_idx, None, :3, 3].expand(p.shape)
        rays_v = p-rays_o
        rays_v = rays_v / torch.linalg.norm(rays_v, ord=2, dim=-1, keepdim=True)

        #visibility = torch.sum(rays_v * nmles, 1).cpu()
        #print(visibility.shape)

        # poject 3D points into camera image space
        cam_o = self.pose_all[img_idx, None, :3, 3].expand(p.shape)
        p = p-cam_o
        p = torch.matmul(self.pose_all_inv[img_idx,None,:3,:3], p[:,:,None]).squeeze()
        p = torch.matmul(self.intrinsics_all[img_idx, None, :3,:3], p[:,:,None]).squeeze()
        z = torch.stack([p[:,2], p[:,2]], dim=-1)
        pixels = (p[:,:2]/z[:]).cpu().detach().numpy()
        pixels = pixels.astype(np.int32)
        pixels[pixels[:,0] < 0,:] = 0
        pixels[pixels[:,1] < 0,:] = 0
        pixels[pixels[:,0] > self.W-2,:] = 0
        pixels[pixels[:,1] > self.H-2,:] = 0
        #pixels[visibility > 0, :] = 0

        #print(self.W, self.H)
        #print(pixels)
        z_buff = self.z_buff[img_idx][pixels[:,1], pixels[:,0]]  
        color = (self.images_smooth[img_idx][(pixels[:,1], pixels[:,0])]).cuda()
        color_smooth = (self.images_smooth[img_idx][(pixels[:,1], pixels[:,0])]).cuda()
        mask = (self.masks[img_idx][(pixels[:,1], pixels[:,0])]).cuda() 
        #print("z_buff", z_buff.shape)
        #print("pixels", pixels.shape)
        norm_pix = np.linalg.norm(pixels, ord=2, axis=-1, keepdims=True).reshape(-1) 
        #print("norm_pix", norm_pix.shape)
        z_buff[norm_pix == 0] = 0
        #print("z_buff", z_buff.shape)
        """z_buff = z_buff[norm_pix > 0]
        rays_o = rays_o[norm_pix > 0]
        rays_v = rays_v[norm_pix > 0]
        color = color[norm_pix > 0]
        mask = mask[norm_pix > 0]

        rdm_smpls = torch.randint(z_buff.shape[0], (z_buff.shape[0] // fact,))"""

        #print("z_buff", z_buff[norm_pix > 0].shape)
        #print("color", color[norm_pix > 0].shape)
        #print("mask", mask[norm_pix > 0].shape)
        #print("rays_o", rays_o[norm_pix > 0].shape)
        #print("rays_v", rays_v[norm_pix > 0].shape)
        #return z_buff, rays_o.cpu().numpy(), rays_v.cpu().numpy()
        return z_buff[norm_pix > 0], torch.cat([rays_o[norm_pix > 0], rays_v[norm_pix > 0], color[norm_pix > 0], color_smooth[norm_pix > 0], mask[norm_pix > 0]], dim=-1)
        #return z_buff[rdm_smpls], torch.cat([rays_o[rdm_smpls], rays_v[rdm_smpls], color[rdm_smpls], mask[rdm_smpls]], dim=-1)
    

    def near_far_from_sphere(self, rays_o, rays_d):
        a = torch.sum(rays_d**2, dim=-1, keepdim=True)
        b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
        mid = 0.5 * (-b) / a
        near = mid - 1.0
        far = mid + 1.0
        return near, far

    def image_at(self, idx, resolution_level):
        img = cv.imread(self.images_lis[idx])
        return (cv.resize(img, (self.W // resolution_level, self.H // resolution_level))).clip(0, 255)

