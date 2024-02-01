import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.sparse
import numpy as np
import logging
from icecream import ic
from timeit import default_timer as timer 
import ctypes
import os
import cv2
from numpy import random
from PIL import Image, ImageDraw
import matplotlib
import matplotlib.pyplot as plt

from torch.utils.cpp_extension import load
renderer_cuda = load(
    'renderer_cuda', ['src/Models/renderer.cpp', 'src/Models/renderer.cu'], verbose=True)

class VortSDFRenderer:
    def __init__(self, n_samples):
        self.n_samples = n_samples
        self.mask_reg = 1.0

        self.grads_color = torch.zeros([1], dtype=torch.float32).to(torch.device('cuda')).contiguous()
        self.grads_sdf = torch.zeros([1], dtype=torch.float32).to(torch.device('cuda')).contiguous()
        self.colors_loss = torch.zeros([1], dtype=torch.float32).to(torch.device('cuda')).contiguous()
        self.mask_loss = torch.zeros([1], dtype=torch.float32).to(torch.device('cuda')).contiguous()
       

    def render_gpu(self, num_rays, inv_s, sdf_seg, knn_sites, weights_seg, color_samples, true_color, mask, cell_ids, offsets):   
        self.grads_color[:, :] = 0
        self.grads_sdf[:, :] = 0
        
        renderer_cuda.render(num_rays, inv_s, self.mask_reg, sdf_seg, knn_sites, weights_seg, color_samples, true_color, mask, cell_ids, offsets, self.grads_sdf, self.grads_color, self.colors_loss, self.mask_loss)
        
        return self.colors_loss, self.grads_color
    
    def prepare_buffs(self, nb_points, nb_samples, nb_sites):
        del self.grads_sdf
        del self.grads_color

        self.grads_color = torch.zeros([nb_points * nb_samples, 3], dtype=torch.float32).to(torch.device('cuda')).contiguous()
        self.grads_sdf = torch.zeros([nb_sites], dtype=torch.float32).to(torch.device('cuda')).contiguous()
        
        self.grads_color = self.grads_color.contiguous()
        self.grads_sdf = self.grads_sdf.contiguous()
        
        self.colors_loss = torch.zeros([nb_points,1]).to(torch.device('cuda'))
        self.mask_loss = torch.zeros([nb_points,1]).to(torch.device('cuda'))
        self.colors_loss = self.colors_loss.contiguous()
        self.mask_loss = self.mask_loss.contiguous()

        
class VortSDFRenderingFunction(autograd.Function):  
    @staticmethod
    def forward(ctx, cvt_renderer, num_rays, inv_s, sdf_seg, knn_sites, weights_seg, color_samples, true_color, mask, cell_ids, offsets):
        
        color_error, grads_color = cvt_renderer.render_gpu(num_rays, inv_s, sdf_seg, knn_sites, weights_seg, color_samples, true_color, mask, cell_ids, offsets)

        grads_color.requires_grad_(True)
        ctx.save_for_backward(grads_color, torch.tensor([color_samples.shape[0]]))   
            
        """norm = matplotlib.colors.Normalize(vmin=0.0, vmax=1.0 , clip = False)
        #print(color_error.shape)
        #print(color_error.max())
        #error_rgb = plt.cm.jet(norm(color_error.cpu().numpy())).astype(np.float32)
        error_rgb = plt.cm.jet(norm(cvt_renderer.mask_loss.cpu().numpy())).astype(np.float32)
        #print(error_rgb.shape)
        Errorimg = Image.fromarray((255.0*error_rgb[:num_rays,:,:3].reshape(1,-1,3)).astype(dtype=np.uint8), 'RGB')
        #Errorimg = Image.fromarray((255.0*color_error[:num_rays,:].reshape(1,-1,3)).cpu().numpy().astype(dtype=np.uint8), 'RGB')
        Errorimg.save('Exp/Errorimg.png')"""
        return color_error.sum()   

    @staticmethod
    def backward(ctx, grad_colors):
        grads, nb_samples_T = ctx.saved_tensors
        nb_samples = int(nb_samples_T.numpy()[0])
        grads = grads[:nb_samples,:]

        return None, None, None, None, None, None, grads, None, None, None, None
        