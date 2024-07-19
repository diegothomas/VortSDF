import numpy as np
import torch
import time 
import argparse
import sys
import glob
import os
from tqdm import tqdm
from natsort import natsorted 
import cv2 as cv
import scipy.signal
import trimesh
from evalPSNR import computePSNRArrays
import SimpleRenderer
import xml.etree.cElementTree as ET


def load_Rt_from(filename):
    lines = open(filename).read().splitlines()
    lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
    pose = np.asarray(lines).astype(np.float32).squeeze()

    return pose

def generateEvalMasks(gtMeshPath, predMeshPath, avgTransform, calibFile, masksPath):
    print(f"Loading gt mesh: {gtMeshPath}..")
    gt_mesh = trimesh.load_mesh(gtMeshPath)
    gt_mesh.vertices = (gt_mesh.vertices @ avgTransform[:3,:3].T + avgTransform[:3, 3].reshape((3,1)).T)
    gt_mesh = gt_mesh.slice_plane((0,0,0.05), (0,0,1))
    
    print("Loading {} ...".format(predMeshPath))
    pred_mesh = trimesh.load_mesh(predMeshPath)
    pred_mesh.vertices = (pred_mesh.vertices @ avgTransform[:3,:3].T + avgTransform[:3, 3].reshape((3,1)).T)
    pred_mesh = pred_mesh.slice_plane((0,0,0.05), (0,0,1))

    print("Computing largest connected component of test mesh...")
    pred_mesh = sorted(pred_mesh.split(only_watertight=False), key=lambda x : x.vertices.shape[0], reverse=True)[0]

    print("Computing icp...")
    m, transformed, cost = trimesh.registration.icp(gt_mesh.vertices, pred_mesh.vertices, initial=np.eye(4), threshold=1.0e-7, max_iterations=20)
    gt_mesh.vertices = transformed

    calib = []
    tree = ET.parse(calibFile)
    root = tree.getroot()
    for c in root.findall("Camera"):
        id = int(c.attrib["id"])
        w = int(c.attrib["width"])
        h = int(c.attrib["height"])
        d = c.find("Distortion")
        K = np.array([float(f) for f in c.find("K").text.split(" ")]).reshape((3,3))
        R = np.array([float(f) for f in c.find("R").text.split(" ")]).reshape((3,3))
        T = np.array([float(f) for f in c.find("T").text.split(" ")]).reshape((3,1))
        calib.append((id, w, h, K, R, T))

    # Create evaluation masks using the reprojection of the aligned gt mesh
    os.makedirs(masksPath, exist_ok=True)
    glMesh = SimpleRenderer.SimpleMesh(2048, 2048, gt_mesh.vertices, gt_mesh.faces)
    for i, c in enumerate(calib):
        id, w, h, K, R, T = c
        maskImg = glMesh.render(K, R, T, np.eye(4), (w,h))
        maskImg = (maskImg * 255).astype(np.uint8)
        kernel = np.ones((5, 5), np.uint8) 
        img_erosion = cv.erode(maskImg, kernel, iterations=1) 
        cv.imwrite(masksPath + f"cam-{id}.png", img_erosion)
    glMesh.delete()

def evalPSNR(rendersPath, gtImagesPath, evalMasksPath, diffPath, dstPath):
    psnrs = [0 for _ in range(68)]
    #Evaluate the psnr within the reprojection of the gt mesh
    for i in range(68):
        gtImg = cv.imread(gtImagesPath + f"/cam-{i+1}.png")
        testImg = cv.imread(rendersPath + f"/cam-{i+1}.png")
        #testImg = cv.imread(rendersPath + f"/img_{i}.png")
        #testImg = cv.imread(rendersPath + f"/pred_"+str(0).zfill(4)+".png")
        maskImg = cv.imread(evalMasksPath + f"/cam-{i+1}.png")
        psnrs[i] = computePSNRArrays(gtImg, maskImg, testImg, diffPath + f"/diff-{i+1}.png")
    np.savetxt(dstPath + "/psnrs", np.array(psnrs))
    np.savetxt(dstPath + "/psnrs-avg", np.array([np.array(psnrs).mean()]))




''' Evaluation metrics (ssim, lpips)
'''
def rgb_ssim(img0, img1, max_val,
             filter_size=11,
             filter_sigma=1.5,
             k1=0.01,
             k2=0.03,
             return_map=False):
    # Modified from https://github.com/google/mipnerf/blob/16e73dfdb52044dcceb47cda5243a686391a6e0f/internal/math.py#L58
    assert len(img0.shape) == 3
    assert img0.shape[-1] == 3
    assert img0.shape == img1.shape

    # Construct a 1D Gaussian blur filter.
    hw = filter_size // 2
    shift = (2 * hw - filter_size + 1) / 2
    f_i = ((np.arange(filter_size) - hw + shift) / filter_sigma)**2
    filt = np.exp(-0.5 * f_i)
    filt /= np.sum(filt)

    # Blur in x and y (faster than the 2D convolution).
    def convolve2d(z, f):
        return scipy.signal.convolve2d(z, f, mode='valid')

    filt_fn = lambda z: np.stack([
        convolve2d(convolve2d(z[...,i], filt[:, None]), filt[None, :])
        for i in range(z.shape[-1])], -1)
    mu0 = filt_fn(img0)
    mu1 = filt_fn(img1)
    mu00 = mu0 * mu0
    mu11 = mu1 * mu1
    mu01 = mu0 * mu1
    sigma00 = filt_fn(img0**2) - mu00
    sigma11 = filt_fn(img1**2) - mu11
    sigma01 = filt_fn(img0 * img1) - mu01

    # Clip the variances and covariances to valid values.
    # Variance must be non-negative:
    sigma00 = np.maximum(0., sigma00)
    sigma11 = np.maximum(0., sigma11)
    sigma01 = np.sign(sigma01) * np.minimum(
        np.sqrt(sigma00 * sigma11), np.abs(sigma01))
    c1 = (k1 * max_val)**2
    c2 = (k2 * max_val)**2
    numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
    denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
    ssim_map = numer / denom
    ssim = np.mean(ssim_map)
    return ssim_map if return_map else ssim

def ssim_all(path1, path2, path3,device = None):
    start = time.time()
    #print(path1)
    #print(path2)

    n_images = len(path1)

    images_pred = np.stack([cv.imread(im_name) for im_name in path1]) / 255.0
    images_pred = torch.from_numpy(images_pred.astype(np.float32)).cpu()
    images_GT = np.stack([cv.imread(im_name) for im_name in path2]) / 255.0
    images_GT = torch.from_numpy(images_GT.astype(np.float32)).cpu()
    images_Mask = np.stack([cv.imread(im_name) for im_name in path3]) / 255.0
    images_Mask = torch.from_numpy(images_Mask.astype(np.float32)).cpu()

        
    print("start computing SSIM")
    avg_ssim = 0.0
    for i in tqdm(range(n_images)):
        im_curr = images_pred[i] * images_Mask[i]       
        im_curr_gt = images_GT[i] * images_Mask[i] 
        ssim_val = rgb_ssim(im_curr, im_curr_gt, max_val=1)
        avg_ssim = avg_ssim + ssim_val

    return (avg_ssim/n_images)

def psnr(exp_path, path1, path2, path3, device = None):
    start = time.time()
    #print(path1)
    #print(path2)

    n_images = len(path1)

    images_pred = np.stack([cv.imread(im_name) for im_name in path1]) / 255.0
    images_pred = torch.from_numpy(images_pred.astype(np.float32)).cpu()
    images_GT = np.stack([cv.imread(im_name) for im_name in path2]) / 255.0
    images_GT = torch.from_numpy(images_GT.astype(np.float32)).cpu()
    images_Mask = np.stack([cv.imread(im_name) for im_name in path3]) / 255.0
    images_Mask = torch.from_numpy(images_Mask.astype(np.float32)).cpu()

        
    print("start computing PSNR")
    avg_psnr = 0.0
    for i in tqdm(range(n_images)):
        im_curr = images_pred[i]
        mag = torch.sum(im_curr, 2) #torch.linalg.norm(im_curr, ord=2, axis=-1, keepdims=True)[:,:]
        mask = torch.zeros([im_curr.shape[0], im_curr.shape[1], 1], dtype=torch.float32)
        mask[mag > 0.0] = 1
        mask = mask * images_Mask[i]
        #mask[im_curr[:,:,0] == 0.5] = 1
        kernel = np.ones((5,5),np.uint8)
        mask = torch.from_numpy(cv.erode(mask.numpy(),kernel,iterations = 1))
        cv.imwrite(exp_path + '/mask'+str(0).zfill(3)+'.png', 255*mask.numpy()[:,:])
        #mask = torch.from_numpy(mask).reshape([im_curr.shape[0], im_curr.shape[1], 1])

        im_curr = im_curr * mask #* images_Mask[i]        #* mask
        im_curr_gt = images_GT[i] * mask #* images_Mask[i]  # * mask
        cv.imwrite(exp_path + '/GT'+str(0).zfill(3)+'.png', 255*im_curr_gt.numpy()[:,:])
        cv.imwrite(exp_path + '/im_curr'+str(0).zfill(3)+'.png', 255*im_curr.numpy()[:,:])

        err = (im_curr_gt-im_curr) #.reshape(-1,1)
        err = torch.square(err)
        cv.imwrite(exp_path + '/err/err_map_'+str(i).zfill(3)+'.png', 10000*err.numpy()[:,:])
        #err = (im_curr_gt.reshape(-1,3)-im_curr.reshape(-1,3)) #.reshape(-1,1)
        #err = torch.square(err) #torch.linalg.norm(err, ord=2, axis=-1, keepdims=True)   

        psnr = -10. * torch.log10(err.sum() / images_Mask[i].sum()) # => VOXURF version
        #psnr = -10. * torch.log10(err.sum() / (mask.sum() * 3.0))
        #print("psnr = ", psnr)

        avg_psnr = avg_psnr + psnr

    return (avg_psnr/n_images)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PSNR')
    parser.add_argument(
        '--data_path',
        default= "Data",
        type=str,
        help='Where logs and weights will be saved')
    
    parser.add_argument(
        '--exp_path',
        default= "Exp",
        type=str,
        help='Where logs and weights will be saved')

    parser.add_argument(
        '--data_name',
        default= "bmvs_bear",
        type=str,
        help='Where logs and weights will be saved')
    

    args = parser.parse_args()
    if args.data_path == "None":
        print("select path of data folder")
        sys.exit()
    else:
        data_path = args.data_path
        
    if args.data_name == "None":
        print("select data_name")
        sys.exit()
    else:
        data_name = args.data_name

    if args.exp_path == "None":
        print("select path of exp folder")
        sys.exit()
    else:
        exp_path = args.exp_path

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    pred_paths = natsorted(glob.glob(os.path.join(exp_path , data_name, "validations_fine/*.png"))) 
    pred_path = os.path.join(exp_path , data_name, "validations_fine/")
    diff_path = os.path.join(exp_path , data_name, "validations_err/")
    gt_path = os.path.join(data_path , data_name, "image/")
    gt_paths = natsorted(glob.glob(os.path.join(data_path, data_name, "image/*.png")))
    mask_paths = natsorted(glob.glob(os.path.join(data_path, data_name, "mask/*.png")))
    if len(gt_paths) == 0:
        gt_path = os.path.join(data_path , data_name, "ImagesUndistorted/")
        gt_paths = natsorted(glob.glob(os.path.join(data_path, data_name, "ImagesUndistorted/*.png")))
        mask_paths = natsorted(glob.glob(os.path.join(data_path, data_name, "Masks/*.png")))

    gtMeshPath = os.path.join(data_path, data_name, "GTMeshRaw.ply")
    predMeshPath = os.path.join(exp_path , data_name, "Final_MT_smooth.ply")
    trans_file = os.path.join(data_path, data_name, 'transform_{}.txt'.format(data_name))
    print(trans_file)
    avgTransform = load_Rt_from(trans_file) 
    calibFile = os.path.join(data_path, data_name, 'calibration_undistorted.xml')
    masksPath = os.path.join(data_path, data_name, 'EvalMasks/')
    
    generateEvalMasks(gtMeshPath, predMeshPath, avgTransform, calibFile, masksPath)
    evalPSNR(pred_path, gt_path, masksPath, diff_path, os.path.join(exp_path , data_name))    
    
    exit()

    avg_psnr = psnr(os.path.join(exp_path , data_name), pred_paths, gt_paths, masksPath, device)
    print("avg_psnr : ",avg_psnr)
    
    avg_ssim = ssim_all(pred_paths, gt_paths, masksPath, device)
    print("avg_ssim : ",avg_ssim)

    result_csv = os.path.join(exp_path,"PSNR.csv")
    if not os.path.exists(result_csv):
        # Creat file to save all results
        # Write Header file
        f = open(result_csv,"w")
        f.write(" " +  ",")
        #for mesh in mesh_list:
        f.write(" " +  "," ) 
        f.write("\n")
        
        f.write(" " +  ",")
        #for mesh in mesh_list:
        f.write("PSNR" +  "," + "SSIM" +  ",") 
        f.write("\n")
    else:
        f = open(result_csv,"a")


    f.write(data_name +  ",")
    #for chm, iou in zip(Chamfer_results, IoU_results):
    f.write("{} , {} ,".format(avg_psnr, avg_ssim)) 
    f.write("\n")

    f.close()

    exit()

    for i, (gt_path , pred_path) in tqdm.tqdm(enumerate(zip(gt_paths,pred_paths))):
        print(gt_path)
        print(pred_path)
        avg_psnr = psnr(pred_path, gt_path , device)
        print("avg_psnr : ",avg_psnr)
