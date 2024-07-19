import os
import cv2
import torch
import argparse
import math
import numpy as np
from pyhocon import ConfigFactory
from tqdm import tqdm
from timeit import default_timer as timer 
from numpy import random
import subprocess


if __name__=='__main__':
    print("Code by Diego Thomas")

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='Data')    
    args = parser.parse_args()

    print("Running batch experiments on Blended MVS data @ ", args.data_path)
    
    for dirs in os.listdir(args.data_path):
        print("Processing ", dirs)
        if dirs == "train.conf":
            continue
        
        if dirs == "bmvs_stone" or dirs == "bmvs_durian" :
            continue
        
        if dirs == "kinette-hx" or dirs == "kino-hx" or dirs == "kinette-cos-hx" or dirs == "kinette-jea-hx": # or dirs == "kinette-hx" or dirs == "kino-hx" or dirs == "kinette-cos-hx" or dirs == "kinette-opt1-hx" or dirs == "kinette-opt2-hx" or dirs == "kinette-opt3-hx"or dirs == "kino-jea-hx" or dirs == "kino-opt-hx":
            continue

        ### Run optimization process ###
        ## the process must output a final checkpoint in folder/checkpoints that contians final SDF, features and network state
        subprocess.run(["C:/Users/thomas/.conda/envs/VortSDF-2/python.exe", "run.py", "--position_encoding", "--double_net", "--conf", args.data_path+"/train.conf", "--data_name", dirs])

        ### Compute Chamfer, IoU errors and HeatMap meshes ###
        subprocess.run(["C:/Users/thomas/.conda/envs/VortSDF-2/python.exe", "src/Eval/chamfer_distance.py", "--GT_path", "GT/"+dirs+"/GTMeshRaw.ply", "--dir_path", "Exp/"+dirs])
        
        ## Render all images
        #subprocess.run(["C:/Users/thomas/.conda/envs/VortSDF-2/python.exe", "run.py", "--position_encoding", "--double_net", "--data_name", dirs , "--is_continue", "--mode", "render_images"])

        ### Compute PSNR and SSIM => output error image in Exp/data_name/err ###
        subprocess.run(["C:/Users/thomas/.conda/envs/VortSDF-2/python.exe", "src/Eval/PSNR.py", "--data_path", args.data_path, "--exp_path", "Exp", "--data_name", dirs])


