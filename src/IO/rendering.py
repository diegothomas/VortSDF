import bpy

#from bpy import context
import glob
#from natsort import natsorted

import re
import argparse
import sys
import shutil
import os

from pathlib import Path
def parentpath(path='.', f=0):
    return Path(path).resolve().parents[f]

sys.path.append(".")
from blender_render import render

def numericalSort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

if __name__ == "__main__":

    """parser = argparse.ArgumentParser(description='Render reconstructed 3D mesh')

    parser.add_argument(
        '--path',
        default= None,
        type=str)
    parser.add_argument(
        '--extension',
        default= None,
        type=str)
    parser.add_argument(
        '--dataname',
        default= None,
        type=str)

    args = parser.parse_args()"""

    ###no color
    rgb_flg = True

    save_rendering_root_path = "C:/Users/thomas/Documents/Projects/Human-AI/VortSDF/Exp/bmvs_man/rendering"
    if os.path.isdir(save_rendering_root_path) == False:
        os.mkdir(save_rendering_root_path)    

    mesh_path = "C:/Users/thomas/Documents/Projects/Human-AI/VortSDF/Exp/bmvs_man/final_MT_smooth_b_raw.ply" #sorted(glob.glob(os.path.join(args.path, args.dataname, ".ply")))
    
    render(mesh_path, save_rendering_root_path, rgb_flg)
