#from pytorch3d.utils import ico_sphere
import torch

from natsort import natsorted 

import os
import sys
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import pymeshlab
from src.IO.dataset import Dataset
from pyhocon import ConfigFactory
from scipy.spatial.transform import Rotation as scipy_rot

import shutil

sys.path.append(r"C:\Users\thomas\Documents\Projects\Human-AI\inria-cvt\Python")      #FIXME:Hard coding
from IO.ply import save_ply

THRESH = 0.02



def load_Rt_from(filename):
    lines = open(filename).read().splitlines()
    lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
    pose = np.asarray(lines).astype(np.float32).squeeze()

    return pose

def getNeus2Transform():
    Gx = scipy_rot.from_euler('xyz', angles=(90, 0, 0), degrees=True).as_matrix()
    Gz = scipy_rot.from_euler('xyz', angles=(0, 0, 45), degrees=True).as_matrix()

    G4x4 = np.eye(4)
    G4x4[:3, :3] = (Gz @ Gx)
    G4x4[:3, 3] = (Gz @ Gx) @ np.array([.5, 1, .5])
    return G4x4

def if_not_exists_makedir(path ,comment = None , delete = False , confirm_delete = True):
    if not os.path.exists(path):
        os.makedirs(path)
        if comment is not None:
            print(comment)
    elif delete == False :
        print("folder will be overwritten")
    elif delete == True :
        if confirm_delete == True:
            print()
            print("Can I delete this folder? ")
            print(path)
            print()
            print("yes → enter 'y'")
            print()
            res = input()

            if res == "y":
                shutil.rmtree(path)
                os.makedirs(path)
            else :
                print("Please empty the folder")
                sys.exit()
        else:
            shutil.rmtree(path)
            os.makedirs(path)



def load_ply(path, min_B = [-np.inf, -np.inf, -np.inf], max_B = [np.inf, np.inf, np.inf]):
    """
    return -> vtx , nml , rgb , face , vtx_num , face_num
    """
    f = open(path,"r")
    ply = f.read()
    lines=ply.split("\n")

    rgb_flg = False
    nml_flg = False
    face_flg = False
    vtx_num = 0
    face_num = 0
    i = 0
    while(1):
        if "end_header" in lines[i]:
            #print("finish reading header")
            break
        if "element vertex" in lines[i]:
            vtx_num = int(lines[i].split(" ")[-1])         
        if "element face" in lines[i]:
            face_num = int(lines[i].split(" ")[-1])
            face_flg = True
        if "red" in lines[i] or "green" in lines[i] or "blue" in lines[i]:
            rgb_flg = True
        if "nx" in lines[i] or "ny" in lines[i] or "nz" in lines[i]:
            nml_flg = True
        i += 1
        if i == 100:
            print("load header error")
            sys.exit()
        header_len = i + 1
    print("vtx :" , vtx_num ,"  face :" , face_num ,"  nml_flg: " , nml_flg,"  rgb_flg :" , rgb_flg,"  face_flg :" , face_flg)

    vtx = []
    nml = []
    rgb = []
    face = []
    indx = []

    if nml_flg and rgb_flg and face_flg:
        for i in range(header_len,vtx_num+header_len):
            curr_p = list(map(float,lines[i].split(" ")[0:3]))
            if (curr_p[0] > min_B[0] and curr_p[0] < max_B[0] and \
                curr_p[1] > min_B[1] and curr_p[1] < max_B[1] and \
                curr_p[2] > min_B[2] and curr_p[2] < max_B[2]):
                indx.append(len(vtx))
                vtx.append(list(map(float,lines[i].split(" ")[0:3])))
                nml.append(list(map(float,lines[i].split(" ")[3:6])))
                rgb.append(list(map(int,lines[i].split(" ")[6:9])))
            else:                
                indx.append(-1)

        for i in range(header_len+vtx_num,face_num+header_len+vtx_num):
            curr_id = list(map(int,lines[i].split(" ")[1:4]))
            curr_id = [indx[curr_id[0]], indx[curr_id[1]], indx[curr_id[2]]]
            if curr_id[0] > -1 and curr_id[1] > -1 and curr_id[2] > -1:
                face.append(curr_id)
            #face.append(list(map(int,lines[i].split(" ")[1:4])))
        

    elif nml_flg and rgb_flg            :
        for i in range(header_len,vtx_num+header_len):
            curr_p = list(map(float,lines[i].split(" ")[0:3]))
            if (curr_p[0] > min_B[0] and curr_p[0] < max_B[0] and \
                curr_p[1] > min_B[1] and curr_p[1] < max_B[1] and \
                curr_p[2] > min_B[2] and curr_p[2] < max_B[2]):
                indx.append(len(vtx))
                vtx.append(list(map(float,lines[i].split(" ")[0:3])))
                nml.append(list(map(float,lines[i].split(" ")[3:6])))
                rgb.append(list(map(int,lines[i].split(" ")[6:9])))
            else:                
                indx.append(-1)
        

    elif nml_flg           and  face_flg:
        for i in range(header_len,vtx_num+header_len):
            curr_p = list(map(float,lines[i].split(" ")[0:3]))
            if (curr_p[0] > min_B[0] and curr_p[0] < max_B[0] and \
                curr_p[1] > min_B[1] and curr_p[1] < max_B[1] and \
                curr_p[2] > min_B[2] and curr_p[2] < max_B[2]):
                indx.append(len(vtx))
                vtx.append(list(map(float,lines[i].split(" ")[0:3])))
                nml.append(list(map(float,lines[i].split(" ")[3:6])))
            else:                
                indx.append(-1)

        for i in range(header_len+vtx_num,face_num+header_len+vtx_num):
            curr_id = list(map(int,lines[i].split(" ")[1:4]))
            curr_id = [indx[curr_id[0]], indx[curr_id[1]], indx[curr_id[2]]]
            if curr_id[0] > -1 and curr_id[1] > -1 and curr_id[2] > -1:
                face.append(curr_id)
            #face.append(list(map(int,lines[i].split(" ")[1:4])))
        
    elif            rgb_flg and face_flg:
        for i in range(header_len,vtx_num+header_len):
            curr_p = list(map(float,lines[i].split(" ")[0:3]))
            if (curr_p[0] > min_B[0] and curr_p[0] < max_B[0] and \
                curr_p[1] > min_B[1] and curr_p[1] < max_B[1] and \
                curr_p[2] > min_B[2] and curr_p[2] < max_B[2]):
                indx.append(len(vtx))
                vtx.append(list(map(float,lines[i].split(" ")[0:3])))
                rgb.append(list(map(int,lines[i].split(" ")[3:6])))
            else:                
                indx.append(-1)

        for i in range(header_len+vtx_num,face_num+header_len+vtx_num):
            curr_id = list(map(int,lines[i].split(" ")[1:4]))
            curr_id = [indx[curr_id[0]], indx[curr_id[1]], indx[curr_id[2]]]
            if curr_id[0] > -1 and curr_id[1] > -1 and curr_id[2] > -1:
                face.append(curr_id)
            #face.append(list(map(int,lines[i].split(" ")[1:4])))

    elif nml_flg                       :
        for i in range(header_len,vtx_num+header_len):
            curr_p = list(map(float,lines[i].split(" ")[0:3]))
            if (curr_p[0] > min_B[0] and curr_p[0] < max_B[0] and \
                curr_p[1] > min_B[1] and curr_p[1] < max_B[1] and \
                curr_p[2] > min_B[2] and curr_p[2] < max_B[2]):
                indx.append(len(vtx))
                vtx.append(list(map(float,lines[i].split(" ")[0:3])))
                nml.append(list(map(float,lines[i].split(" ")[3:6])))
            else:                
                indx.append(-1)

    elif            rgb_flg            :
        for i in range(header_len,vtx_num+header_len):
            curr_p = list(map(float,lines[i].split(" ")[0:3]))
            if (curr_p[0] > min_B[0] and curr_p[0] < max_B[0] and \
                curr_p[1] > min_B[1] and curr_p[1] < max_B[1] and \
                curr_p[2] > min_B[2] and curr_p[2] < max_B[2]):
                indx.append(len(vtx))
                vtx.append(list(map(float,lines[i].split(" ")[0:3])))
                rgb.append(list(map(int,lines[i].split(" ")[3:6])))
            else:                
                indx.append(-1)

    elif                       face_flg:
        for i in range(header_len,vtx_num+header_len):
            curr_p = list(map(float,lines[i].split(" ")[0:3]))
            if (curr_p[0] > min_B[0] and curr_p[0] < max_B[0] and \
                curr_p[1] > min_B[1] and curr_p[1] < max_B[1] and \
                curr_p[2] > min_B[2] and curr_p[2] < max_B[2]):
                indx.append(len(vtx))
                vtx.append(list(map(float,lines[i].split(" ")[0:3])))
            else:                
                indx.append(-1)

        for i in range(header_len+vtx_num,face_num+header_len+vtx_num):
            curr_id = list(map(int,lines[i].split(" ")[1:4]))
            curr_id = [indx[curr_id[0]], indx[curr_id[1]], indx[curr_id[2]]]
            if curr_id[0] > -1 and curr_id[1] > -1 and curr_id[2] > -1:
                face.append(curr_id)
            #face.append(list(map(int,lines[i].split(" ")[1:4])))
    f.close()

    print(len(vtx))
    
    return vtx , nml , rgb , face , vtx_num , face_num

def load_obj(path):
    """
    return -> vtxs , nmls , uvs , faceID2vtxIDset , faceID2uvIDset , faceID2normalIDset , vtx_num , face_num
    """
    f = open(path,"r")
    obj = f.read()
    lines=obj.split("\n")

    vtx_num = 0
    face_num = 0

    vtxs = []
    nmls = []
    uvs = []
    faceID2vtxIDset = []
    faceID2uvIDset  = []
    faceID2normalIDset = []
    #i = 0
    for i in range(len(lines)):
        #if lines[i] == '':
        #    break
        line = lines[i].split(" ")
        if "v" == line[0]:
            vtxs.append(list(map(float,line[1:4])))
            vtx_num += 1
        if "vt" == line[0]:
            uvs.append(list(map(float,line[1:3])))
        if "vn" == line[0]:
            nmls.append(list(map(float,line[1:4])))
        if "f" == line[0]:
            face_num += 1
            faceID2vtxIDset.append([int(line[1])-1,int(line[2])-1,int(line[3])-1]) 
            """if line[1].split("/")[0] != '' and line[2].split("/")[0] != '' and line[3].split("/")[0] != '':
                faceID2vtxIDset   .append([int(line[1].split("/")[0])-1,int(line[2].split("/")[0])-1,int(line[3].split("/")[0])-1]) 
            if line[1].split("/")[1] != '' and line[2].split("/")[1] != '' and line[3].split("/")[1] != '':
                faceID2uvIDset    .append([int(line[1].split("/")[1])-1,int(line[2].split("/")[1])-1,int(line[3].split("/")[1])-1]) 
            if line[1].split("/")[2] != '' and line[2].split("/")[2] != '' and line[3].split("/")[2] != '':
                faceID2normalIDset.append([int(line[1].split("/")[2])-1,int(line[2].split("/")[2])-1,int(line[3].split("/")[2])-1])""" 
    #i+=1
    f.close()
    return vtxs , nmls , uvs , faceID2vtxIDset , faceID2uvIDset , faceID2normalIDset , vtx_num , face_num


def save_color_mesh(distance , verts , face , save_path , cmap_type = "jet" , cmap_inverse = False , vmin=0 , vmax=1e-1):
    if type(distance) == torch.Tensor:
        disance_np = distance.to('cpu').detach().numpy().copy()
    else:
        disance_np = distance
    
    norm = matplotlib.colors.Normalize(vmin=vmin , vmax=vmax , clip = False)   #1mm(0.001m) ~ 10cm(0.1m) → to [0-1]
    norm_distance = norm(disance_np)
    
    if cmap_type =="jet" :
        if cmap_inverse :
            distance_color = plt.cm.jet_r(norm_distance)  
        else:
            distance_color = plt.cm.jet(norm_distance)                                                    #→RGBA : "jet" color map(R is biggest & B is smallest))
    elif cmap_type =="viridis" :
        if cmap_inverse :
            distance_color = plt.cm.viridis_r(norm_distance)                                                    
        else:     
            distance_color = plt.cm.viridis(norm_distance)                                                    
    elif cmap_type =="gray" :
        if cmap_inverse :
            distance_color = plt.cm.gray_r(norm_distance) 
        else:
            distance_color = plt.cm.gray(norm_distance) 
    
    distance_color[norm_distance >= 1.0e3,:3] = 0
    
    ms = pymeshlab.MeshSet()
    new_mesh = pymeshlab.Mesh(vertex_matrix = verts ,face_matrix = face , v_color_matrix = distance_color)  
    ms.add_mesh(new_mesh)
    ms.save_current_mesh(save_path , binary = False) 

    distance_color[:,:3] = (distance_color[:,:3] * 255)
    distance_color = distance_color.astype(np.uint8)

    return distance_color

def save_mesh(verts , face , save_path):
    ms = pymeshlab.MeshSet()
    new_mesh = pymeshlab.Mesh(vertex_matrix = verts ,face_matrix = face)  
    ms.add_mesh(new_mesh)
    ms.save_current_mesh(save_path , binary = False) 


def load_factory(path , min_B = [-np.inf, -np.inf, np.inf], max_B = [np.inf, np.inf, np.inf]):
    '''
    return verts_tensor , face_tensor , face_vertices , vertices_normal , face_normal
    '''
    #Load GT Mesh
    ext = os.path.splitext(path)[-1]
    #verts = torch.Tensor(np.load(GT_path)["scan_pc"])   #from cape npz
    if ext == ".ply":
        verts , _ , _ , face , _ , _ =  load_ply(path, min_B, max_B)
        print(len(verts))
        face         = np.array(face)
        verts         = np.array(verts)
        print(path)
        print(verts.shape)
        print(face.shape)
    elif ext == ".obj":
        """
        verts_tensor, face_tensor , _ , _ , _ , _ , _ , _ = kaolin.io.obj.import_mesh(GT_path) 
        verts = verts_tensor.to('cpu').detach().numpy().copy()
        face  = face_tensor.to('cpu').detach().numpy().copy()
        """
        verts , _ , _ , face , _ , _ , _ , _ = load_obj(path) 
        verts_tensor = torch.tensor(verts)
        face_tensor  = torch.tensor(face)
        face         = np.array(face)
        verts         = np.array(verts)
        print(path)
        print(verts_tensor.shape)
        print(face_tensor.shape)
        print(face.shape)


    return verts , face


if __name__ == "__main__":
    
    #conf_path = "D:/MV_data/DTU/data/train.conf"
    conf_path = "D:/MV_data/manikins/data/train.conf"
    f = open(conf_path)
    conf_text = f.read()
    f.close()

    conf = ConfigFactory.parse_string(conf_text)
    dataset = Dataset(conf['dataset'], "kinette-cos-hx")
    verts_GT    , face_GT   = load_factory("D:/MV_data/manikins/voxurf_meshes/kino-sho-hx/mesh.ply", [-1000.0, -1000.0, -1000.0], [1000.0, 1000.0, 1000.0])
    #verts_GT    , face_GT   = load_factory("D:/MV_data/manikins/Neus2/kino-sho-hx/_kino-sho-hx.ply", [-1000.0, -1000.0, -1000.0], [1000.0, 1000.0, 1000.0])
    
    translate = dataset.scale_mats_np[0][:3, 3][None]
    scale = dataset.scale_mats_np[0][0, 0]
    translate = np.ascontiguousarray(translate, dtype=np.float32)
    
    verts_GT = verts_GT * scale + translate


    """translate = dataset.scale_mats_np[0][:3, 3][None]
    rotate = dataset.scale_mats_np[0][:3, :3]
    translate = np.ascontiguousarray(translate, dtype=np.float32)
    verts_GT = np.dot(verts_GT, rotate) + translate"""
    
    
    """transfo = getNeus2Transform()
    translate = transfo[:3, 3][None]
    rotate = transfo[:3, :3]
    translate = np.ascontiguousarray(translate, dtype=np.float32)
    verts_GT = verts_GT @ rotate.T + translate"""
    
    """transfo = np.array([[-0.064594872296, -0.997774839401, -0.016518760473, 0.714533984661],
                        [-0.030686499551, 0.018531566486, -0.999357283115, 0.601175248623],
                        [0.997439622879, -0.064046449959, -0.031815256923, 1.163400173187],
                        [0.000000000000, 0.000000000000, 0.000000000000, 1.000000000000]
                        ])"""
    
    transfo = load_Rt_from('D:/MV_data/manikins/Neus2/kino-sho-hx/transform.txt') 
    
    translate = transfo[:3, 3][None]
    rotate = transfo[:3, :3]
    translate = np.ascontiguousarray(translate, dtype=np.float32)
    verts_GT = (verts_GT - translate) @ rotate 

    save_mesh( verts_GT , face_GT , "D:/MV_data/manikins/voxurf_meshes/kino-sho-hx/GTMeshRaw.ply")
    

