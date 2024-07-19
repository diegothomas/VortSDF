#from pytorch3d.utils import ico_sphere
import torch
import argparse
import os
from glob import glob
import sys
import numpy as np
import time 
from pyhocon import ConfigFactory

import matplotlib
import matplotlib.pyplot as plt
import pymeshlab
import trimesh
import colormap

from pysdf import SDF
import shutil

#THRESH = 0.02


def load_Rt_from(filename):
    lines = open(filename).read().splitlines()
    lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
    pose = np.asarray(lines).astype(np.float32).squeeze()

    return pose

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

def load_ply(path, min_B = [-np.inf, -np.inf, np.inf], max_B = [np.inf, np.inf, np.inf]):
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
    #print("vtx :" , vtx_num ,"  face :" , face_num ,"  nml_flg: " , nml_flg,"  rgb_flg :" , rgb_flg,"  face_flg :" , face_flg)

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

def load_factory(path , device, min_B = [-np.inf, -np.inf, np.inf], max_B = [np.inf, np.inf, np.inf]):
    '''
    return verts_tensor , face_tensor , face_vertices , vertices_normal , face_normal
    '''
    #Load GT Mesh
    ext = os.path.splitext(path)[-1]
    #verts = torch.Tensor(np.load(GT_path)["scan_pc"])   #from cape npz
    if ext == ".ply":
        verts , _ , _ , face , _ , _ =  load_ply(path, min_B, max_B)
        #print(len(verts))
        verts_tensor = torch.tensor(verts)
        face_tensor  = torch.tensor(face)
        face         = np.array(face)
        #print(verts_tensor.shape)
        #print(face_tensor.shape)
        #print(face.shape)
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
        #print(path)
        #print(verts_tensor.shape)
        #print(face_tensor.shape)
        #print(face.shape)

    verts_tensor  = verts_tensor.unsqueeze(0).to(device)       #.to('cpu')
    face_tensor   = face_tensor.to(device)                     #.to('cpu')
    #print(face_tensor.shape)
    """if face_tensor.shape[0] > 0:
        face_vertices  = None #kaolin.ops.mesh.index_vertices_by_faces(verts_tensor,face_tensor)

        #compute vertices & face normal 
        face_normal_tensor = None #kaolin.ops.mesh.face_normals(face_vertices , True)
        face_normal = None #face_normal_tensor[0].to('cpu').detach().numpy().copy()
        ms = pymeshlab.MeshSet()
        new_mesh = pymeshlab.Mesh(vertex_matrix = verts ,face_matrix = face , f_normals_matrix = face_normal)
        ms.add_mesh(new_mesh)
        ms.compute_normal_per_vertex()
        curr_mesh = ms.mesh(0)
        vertices_normal = curr_mesh.vertex_normal_matrix()
    else:"""
    face_vertices = None
    vertices_normal = None
    face_normal = None

    return verts , verts_tensor , face_tensor , face_vertices , vertices_normal , face_normal

def chamfer(GT_path, dir_path, heat_map_path, mesh_list, THRESH, device = None, min_B = [-np.inf, -np.inf, -np.inf], max_B = [np.inf, np.inf, np.inf]):
    print("Loading {} ...".format(GT_path))
    #verts_GT, verts_GT_tensor, face_GT, face_vertices_GT, vertices_normal_GT, face_normal_GT = load_factory(GT_path, device, min_B, max_B)
    gt_mesh = trimesh.load_mesh(GT_path)

    trans_file = os.path.join(dir_path, 'transform.txt')
    print(trans_file)
    transfo = load_Rt_from(trans_file) 
    
    gt_mesh.vertices = (gt_mesh.vertices @ transfo[:3,:3].T + transfo[:3, 3].reshape((3,1)).T)
    gt_mesh = gt_mesh.slice_plane((0,0,0.05), (0,0,1)) ##gt_mesh = gt_mesh.slice_plane((0,0,0.05), (0,0,1))
    #gt_mesh = gt_mesh.slice_plane((0,-0.34,0), (0,1,0)) 
    """gt_mesh = gt_mesh.slice_plane((-2.8,0,0), (1,0,0)) 
    gt_mesh = gt_mesh.slice_plane((0.2,0,0), (-1,0,0)) 
    gt_mesh = gt_mesh.slice_plane((0,-2.0,0), (0,1,0))
    gt_mesh = gt_mesh.slice_plane((0,1.6,0), (0,-1,0))"""
    """gt_mesh = gt_mesh.slice_plane((0,0,35.6), (0,0,1))
    gt_mesh = gt_mesh.slice_plane((0,0,42), (0,0,-1))
    gt_mesh = gt_mesh.slice_plane((-6.9,0,0), (1,0,0))
    gt_mesh = gt_mesh.slice_plane((-0.9,0,0), (-1,0,0))
    gt_mesh = gt_mesh.slice_plane((0,-3.5,0), (0,1,0))
    gt_mesh = gt_mesh.slice_plane((0,3.1,0), (0,-1,0))"""
    
    #print("Computing largest connected component of test mesh...")
    #gt_mesh = sorted(gt_mesh.split(only_watertight=False), key=lambda x : x.vertices.shape[0], reverse=True)[0]

    c = colormap.Colormap()
    jet = c.cmap("jet")

    chamfer_list1 = []
    chamfer_list2 = []
    for mesh in mesh_list:
        print("Loading {} ...".format(mesh))
        #verts, verts_tensor, face, face_vertices, vertices_normal, face_normal = load_factory(mesh, device, min_B, max_B)
        pred_mesh = trimesh.load_mesh(mesh)
        pred_mesh.vertices = (pred_mesh.vertices @ transfo[:3,:3].T + transfo[:3, 3].reshape((3,1)).T)
        pred_mesh = pred_mesh.slice_plane((0,0,0.05), (0,0,1))
        #pred_mesh = pred_mesh.slice_plane((0,-0.34,0), (0,1,0))
        """pred_mesh = pred_mesh.slice_plane((0,0,35), (0,0,1))
        pred_mesh = pred_mesh.slice_plane((0,0,42), (0,0,-1))
        pred_mesh = pred_mesh.slice_plane((-6.9,0,0), (1,0,0))
        pred_mesh = pred_mesh.slice_plane((-0.9,0,0), (-1,0,0))
        pred_mesh = pred_mesh.slice_plane((0,-3.5,0), (0,1,0))
        pred_mesh = pred_mesh.slice_plane((0,3.1,0), (0,-1,0))"""
        """pred_mesh = pred_mesh.slice_plane((-2.8,0,0), (1,0,0)) 
        pred_mesh = pred_mesh.slice_plane((0.2,0,0), (-1,0,0)) 
        pred_mesh = pred_mesh.slice_plane((0,-2.0,0), (0,1,0)) 
        pred_mesh = pred_mesh.slice_plane((0,1.6,0), (0,-1,0)) """

        print("Computing largest connected component of test mesh...")
        pred_mesh = sorted(pred_mesh.split(only_watertight=False), key=lambda x : x.vertices.shape[0], reverse=True)[0]

        print("Computing icp...")
        m, transformed, cost = trimesh.registration.icp(gt_mesh.vertices, pred_mesh.vertices, initial=np.eye(4), threshold=1.0e-7, max_iterations=20)
        #transformed = gt_mesh.vertices

        print("Computing chamfer distances...")
        sdf_gt = SDF(transformed, gt_mesh.faces)
        sdf_test = SDF(pred_mesh.vertices, pred_mesh.faces)

        dists0 = np.abs(sdf_test(transformed))
        dists1 = np.abs(sdf_gt(pred_mesh.vertices))

        # Some values can be very high due to numerical errors
        dists0[dists0 > 0.1] = 0.1
        dists1[dists1 > 0.1] = 0.1

        cd0 = float(np.mean(dists0))
        cd1 = float(np.mean(dists1))

        print("Chamfer dist gt -> test:", cd0)
        print("Chamfer dist test -> gt:", cd1)

        chamfer_list1.append(cd0)
        chamfer_list2.append(cd1)

        mesh_name = mesh.split("\\")[-1] # "/"
        colors0 = jet(dists0 / THRESH)
        gt_mesh.visual = trimesh.visual.color.ColorVisuals(mesh=gt_mesh, vertex_colors=np.uint8(colors0 * 255))
        gt_mesh.apply_transform(m)
        gt_mesh.export(os.path.join(heat_map_path, mesh_name), encoding='binary')

        colors1 = jet(dists1 / THRESH)
        pred_mesh.visual = trimesh.visual.color.ColorVisuals(mesh=pred_mesh, vertex_colors=np.uint8(colors1 * 255))
        pred_mesh.export(os.path.join(heat_map_path, "pred_"+mesh_name), encoding='binary')

        #_ = save_color_mesh(dists0, transformed , gt_mesh.faces, os.path.join(heat_map_path, mesh_name), vmin=0.0 , vmax=1.0*THRESH)   
        #_ = save_color_mesh(dists1, pred_mesh.vertices , pred_mesh.faces, os.path.join(heat_map_path, "pred_"+mesh_name), vmin=0.0 , vmax=1.0*THRESH)   
        
        
    return chamfer_list1, chamfer_list2      
  

def main():
    parser = argparse.ArgumentParser(description='chamfer distance')
    parser.add_argument(
        '--dir_path',
        type=str,
        help='')    
    
    parser.add_argument(
        '--GT_path',
        type=str,
        help='')    
    
    args = parser.parse_args()
    
    # Get list of 3D meshes in the folder
    mesh_list = sorted(glob(os.path.join(args.dir_path, '*.ply'))) #os.listdir(args.dir_path)
    mesh_list.append(sorted(glob(os.path.join(args.dir_path, '*.obj')))) #os.listdir(args.dir_path)
    mesh_list.pop()
    print(mesh_list)

    # Get bounding box
    f = open(args.dir_path + "/bbox.conf")
    conf_text = f.read()
    f.close()
    conf_bbox = ConfigFactory.parse_string(conf_text)
    visual_hull = conf_bbox.get_list('data_info.visual_hull')
    data_name = conf_bbox.get_string('data_info.data_name')
    min_B = visual_hull[0:3]
    max_B = visual_hull[3:6]

    # Make folder to save heat maps
    heat_map_path = os.path.join(args.dir_path , "HeatMaps")
    if_not_exists_makedir(heat_map_path , confirm_delete = False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    root_dir = os.path.abspath(os.path.join(args.dir_path, os.pardir))
    result_csv = os.path.join(root_dir,"results.csv")
    if not os.path.exists(result_csv):
        # Creat file to save all results
        # Write Header file
        f = open(result_csv,"w")
        f.write(" " +  ",")
        for mesh in mesh_list:
            mesh_name = mesh.split("\\")[-1].split(".")[0]
            f.write(mesh_name +  "," + " " +  "," ) 
        f.write("\n")
        
        f.write(" " +  ",")
        for mesh in mesh_list:
            f.write("Chamfer (mm)" +  "," + "iou" +  ",") 
        f.write("\n")
    else:
        f = open(result_csv,"a")

    Chamfer_results1, Chamfer_results2 = chamfer(args.GT_path, args.dir_path, heat_map_path,mesh_list,  0.03, device) #, min_B, max_B)
    
    f.write(data_name +  ",")
    for chm1, chm2 in zip(Chamfer_results1, Chamfer_results2):
        f.write("{} , {} ,".format(chm1, chm2)) 
    f.write("\n")

    f.close()
    

if __name__ == "__main__":
    main()

