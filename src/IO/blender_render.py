import bpy

#from bpy import context
import os
import time
import tempfile
from pathlib import Path
import sys
import glob
#from natsort import natsorted

import re
import argparse
import math
import numpy

import meshes_utils

def numericalSort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def render(mesh_path,save_folder_path,rgb_flg,data = "whole"):
    camera = bpy.data.objects["Camera"]
    
    scene = bpy.data.scenes['Scene']
    render_setting = scene.render
    render_setting.resolution_x = 1280 #1920
    render_setting.resolution_y = 1280 #1920

    camera.location =  (0.0, 0.0, 3.0)  #ours & scanimate3(range of motion)

    pi = math.pi
    camera.rotation_euler = (0.0, 0.0, -pi / 2.0)

    lamp = bpy.data.objects["Light"]
    #lamp.location = (0.0, 1.8, 7.0)    #scanimate
    #lamp.location = (0.0, 0.0, 9.0)   #ours
    #lamp.location = (0.0, 2.0, 8.0)    #aft_trans(4ddata , cape33 )
    #lamp.location = (0.0, 0.0, 12.0)   #ours(Dancing 1,Dancing2)
    lamp.location = (0.0, 0.0, 8.0)    #ours & scanimate3(range of motion , Muscle_ref_bundle)

    targetob = bpy.data.objects.get("Cube")
    if targetob != None:
        bpy.data.objects.remove(targetob)
        #bpy.data.objects.remove(bpy.data.objects["Cube"], do_unlink=True)
        
    if os.path.exists(save_folder_path) == False:
        os.mkdir(save_folder_path)

    frame_num = 0

    print("input_path:",mesh_path)
    #開始のフレームを設定する
    bpy.context.scene.frame_set(frame_num)
    
    data_type = os.path.basename(mesh_path).split(".")[-1]

    if data_type == "ply":
        vtx , nml , rgb , face , vtx_num , face_num = meshes_utils.load_ply(mesh_path)
    elif data_type == "obj":
        vtx , nml , txr , face , vtx2txr , vtx2nml , vtx_num , face_num = meshes_utils.load_obj(mesh_path)          

    mesh_center = numpy.array(vtx).mean(axis = 0, keepdims = True)[0]
    print("Data loaded, ", numpy.array(vtx).shape, mesh_center)
    #vtx = vtx - mesh_center    
    for id, v_id in enumerate(vtx):
        vtx[id] = vtx[id] - mesh_center

    targetob = bpy.data.objects.get("Cube")
    if targetob != None:
        bpy.data.objects.remove(targetob)
            
    msh = bpy.data.meshes.new("cubemesh") #create mesh data
    msh.from_pydata(vtx, [], face) # 頂点座標と各面の頂点の情報でメッシュを作成
    obj = bpy.data.objects.new("Cube", msh) # メッシュデータでオブジェクトを作成

    print("mesh data created")
    if rgb_flg == True:
        print("rgb_flg ON")
        # make vertex colour data
        msh.vertex_colors.new(name='col')

        print("len:",len(msh.vertex_colors['col'].data))
        print("face_len:",face_num)
        for idx, vc in enumerate(msh.vertex_colors['col'].data):
            vtx_cnt = idx%3
            vc.color =  [0.8,0.8,0.8,1.0]    
            #vc.color =  [rgb[face[int(idx/3)][vtx_cnt]][0]/255,rgb[face[int(idx/3)][vtx_cnt]][1]/255,rgb[face[int(idx/3)][vtx_cnt]][2]/255,1.0]

        # make its first material slot
        mat = bpy.data.materials.new(name = "Skin")
        mat.use_nodes = True #Make so it has a node tree

        #Get the shader
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        base_color = bsdf.inputs['Base Color'] #Or principled.inputs[0]

        #Add the vertex color node
        vtxclr = mat.node_tree.nodes.new('ShaderNodeVertexColor')
        #Assign its layer
        vtxclr.layer_name = "col"
        #Link the vertex color to the shader
        mat.node_tree.links.new( vtxclr.outputs[0], base_color )
        obj.data.materials.append(mat)
    else:
        print("rgb_flg OFF")

    bpy.context.scene.render.film_transparent = True
    bpy.context.scene.collection.objects.link(obj) # シーンにオブジェクトを配置
    
    print("obj linked")    
    #bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)   #sleep
    #time.sleep(5)

    #rendering with image
    print("rotating object")    
    #obj.rotation_euler = (-pi / 2.0, 0.0, pi / 3.0)
    #obj.rotation_euler = (0.0, 0.0, 0.0)
    #obj.rotation_euler.rotate_axis("X", -pi / 2.0)
    #obj.rotation_euler.rotate_axis("Y", -pi / 3.0)
    #obj.rotation_euler.rotate((-pi / 2.0, 0.0, pi / 3.0))
    obj.rotation_euler.rotate_axis("X", 2.0 * pi * 35 / 48.0)
    obj.rotation_euler.rotate_axis("Y", -pi / 3.0)

    print("rendering image")    

    for i in range(48):
        obj.rotation_euler.rotate_axis("X", 2.0 * pi / 48.0)
        bpy.ops.render.render()
        save_path = os.path.join(save_folder_path , 'rendering_' + str(i) + '.png')
        bpy.data.images['Render Result'].save_render(filepath = save_path)

    print("save_path:",save_path)


if __name__ == "__main__":
    #print("please call rendering.py or renderingWithColor.py")
    mesh_folder_path = sorted(glob.glob(r"D:\Project\Human\Pose2Texture\aftprocess\result\reconst_posed\*"),key=numericalSort)
    #mesh_folder_path = sorted(glob.glob(r"D:\Data\Human\HUAWEI\MIT\MIT\data\FittedMesh_ply\*.ply"),key=numericalSort)
    save_folder_path = r"D:\Project\Human\Pose2Texture\aftprocess\result\rendering"

    rgb_flg = True
    render(mesh_folder_path,save_folder_path,rgb_flg)