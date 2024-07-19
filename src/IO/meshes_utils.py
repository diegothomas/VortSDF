import sys
import struct
import numpy as np

def load_ply(path):
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

    if nml_flg and rgb_flg and face_flg:
        for i in range(header_len,vtx_num+header_len):
            vtx.append(list(map(float,lines[i].split(" ")[0:3])))
            nml.append(list(map(float,lines[i].split(" ")[3:6])))
            rgb.append(list(map(int,lines[i].split(" ")[6:9])))

        for i in range(header_len+vtx_num,face_num+header_len+vtx_num):
            face.append(list(map(int,lines[i].split(" ")[1:4])))
        

    elif nml_flg and rgb_flg            :
        for i in range(header_len,vtx_num+header_len):
            vtx.append(list(map(float,lines[i].split(" ")[0:3])))
            nml.append(list(map(float,lines[i].split(" ")[3:6])))
            rgb.append(list(map(int,lines[i].split(" ")[6:9])))
        

    elif nml_flg           and  face_flg:
        for i in range(header_len,vtx_num+header_len):
            vtx.append(list(map(float,lines[i].split(" ")[0:3])))
            nml.append(list(map(float,lines[i].split(" ")[3:6])))

        for i in range(header_len+vtx_num,face_num+header_len+vtx_num):
            face.append(list(map(int,lines[i].split(" ")[1:4])))
        
    elif            rgb_flg and face_flg:
        for i in range(header_len,vtx_num+header_len):
            vtx.append(list(map(float,lines[i].split(" ")[0:3])))
            rgb.append(list(map(int,lines[i].split(" ")[3:6])))

        for i in range(header_len+vtx_num,face_num+header_len+vtx_num):
            face.append(list(map(int,lines[i].split(" ")[1:4])))

    elif nml_flg                       :
        for i in range(header_len,vtx_num+header_len):
            vtx.append(list(map(float,lines[i].split(" ")[0:3])))
            nml.append(list(map(float,lines[i].split(" ")[3:6])))

    elif            rgb_flg            :
        for i in range(header_len,vtx_num+header_len):
            vtx.append(list(map(float,lines[i].split(" ")[0:3])))
            rgb.append(list(map(int,lines[i].split(" ")[3:6])))

    elif                       face_flg:
        for i in range(header_len,vtx_num+header_len):
            vtx.append(list(map(float,lines[i].split(" ")[0:3])))

        for i in range(header_len+vtx_num,face_num+header_len+vtx_num):
            face.append(list(map(int,lines[i].split(" ")[1:4])))
    return vtx , nml , rgb , face , vtx_num , face_num

def save_ply(path, vtx , nml = None , rgb = None, face = None , vtx_num = 0 , face_num = 0):
    if nml != None and len(vtx) != len(nml):
        print("vtx num and nml num is different")
        print("please check you don't use nml from obj")
        sys.exit()
        
    if rgb != None and len(vtx) != len(rgb):
        print("vtx_num and rgb num is different")
        print("len(vtx) : " , len(vtx))
        print("len(rgb) : " , len(rgb))
        sys.exit()
    f = open(path,"w")
    
    header1 = [
        "ply",
        "format ascii 1.0",
        "comment ply file created by Kitamrua",
        "element vertex " + str(vtx_num),
        "property float x",
        "property float y",
        "property float z"
    ]
    
    header_nml = []
    if nml != None:
        header_nml = [
            "property float nx",
            "property float ny",
            "property float nz"
        ]

    header_rgb = []
    if rgb != None:
        header_rgb =[
            "property uchar red",
            "property uchar green",
            "property uchar blue"
        ]

    header_face = []
    if face != None:
        header_face = [
            "element face " + str(face_num)
        ]

    header2 = [
        "property list uchar int vertex_index",
        "end_header"
    ]

    vtx_lines = []
    if nml != None and rgb != None:
        for i in range(vtx_num):
            vtx_lines.append(str(vtx[i][0]) + " "  + str(vtx[i][1]) + " "  + str(vtx[i][2]) + " " + str(nml[i][0]) + " "  + str(nml[i][1]) + " "  + str(nml[i][2]) + " " + str(rgb[i][0]) + " "  + str(rgb[i][1]) + " "  + str(rgb[i][2]))

    elif nml != None :
        for i in range(vtx_num):
            vtx_lines.append(str(vtx[i][0]) + " "  + str(vtx[i][1]) + " "  + str(vtx[i][2]) + " " + str(nml[i][0]) + " "  + str(nml[i][1]) + " "  + str(nml[i][2]))

    elif rgb != None:
        for i in range(vtx_num):
            vtx_lines.append(str(vtx[i][0]) + " "  + str(vtx[i][1]) + " "  + str(vtx[i][2]) + " " + str(rgb[i][0]) + " "  + str(rgb[i][1]) + " "  + str(rgb[i][2]))
    else :
        for i in range(vtx_num):
            vtx_lines.append(str(vtx[i][0]) + " "  + str(vtx[i][1]) + " "  + str(vtx[i][2])) 


    face_lines = []
    if face != None:
        for i in range(face_num):
            face_lines.append(str(3) + " " + str(face[i][0]) + " "  + str(face[i][1]) + " "  + str(face[i][2]))

    eof = ['']
    lines = header1 + header_nml + header_rgb + header_face + header2 + vtx_lines + face_lines + eof
    f.write("\n".join(lines))


def load_obj(path):
    """
    return -> vtx , nml , txr , face , vtx2txr , vtx2nml , vtx_num , face_num
    """
    f = open(path,"r")
    obj = f.read()
    lines=obj.split("\n")

    vtx_num = 0
    face_num = 0

    vtx = []
    txr = []
    nml = []
    face = []
    vtx2txr = []
    vtx2nml = []
            
    i = 0
    while(1):
        if lines[i] == '':
            #print("end of file")
            #print(i)
            break

        line = lines[i].split(" ")
        if "v" == line[0]:
            vtx.append(list(map(float,line[1:4])))
            vtx_num += 1
        if "vt" == line[0]:
            txr.append(list(map(float,line[1:3])))
        if "vn" == line[0]:
            nml.append(list(map(float,line[1:4])))
        if "f" == line[0]:
            if len(line[1].split("/")) == 3:
                face.append(   [int(line[1].split("/")[0])-1,int(line[2].split("/")[0])-1,int(line[3].split("/")[0])-1])
                vtx2txr.append([int(line[1].split("/")[1])-1,int(line[2].split("/")[1])-1,int(line[3].split("/")[1])-1])
                vtx2nml.append([int(line[1].split("/")[2])-1,int(line[2].split("/")[2])-1,int(line[3].split("/")[2])-1])
            if len(line[1].split("/")) == 1:
                face.append(   [int(line[1].split("/")[0])-1,int(line[2].split("/")[0])-1,int(line[3].split("/")[0])-1])
                vtx2txr=None
                vtx2nml=None
            face_num += 1
        i+=1

    return vtx , nml , txr , face , vtx2txr , vtx2nml , vtx_num , face_num

def load_obj2(path):
    """
    return -> vtx , nml , txr , face , vtx2txr , vtx2nml , vtx_num , face_num
    """
    f = open(path,"r")
    obj = f.read()
    lines=obj.split("\n")

    vtx_num = 0
    face_num = 0

    vtx = []
    txr = []
    nml = []
    face = []
    vtx2txr = {}
    vtx2nml = []
            
    i = 0
    while(1):
        if lines[i] == '':
            #print("end of file")
            #print(i)
            break

        line = lines[i].split(" ")
        if "v" == line[0]:
            vtx.append(list(map(float,line[1:4])))
            vtx_num += 1
        if "vt" == line[0]:
            txr.append(list(map(float,line[1:3])))
        if "vn" == line[0]:
            nml.append(list(map(float,line[1:4])))
        if "f" == line[0]:
            face.append(   [int(line[1].split("/")[0])-1,int(line[2].split("/")[0])-1,int(line[3].split("/")[0])-1])
            #vtx2txr.append([int(line[1].split("/")[1])-1,int(line[2].split("/")[1])-1,int(line[3].split("/")[1])-1])
            if (int(line[1].split("/")[0])-1 in vtx2txr) == False:
                vtx2txr[int(line[1].split("/")[0])-1] = []
            if (int(line[2].split("/")[0])-1 in vtx2txr) == False:
                vtx2txr[int(line[2].split("/")[0])-1] = []
            if (int(line[3].split("/")[0])-1 in vtx2txr) == False:
                vtx2txr[int(line[3].split("/")[0])-1] = []

            vtx2txr[int(line[1].split("/")[0])-1].append(int(line[1].split("/")[1])-1)
            vtx2txr[int(line[2].split("/")[0])-1].append(int(line[2].split("/")[1])-1)
            vtx2txr[int(line[3].split("/")[0])-1].append(int(line[3].split("/")[1])-1)

            vtx2nml.append([int(line[1].split("/")[2])-1,int(line[2].split("/")[2])-1,int(line[3].split("/")[2])-1])
            face_num += 1
        i+=1

    for vtxID in vtx2txr:
        vtx2txr[vtxID] = list(set(vtx2txr[vtxID]))

    return vtx , nml , txr , face , vtx2txr , vtx2nml , vtx_num , face_num

def save_obj(path ,vtx , nml = None , txr = None , face=None , vtx2txr=None , vtx2nml=None , vtx_num=None , face_num=None):  
    f = open(path,"w")
    header1 = [ 
    "# Blender v2.93.0 OBJ File: 'unskin2.blend'",
    "# www.blender.org",
    "mtllib uvskin.mtl",
    "o skin"
    ]

    vtx_lines = []
    for i in range(vtx_num):
        vtx_lines.append("v " + str(vtx[i][0]) + " "  + str(vtx[i][1]) + " "  + str(vtx[i][2]))

    if txr != None:
        txr_lines = []
        for t in txr:
            txr_lines.append("vt " + str(t[0]) + " "  + str(t[1]))
    else:
        txr_lines = []

    if nml != None:
        nml_lines = []
        for n in nml:
            nml_lines.append("vn " + str(n[0]) + " "  + str(n[1])+ " "  + str(n[2]))
    else:
        nml_lines = []


    header2 = [
    "usemtl None",
    "s off"
    ]

    if face != None:
        face_lines = []
        for i,fc in enumerate(face):
            if vtx2txr != None and vtx2nml != None: 
                face_lines.append("f " + str(fc[0]+1) + "/" + str(vtx2txr[i][0]+1) + "/" + str(vtx2nml[i][0]+1) + " " + str(fc[1]+1) + "/" + str(vtx2txr[i][1]+1) + "/" + str(vtx2nml[i][1]+1) + " " + str(fc[2]+1) + "/" + str(vtx2txr[i][2]+1) + "/" + str(vtx2nml[i][2]+1))
            elif vtx2txr != None :
                face_lines.append("f " + str(fc[0]+1) + "/" + str(vtx2txr[i][0]+1) + "/" + " " + str(fc[1]+1) + "/" + str(vtx2txr[i][1]+1) + "/" + " " + str(fc[2]+1) + "/" + str(vtx2txr[i][2]+1) + "/")
            elif vtx2nml != None: 
                face_lines.append("f " + str(fc[0]+1) + "/" + "/" + str(vtx2nml[i][0]+1) + " " + str(fc[1]+1) + "/" + "/" + str(vtx2nml[i][1]+1) + " " + str(fc[2]+1) + "/" + "/" + str(vtx2nml[i][2]+1))
            else:
                face_lines.append("f " + str(fc[0]+1) + "/" + "/"  + " " + str(fc[1]+1) + "/" + "/" + " " + str(fc[2]+1) + "/" + "/" )
    else:
        face_lines = []

    eof = ['']
    lines = header1 + vtx_lines + txr_lines + nml_lines + header2 + face_lines + eof
    f.write("\n".join(lines))

def load_skinweight_bin(sw_path,vtx_num = 35010):   #vtx_num = 35010 is hardcoding for our task
    f = open(sw_path,"rb")
    sw_bin = f.read()
    sw_list = []

    for i in range(24*vtx_num):
        sw_list.append(struct.unpack("f",sw_bin[4*i:4*i+4])[0])
    sw = np.array(sw_list)
    sw = sw.astype(np.float32)
    return sw

def save_skinweight_bin(sw_path,data,vtx_num = 35010):   #vtx_num = 35010 is hardcoding for our task
    """
    f = open(sw_path,"wb")
    for i in range(24*vtx_num):
            packed_data = struct.pack("f", data[i])
            f.write(packed_data)
    f.close()
    """
    data = np.array(data).astype(np.float32)
    data.tofile(sw_path)

def mapVtx2UV(uvskin_path):
    uv_vtx , uv_nml , uv_txr , uv_face , uv_vtx2txr , uv_vtx2nml , uv_vtx_num , uv_face_num = load_obj(uvskin_path)

    #map generation is needed only first time
    mapVtx2UV = {}
    for i, fc in enumerate(uv_face):
        for j in range(3):
            vtx_id = fc[j]
            txr_id = uv_vtx2txr[i][j]
            if vtx_id not in mapVtx2UV.keys():
                mapVtx2UV[vtx_id] = []
            else:
                mapVtx2UV[vtx_id].append(txr_id)
            mapVtx2UV[vtx_id] = list(set(mapVtx2UV[vtx_id]))

    #with open('mapVtx2UV.json', 'w') as fp:
    #    json.dump(mapVtx2UV, fp)
    return mapVtx2UV

if __name__ == "__main__":
    ply_path = r"D:\Project\Human\Pose2Texture\aftprocess\result\trainWithHuawei\reconst_Tpose_HuaweiGT\reconst_Tpose_0000.ply"
    obj_path = r"D:\Data\Human\Template-star-0.015\uvskin.obj"
    #load_ply(ply_path)
    """vtx , nml , txr , face , vtx2txr , vtx2nml , vtx_num , face_num = load_obj(obj_path)
    print(vtx[0] ," ", nml[0] ," " , txr[0] ," " , face[0]  ," ", vtx2txr[0] ," " , vtx2nml[0] ," " , vtx_num ," " , face_num ," ")
    save_obj("test.obj",vtx , nml , txr , face , vtx2txr , vtx2nml , vtx_num , face_num)
    s_vtx , s_nml , s_txr , s_face , s_vtx2txr , s_vtx2nml , s_vtx_num , s_face_num = load_obj("test.obj")

    print(vtx == s_vtx)
    print(nml == s_nml)
    print(txr == s_txr)
    print(face == s_face)
    print(vtx2txr == s_vtx2txr )
    print(vtx2nml == s_vtx2nml)
    print(vtx_num == s_vtx_num)
    print(face_num == s_face_num)
    """
    vtx , nml , rgb , face , vtx_num , face_num = load_ply(ply_path)
    print(len(rgb))
    print(rgb[0][0])
    #save_ply("test.ply" , vtx , nml , rgb , face , vtx_num , face_num)
    #s_vtx , s_nml , s_rgb , s_face , s_vtx_num , s_face_num = load_ply("test.ply")

    #print(vtx == s_vtx)
    #print(nml == s_nml)
    #print(rgb == s_rgb)
    #print(face == s_face)
    #print(vtx_num == s_vtx_num )
    #print(face_num == s_face_num)

