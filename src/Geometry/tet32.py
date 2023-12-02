import torch
import open3d as o3d
import numpy as np
import sys
from timeit import default_timer as timer 
sys.path.append('/root/VortSDF/')
import src.IO.ply as ply

from torch.utils.cpp_extension import load
tet_utils = load('tet_utils', ['src/Geometry/tet_utils.cpp'], verbose=True)

def get_faces_list_from_tetrahedron(tetrahedron):
    """
    Gets the faces from the given tetrahedron.

    Parameters:
    - tetrahedron: The tetrahedron to get the faces from.

    Returns:
    - faces: The faces of the tetrahedron in a list of immutable set
    """
    return [
        [tetrahedron[1], tetrahedron[2], tetrahedron[3]],
        [tetrahedron[0], tetrahedron[2], tetrahedron[3]],
        [tetrahedron[0], tetrahedron[1], tetrahedron[3]],
        [tetrahedron[0], tetrahedron[1], tetrahedron[2]]
    ]

def get_faces_from_tetrahedron(tetrahedron):
    """
    Gets the faces from the given tetrahedron.

    Parameters:
    - tetrahedron: The tetrahedron to get the faces from.

    Returns:
    - faces: The faces of the tetrahedron in a list of immutable set
    """
    return [
        frozenset([tetrahedron[1], tetrahedron[2], tetrahedron[3]]),
        frozenset([tetrahedron[0], tetrahedron[2], tetrahedron[3]]),
        frozenset([tetrahedron[0], tetrahedron[1], tetrahedron[3]]),
        frozenset([tetrahedron[0], tetrahedron[1], tetrahedron[2]])
    ]

## Implements the 32 bits tetrahedral mesh data structure
class Tet32:
    ## sites must be a (n,3) numpy array
    def __init__(self, sites):
        self.device = torch.device('cuda')

        ## Build the tetrahedral mesh from the sites
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(sites)
        
        start = timer()
        KDtree = o3d.geometry.KDTreeFlann(point_cloud)
        print('KDTreeFlann time:', timer() - start)    

        o3d_mesh, _ = o3d.geometry.TetraMesh.create_from_point_cloud(point_cloud)
        
        self.vertices = o3d_mesh.vertices
        self.tetras = o3d_mesh.tetras

        ## 4 values for indices of summits
        self.summits = np.asarray(self.tetras)
        self.nb_tets = self.summits.shape[0]

        print("nb tets: ", self.nb_tets)
        return

        ## 4 values for indices of neighbors

        # first iterate over all tetrahedron and compute adjacencies for each face
        faces_to_tetrahedron = {}
        for i, tetrahedron in enumerate(self.tetras):
            faces = get_faces_from_tetrahedron(tetrahedron)
            for face in faces:
                try :
                    faces_to_tetrahedron[face].append(i)
                except KeyError:
                    faces_to_tetrahedron[face] = [i]

        print("Faces adjacencies computed")

        # second iterate over all tetrahedron and get neighbors from faces
        self.neighbors = torch.zeros([self.nb_tets, 4], dtype = torch.int32).cuda().contiguous()

        for i, tetrahedron in enumerate(self.tetras):
            faces = get_faces_from_tetrahedron(tetrahedron)
            for j, face in enumerate(faces):
                if len(faces_to_tetrahedron[face]) == 1:
                    self.neighbors[i,j] = -1 
                elif faces_to_tetrahedron[face][0] == i:
                    self.neighbors[i,j] = faces_to_tetrahedron[face][1] 
                else:
                    self.neighbors[i,j] = faces_to_tetrahedron[face][0] 

            # make last summit index as xor 
            self.summits[i,3] = self.summits[i,0] ^ self.summits[i,1] ^ self.summits[i,2] ^ self.summits[i,3] 

        print("Neighboors computed")

        values = np.tile(np.array([1.]),len(sites))
        tri_mesh = o3d_mesh.extract_triangle_mesh(o3d.utility.DoubleVector(values),1.)
        print(tri_mesh)
        self.tri_vertices = np.asarray(tri_mesh.vertices)
        self.tri_faces = np.asarray(tri_mesh.triangles)



    def save(self, filename):
        faces_list = [] 
        for i, tetrahedron in enumerate(self.tetras):
            for x in get_faces_list_from_tetrahedron(tetrahedron):
                curr_face = list(x)
                if len(curr_face) == 3:
                    faces_list.append(curr_face)
        print(np.asarray(faces_list).shape)

        ply.save_ply(filename, np.asarray(self.vertices).transpose(), f=(np.asarray(faces_list)).transpose())




if __name__=='__main__':
    tet_utils.helloworld()
    visual_hull = [-1, -1, -1, 1, 1, 1]
    import sampling as sampler
    sites = sampler.sample_Bbox(visual_hull[0:3], visual_hull[3:6], 16, perturb_f =  visual_hull[3]*0.005) 
    T32 = Tet32(sites)
    T32.save("data/bmvs_man/test.ply")