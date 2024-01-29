import torch
import open3d as o3d
import numpy as np
import sys
from timeit import default_timer as timer 
sys.path.append('/root/VortSDF/')
import src.IO.ply as ply

from torch.utils.cpp_extension import load
tet32_march_cuda = load(
    'tet32_march_cuda', ['src/Cuda/tet32_march_cuda.cpp', 'src/Cuda/tet32_march_cuda.cu'], verbose=True)
#help(cvt_march_cuda)


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

        self.o3d_mesh, _ = o3d.geometry.TetraMesh.create_from_point_cloud(point_cloud)
        
        self.vertices = self.o3d_mesh.vertices
        self.tetras = self.o3d_mesh.tetras

        ## 4 values for indices of summits
        self.nb_tets = np.asarray(self.tetras).shape[0]

        self.summits = np.zeros([self.nb_tets, 4], dtype = np.int32)
        self.summits[:,] = np.asarray(self.tetras)[:,:]

        print("nb tets: ", self.nb_tets)

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

        self.summits = torch.from_numpy(self.summits).int().cuda()
        self.sites = torch.from_numpy(np.asarray(self.vertices)).float().cuda()
        print("Neighboors computed")




    ## Make adjacencies for cameras
    def make_adjacencies(self, cam_ids):
        cam_tets = [[] for _ in range(cam_ids.shape[0])]

        summ_cpu = self.summits.cpu().numpy()
        for i, _ in enumerate(self.tetras):
            if summ_cpu[i,0] in cam_ids:
                tet_id = np.where(summ_cpu[i,0] == cam_ids)[0][0]
                cam_tets[tet_id].append(i)
            if summ_cpu[i,1] in cam_ids:
                tet_id = np.where(summ_cpu[i,1] == cam_ids)[0][0]
                cam_tets[tet_id].append(i)
            if summ_cpu[i,2] in cam_ids:
                tet_id = np.where(summ_cpu[i,2] == cam_ids)[0][0]
                cam_tets[tet_id].append(i)
            if summ_cpu[i,0] ^ summ_cpu[i,1] ^ summ_cpu[i,2] ^ summ_cpu[i,3] in cam_ids:
                tet_id = np.where(summ_cpu[i,0] ^ summ_cpu[i,1] ^ summ_cpu[i,2] ^ summ_cpu[i,3] == cam_ids)[0][0]
                cam_tets[tet_id].append(i)
            
        self.offsets_cam = np.zeros(cam_ids.shape[0])
        self.cam_tets = []
        offset_curr = 0
        for i in range(cam_ids.shape[0]):
            self.cam_tets = self.cam_tets + cam_tets[i]
            offset_curr = offset_curr + len(cam_tets[i])
            self.offsets_cam[i] = offset_curr

        self.cam_tets = np.stack(self.cam_tets)

        self.offsets_cam = torch.from_numpy(self.offsets_cam).int().cuda()
        self.cam_tets = torch.from_numpy(self.cam_tets).int().cuda()



    def surface_from_sdf(self, values, filename = ""):
        print(values.shape)
        tri_mesh = self.o3d_mesh.extract_triangle_mesh(o3d.utility.DoubleVector(values.astype(np.float64)),0.0)
        print(tri_mesh)
        self.tri_vertices = np.asarray(tri_mesh.vertices)
        self.tri_faces = np.asarray(tri_mesh.triangles)
        if not filename == "":
            ply.save_ply(filename, np.asarray(self.tri_vertices).transpose(), f=(np.asarray(self.tri_faces)).transpose())


    def save(self, filename):
        faces_list = [] 
        for i, tetrahedron in enumerate(self.tetras):
            for x in get_faces_list_from_tetrahedron(tetrahedron):
                curr_face = list(x)
                if len(curr_face) == 3:
                    faces_list.append(curr_face)

        ply.save_ply(filename, np.asarray(self.vertices).transpose(), f=(np.asarray(faces_list)).transpose())


    ## Sample points along a ray at the faces of Tet32 structure
    def sample_rays_cuda(self, cam_id, ray_d, sdf, cam_ids, in_z, in_sdf, in_ids, offset, nb_samples = 256):
        nb_rays = ray_d.shape[0]
        nb_samples = tet32_march_cuda.tet32_march(nb_rays, nb_samples, cam_id, ray_d, self.sites, sdf, self.summits, sdf, cam_ids, self.offsets_cam, self.cam_tets, in_z) #, in_sdf, in_ids, offset)

        return nb_samples

if __name__=='__main__':
    visual_hull = [-1, -1, -1, 1, 1, 1]
    import sampling as sampler
    sites = sampler.sample_Bbox(visual_hull[0:3], visual_hull[3:6], 16, perturb_f =  visual_hull[3]*0.005) 
    T32 = Tet32(sites)
    T32.save("data/bmvs_man/test.ply")