import torch
import open3d as o3d
import numpy as np
import sys
from timeit import default_timer as timer 
sys.path.append('/root/VortSDF/')
import src.IO.ply as ply
import scipy.spatial
from tqdm import tqdm
from multiprocessing import Process, Value, Array, Manager
import time 

from torch.utils.cpp_extension import load
tet32_march_cuda = load(
    'tet32_march_cuda', ['src/Cuda/tet32_march_cuda.cpp', 'src/Cuda/tet32_march_cuda.cu'], verbose=True)
#help(cvt_march_cuda)

mt_cuda_kernel = load(
    'mt_cuda_kernel', ['src/Geometry/mt_cuda_kernel.cpp', 'src/Geometry/mt_cuda_kernel.cu'], verbose=True)

cvt_grad_cuda = load(
    'cvt_grad_cuda', ['src/Geometry/CVT_gradients.cpp', 'src/Geometry/CVT_gradients.cu'], verbose=True)

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
class Tet32(Process):
    ## sites must be a (n,3) numpy array
    def __init__(self, sites, id = 0, KNN = 24):
        super(Tet32, self).__init__() 
        self.id = id
        self.KNN = KNN
        self.sites = sites
        self.device = torch.device('cuda')
        self.manager = Manager()
        self.d = self.manager.dict()

    def run(self): 
        time.sleep(1) 
        print("I'm the process with id: {}".format(self.id)) 

        ## Build the tetrahedral mesh from the sites
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(self.sites)
        
        self.o3d_mesh, _ = o3d.geometry.TetraMesh.create_from_point_cloud(point_cloud)
        self.o3d_edges = o3d.geometry.LineSet.create_from_tetra_mesh(self.o3d_mesh)
        
        self.vertices = self.o3d_mesh.vertices
        self.edges = np.asarray(self.o3d_edges.lines)
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
        self.neighbors = -np.ones([self.nb_tets, 4], dtype = np.int32)

        for i, tetrahedron in tqdm(enumerate(self.tetras)):
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

        self.sites = np.asarray(self.vertices)
        print("Neighboors computed")
        
        start = timer()
        self.KDtree = scipy.spatial.KDTree(self.sites)
        self.knn_sites = -1 * np.ones((self.sites.shape[0], self.KNN))
        _, idx = self.KDtree.query(self.sites, k=self.KNN+1)
        self.knn_sites[:,:] = np.asarray(idx[:,1:])
        print('KDTreeFlann time:', timer() - start)    

        self.d['summits'] = self.summits
        self.d['edges'] = self.edges
        self.d['neighbors'] = self.neighbors
        self.d['sites'] = self.sites
        self.d['knn_sites'] = self.knn_sites
        #self.d['tetras'] = self.tetras

    def load_cuda(self):
        self.edges = torch.from_numpy(self.d['edges']).int().cuda().contiguous()
        self.summits = torch.from_numpy(self.d['summits']).int().cuda().contiguous()  
        self.neighbors = torch.from_numpy(self.d['neighbors']).int().cuda().contiguous()        
        self.knn_sites = torch.from_numpy(self.d['knn_sites']).int().cuda().contiguous()
        self.sites = torch.from_numpy(self.d['sites']).float().cuda().contiguous()  
        #self.tetras = self.d['tetras']
        print("nb edges: ", self.edges.shape)

    def make_knn(self):
        start = timer()
        self.KDtree = scipy.spatial.KDTree(self.sites.cpu().numpy())
        self.knn_sites = -1 * np.ones((self.sites.shape[0], self.KNN))
        _, idx = self.KDtree.query(self.sites.cpu().numpy(), k=self.KNN+1)
        self.knn_sites[:,:] = np.asarray(idx[:,1:])  
        self.knn_sites = torch.from_numpy(self.knn_sites).int().cuda().contiguous()
        #print('KDTreeFlann time:', timer() - start)    
        
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


    def CVT(self, outside_flag, cam_ids, sdf, fine_features, nb_iter = 1000, sdf_weight = 0.0, lr = 1.0e-4):
        grad_sdf_space = torch.zeros([self.sites.shape[0], 3]).float().cuda().contiguous()
        grad_feat_space = torch.zeros([self.sites.shape[0], 3, 6]).float().cuda().contiguous()
        weights_grad = torch.zeros([3*self.KNN*self.sites.shape[0]]).float().cuda().contiguous()

        grad_sites = torch.zeros(self.sites.shape).cuda()       
        grad_sites = grad_sites.contiguous()

        grad_sites_sdf = torch.zeros(self.sites.shape).cuda()       
        grad_sites_sdf = grad_sites_sdf.contiguous()

        mask_grad = torch.zeros(self.sites.shape).cuda()       
        mask_grad = mask_grad.contiguous()

        delta_sites = torch.zeros(self.sites.shape).float().cuda()
        with torch.no_grad():  
            delta_sites[:] = self.sites[:]
            
        self.sites.requires_grad_(True)
        learning_rate_cvt = lr
        learning_rate_alpha = 1.0e-4
        optimizer_cvt = torch.optim.Adam([self.sites], lr=learning_rate_cvt) 

        for iter_step in tqdm(range(nb_iter)):
            thetas = torch.from_numpy(np.random.rand(self.sites.shape[0])).float().cuda()
            phis = torch.from_numpy(np.random.rand(self.sites.shape[0])).float().cuda()
            gammas = torch.from_numpy(np.random.rand(self.sites.shape[0])).float().cuda()

            ############ Compute spatial SDF gradients
            cvt_grad_cuda.sdf_space_grad(self.sites.shape[0], self.KNN, self.knn_sites, self.sites, sdf, fine_features, weights_grad, grad_sdf_space, grad_feat_space)
            
            ############ Compute approximated CVT gradients at sites
            grad_sites[:] = 0.0
            loss_cvt = cvt_grad_cuda.cvt_grad(self.sites.shape[0], self.KNN, thetas, phis, gammas, self.knn_sites, self.sites, grad_sites)
            grad_sites = grad_sites / self.sites.shape[0]
            
            grad_sites_sdf[:] = 0.0
            cvt_grad_cuda.sdf_grad(self.sites.shape[0], self.KNN, self.knn_sites, self.sites, sdf, grad_sites_sdf)
            
            mask_grad[:,:] = 1.0
            if sdf_weight > 0.0:
                mask_grad[(torch.linalg.norm(grad_sites_sdf, ord=2, axis=-1, keepdims=True) > 0.0).reshape(-1),:] = 1.0e-3

            grad_sites[outside_flag == 1.0, :] = 0.0
            grad_sites[cam_ids, :] = 0.0
            grad_sites_sdf[outside_flag == 1.0] = 0.0
            optimizer_cvt.zero_grad()
            self.sites.grad = grad_sites*mask_grad + sdf_weight*grad_sites_sdf
            optimizer_cvt.step()

            with torch.no_grad():
                self.sites[cam_ids] = delta_sites[cam_ids]
                delta_sites[:] = self.sites[:] - delta_sites[:] 
                if iter_step % 100 == 0:
                    #print(delta_sites.mean())
                    print('iter:{:8>d} loss CVT = {} lr={}'.format(iter_step, loss_cvt, optimizer_cvt.param_groups[0]['lr']))

            with torch.no_grad():
                sdf[:] = sdf[:] + (delta_sites*grad_sdf_space).sum(dim = 1)[:] #  + self.sdf_diff
                for i in range(6):
                    fine_features[:, i] = fine_features[:, i] + (delta_sites*grad_feat_space[:, :, i]).sum(dim = 1)[:] # self.feat_diff[:]

            if iter_step % 100 == 0:
                with torch.no_grad():
                    self.make_knn()

            with torch.no_grad():
                delta_sites[:] = self.sites[:]

                
            alpha = learning_rate_alpha
            progress = iter_step / nb_iter
            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha

            for g in optimizer_cvt.param_groups:
                g['lr'] = learning_rate_cvt * learning_factor

        self.sites = self.sites.detach().cpu().numpy()

    def upsample(self, sdf, feat, visual_hull, res, cam_sites, lr):        
        sites = self.sites.cpu().numpy()
        in_sdf = sdf
        in_feat = feat

        new_sites = []
        new_sdf = []
        new_feat = []
        for _, edge in enumerate(self.o3d_edges.lines):
            edge_length = np.linalg.norm(sites[edge[0]] - sites[edge[1]], ord=2, axis=-1, keepdims=True)
            if sdf[edge[0]]*sdf[edge[1]] <= 0.0: # or min(abs(sdf[edge[0]]), abs(sdf[edge[1]])) < edge_length:
                new_sites.append((sites[edge[0]] + sites[edge[1]])/2.0)
                new_sdf.append((sdf[edge[0]] + sdf[edge[1]])/2.0)
                new_feat.append((feat[edge[0]] + feat[edge[1]])/2.0)

        new_sites = np.stack(new_sites)
        new_sdf = np.stack(new_sdf)
        new_feat = np.stack(new_feat)
        print("nb new sites: ", new_sites.shape)
        self.sites = np.concatenate((sites, new_sites))
        in_sdf = np.concatenate((sdf, new_sdf))
        in_feat = np.concatenate((feat, new_feat))
        new_sites = self.sites

        outside_flag = np.zeros(self.sites.shape[0], np.int32)
        outside_flag[self.sites[:,0] < visual_hull[0] + (visual_hull[3]-visual_hull[0])/(2*res)] = 1
        outside_flag[self.sites[:,1] < visual_hull[1] + (visual_hull[4]-visual_hull[1])/(2*res)] = 1
        outside_flag[self.sites[:,2] < visual_hull[2] + (visual_hull[5]-visual_hull[2])/(2*res)] = 1
        outside_flag[self.sites[:,0] > visual_hull[3] - (visual_hull[3]-visual_hull[0])/(2*res)] = 1
        outside_flag[self.sites[:,1] > visual_hull[4] - (visual_hull[4]-visual_hull[1])/(2*res)] = 1
        outside_flag[self.sites[:,2] > visual_hull[5] - (visual_hull[5]-visual_hull[2])/(2*res)] = 1

        cam_ids = np.stack([np.where((self.sites == cam_sites[i,:]).all(axis = 1))[0] for i in range(cam_sites.shape[0])]).reshape(-1)
        cam_ids = torch.from_numpy(cam_ids).int().cuda()
                
        self.sites = torch.from_numpy(self.sites).float().cuda()
        self.make_knn()
        self.CVT(outside_flag, cam_ids, torch.from_numpy(in_sdf).float().cuda(), torch.from_numpy(in_feat).float().cuda(), 1000, 1.0, lr)

        prev_kdtree = scipy.spatial.KDTree(new_sites)
        self.run()
        new_sites = np.asarray(self.vertices)  

        _, idx = prev_kdtree.query(new_sites, k=1)
        out_sdf = np.zeros(new_sites.shape[0])
        out_sdf[:] = in_sdf[idx[:]]
        
        out_feat = np.zeros([new_sites.shape[0],6])
        out_feat[:] = in_feat[idx[:]]

        return torch.from_numpy(out_sdf).float().cuda(), torch.from_numpy(out_feat).float().cuda()

    def marching_tets(self, sdf, filename, m_iso = 0.0):
        nb_tets = self.nb_tets
        faces = torch.zeros([3*2*nb_tets], dtype=torch.int32).cuda()
        faces = faces.contiguous()
        normals = torch.zeros([3*2*nb_tets], dtype=torch.float32).cuda()
        normals = normals.contiguous()

        nb_edges = self.edges.shape[0]
        vertices = torch.zeros([3*nb_edges], dtype=torch.float32).cuda()
        vertices = vertices.contiguous()

        mt_cuda_kernel.marching_tets(self.sites.shape[0], nb_edges, nb_tets, m_iso, faces, vertices, normals, self.sites, sdf, self.edges, self.summits)

        # Reshape all outputs
        faces_out = faces.reshape((2*nb_tets, 3))
        normals_out = normals.reshape((2*nb_tets, 3))
        vertices_out = vertices.reshape((nb_edges, 3))

        # Remove 0 entries
        nnz = torch.nonzero(torch.sum(vertices_out,1))[:,0]
        nnz_vertices_out = vertices_out[nnz,:]

        indices_v = torch.zeros((vertices_out.shape[0]), dtype=torch.int32).cuda()
        indices_nnz_v = torch.arange(nnz_vertices_out.shape[0], dtype=torch.int32).cuda()
        indices_v[nnz] = indices_nnz_v
        
        nnz_f = torch.nonzero(torch.sum(faces_out,1))[:,0]
        nnz_faces = faces_out[nnz_f,:]
        nnz_normals = normals_out[nnz_f,:]
        nnz_faces[:,:] = indices_v[nnz_faces[:,:].long()]
        nnz_normals[:,:] = indices_v[nnz_normals[:,:].long()]        
        ply.save_ply(filename, np.transpose(nnz_vertices_out.cpu()),  f = np.transpose(nnz_faces.cpu()))  


    def surface_from_sdf(self, values, filename = ""):
        #print(values.shape)
        tri_mesh = self.o3d_mesh.extract_triangle_mesh(o3d.utility.DoubleVector(values.astype(np.float64)),0.0)
        #print(tri_mesh)
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

        ply.save_ply(filename, self.sites.cpu().numpy().transpose(), f=(np.asarray(faces_list)).transpose())


    ## Sample points along a ray at the faces of Tet32 structure
    def sample_rays_cuda(self, cam_id, ray_d, sdf, fine_features, cam_ids, weights, in_z, in_sdf, in_feat, in_ids, offset, nb_samples = 256):
        #ply.save_ply("Exp/bmvs_man/cam.ply", (self.sites[cam_ids[cam_id]]).reshape(1,3).cpu().numpy().transpose())
        nb_rays = ray_d.shape[0]
        nb_samples = tet32_march_cuda.tet32_march(nb_rays, 24, nb_samples, cam_id, ray_d, self.knn_sites, self.sites, sdf, fine_features, self.summits, self.neighbors,
                                                  cam_ids, self.offsets_cam, self.cam_tets, weights, in_z, in_sdf, in_feat, 
                                                  in_ids, offset)
        #for i in range(nb_rays):
        #    print(in_ids[3*nb_samples*i], ", ", in_ids[3*nb_samples*i + 1] , ", ", in_ids[3*nb_samples*i + 2])

        return nb_samples

if __name__=='__main__':
    visual_hull = [-1, -1, -1, 1, 1, 1]
    import sampling as sampler
    sites = sampler.sample_Bbox(visual_hull[0:3], visual_hull[3:6], 16, perturb_f =  visual_hull[3]*0.005) 
    T32 = Tet32(sites)
    T32.save("data/bmvs_man/test.ply")