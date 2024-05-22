import torch
import open3d as o3d
import numpy as np
import sys
from timeit import default_timer as timer 
#sys.path.append('C:/Users/thomas/Documents/Projects/Human-AI/VortSDF/')
import src.IO.ply as ply
import scipy.spatial
from tqdm import tqdm
from multiprocessing import Process, Value, Array, Manager
import time 
from pysdf import SDF

from torch.utils.cpp_extension import load
tet32_march_cuda = load('tet32_march_cuda', ['src/Cuda/tet32_march_cuda.cpp', 'src/Cuda/tet32_march_cuda.cu'], verbose=True)

mt_cuda_kernel = load('mt_cuda_kernel', ['src/Geometry/mt_cuda_kernel.cpp', 'src/Geometry/mt_cuda_kernel.cu'], verbose=True)

cvt_grad_cuda = load('cvt_grad_cuda', ['src/Geometry/CVT_gradients.cpp', 'src/Geometry/CVT_gradients.cu'], verbose=True)

tet_utils = load('tet_utils', ['src/Geometry/tet_utils.cpp', 'src/Geometry/tet_utils.cu'], verbose=True)

backprop_cuda = load('backprop_cuda', ['src/Models/backprop.cpp', 'src/Models/backprop.cu'], verbose=True)

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
        self.lvl = 0
        self.lvl_sites = []
        self.nb_pre_sites = 0

    def run(self, radius = 0.3): 
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
        self.neighbors = -np.ones([self.nb_tets, 4], dtype = np.int32)

        print("nb tets: ", self.nb_tets)
        

        start = timer()
        self.summits = torch.from_numpy(self.summits).cuda().contiguous()
        self.neighbors = torch.from_numpy(self.neighbors).cuda().contiguous()
        #adj = torch.zeros((self.sites.shape[0], 128)).int().cuda().contiguous()
        tet_utils.compute_neighbors(self.nb_tets, self.sites.shape[0], torch.from_numpy(np.asarray(self.tetras)).cuda().contiguous(),
                                    self.summits, self.neighbors)
        self.summits = self.summits.cpu().numpy()
        self.neighbors = self.neighbors.cpu().numpy()
        print('C++ time:', timer() - start)  

        ## 4 values for indices of neighbors
        """self.neighbors = -np.ones([self.nb_tets, 4], dtype = np.int32)
        
        start = timer()
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

        print("Neighboors computed")
        print('Python time:', timer() - start)   

        print(self.neighbors[:10,:])

        input()"""

        if self.lvl > 0:
            new_sites = np.copy(self.sites)

        self.sites = np.asarray(self.vertices)

        start = timer()
        self.KDtree = scipy.spatial.KDTree(self.sites)
        self.knn_sites = -1 * np.ones((self.sites.shape[0], 96))
        _, idx = self.KDtree.query(self.sites, k=32)
        self.knn_sites[:,:32] = np.asarray(idx[:,:])
        print('KDTreeFlann time:', timer() - start)
        """print('radius time:', radius)
        start = timer()
        res_id = self.KDtree.query_ball_point(self.sites, radius, return_sorted = True) # * np.ones((self.sites.shape[0])))   
        print('KDTree Ball point time:', timer() - start) 
        #print(res_id.shape)
        self.offset_bnn = np.zeros((2*self.sites.shape[0]))
        tot_points = 0
        max_length = 0
        self.bnn_sites = []
        for i in range(self.sites.shape[0]):
            self.offset_bnn[2*i] = tot_points
            #self.offset_bnn[2*i+1] = len(res_id[i])
            #tot_points = tot_points + len(res_id[i])
            max_length = max(max_length, len(res_id[i]))
            self.offset_bnn[2*i+1] = min(400, len(res_id[i]))
            tot_points = tot_points + min(400, len(res_id[i]))
            if len(res_id[i]) < 400:
                self.bnn_sites.append(np.array(res_id[i]))
            else:
                self.bnn_sites.append(np.array(res_id[i])[:300])
                self.bnn_sites.append(np.array(res_id[i])[-100:])
        self.bnn_sites = np.concatenate(self.bnn_sites) #[np.array(res_id[id]) for id in range(self.sites.shape[0])])
        #self.bnn_sites = np.concatenate([np.array(res_id[id]) for id in range(self.sites.shape[0])])
        print(max_length)
        print(self.bnn_sites.shape)"""
        #input()

        if self.lvl > 0:
            for lvl_curr in range(self.lvl+1):
                _, idx = self.KDtree.query(new_sites[self.lvl_sites[lvl_curr][:]], k=1)
                self.lvl_sites[lvl_curr][:] = idx[:]

        self.d['summits'] = self.summits
        self.d['edges'] = self.edges
        self.d['neighbors'] = self.neighbors
        self.d['sites'] = self.sites
        self.d['knn_sites'] = self.knn_sites
        #self.d['bnn_sites'] = self.bnn_sites.reshape(-1)
        #self.d['offset_bnn'] = self.offset_bnn
        #self.d['tetras'] = self.tetras


    def load_cuda(self):
        self.edges = torch.from_numpy(self.d['edges']).int().cuda().contiguous()
        self.summits = torch.from_numpy(self.d['summits']).int().cuda().contiguous()  
        self.neighbors = torch.from_numpy(self.d['neighbors']).int().cuda().contiguous()        
        self.knn_sites = torch.from_numpy(self.d['knn_sites']).int().cuda().contiguous()   
        #self.bnn_sites = torch.from_numpy(self.d['bnn_sites']).int().cuda().contiguous()
        #self.offset_bnn = torch.from_numpy(self.d['offset_bnn']).int().cuda().contiguous()
        self.sites = torch.from_numpy(self.d['sites']).float().cuda().contiguous()  
        #self.tetras = self.d['tetras']
        print("nb edges: ", self.edges.shape)

    def make_knn(self):
        start = timer()
        self.KDtree = scipy.spatial.KDTree(self.sites.cpu().numpy())
        self.knn_sites = -1 * np.ones((self.sites.shape[0], self.KNN))
        _, idx = self.KDtree.query(self.sites.cpu().numpy(), k=self.KNN+1)
        self.knn_sites[:,:self.KNN] = np.asarray(idx[:,1:])  
        self.knn_sites = torch.from_numpy(self.knn_sites).int().cuda().contiguous()
        #print('KDTreeFlann time:', timer() - start)    

    def make_tet(self, lvl = 0):
        sites_lvl = self.sites[self.lvl_sites[lvl][:]].cpu().numpy()

        ## Build the tetrahedral mesh from the sites
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(sites_lvl)
        
        o3d_mesh, _ = o3d.geometry.TetraMesh.create_from_point_cloud(point_cloud)
        
        vertices = o3d_mesh.vertices
        tetras = o3d_mesh.tetras

        
        ## 4 values for indices of summits
        nb_tets = np.asarray(tetras).shape[0]
        summits = np.zeros([nb_tets, 4], dtype = np.int32)
        summits[:,] = np.asarray(tetras)[:,:]
        neighbors = -np.ones([nb_tets, 4], dtype = np.int32)

        print("nb tets: ", self.nb_tets)

        start = timer()
        summits = torch.from_numpy(summits).cuda().contiguous()
        neighbors = torch.from_numpy(neighbors).cuda().contiguous()
        tet_utils.compute_neighbors(nb_tets, sites_lvl.shape[0], torch.from_numpy(np.asarray(tetras)).cuda().contiguous(),
                                    summits, neighbors)
        summits = summits.cpu().numpy()
        neighbors = neighbors.cpu().numpy()
        print('C++ time:', timer() - start)  

        new_sites = np.asarray(vertices)

        # re-index summits
        _, idx = self.KDtree.query(new_sites, k=1)
        summits[:,:] = idx[summits[:,:]]

        self.tet_lvl[lvl] = summits
        self.neighbors_lvl[lvl] = neighbors


        
    def make_multilvl_knn(self):
        start = timer()

        self.knn_sites = -1 * np.ones((self.sites.shape[0], 96))
        
        self.KDtree = scipy.spatial.KDTree(self.sites.cpu().numpy())
        _, idx = self.KDtree.query(self.sites.cpu().numpy(), k=32)
        self.knn_sites[:,:32] = np.asarray(idx[:,:])  

        print("self.lvl => ", self.lvl)
        
        curr_it = 1
        start_lvl = max(0, self.lvl-2)
        for lvl_curr in range(start_lvl,self.lvl):
            KDtree = scipy.spatial.KDTree(self.sites[self.lvl_sites[self.lvl-curr_it][:]].cpu().numpy())
            _, idx = KDtree.query(self.sites.cpu().numpy(), k=32)
            self.knn_sites[:,32*curr_it:32*(curr_it+1)] = np.asarray(self.lvl_sites[self.lvl-curr_it][idx[:,:]])  
            curr_it = curr_it + 1

        self.knn_sites = torch.from_numpy(self.knn_sites).int().cuda().contiguous()
        #print('KDTreeFlann time:', timer() - start)    
        
    ## Make adjacencies for cameras
    def make_adjacencies(self, cam_ids):
        self.offsets_cam = torch.zeros(cam_ids.shape[0]).int().cuda()

        tot_vol = tet_utils.count_cam_neighbors(self.nb_tets, cam_ids.shape[0], 
                                      torch.from_numpy(np.asarray(self.tetras)).cuda().contiguous(), 
                                      torch.from_numpy(cam_ids).int().cuda(), self.offsets_cam)
    
        self.cam_tets = torch.zeros(tot_vol).int().cuda()

        tet_utils.compute_cam_neighbors(self.nb_tets, cam_ids.shape[0], 
                                        torch.from_numpy(np.asarray(self.tetras)).cuda().contiguous(), 
                                        torch.from_numpy(cam_ids).int().cuda(), 
                                        self.cam_tets, self.offsets_cam)
        
        """tmp_offsets = torch.zeros(cam_ids.shape[0]).int().cuda()
        tmp_offsets[:] =  self.offsets_cam[:]
 
        print(self.cam_tets[0])
        to_print = self.cam_tets[1:self.cam_tets[0]].sort().values
        print(to_print[:10])
        print(self.offsets_cam[:])
        input()

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

        print(abs(tmp_offsets - self.offsets_cam).sum())
        input()

        print(self.offsets_cam[:])
        print(self.cam_tets[:10])
        input()"""

    def move_sites(self,  outside_flag, cam_ids, sdf, fine_features):
        delta = sdf - self.sdf_init

        delta_sites = torch.zeros([self.sites.shape[0], 3]).float().cuda().contiguous()
        grad_sdf_space = torch.zeros([self.sites.shape[0], 3]).float().cuda().contiguous()
        grad_feat_space = torch.zeros([self.sites.shape[0], 3, fine_features.shape[1]]).float().cuda().contiguous()
        weights_grad = torch.zeros([3*self.KNN*self.sites.shape[0]]).float().cuda().contiguous()
        activated = torch.ones([self.sites.shape[0]]).int().cuda().contiguous()

        ############ Compute spatial SDF gradients
        grad_sdf_space[:] = 0.0
        grad_feat_space[:] = 0.0
        weights_grad[:] = 0.0
        cvt_grad_cuda.knn_sdf_space_grad(self.sites.shape[0], self.KNN, self.knn_sites, self.sites, activated, sdf, fine_features, grad_sdf_space, grad_feat_space, weights_grad)

        for i in range(3):
            delta_sites[:,i] = delta[:]*grad_sdf_space[:,i]
        delta_sites[cam_ids, :] = 0.0
        delta_sites[outside_flag == 1.0] = 0.0

        self.sites = self.sites + delta_sites

        for i in range(fine_features.shape[1]):
            fine_features[:, i] = fine_features[:, i] + (delta_sites*grad_feat_space[:, :, i]).sum(dim = 1)[:] # self.feat_diff[:]




    def CVT(self, outside_flag, cam_ids, sdf, fine_features, nb_iter = 1000, radius = 0.1, sdf_weight = 0.0, lr = 1.0e-4):
        self.make_knn()
        

        """grad_sdf_space = torch.zeros([self.sites.shape[0], 3]).float().cuda().contiguous()
        grad_feat_space = torch.zeros([self.sites.shape[0], 3, fine_features.shape[1]]).float().cuda().contiguous()
        weights_grad = torch.zeros([3*self.KNN*self.sites.shape[0]]).float().cuda().contiguous()
        grad_mean_curve = torch.zeros([self.sites.shape[0]]).float().cuda().contiguous()
        grad_eik = torch.zeros([self.sites.shape[0]]).float().cuda().contiguous()
        grad_norm_smooth = torch.zeros([self.sites.shape[0]]).float().cuda().contiguous()
        eik_loss = torch.zeros([self.sites.shape[0]]).float().cuda().contiguous()
        activated = torch.ones([self.sites.shape[0]]).int().cuda().contiguous()"""

        in_sdf = torch.zeros([self.sites.shape[0]]).float().cuda().contiguous()        
        out_sdf = torch.zeros([self.sites.shape[0]]).float().cuda().contiguous()
        in_feat = torch.zeros([self.sites.shape[0], fine_features.shape[1]]).float().cuda().contiguous()
        out_feat = torch.zeros([self.sites.shape[0], fine_features.shape[1]]).float().cuda().contiguous()

        grad_sites = torch.zeros(self.sites.shape).cuda()       
        grad_sites = grad_sites.contiguous()

        grad_sites_sdf = torch.zeros(self.sites.shape).cuda()       
        grad_sites_sdf = grad_sites_sdf.contiguous()

        mask_grad = torch.zeros(self.sites.shape).cuda()       
        mask_grad = mask_grad.contiguous()

        delta_sites = torch.zeros(self.sites.shape).float().cuda()
        init_sites = torch.zeros(self.sites.shape).float().cuda()
        with torch.no_grad():  
            delta_sites[:] = self.sites[:]
            init_sites[:] = self.sites[:] 
            in_sdf[:] = sdf[:]
            in_feat[:] = fine_features[:]
            
        init_sites = init_sites.cpu().numpy()
        prev_kdtree = scipy.spatial.KDTree(init_sites)
        knn_sites = -1 * np.ones((self.sites.shape[0], 32))
        _, idx = prev_kdtree.query(self.sites.cpu().numpy(), k=32)
        knn_sites[:,:32] = np.asarray(idx[:,:])
            
        self.sites.requires_grad_(True)
        learning_rate_cvt = lr
        learning_rate_alpha = 1.0e-4
        optimizer_cvt = torch.optim.Adam([self.sites], lr=learning_rate_cvt) 

        for iter_step in tqdm(range(nb_iter)):
            thetas = torch.from_numpy(np.random.rand(self.sites.shape[0])).float().cuda()
            phis = torch.from_numpy(np.random.rand(self.sites.shape[0])).float().cuda()
            gammas = torch.from_numpy(np.random.rand(self.sites.shape[0])).float().cuda()

            ############ Compute spatial SDF gradients
            """grad_sdf_space[:] = 0.0
            grad_feat_space[:] = 0.0
            weights_grad[:] = 0.0
            cvt_grad_cuda.knn_sdf_space_grad(self.sites.shape[0], self.KNN, self.knn_sites, self.sites, activated, sdf, fine_features, grad_sdf_space, grad_feat_space, weights_grad)
            """

            """grad_sdf_space[:] = 0.0
            grad_feat_space[:] = 0.0
            grad_mean_curve[:] = 0.0
            weights_grad[:] = 0.0
            grad_eik[:] = 0.0
            grad_norm_smooth[:] = 0.0
            eik_loss[:] = 0.0
            activated[:] = 1
            cvt_grad_cuda.eikonal_grad(self.nb_tets, self.sites.shape[0], self.summits, self.sites, activated, sdf, sdf, fine_features, 
                                        grad_eik, grad_norm_smooth, grad_sdf_space, grad_feat_space, weights_grad, eik_loss)
            """

            #cvt_grad_cuda.sdf_space_grad(self.nb_tets, self.sites.shape[0], self.summits, self.sites, sdf, fine_features, grad_sdf_space, grad_feat_space, weights_grad)
  
            ############ Compute approximated CVT gradients at sites
            grad_sites[:] = 0.0
            loss_cvt = cvt_grad_cuda.cvt_grad(self.sites.shape[0], self.KNN, thetas, phis, gammas, self.knn_sites, self.sites, torch.from_numpy(outside_flag).int().cuda().contiguous(), sdf, grad_sites)
            grad_sites = grad_sites / self.sites.shape[0]
            
            grad_sites_sdf[:] = 0.0
            #cvt_grad_cuda.sdf_grad(self.sites.shape[0], self.KNN, self.knn_sites, self.sites, sdf, grad_sites_sdf)
            
            mask_grad[:,:] = 1.0
            #if sdf_weight > 0.0:
            #    mask_grad[(torch.linalg.norm(grad_sites_sdf, ord=2, axis=-1, keepdims=True) > 0.0).reshape(-1),:] = 1.0e-3

            grad_sites[outside_flag == 1.0, :] = 0.0
            grad_sites[cam_ids, :] = 0.0
            grad_sites_sdf[outside_flag == 1.0] = 0.0
            grad_sites_sdf[cam_ids, :] = 0.0

            #grad_sites[:self.nb_pre_sites,:] = 0.0
            #grad_sites_sdf[:self.nb_pre_sites,:] = 0.0
            optimizer_cvt.zero_grad()
            self.sites.grad = grad_sites*mask_grad + sdf_weight*grad_sites_sdf
            optimizer_cvt.step()

            with torch.no_grad():
                self.sites[cam_ids] = delta_sites[cam_ids]
                delta_sites[:] = self.sites[:] - delta_sites[:] 
                if iter_step % 100 == 0:
                    #print(delta_sites.mean())
                    print('iter:{:8>d} loss CVT = {}, grad = {}, lr={}'.format(iter_step, loss_cvt,abs(grad_sites[abs(grad_sites) > 0.0]).mean(), optimizer_cvt.param_groups[0]['lr']))

            with torch.no_grad():
                out_sdf[:] = 0
                backprop_cuda.knn_interpolate(self.sites.shape[0], 32, radius/2.0, 1, torch.from_numpy(init_sites).float().cuda().contiguous(), 
                                        self.sites, in_sdf, 
                                        torch.from_numpy(knn_sites).int().cuda().contiguous(), out_sdf)
                sdf[:] = out_sdf[:]
                
                out_feat[:] = 0
                backprop_cuda.knn_interpolate(self.sites.shape[0], 32, radius/2.0, fine_features.shape[1], torch.from_numpy(init_sites).float().cuda().contiguous(), 
                                                self.sites, in_feat, 
                                                torch.from_numpy(knn_sites).int().cuda().contiguous(), out_feat)
                fine_features[:] = out_feat[:]
                

                """sdf[:] = sdf[:] + (delta_sites*grad_sdf_space).sum(dim = 1)[:] #  + self.sdf_diff
                grad_feat_space[abs(grad_feat_space) > 10.0*abs(grad_feat_space.mean())] = 0.0
                print(grad_feat_space.mean())
                print(grad_feat_space.min())
                print(grad_feat_space.max())
                print(delta_sites.mean())
                print(delta_sites.min())
                print(delta_sites.max())
                for i in range(fine_features.shape[1]):
                    fine_features[:, i] = fine_features[:, i] + (delta_sites*grad_feat_space[:, :, i]).sum(dim = 1)[:] # self.feat_diff[:]"""

            if iter_step % 100 == 0:
                with torch.no_grad():
                    self.make_knn()
                    _, idx = prev_kdtree.query(self.sites.detach().cpu().numpy(), k=32)
                knn_sites[:,:32] = np.asarray(idx[:,:])

            with torch.no_grad():
                delta_sites[:] = self.sites[:]

                
            alpha = learning_rate_alpha
            progress = iter_step / nb_iter
            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha

            for g in optimizer_cvt.param_groups:
                g['lr'] = learning_rate_cvt * learning_factor

        self.sites = self.sites.detach().cpu().numpy()

        return sdf.cpu().numpy(), fine_features.cpu().numpy()

    def upsample(self, sdf, feat, visual_hull, res, cam_sites, lr, flag = True, radius = 0.3):    
        self.nb_pre_sites = self.sites.shape[0]
        if True: #self.nb_pre_sites < 1.0e6:  
            self.lvl_sites.append(np.arange(self.sites.shape[0]))

        ## Smooth current mesh and build sdf        
        tri_mesh = self.o3d_mesh.extract_triangle_mesh(o3d.utility.DoubleVector(sdf.astype(np.float64)),0.0)
        #tri_mesh.filter_smooth_laplacian(number_of_iterations=3)

        self.tri_vertices = np.asarray(tri_mesh.vertices)
        self.tri_faces = np.asarray(tri_mesh.triangles)
        f = SDF(np.asarray(self.tri_vertices), np.asarray(self.tri_faces))
        true_sdf = -f(self.sites.cpu().numpy())

        if True: #self.nb_pre_sites < 1.0e6:            
            nb_new_sites = tet_utils.upsample_counter(self.edges.shape[0], radius, self.edges, self.sites, torch.from_numpy(true_sdf).float().cuda())
                    
            #nb_new_sites = tet_utils.upsample_counter_tet(self.nb_tets, radius, self.summits, self.sites, torch.from_numpy(sdf).float().cuda())
                    
            new_sites = torch.zeros([nb_new_sites,3]).float().cuda().contiguous()
            new_sdf = torch.zeros([nb_new_sites]).float().cuda().contiguous()
            new_feat = torch.zeros([nb_new_sites,feat.shape[1]]).float().cuda().contiguous()

            tet_utils.upsample(self.edges.shape[0], radius, self.edges, self.sites, torch.from_numpy(sdf).float().cuda().contiguous(), torch.from_numpy(true_sdf).float().cuda().contiguous(), torch.from_numpy(feat).float().cuda().contiguous(),
                            new_sites, new_sdf, new_feat)
            
            #tet_utils.upsample_tet(self.nb_tets, radius, self.summits, self.sites, torch.from_numpy(sdf).float().cuda().contiguous(), torch.from_numpy(feat).float().cuda().contiguous(),
            #                   new_sites, new_sdf, new_feat)
            
            new_sites = new_sites.cpu().numpy()
            new_sdf = new_sdf.cpu().numpy()
            new_feat = new_feat.cpu().numpy()

            sites = self.sites.cpu().numpy()

            """in_sdf = sdf
            in_feat = feat

            new_sites = []
            new_sdf = []
            new_feat = []
            for _, edge in enumerate(self.o3d_edges.lines):
                edge_length = np.linalg.norm(sites[edge[0]] - sites[edge[1]], ord=2, axis=-1, keepdims=True)
                if sdf[edge[0]]*sdf[edge[1]] <= 0.0 or min(abs(sdf[edge[0]]), abs(sdf[edge[1]])) < edge_length:
                    new_sites.append((sites[edge[0]] + sites[edge[1]])/2.0)
                    new_sdf.append((sdf[edge[0]] + sdf[edge[1]])/2.0)
                    new_feat.append((feat[edge[0]] + feat[edge[1]])/2.0)

            new_sites = np.stack(new_sites)
            new_sdf = np.stack(new_sdf)
            new_feat = np.stack(new_feat)"""

            print("nb new sites: ", new_sites.shape)
            self.sites = np.concatenate((sites, new_sites))
            in_sdf = np.concatenate((sdf, new_sdf))
            in_feat = np.concatenate((feat, new_feat))
        else:
            in_sdf = np.copy(sdf)
            in_feat = np.copy(feat)
            self.sites = self.sites.cpu().numpy()

        true_sdf = -f(self.sites)
        new_sites = np.copy(self.sites)

        outside_flag = np.zeros(self.sites.shape[0], np.int32)
        outside_flag[self.sites[:,0] < visual_hull[0] + (visual_hull[3]-visual_hull[0])/(2*res)] = 1
        outside_flag[self.sites[:,1] < visual_hull[1] + (visual_hull[4]-visual_hull[1])/(2*res)] = 1
        outside_flag[self.sites[:,2] < visual_hull[2] + (visual_hull[5]-visual_hull[2])/(2*res)] = 1
        outside_flag[self.sites[:,0] > visual_hull[3] - (visual_hull[3]-visual_hull[0])/(2*res)] = 1
        outside_flag[self.sites[:,1] > visual_hull[4] - (visual_hull[4]-visual_hull[1])/(2*res)] = 1
        outside_flag[self.sites[:,2] > visual_hull[5] - (visual_hull[5]-visual_hull[2])/(2*res)] = 1
        outside_flag[abs(true_sdf) > 2*radius] = 1

        cam_ids = np.stack([np.where((self.sites == cam_sites[i,:]).all(axis = 1))[0] for i in range(cam_sites.shape[0])]).reshape(-1)
        cam_ids = torch.from_numpy(cam_ids).int().cuda()
                
        self.sites = torch.from_numpy(self.sites).float().cuda()
        self.CVT(outside_flag, cam_ids.long(), torch.from_numpy(in_sdf).float().cuda(), torch.from_numpy(in_feat).float().cuda(), 300, radius, 0.1, lr)
        #in_sdf, in_feat

        #ply.save_ply("Exp/bmvs_man/testprevlvlv.ply", (self.sites[self.lvl_sites[0][:]]).transpose())
        prev_kdtree = scipy.spatial.KDTree(new_sites)

        self.run(radius)

        #new_sites = np.asarray(self.vertices)  
        self.KDtree = scipy.spatial.KDTree(self.sites)
        
        if True: #self.nb_pre_sites < 1.0e6:  
            self.lvl = self.lvl + 1
            
        for lvl_curr in range(self.lvl):
            _, idx = self.KDtree.query(new_sites[self.lvl_sites[lvl_curr][:]], k=1)
            self.lvl_sites[lvl_curr][:] = idx[:]

        #_, idx = prev_kdtree.query(self.sites, k=1)
        #out_sdf = np.zeros(self.sites.shape[0])
        
        
        knn_sites = -1 * np.ones((self.sites.shape[0], 32))
        _, idx = prev_kdtree.query(self.sites, k=32)
        knn_sites[:,:32] = np.asarray(idx[:,:])
        out_sdf = torch.zeros(self.sites.shape[0]).float().cuda().contiguous()
        backprop_cuda.knn_interpolate(self.sites.shape[0], 32, radius/4.0, 1, torch.from_numpy(new_sites).float().cuda().contiguous(), 
                                        torch.from_numpy(self.sites).float().cuda().contiguous(), torch.from_numpy(in_sdf).float().cuda().contiguous(), 
                                        torch.from_numpy(knn_sites).int().cuda().contiguous(), out_sdf)
        out_sdf = out_sdf.cpu().numpy()
        
        #out_sdf = -f(self.sites)
        mask_background = -f(self.sites) > 2*radius

        #if flag:
        #    out_sdf[mask_background[:] == True] = radius

        if flag:
            out_sdf = -f(self.sites)
            #out_sdf[abs(out_sdf) > 0] = -f(self.sites)[abs(out_sdf) > 0]
            #out_sdf[out_sdf < 0] = -radius/2.0

        """lap_sdf = -f(self.sites)
        out_sdf[abs(out_sdf[:]) > radius] = lap_sdf[abs(out_sdf[:]) > radius]"""
        print("out_sdf => ", out_sdf.sum())
        print("out_sdf => ", out_sdf.min())
        print("out_sdf => ", out_sdf.max())
        
        #out_feat = np.zeros([self.sites.shape[0],feat.shape[1]])
        #out_feat[:] = in_feat[idx[:]]

        
        out_feat = torch.zeros(self.sites.shape[0],feat.shape[1]).float().cuda().contiguous()
        backprop_cuda.knn_interpolate(self.sites.shape[0], 32, radius/4.0, feat.shape[1], torch.from_numpy(new_sites).float().cuda().contiguous(), 
                                        torch.from_numpy(self.sites).float().cuda().contiguous(), torch.from_numpy(in_feat).float().cuda().contiguous(), 
                                        torch.from_numpy(knn_sites).int().cuda().contiguous(), out_feat)
        out_feat = out_feat.cpu().numpy()
                
        self.sdf_init = torch.from_numpy(out_sdf).float().cuda()

        return torch.from_numpy(out_sdf).float().cuda(), torch.from_numpy(out_feat).float().cuda(), torch.from_numpy(mask_background).float().cuda()

    def upsample2(self, sdf, feat, visual_hull, res, cam_sites, lr, flag = True, radius = 0.3):    
        self.lvl_sites.append(np.arange(self.sites.shape[0]))
        self.nb_pre_sites = self.sites.shape[0]
        
        ## Smooth current mesh and build sdf        
        tri_mesh = self.o3d_mesh.extract_triangle_mesh(o3d.utility.DoubleVector(sdf.astype(np.float64)),0.0)
        #tri_mesh.filter_smooth_laplacian(number_of_iterations=3)

        self.tri_vertices = np.asarray(tri_mesh.vertices)
        self.tri_faces = np.asarray(tri_mesh.triangles)
        f = SDF(np.asarray(self.tri_vertices), np.asarray(self.tri_faces))
        true_sdf = -f(self.sites.cpu().numpy())

        if self.nb_pre_sites < 1.0e6:
            nb_new_sites = tet_utils.upsample_counter(self.edges.shape[0], radius, self.edges, self.sites, torch.from_numpy(true_sdf).float().cuda())
                    
            new_sites = torch.zeros([nb_new_sites,3]).float().cuda().contiguous()
            new_sdf = torch.zeros([nb_new_sites]).float().cuda().contiguous()
            new_feat = torch.zeros([nb_new_sites,feat.shape[1]]).float().cuda().contiguous()

            tet_utils.upsample(self.edges.shape[0], radius, self.edges, self.sites, torch.from_numpy(sdf).float().cuda().contiguous(), torch.from_numpy(true_sdf).float().cuda().contiguous(), torch.from_numpy(feat).float().cuda().contiguous(),
                            new_sites, new_sdf, new_feat)
            
            new_sites = new_sites.cpu().numpy()
            new_sdf = new_sdf.cpu().numpy()
            new_feat = new_feat.cpu().numpy()
            sites = self.sites.cpu().numpy()

            print("nb new sites: ", new_sites.shape)
            self.sites = np.concatenate((sites, new_sites))
            in_sdf = np.concatenate((sdf, new_sdf))
            true_in_sdf = np.concatenate((true_sdf, np.zeros(new_sdf.shape[0], np.float32)))
            in_feat = np.concatenate((feat, new_feat))
        else:
            in_sdf = np.copy(sdf)
            in_feat = np.copy(feat)
            self.sites = self.sites.cpu().numpy()

        new_sites = self.sites

        outside_flag = np.zeros(self.sites.shape[0], np.int32)
        outside_flag[abs(true_in_sdf) > 2*radius] = 1

        cam_ids = np.stack([np.where((self.sites == cam_sites[i,:]).all(axis = 1))[0] for i in range(cam_sites.shape[0])]).reshape(-1)
        cam_ids = torch.from_numpy(cam_ids).int().cuda()
        outside_flag[cam_ids.cpu().numpy()] = 1
                
        self.sites = torch.from_numpy(self.sites).float().cuda()
        in_sdf, in_feat = self.CVT(outside_flag, cam_ids.long(), torch.from_numpy(in_sdf).float().cuda(), torch.from_numpy(in_feat).float().cuda(), 300, radius, 0.1, lr)

        print("nb sites: ", self.sites.shape)
        outside_flag[:] = 0
        outside_flag[-f(self.sites) < 2*radius] = 1
        outside_flag[cam_ids.cpu().numpy()] = 1
        self.sites = self.sites[outside_flag == 1]
        in_feat = in_feat[outside_flag == 1]
        in_sdf = in_sdf[outside_flag == 1]
        print("nb sites: ", self.sites.shape)

        #ply.save_ply("Exp/bmvs_man/testprevlvlv.ply", (self.sites[self.lvl_sites[0][:]]).transpose())
        prev_kdtree = scipy.spatial.KDTree(new_sites)

        self.run(radius)

        self.KDtree = scipy.spatial.KDTree(self.sites)
        
        for lvl_curr in range(self.lvl+1):
            _, idx = self.KDtree.query(new_sites[self.lvl_sites[lvl_curr][:]], k=1)
            self.lvl_sites[lvl_curr][:] = idx[:]

        knn_sites = -1 * np.ones((self.sites.shape[0], 32))
        _, idx = prev_kdtree.query(self.sites, k=32)
        knn_sites[:,:32] = np.asarray(idx[:,:])
        out_sdf = torch.zeros(self.sites.shape[0]).float().cuda().contiguous()
        backprop_cuda.knn_interpolate(self.sites.shape[0], 32, radius/8.0, 1, torch.from_numpy(new_sites).float().cuda().contiguous(), 
                                        torch.from_numpy(self.sites).float().cuda().contiguous(), torch.from_numpy(in_sdf).float().cuda().contiguous(), 
                                        torch.from_numpy(knn_sites).int().cuda().contiguous(), out_sdf)
        out_sdf = out_sdf.cpu().numpy()
        
        mask_background = -f(self.sites) > 2*radius
        
        out_feat = torch.zeros(self.sites.shape[0],feat.shape[1]).float().cuda().contiguous()
        backprop_cuda.knn_interpolate(self.sites.shape[0], 32, radius/8.0, feat.shape[1], torch.from_numpy(new_sites).float().cuda().contiguous(), 
                                        torch.from_numpy(self.sites).float().cuda().contiguous(), torch.from_numpy(in_feat).float().cuda().contiguous(), 
                                        torch.from_numpy(knn_sites).int().cuda().contiguous(), out_feat)
        out_feat = out_feat.cpu().numpy()
                
        self.lvl = self.lvl + 1

        self.sdf_init = torch.from_numpy(out_sdf).float().cuda()

        return torch.from_numpy(out_sdf).float().cuda(), torch.from_numpy(out_feat).float().cuda(), torch.from_numpy(mask_background).float().cuda()

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


    def surface_from_sdf(self, values, filename = "", translate = None, scale = None):
        #print(values.shape)
        tri_mesh = self.o3d_mesh.extract_triangle_mesh(o3d.utility.DoubleVector(values.astype(np.float64)),0.0)

        """filter_smooth_laplacian(self, number_of_iterations=1, lambda_filter=0.5, filter_scope=<FilterScope.All: 0>)"""
        
        if translate is not None:
            translate = np.ascontiguousarray(translate, dtype=np.float32)
        
        #print(tri_mesh)
        self.tri_vertices = np.asarray(tri_mesh.vertices)

        if translate is not None:
            self.tri_vertices = np.dot(self.tri_vertices, scale) + translate

        self.tri_faces = np.asarray(tri_mesh.triangles)
        if not filename == "":
            ply.save_ply(filename, np.asarray(self.tri_vertices).transpose(), f=(np.asarray(self.tri_faces)).transpose())

    def smooth_sdf(self, sdf):
        ## Smooth current mesh and build sdf        
        tri_mesh = self.o3d_mesh.extract_triangle_mesh(o3d.utility.DoubleVector(sdf.astype(np.float64)),0.0)
        tri_mesh.filter_smooth_laplacian(number_of_iterations=3)

        self.tri_vertices = np.asarray(tri_mesh.vertices)
        self.tri_faces = np.asarray(tri_mesh.triangles)
        f = SDF(np.asarray(self.tri_vertices), np.asarray(self.tri_faces))

        out_sdf = -f(self.sites.cpu().numpy())
        return torch.from_numpy(out_sdf).float().cuda()


    def clipped_cvt(self, sdf, feat, outside_flag, cam_ids, lr, filename = "", translate = None, scale = None):
        #self.sites = torch.from_numpy(self.sites).float().cuda()
        in_sdf, in_feat = self.CVT(outside_flag, cam_ids, sdf, feat, 300, 1.0, lr)

        faces_list = [] 
        vtx_list = [] 
        offset = 0
        voro = scipy.spatial.Voronoi(self.sites)
        for i in tqdm(range(self.sites.shape[0])):            
            if voro.point_region[i] == -1 or in_sdf[i] > 0:
                continue
            infinite_reg = sum([x == -1 for x in voro.regions[voro.point_region[i]]])

            if not infinite_reg and len(voro.regions[voro.point_region[i]]) > 3:
                points = voro.vertices[voro.regions[voro.point_region[i]]]
                vtx_list.append(points)
                if len(voro.regions[voro.point_region[i]]) == 3:
                    print(points)
                hull = scipy.spatial.ConvexHull(points)
                # 12 = 2 * 6 faces are the simplices (2 simplices per square face)
                for s in hull.simplices:
                    #s = np.append(s, s[0]) 
                    faces_list.append(s.reshape(1,3) + offset)
                #draw.polygon(list(map(tuple, points[hull.vertices])), outline=(0, 0, 0))
                offset = offset + points.shape[0]

        vtx_list = np.concatenate(vtx_list)
        faces_list = np.concatenate(faces_list)
        print(vtx_list.shape)
        print(faces_list.shape)
        ply.save_ply(filename, vtx_list.transpose(), f=faces_list.transpose())

    def save(self, filename):
        faces_list = [] 
        for i, tetrahedron in enumerate(self.tetras):
            for x in get_faces_list_from_tetrahedron(tetrahedron):
                curr_face = list(x)
                if len(curr_face) == 3:
                    faces_list.append(curr_face)

        ply.save_ply(filename, self.sites.cpu().numpy().transpose(), f=(np.asarray(faces_list)).transpose())

    def save_multi_lvl(self, filename):
        for lvl_curr in range(self.lvl):
            ply.save_ply(filename+"{:0>2d}.ply".format(lvl_curr), (self.sites[self.lvl_sites[lvl_curr][:]]).cpu().numpy().transpose())
        ply.save_ply(filename+".ply", self.sites.cpu().numpy().transpose())


    ## Sample points along a ray at the faces of Tet32 structure
    def sample_rays_cuda(self, inv_s, cam_id, ray_d, sdf, cam_ids, weights, in_z, in_sdf, in_ids, offset, activated, nb_samples = 256):
        #ply.save_ply("Exp/bmvs_man/cam.ply", (self.sites[cam_ids[cam_id]]).reshape(1,3).cpu().numpy().transpose())
        nb_rays = ray_d.shape[0]
        nb_samples = tet32_march_cuda.tet32_march(inv_s, nb_rays, nb_samples, cam_id, ray_d, self.sites, sdf, self.summits, self.neighbors,
                                                  cam_ids, self.offsets_cam, self.cam_tets, weights, in_z, in_sdf, 
                                                  in_ids, activated, offset)
        #for i in range(nb_rays):
        #    print(in_ids[3*nb_samples*i], ", ", in_ids[3*nb_samples*i + 1] , ", ", in_ids[3*nb_samples*i + 2])

        return nb_samples
    
    def make_clipped_CVT(self, sdf, sigma, gradients, bbox, filename = "", translate = None, scale = None):
        print("Start clipping CVT")

        cpy_sites = self.sites[abs(sdf) < sigma]
        down_gradients = gradients[abs(sdf) < sigma]
        down_sdf = sdf[abs(sdf) < sigma]

        print(cpy_sites.shape)

        import ctypes
        
        self.params = torch.from_numpy(np.asarray(bbox).astype(np.float32)).cuda()
        
        translate = np.ascontiguousarray(translate, dtype=np.float32)
        translate_c = translate.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        #libnameCGAL = r"C:/Users/Diego Thomas/Documents/Project/inria-cvt/Python/CVT.dll"
        libnameCGAL = "C:/Users/thomas/Documents/Projects/Human-AI/inria-cvt/Python/CVT.dll"
        cvt_libCGAL = ctypes.CDLL(libnameCGAL)

        #libname = r"C:/Users/Diego Thomas/Documents/Project/inria-cvt/Python/DiscreteCVT.dll"
        libname = "C:/Users/thomas/Documents/Projects/Human-AI/inria-cvt/Python/DiscreteCVT.dll"
        cvt_lib = ctypes.CDLL(libname)
        
        cvt_lib.Get_nb_tets32.restype = ctypes.c_int32
        cvt_lib.Get_nb_tets32.argtypes = [ctypes.c_void_p]
        cvt_libCGAL.CVT.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32, ctypes.c_float, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32]
        
        cvt_lib.NewTet32.restype = ctypes.c_void_p
        cvt_lib.NewTet32.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32]

        active_flag = torch.zeros([cpy_sites.shape[0]], dtype=torch.int32).cuda()
        active_flag = active_flag.contiguous()
        
        counter_sites = torch.zeros([cpy_sites.shape[0]], dtype=torch.int32).cuda()
        counter_sites = counter_sites.contiguous()
        
        cvt_lib.Gradient_sites32.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        cvt_lib.Compute_gradient_field32.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]

        #cpy_sites = torch.clone(down_sites)
        cvt_libCGAL.CVT(cpy_sites.data_ptr(), down_sdf.data_ptr(), down_gradients.data_ptr(), self.params.data_ptr(),
                                                cpy_sites.shape[0], scale, translate_c, filename.encode(), 1) #0
    
        """active_flag[:] = 1

        for _ in range(2):
            Tet_32 = cvt_lib.NewTet32(cpy_sites.data_ptr(), self.params.data_ptr(), cpy_sites.shape[0])            
            nb_tets = cvt_lib.Get_nb_tets32(Tet_32)

            gradients_field = torch.zeros([3*nb_tets], dtype=torch.float32).cuda()
            gradients_field_weights = torch.zeros([4*3*nb_tets], dtype=torch.float32).cuda()
            cvt_lib.Compute_gradient_field32(Tet_32, gradients_field.data_ptr(), gradients_field_weights.data_ptr(), sdf.data_ptr(), cpy_sites.data_ptr(), active_flag.data_ptr())
            
            counter_sites[:] = 0
            gradients[:] = 0
            cvt_lib.Gradient_sites32(Tet_32, gradients.data_ptr(), counter_sites.data_ptr(), gradients_field.data_ptr(), active_flag.data_ptr())
            
            cvt_libCGAL.CVT(cpy_sites.data_ptr(), sdf.data_ptr(), gradients.data_ptr(), self.params.data_ptr(),
                                cpy_sites.shape[0], scale, translate_c, filename.encode(), 0)
        
        Tet_32 = cvt_lib.NewTet32(cpy_sites.data_ptr(), self.params.data_ptr(), cpy_sites.shape[0])
        
        nb_tets = cvt_lib.Get_nb_tets32(Tet_32)
        gradients_field = torch.zeros([3*nb_tets], dtype=torch.float32).cuda()
        gradients_field_weights = torch.zeros([4*3*nb_tets], dtype=torch.float32).cuda()
        cvt_lib.Compute_gradient_field32(Tet_32, gradients_field.data_ptr(), gradients_field_weights.data_ptr(), sdf.data_ptr(), cpy_sites.data_ptr(), active_flag.data_ptr())
        
        counter_sites[:] = 0
        gradients[:] = 0
        cvt_lib.Gradient_sites32(Tet_32, gradients.data_ptr(), counter_sites.data_ptr(), gradients_field.data_ptr(), active_flag.data_ptr())#self.active_flag.data_ptr())
        
        cvt_libCGAL.CVT(cpy_sites.data_ptr(), sdf.data_ptr(), gradients.data_ptr(), self.params.data_ptr(),
                                cpy_sites.shape[0], scale, translate_c, filename.encode(), 1)"""


if __name__=='__main__':
    visual_hull = [-1, -1, -1, 1, 1, 1]
    import sampling as sampler
    sites = sampler.sample_Bbox(visual_hull[0:3], visual_hull[3:6], 16, perturb_f =  visual_hull[3]*0.005) 
    T32 = Tet32(sites)
    T32.save("data/bmvs_man/test.ply")