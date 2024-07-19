#include <torch/extension.h>

#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <thrust/device_vector.h>

#include <device_launch_parameters.h>
#include "cudaType.cuh"

#ifdef __INTELLISENSE__
#define CUDA_KERNEL(...)
#define FAKEINIT = {0}
#else
#define CUDA_KERNEL(...) <<< __VA_ARGS__ >>>
#define FAKEINIT
#endif

#define DIM_L_FEAT 16
#define PI 3.141592653589793238462643383279502884197

/** Device functions **/
/** Device functions **/
/** Device functions **/

__global__ void setup_kernel(curandState* state, uint64_t seed)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, tid, 0, &state[tid]);
}

inline __device__ double dot3D_gpu_d(double a[3], double b[3]) {
	return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

__device__ double volume_tetrahedron_32(float a[3], float b[3], float c[3], float d[3]) {
	double ad[3] = { a[0] - d[0], a[1] - d[1], a[2] - d[2] };
	double bd[3] = { b[0] - d[0], b[1] - d[1], b[2] - d[2] };
	double cd[3] = { c[0] - d[0], c[1] - d[1], c[2] - d[2] };

	double n[3] = { bd[1] * cd[2] - bd[2] * cd[1],
					-(bd[0] * cd[2] - bd[2] * cd[0]),
					bd[0] * cd[1] - bd[1] * cd[0] };

	double res = abs(dot3D_gpu_d(ad, n)) / 6.0;
	return res;
}


__device__ float get_sdf_cvt(float weights[4], float p[3], float* sites, float* sdf, int* tets, int tet_id, float s_inv) {
    int id0 = tets[4 * tet_id];
	int id1 = tets[4 * tet_id + 1];
	int id2 = tets[4 * tet_id + 2];
	int id3 = id0 ^ id1 ^ id2 ^ tets[4 * tet_id + 3];
    
    float tot_vol = __double2float_rn(volume_tetrahedron_32(&sites[3 * id0], &sites[3 * id1],
		&sites[3 * id2], &sites[3 * id3]));

    weights[0] = tot_vol == 0.0f ? 0.25f : __double2float_rn(volume_tetrahedron_32(p, &sites[3 * id1],
		&sites[3 * id2], &sites[3 * id3])) / tot_vol;
    weights[1] = tot_vol == 0.0f ? 0.25f : __double2float_rn(volume_tetrahedron_32(p, &sites[3 * id0],
		&sites[3 * id2], &sites[3 * id3])) / tot_vol;
    weights[2] = tot_vol == 0.0f ? 0.25f : __double2float_rn(volume_tetrahedron_32(p, &sites[3 * id0],
		&sites[3 * id1], &sites[3 * id3])) / tot_vol;
    weights[3] = tot_vol == 0.0f ? 0.25f : __double2float_rn(volume_tetrahedron_32(p, &sites[3 * id0],
		&sites[3 * id1], &sites[3 * id2])) / tot_vol;

    float sum_weights = weights[0] + weights[1] + weights[2] + weights[3];
	if (sum_weights > 0.0f) {
		weights[0] = weights[0] / sum_weights;
		weights[1] = weights[1] / sum_weights;
		weights[2] = weights[2] / sum_weights;
		weights[3] = weights[3] / sum_weights;
	}
	else {
		weights[0] = 0.25f;
		weights[1] = 0.25f;
		weights[2] = 0.25f;
		weights[3] = 0.25f;
	}

    return sdf[id0] * weights[0] + sdf[id1] * weights[1] +
            sdf[id2] * weights[2] + sdf[id3] * weights[3];
}

__global__ void backprop_multi_kernel(
    const size_t num_sites, 
    const size_t num_knn,  
    const size_t dim_feat,
    float *__restrict__ vertices,     // [N_voxels, 4] for each voxel => it's vertices
    int *__restrict__ activated,     // [N_voxels, 4] for each voxel => it's vertices
    float *__restrict__ grad_norm,     // [N_voxels, 4] for each voxel => it's vertices
    float *__restrict__ grad_norm_feat,     // [N_voxels, 4] for each voxel => it's vertices
    float *__restrict__ grad_feat,     // [N_voxels, 4] for each voxel => it's vertices
    int *__restrict__ neighbors)
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_sites)
    {
        return;
    }

    if (activated[idx] != 2)
        return;

    int nb_lvl = num_knn / 32;
    int knn_id; 

    for (int lvl_curr = 0; lvl_curr < nb_lvl; lvl_curr++) {
        for (int i = 0; i < 32; i++) {
            knn_id = neighbors[num_knn*idx + lvl_curr*32 + i];
            if (knn_id == -1)
                return;
            
            for (int i = 0; i < dim_feat; i++) {
                atomicAdd(&grad_feat[4*dim_feat*knn_id + i], grad_feat[dim_feat*(4*idx+lvl_curr+1) + i]/32.0f);
            }  
            for (int i = 0; i < 3; i++) {
                atomicAdd(&grad_norm[3*knn_id + i], grad_norm_feat[3*(4*idx+lvl_curr+1) + i]/32.0f);
            }                       
        }
    }

    return;
}

__global__ void backprop_feat_kernel(
    const size_t num_samples,
    const size_t dim_feats,
    float *__restrict__ sdf,
    float *__restrict__ grad_feat,
    float *__restrict__ counter,
    const float *__restrict__ grad_samples,
    const int *__restrict__ cell_ids,
    const float *__restrict__ cell_weights)
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_samples)
    {
        return;
    }

    int id_prev, id;
    ////////////////////////Linear interpolation//////////////////////////
    //////////////////////////////////////////////////////////////
    float lamda = cell_weights[7*idx + 6] ;
    float fact = lamda == 0.5? 1.0f : 1.0f;///3.0f;
    for (int i = 0; i < 3; i++) {
        id_prev = cell_ids[6 * idx + i];
        id = cell_ids[6 * idx + 3 + i];
        for (int k = 0; k < dim_feats; k++) {    
            atomicAdd(&grad_feat[dim_feats * id_prev + k], cell_weights[7*idx + i] * fact * lamda * grad_samples[dim_feats * idx + k]);              
            atomicAdd(&grad_feat[dim_feats * id + k], cell_weights[7*idx + 3 + i] * fact * (1.0f - lamda) * grad_samples[dim_feats * idx + k]);
        }
    }
    return;
}

__global__ void backprop_feat_kernel_o(
    const size_t num_samples,
    const size_t dim_feats,
    float3 *__restrict__ sdf,
    float4 *__restrict__ grad_feat,
    float *__restrict__ counter,
    const float4 *__restrict__ grad_samples,
    const int3 *__restrict__ cell_ids,
    const float3 *__restrict__ cell_weights)
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_samples)
    {
        return;
    }

    ////////////////////////Linear interpolation//////////////////////////
    //////////////////////////////////////////////////////////////
    float lamda = sdf[idx].z ;
    int3 id_prev = cell_ids[2 * idx ];
    int3 id = cell_ids[2 * idx + 1];

    float3 cell_weights_prev = cell_weights[2 * idx ];
    float3 cell_weights_curr = cell_weights[2 * idx + 1];

    float fact = lamda == 0.5? 1.0f : 1.0f/3.0f;
    float4 val;
    for (int k = 0; k < dim_feats; k++) {    
        val = grad_samples[dim_feats * idx + k];
        atomicAdd(&(grad_feat[dim_feats * id_prev.x + k].x), cell_weights_prev.x * fact * lamda * val.x);
        atomicAdd(&(grad_feat[dim_feats * id_prev.x + k].y), cell_weights_prev.x * fact * lamda * val.y);
        atomicAdd(&(grad_feat[dim_feats * id_prev.x + k].z), cell_weights_prev.x * fact * lamda * val.z);
        atomicAdd(&(grad_feat[dim_feats * id_prev.x + k].w), cell_weights_prev.x * fact * lamda * val.w);
        //atomicAdd(&grad_feat[dim_feats * id_prev.x + k], cell_weights_prev.x * lamda * grad_samples[dim_feats * idx + k]);
        
        //val = cell_weights_curr.x * (1.0f - lamda) * grad_samples[dim_feats * idx + k];   
        atomicAdd(&(grad_feat[dim_feats * id.x + k].x), cell_weights_curr.x * fact * (1.0f - lamda) * val.x);
        atomicAdd(&(grad_feat[dim_feats * id.x + k].y), cell_weights_curr.x * fact * (1.0f - lamda) * val.y);
        atomicAdd(&(grad_feat[dim_feats * id.x + k].z), cell_weights_curr.x * fact * (1.0f - lamda) * val.z);      
        atomicAdd(&(grad_feat[dim_feats * id.x + k].w), cell_weights_curr.x * fact * (1.0f - lamda) * val.w);     
        //atomicAdd(&grad_feat[dim_feats * id.x + k], cell_weights_curr.x * (1.0f - lamda) * grad_samples[dim_feats * idx + k]);
        
        //val = cell_weights_prev.y * lamda * grad_samples[dim_feats * idx + k];     
        atomicAdd(&(grad_feat[dim_feats * id_prev.y + k].x), cell_weights_prev.y * fact * lamda * val.x);
        atomicAdd(&(grad_feat[dim_feats * id_prev.y + k].y), cell_weights_prev.y * fact * lamda * val.y);
        atomicAdd(&(grad_feat[dim_feats * id_prev.y + k].z), cell_weights_prev.y * fact * lamda * val.z);     
        atomicAdd(&(grad_feat[dim_feats * id_prev.y + k].w), cell_weights_prev.y * fact * lamda * val.w);      
        //atomicAdd(&grad_feat[dim_feats * id_prev.y + k], cell_weights_prev.y * lamda * grad_samples[dim_feats * idx + k]);     

        //val = cell_weights_curr.y * (1.0f - lamda) * grad_samples[dim_feats * idx + k];   
        atomicAdd(&(grad_feat[dim_feats * id.y + k].x), cell_weights_curr.y * fact * (1.0f - lamda) * val.x);
        atomicAdd(&(grad_feat[dim_feats * id.y + k].y), cell_weights_curr.y * fact * (1.0f - lamda) * val.y);
        atomicAdd(&(grad_feat[dim_feats * id.y + k].z), cell_weights_curr.y * fact * (1.0f - lamda) * val.z);     
        atomicAdd(&(grad_feat[dim_feats * id.y + k].w), cell_weights_curr.y * fact * (1.0f - lamda) * val.w);  
        //atomicAdd(&grad_feat[dim_feats * id.y + k], cell_weights_curr.y * (1.0f - lamda) * grad_samples[dim_feats * idx + k]);
        
        //val = cell_weights_prev.z * lamda * grad_samples[dim_feats * idx + k];   
        atomicAdd(&(grad_feat[dim_feats * id_prev.z + k].x), cell_weights_prev.z * fact * lamda * val.x);
        atomicAdd(&(grad_feat[dim_feats * id_prev.z + k].y), cell_weights_prev.z * fact * lamda * val.y);
        atomicAdd(&(grad_feat[dim_feats * id_prev.z + k].z), cell_weights_prev.z * fact * lamda * val.z);    
        atomicAdd(&(grad_feat[dim_feats * id_prev.z + k].w), cell_weights_prev.z * fact * lamda * val.w); 
        //atomicAdd(&grad_feat[dim_feats * id_prev.z + k], cell_weights_prev.z * lamda * grad_samples[dim_feats * idx + k]);              
        
        //val = cell_weights_curr.z * (1.0f - lamda) * grad_samples[dim_feats * idx + k];   
        atomicAdd(&(grad_feat[dim_feats * id.z + k].x), cell_weights_curr.z * fact * (1.0f - lamda) * val.x);
        atomicAdd(&(grad_feat[dim_feats * id.z + k].y), cell_weights_curr.z * fact * (1.0f - lamda) * val.y);
        atomicAdd(&(grad_feat[dim_feats * id.z + k].z), cell_weights_curr.z * fact * (1.0f - lamda) * val.z);     
        atomicAdd(&(grad_feat[dim_feats * id.z + k].w), cell_weights_curr.z * fact * (1.0f - lamda) * val.w);  
        //atomicAdd(&grad_feat[dim_feats * id.xz + k], cell_weights_curr.z * (1.0f - lamda) * grad_samples[dim_feats * idx + k]);*/
    }

    return;
}


__global__ void backprop_grad_kernel(
    const size_t num_samples,
    float *__restrict__ sdf,
    float *__restrict__ grad_sites,
    const float *__restrict__ grad_samples,
    const int *__restrict__ cell_ids,
    const float *__restrict__ cell_weights)
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_samples)
    {
        return;
    }

    int id_prev, id;
    ////////////////////////Linear interpolation//////////////////////////
    //////////////////////////////////////////////////////////////
    float lamda = cell_weights[7*idx + 6] ;
    float fact = lamda == 0.5? 1.0f : 1.0f;///3.0f;
    for (int i = 0; i < 3; i++) {
        id_prev = cell_ids[6 * idx + i];
        id = cell_ids[6 * idx + 3 + i];
        for (int k = 0; k < 3; k++) {    
            atomicAdd(&grad_sites[3 * id_prev + k], cell_weights[7*idx + i] * fact * lamda * grad_samples[3 * idx + k]);              
            atomicAdd(&grad_sites[3 * id + k], cell_weights[7*idx + 3 + i] * fact * (1.0f - lamda) * grad_samples[3 * idx + k]);
        }
    }
    return;
}

__global__ void backprop_grad_kernel_o(
    const size_t num_samples,
    float3 *__restrict__ sdf,
    float3 *__restrict__ grad_sites,
    const float3 *__restrict__ grad_samples,
    const int3 *__restrict__ cell_ids,
    const float3 *__restrict__ cell_weights)
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_samples)
    {
        return;
    }

    int3 id_prev, id;
    ////////////////////////Linear interpolation//////////////////////////
    //////////////////////////////////////////////////////////////
    float lamda = sdf[idx].z ;
    id_prev = cell_ids[2 * idx ];
    id = cell_ids[2 * idx + 1];

    float3 cell_weights_prev = cell_weights[2 * idx ];
    float3 cell_weights_curr = cell_weights[2 * idx + 1];

    float3 val;
    for (int k = 0; k < 4; k++) {    
        val = grad_samples[4 * idx + k];
        //val = cell_weights_prev.x * lamda * grad_samples[4 * idx + k];
        atomicAdd(&(grad_sites[4 * id_prev.x + k].x), cell_weights_prev.x * lamda * val.x);
        atomicAdd(&(grad_sites[4 * id_prev.x + k].y), cell_weights_prev.x * lamda * val.y);
        atomicAdd(&(grad_sites[4 * id_prev.x + k].z), cell_weights_prev.x * lamda * val.z);
        //atomicAdd(&grad_sites[4 * id_prev.x + k], cell_weights_prev.x * lamda * grad_samples[4 * idx + k]);              
        
        //val = cell_weights_curr.x * (1.0f - lamda) * grad_samples[4 * idx + k];
        atomicAdd(&(grad_sites[4 * id.x + k].x), cell_weights_curr.x * (1.0f - lamda) * val.x);
        atomicAdd(&(grad_sites[4 * id.x + k].y), cell_weights_curr.x * (1.0f - lamda) * val.y);
        atomicAdd(&(grad_sites[4 * id.x + k].z), cell_weights_curr.x * (1.0f - lamda) * val.z);
        //atomicAdd(&grad_sites[4 * id.x + k], cell_weights_curr.x * (1.0f - lamda) * grad_samples[4 * idx + k]);
        
        //val = cell_weights_prev.y * lamda * grad_samples[4 * idx + k];
        atomicAdd(&(grad_sites[4 * id_prev.y + k].x), cell_weights_prev.y * lamda * val.x);
        atomicAdd(&(grad_sites[4 * id_prev.y + k].y), cell_weights_prev.y * lamda * val.y);
        atomicAdd(&(grad_sites[4 * id_prev.y + k].z), cell_weights_prev.y * lamda * val.z);
        //atomicAdd(&grad_feat[4 * id_prev.y + k], cell_weights_prev.y * lamda * grad_samples[4 * idx + k]);              
        
        //val = cell_weights_curr.y * (1.0f - lamda) * grad_samples[4 * idx + k];
        atomicAdd(&(grad_sites[4 * id.y + k].x), cell_weights_curr.y * (1.0f - lamda) * val.x);
        atomicAdd(&(grad_sites[4 * id.y + k].y), cell_weights_curr.y * (1.0f - lamda) * val.y);
        atomicAdd(&(grad_sites[4 * id.y + k].z), cell_weights_curr.y * (1.0f - lamda) * val.z);
        //atomicAdd(&grad_feat[4 * id.y + k], cell_weights_curr.y * (1.0f - lamda) * grad_samples[4 * idx + k]);
        
        //val = cell_weights_prev.z * lamda * grad_samples[4 * idx + k];
        atomicAdd(&(grad_sites[4 * id_prev.z + k].x), cell_weights_prev.z * lamda * val.x);
        atomicAdd(&(grad_sites[4 * id_prev.z + k].y), cell_weights_prev.z * lamda * val.y);
        atomicAdd(&(grad_sites[4 * id_prev.z + k].z), cell_weights_prev.z * lamda * val.z);
        //atomicAdd(&grad_feat[4 * id_prev.z + k], cell_weights_prev.z * lamda * grad_samples[4 * idx + k]);              
        
        //val = cell_weights_curr.z * (1.0f - lamda) * grad_samples[4 * idx + k];
        atomicAdd(&(grad_sites[4 * id.z + k].x), cell_weights_curr.z * (1.0f - lamda) * val.x);
        atomicAdd(&(grad_sites[4 * id.z + k].y), cell_weights_curr.z * (1.0f - lamda) * val.y);
        atomicAdd(&(grad_sites[4 * id.z + k].z), cell_weights_curr.z * (1.0f - lamda) * val.z);
        //atomicAdd(&grad_feat[4 * id.xz + k], cell_weights_curr.z * (1.0f - lamda) * grad_samples[4 * idx + k]);
    }

    return;
}


__global__ void backprop_sdf_kernel(
    const size_t num_samples,
    float4 *__restrict__ sdf,
    float *__restrict__ grad_sdf,
    const float2 *__restrict__ grad_sdf_samples,
    const int3 *__restrict__ cell_ids,
    const float *__restrict__ cell_weights)
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_samples)
    {
        return;
    }

    int3 id_prev, id;
    ////////////////////////Linear interpolation//////////////////////////
    //////////////////////////////////////////////////////////////
    //float lamda = sdf[idx].z ;
    id_prev = cell_ids[2 * idx ];
    id = cell_ids[2 * idx + 1];

    float2 grad_curr = grad_sdf_samples[idx];

    atomicAdd(&grad_sdf[id_prev.x], cell_weights[7 * idx] * grad_curr.x);       
    atomicAdd(&grad_sdf[id.x], cell_weights[7 * idx + 3]  * grad_curr.y);  
    
    atomicAdd(&grad_sdf[id_prev.y], cell_weights[7 * idx + 1] * grad_curr.x);       
    atomicAdd(&grad_sdf[id.y], cell_weights[7 * idx + 4]  * grad_curr.y);  
    
    atomicAdd(&grad_sdf[id_prev.z], cell_weights[7 * idx + 2]  * grad_curr.x);       
    atomicAdd(&grad_sdf[id.z], cell_weights[7 * idx + 5]  * grad_curr.y);  

    return;
}


__global__ void backprop_norm_kernel(
    const size_t num_tets,                // number of rays
    const int *__restrict__ tets,  // [N_voxels, 4] for each voxel => it's neighbors
    float *__restrict__ sites,     // [N_voxels, 4] for each voxel => it's vertices,
    float *__restrict__ vol,     // [N_voxels, 4] for each voxel => it's vertices
    float *__restrict__ weights,     // [N_voxels, 4] for each voxel => it's vertices
    float *__restrict__ weights_tot,     // [N_voxels, 4] for each voxel => it's vertices
    float *__restrict__ grad_norm,
    float *__restrict__ grad_sdf,
    int *__restrict__ activated
    )
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_tets)
    {
        return;
    }

    int ids[4] = {0, 0, 0, 0,};
    ids[0] = tets[4*idx];  ids[1] = tets[4*idx + 1];  ids[2] = tets[4*idx + 2];
    ids[3] = ids[0] ^ ids[1] ^ ids[2] ^ tets[4*idx + 3];

    
    if (activated[ids[0]] == 0 && 
        activated[ids[1]] == 0 && 
        activated[ids[2]] == 0 && 
        activated[ids[3]] == 0)
        return;

    float volume_tet = vol[idx];
    
    float *Weights_curr = &weights[12*idx];   
    
    float grad_curr[3] {};
    if (weights_tot[ids[0]] > 0.0f) {
        grad_curr[0] = (volume_tet/(4.0*weights_tot[ids[0]])) * (3.0*Weights_curr[0]);
        grad_curr[1] = (volume_tet/(4.0*weights_tot[ids[0]])) * (3.0*Weights_curr[1]);
        grad_curr[2] = (volume_tet/(4.0*weights_tot[ids[0]])) * (3.0*Weights_curr[2]);
        atomicAdd(&grad_sdf[ids[0]], grad_norm[3*ids[0]]*grad_curr[0] + grad_norm[3*ids[0]+1]*grad_curr[1] + grad_norm[3*ids[0]+2]*grad_curr[2]);
    }

    if (weights_tot[ids[1]] > 0.0f) {
        grad_curr[0] = (volume_tet/(4.0*weights_tot[ids[1]])) * (3.0*Weights_curr[3*1]);
        grad_curr[1] = (volume_tet/(4.0*weights_tot[ids[1]])) * (3.0*Weights_curr[3*1+1]);
        grad_curr[2] = (volume_tet/(4.0*weights_tot[ids[1]])) * (3.0*Weights_curr[3*1+2]);
        atomicAdd(&grad_sdf[ids[1]], grad_norm[3*ids[1]]*grad_curr[0] + grad_norm[3*ids[1]+1]*grad_curr[1] + grad_norm[3*ids[1]+2]*grad_curr[2]);
    }

    if (weights_tot[ids[2]] > 0.0f) {
        grad_curr[0] = (volume_tet/(4.0*weights_tot[ids[2]])) * (3.0*Weights_curr[3*2]);
        grad_curr[1] = (volume_tet/(4.0*weights_tot[ids[2]])) * (3.0*Weights_curr[3*2+1]);
        grad_curr[2] = (volume_tet/(4.0*weights_tot[ids[2]])) * (3.0*Weights_curr[3*2+2]);
        atomicAdd(&grad_sdf[ids[2]], grad_norm[3*ids[2]]*grad_curr[0] + grad_norm[3*ids[2]+1]*grad_curr[1] + grad_norm[3*ids[2]+2]*grad_curr[2]);
    }
    
    if (weights_tot[ids[3]] > 0.0f) {
        grad_curr[0] = (volume_tet/(4.0*weights_tot[ids[3]])) * (3.0*Weights_curr[3*3]);
        grad_curr[1] = (volume_tet/(4.0*weights_tot[ids[3]])) * (3.0*Weights_curr[3*3+1]);
        grad_curr[2] = (volume_tet/(4.0*weights_tot[ids[3]])) * (3.0*Weights_curr[3*3+2]);
        atomicAdd(&grad_sdf[ids[3]], grad_norm[3*ids[3]]*grad_curr[0] + grad_norm[3*ids[3]+1]*grad_curr[1] + grad_norm[3*ids[3]+2]*grad_curr[2]);
    }
}

__global__ void backprop_unit_norm_kernel(
    const size_t num_tets,                // number of rays
    const int *__restrict__ tets,  // [N_voxels, 4] for each voxel => it's neighbors
    float *__restrict__ sites,     // [N_voxels, 4] for each voxel => it's vertices,
    float *__restrict__ norm_grad,
    float *__restrict__ grad_unornmed,
    float *__restrict__ vol,     // [N_voxels, 4] for each voxel => it's vertices
    float *__restrict__ weights,     // [N_voxels, 4] for each voxel => it's vertices
    float *__restrict__ weights_tot,     // [N_voxels, 4] for each voxel => it's vertices
    float *__restrict__ grad_norm,
    float *__restrict__ grad_sdf,
    int *__restrict__ activated
    )
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_tets)
    {
        return;
    }

    int ids[4] = {0, 0, 0, 0,};
    ids[0] = tets[4*idx];  ids[1] = tets[4*idx + 1];  ids[2] = tets[4*idx + 2];
    ids[3] = ids[0] ^ ids[1] ^ ids[2] ^ tets[4*idx + 3];

    if (activated[ids[0]] == 0 && 
        activated[ids[1]] == 0 && 
        activated[ids[2]] == 0 && 
        activated[ids[3]] == 0)
        return;

    float volume_tet = vol[idx];    
    float *Weights_curr = &weights[12*idx];    

    for (int i = 0; i < 4; i++) {
        float norm_grad = sqrt(grad_unornmed[3*ids[i]]*grad_unornmed[3*ids[i]] + grad_unornmed[3*ids[i]+1]*grad_unornmed[3*ids[i]+1] + grad_unornmed[3*ids[i]+2]*grad_unornmed[3*ids[i]+2]);
        if (norm_grad > 1.0e-8) {
            atomicAdd(&grad_sdf[ids[i]], (grad_norm[3*ids[i]] * (1.0f/norm_grad - grad_unornmed[3*ids[i]]*grad_unornmed[3*ids[i]] / (norm_grad*norm_grad*norm_grad)) * (3.0f*Weights_curr[3*i] - Weights_curr[3*((i+1)%4)] - Weights_curr[3*((i+2)%4)] - Weights_curr[3*((i+3)%4)]) / 4.0f  +
                                    grad_norm[3*ids[i] + 1] * (1.0f/norm_grad - grad_unornmed[3*ids[i] + 1]*grad_unornmed[3*ids[i] + 1] / (norm_grad*norm_grad*norm_grad)) * (3.0f*Weights_curr[3*i + 1] - Weights_curr[3*((i+1)%4) + 1] - Weights_curr[3*((i+2)%4) + 1] - Weights_curr[3*((i+3)%4) + 1]) / 4.0f + 
                                    grad_norm[3*ids[i] + 2] * (1.0f/norm_grad - grad_unornmed[3*ids[i] + 2]*grad_unornmed[3*ids[i] + 2] / (norm_grad*norm_grad*norm_grad)) * (3.0f*Weights_curr[3*i + 2] - Weights_curr[3*((i+1)%4) + 2] - Weights_curr[3*((i+2)%4) + 2] - Weights_curr[3*((i+3)%4) + 2]) / 4.0f) * volume_tet / weights_tot[ids[i]]);
        }   
    }
}


__global__ void eikonal_loss_kernel(
    const size_t num_sites,                // number of rays
    const size_t num_knn,                // number of rays   
    const int *__restrict__ neighbors,  // [N_voxels, 4] for each voxel => it's neighbors
    float *__restrict__ Weights,     // [N_voxels, 4] for each voxel => it's vertices
    float *__restrict__ grad,     // [N_voxels, 4] for each voxel => it's vertices)
    float *__restrict__ grad_eik,     // [N_voxels, 4] for each voxel => it's vertices
    float *Loss
    )
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_sites)
    {
        return;
    }
    
    float *Weights_curr = &Weights[2*num_knn*idx];    
    float norm_grad = grad[2*idx]*grad[2*idx] + grad[2*idx+1]*grad[2*idx+1];

    float diff_loss[2]  {2.0*grad[2*idx], 2.0*grad[2*idx + 1]};
    if (norm_grad < 1.0f) {
        diff_loss[0] = -diff_loss[0];
        diff_loss[1] = -diff_loss[1];
    }

    int knn_id;
    for (int i = 0; i < num_knn; i++) {
        knn_id = neighbors[num_knn*(idx+1) + i];
        atomicAdd(&grad_eik[knn_id], diff_loss[0] * Weights_curr[2*i] + diff_loss[1] * Weights_curr[2*i + 1]);
        atomicAdd(&grad_eik[idx], -(diff_loss[0] * Weights_curr[2*i] + diff_loss[1] * Weights_curr[2*i + 1]));
    }

    atomicAdd(Loss, abs(norm_grad-1));

    return;
}

__global__ void smooth_grad_kernel(
    const size_t num_edges,                // number of rays
    float sigma,
    float *__restrict__ vertices,     // [N_voxels, 4] for each voxel => it's vertices
    int *__restrict__ activated,     // [N_voxels, 4] for each voxel => it's vertices
    float *__restrict__ sdf,     // [N_voxels, 4] for each voxel => it's vertices
    float *__restrict__ feat,     // [N_voxels, 4] for each voxel => it's vertices
    int *__restrict__ edges,     // [N_voxels, 4] for each voxel => it's vertices)
    float *__restrict__ sdf_grad,    // [N_voxels, 4] for each voxel => it's vertices
    float *__restrict__ feat_grad,     // [N_voxels, 4] for each voxel => it's vertices
    float *__restrict__ counter     // [N_voxels, 4] for each voxel => it's vertices
    )
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_edges)
    {
        return;
    }

    if (!(activated[edges[2*idx]] >= 1 || activated[edges[2*idx + 1]] >= 1))
        return;
    
    float length_edge = sqrt((vertices[3*edges[2*idx]] - vertices[3*edges[2*idx+1]])*(vertices[3*edges[2*idx]] - vertices[3*edges[2*idx+1]]) +
                            (vertices[3*edges[2*idx] + 1] - vertices[3*edges[2*idx+1] + 1])*(vertices[3*edges[2*idx] + 1] - vertices[3*edges[2*idx+1] + 1]) +
                            (vertices[3*edges[2*idx] + 2] - vertices[3*edges[2*idx+1] + 2])*(vertices[3*edges[2*idx] + 2] - vertices[3*edges[2*idx+1] + 2]));
    
    float eps= 1.0e-10;
    float denom = 1.0f;//sqrt( eps + (sdf[edges[2*idx]] - sdf[edges[2*idx+1]])*(sdf[edges[2*idx]] - sdf[edges[2*idx+1]]));
    float sdf_a = sdf[edges[2*idx]] + (sigma/length_edge)*(sdf[edges[2*idx+1]] - sdf[edges[2*idx]]);
    atomicAdd(&sdf_grad[edges[2*idx]], (sdf[edges[2*idx]] - sdf_a)) ; //exp(-length_edge/(sigma*sigma)) * (sdf[edges[2*idx]] - sdf[edges[2*idx+1]]) / denom);     
    sdf_a = sdf[edges[2*idx+1]] + (sigma/length_edge)*(sdf[edges[2*idx]] - sdf[edges[2*idx+1]]);    
    atomicAdd(&sdf_grad[edges[2*idx + 1]], (sdf[edges[2*idx+1]] - sdf_a));

    // add bilateral smooth term with features
    /*float length_feat = 0.0f;
    for (int i = 0; i < DIM_L_FEAT; i++) {
        length_feat = length_feat +  (feat[DIM_L_FEAT*edges[2*idx]+ i] - feat[DIM_L_FEAT*edges[2*idx+1]+ i])*
                                        (feat[DIM_L_FEAT*edges[2*idx]+ i] - feat[DIM_L_FEAT*edges[2*idx+1]+ i]);
    }*/

    for (int i = 0; i < DIM_L_FEAT; i++) {
        denom = 1.0f;//sqrt( eps +  (feat[DIM_L_FEAT*edges[2*idx] + i] - feat[DIM_L_FEAT*edges[2*idx+1] + i])* (feat[DIM_L_FEAT*edges[2*idx] + i] - feat[DIM_L_FEAT*edges[2*idx+1] + i]));
        sdf_a = sdf[edges[2*idx]] + (sigma/length_edge)*(feat[DIM_L_FEAT*edges[2*idx+1] + i] - feat[DIM_L_FEAT*edges[2*idx] + i]);
        atomicAdd(&feat_grad[DIM_L_FEAT*edges[2*idx] + i], (feat[DIM_L_FEAT*edges[2*idx] + i] - sdf_a)) ; //exp(-length_edge/(sigma*sigma)) * (feat[DIM_L_FEAT*edges[2*idx] + i] - feat[DIM_L_FEAT*edges[2*idx+1] + i]) / denom); 
        sdf_a = sdf[edges[2*idx]] + (sigma/length_edge)*(feat[DIM_L_FEAT*edges[2*idx] + i] - feat[DIM_L_FEAT*edges[2*idx+1] + i]);         
        atomicAdd(&feat_grad[DIM_L_FEAT*edges[2*idx + 1] + i], (feat[DIM_L_FEAT*edges[2*idx+1] + i] - sdf_a)) ; //-exp(-length_edge/(sigma*sigma)) * (feat[DIM_L_FEAT*edges[2*idx] + i] - feat[DIM_L_FEAT*edges[2*idx+1] + i]) / denom);
    }
    
    //atomicAdd(&counter[edges[2*idx]], exp(-length_edge/(sigma*sigma)));
    //atomicAdd(&counter[edges[2*idx + 1]], exp(-length_edge/(sigma*sigma)));
    
    //atomicAdd(&counter[2*edges[2*idx]+1], exp(-length_edge/(sigma*sigma) - length_feat/(0.2f)));
    //atomicAdd(&counter[2*edges[2*idx + 1]+1], exp(-length_edge/(sigma*sigma) - length_feat/(0.2f)));
    
    /*atomicAdd(&sdf_grad[edges[2*idx]], (sdf[edges[2*idx]] - sdf[edges[2*idx+1]]));          
    atomicAdd(&sdf_grad[edges[2*idx + 1]], -(sdf[edges[2*idx]] - sdf[edges[2*idx+1]]));

    for (int i = 0; i < DIM_L_FEAT; i++) {
        atomicAdd(&feat_grad[DIM_L_FEAT*edges[2*idx] + i],  (feat[DIM_L_FEAT*edges[2*idx] + i] - feat[DIM_L_FEAT*edges[2*idx+1] + i]));          
        atomicAdd(&feat_grad[DIM_L_FEAT*edges[2*idx + 1] + i], - (feat[DIM_L_FEAT*edges[2*idx] + i] - feat[DIM_L_FEAT*edges[2*idx+1] + i]));
    }
    
    atomicAdd(&counter[edges[2*idx]], 1.0f);
    atomicAdd(&counter[edges[2*idx + 1]], 1.0f);*/

    return;
}

__global__ void smooth_grad_n_kernel(
    const size_t num_edges,                // number of rays
    float sigma,
    int  dim,
    float *__restrict__ vertices,     // [N_voxels, 4] for each voxel => it's vertices
    int *__restrict__ activated,     // [N_voxels, 4] for each voxel => it's vertices
    float *__restrict__ nmle,     // [N_voxels, 4] for each voxel => it's vertices
    float *__restrict__ feat,     // [N_voxels, 4] for each voxel => it's vertices
    int *__restrict__ edges,     // [N_voxels, 4] for each voxel => it's vertices)
    float *__restrict__ nmle_grad,    // [N_voxels, 4] for each voxel => it's vertices
    float *__restrict__ feat_grad,     // [N_voxels, 4] for each voxel => it's vertices
    float *__restrict__ counter     // [N_voxels, 4] for each voxel => it's vertices
    )
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_edges)
    {
        return;
    }

    //if (!(activated[edges[2*idx]] >= 1 || activated[edges[2*idx + 1]] >= 1))
    //     return;
    
    float length_edge = (vertices[3*edges[2*idx]] - vertices[3*edges[2*idx+1]])*(vertices[3*edges[2*idx]] - vertices[3*edges[2*idx+1]]) +
                            (vertices[3*edges[2*idx] + 1] - vertices[3*edges[2*idx+1] + 1])*(vertices[3*edges[2*idx] + 1] - vertices[3*edges[2*idx+1] + 1]) +
                            (vertices[3*edges[2*idx] + 2] - vertices[3*edges[2*idx+1] + 2])*(vertices[3*edges[2*idx] + 2] - vertices[3*edges[2*idx+1] + 2]);

    
    int n = edges[2*idx];
    int id_curr = edges[2*idx + 1];

    if (dim == 3) {
	    float norm_grad = sqrt(nmle[3*n]*nmle[3*n] + nmle[3*n + 1]*nmle[3*n + 1] + nmle[3*n + 2]*nmle[3*n + 2]);
	    float norm_grad_curr = sqrt(nmle[3*id_curr]*nmle[3*id_curr] + nmle[3*id_curr + 1]*nmle[3*id_curr + 1] + nmle[3*id_curr + 2]*nmle[3*id_curr + 2]);
        float dot_prod = nmle[3*n]*nmle[3*id_curr] + nmle[3*n + 1]*nmle[3*id_curr + 1] + nmle[3*n + 2]*nmle[3*id_curr + 2];
        
        if (norm_grad > 1.0e-8 && norm_grad_curr > 1.0e-8) {
            for (int i = 0; i < dim; i++) {
                atomicAdd(&nmle_grad[dim*n + i], (1.0f / norm_grad_curr) * (dot_prod * nmle[3*n + i] / (norm_grad*norm_grad*norm_grad) - nmle[3*id_curr + i] / norm_grad) * exp(-length_edge/(sigma*sigma))) ;     // * exp(-length_edge/(sigma*sigma))
                atomicAdd(&nmle_grad[dim*id_curr + i], (1.0f / norm_grad) * (dot_prod * nmle[3*id_curr + i] / (norm_grad_curr*norm_grad_curr*norm_grad_curr) - nmle[3*n + i] / norm_grad_curr) * exp(-length_edge/(sigma*sigma))) ; //* exp(-length_edge/(sigma*sigma))
            }
        }
    } else {
        for (int i = 0; i < dim; i++) {
            atomicAdd(&nmle_grad[dim*n + i], (nmle[dim*n + i] - nmle[dim*id_curr + i]));// * exp(-length_edge/(sigma*sigma))) ;     
            atomicAdd(&nmle_grad[dim*id_curr + i], -(nmle[dim*n + i] - nmle[dim*id_curr + i]));// * exp(-length_edge/(sigma*sigma))) ; 
        }
    }

    return;
}


/*__global__ void geo_feat_kernel(
    const size_t num_samples,                // number of rays
    const size_t num_knn,  
    const float sigma,  
    float *__restrict__ samples,     // [N_voxels, 4] for each voxel => it's vertices
    float *__restrict__ vertices,     // [N_voxels, 4] for each voxel => it's vertices
    float *__restrict__ grads,     // [N_voxels, 4] for each voxel => it's vertices
    float *__restrict__ sdf,     // [N_voxels, 4] for each voxel => it's vertices
    float *__restrict__ geo_feat,     // [N_voxels, 4] for each voxel => it's vertices
    int *__restrict__ neighbors,     // [N_voxels, 4] for each voxel => it's vertices
    int *__restrict__ cell_ids
    )
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_samples)
    {
        return;
    }

    float total_weight = 0.0f;
    float total_sdf = 0.0f;

    int nb_lvl = 1;//num_knn / 32; //fmin(2, num_knn / 32);

    float theta, phi, length_edge, length_edge_2, diff_theta, diff_phi, w_r;
    int knn_id;

    float curr_point[3] {samples[3*idx], samples[3*idx + 1], samples[3*idx + 2]};
    float curr_edge[3] {};
    float counter_feat[8] {};
    int id[2] {};
    float base_theta[2] {PI / 4.0f, 3.0f*PI / 4.0f};
    float base_phi[4] {-3.0f*PI / 4.0f, -PI / 4.0f, PI / 4.0f, 3.0f*PI / 4.0f};

    //select closest site
    float min_dist = 1.0e32;
    int min_id = -1;
    for (int i = 0; i < 6; i++) {        
        curr_edge[0] = (vertices[3*cell_ids[12 * idx + 6 + i]] - curr_point[0]);
        curr_edge[1] = (vertices[3*cell_ids[12 * idx + 6 + i] + 1] - curr_point[1]);
        curr_edge[2] = (vertices[3*cell_ids[12 * idx + 6 + i] + 2] - curr_point[2]);
        length_edge = sqrt(curr_edge[0]*curr_edge[0] + curr_edge[1]*curr_edge[1] + curr_edge[2]*curr_edge[2]);

        if (length_edge < min_dist) {
            min_dist = length_edge;
            min_id = cell_ids[12 * idx + 6 + i];
        }
    }

    if (min_id == -1)
        return;
    
    float radius = sigma;
    float max_dist = -1.0f;
    for (int lvl_curr = 0; lvl_curr < nb_lvl; lvl_curr++) {
        for (int i = 0; i < 32; i++) {
            knn_id = neighbors[num_knn*min_id + lvl_curr*32 + i];
            if (knn_id == -1)
                continue;
            
            curr_edge[0] = (vertices[3*knn_id] - curr_point[0]);
            curr_edge[1] = (vertices[3*knn_id + 1] - curr_point[1]);
            curr_edge[2] = (vertices[3*knn_id + 2] - curr_point[2]);
            length_edge = sqrt(curr_edge[0]*curr_edge[0] + curr_edge[1]*curr_edge[1] + curr_edge[2]*curr_edge[2]);
            
            if (length_edge < 1.0e-10) // || length_edge > radius)
                continue;

            w_r = exp(-4.0f*(length_edge - sigma/2.0f)*(length_edge - sigma/2.0f) / (sigma*sigma));
            theta = acos(curr_edge[2]/length_edge);
            length_edge_2 = sqrt(curr_edge[0]*curr_edge[0] + curr_edge[1]*curr_edge[1]);
            if (length_edge_2 < 1.0e-10) {
                phi = 0.0f;
            } else {
                if (curr_edge[1] > 0.0) {
                    phi = acos(curr_edge[0]/length_edge_2);
                } else {
                    phi = -acos(curr_edge[0]/length_edge_2);
                } 
            }

            for (int t_i = 0; t_i < 2; t_i++) {           
                //diff_theta = max(0.0f, cos(theta - base_theta[t_i]));     
                diff_theta = exp(-16.0f*(theta - base_theta[t_i])*(theta - base_theta[t_i])/(PI*PI)); //max(0.0f, cos(theta - base_theta[t_i]));     
                for (int p_i = 0; p_i < 4; p_i++) {
                    diff_phi = exp(-64.0f*(phi - base_phi[p_i])*(phi - base_phi[p_i])/(PI*PI));  //max(0.0f, cos(phi - base_phi[p_i]));     

                    id[0] = t_i;//int(floorf(2.0f * theta / PI));
                    id[1] = p_i; //int(floorf(2.0f * (phi + PI) / PI));

                    geo_feat[32*idx + (4*id[0] + id[1])] = geo_feat[32*idx + (4*id[0] + id[1])] + w_r*diff_theta*diff_phi*sdf[knn_id];
                    geo_feat[32*idx + 8 + 3*(4*id[0] + id[1])] = geo_feat[32*idx + 8 + 3*(4*id[0] + id[1])] + w_r*diff_theta*diff_phi*grads[3*knn_id];
                    geo_feat[32*idx + 8 + 3*(4*id[0] + id[1]) + 1] = geo_feat[32*idx + 8 + 3*(4*id[0] + id[1]) + 1] + w_r*diff_theta*diff_phi*grads[3*knn_id + 1];
                    geo_feat[32*idx + 8 + 3*(4*id[0] + id[1]) + 2] = geo_feat[32*idx + 8 + 3*(4*id[0] + id[1]) + 2] + w_r*diff_theta*diff_phi*grads[3*knn_id + 2];
                    counter_feat[4*id[0] + id[1]] = counter_feat[4*id[0] + id[1]] + w_r*diff_theta*diff_phi;
                }
            }
        }
    }
    
    for (int i = 0; i < 8; i++) {
        geo_feat[32*idx + i] = counter_feat[i] == 0.0f ? 0.0f : geo_feat[32*idx + i] / counter_feat[i];
        geo_feat[32*idx + 8 + 3*i] = counter_feat[i] == 0.0f ? 0.0f : geo_feat[32*idx + 8 + 3*i] / counter_feat[i];
        geo_feat[32*idx + 8 + 3*i + 1] = counter_feat[i] == 0.0f ? 0.0f : geo_feat[32*idx + 8 + 3*i + 1] / counter_feat[i];
        geo_feat[32*idx + 8 + 3*i + 2] = counter_feat[i] == 0.0f ? 0.0f : geo_feat[32*idx + 8 + 3*i + 2] / counter_feat[i];
    }

    return;
}*/

__global__ void geo_feat_kernel(
    const size_t num_samples,                // number of rays
    const size_t num_knn,  
    float *__restrict__ samples,     // [N_voxels, 4] for each voxel => it's vertices
    float *__restrict__ vertices,     // [N_voxels, 4] for each voxel => it's vertices
    float *__restrict__ grads,     // [N_voxels, 4] for each voxel => it's vertices
    float *__restrict__ sdf,     // [N_voxels, 4] for each voxel => it's vertices
    float *__restrict__ geo_feat,     // [N_voxels, 4] for each voxel => it's vertices
    curandState* my_curandstate, 
    int *__restrict__ tets,     // [N_voxels, 4] for each voxel => it's vertices
    int *__restrict__ neighbors,     // [N_voxels, 4] for each voxel => it's vertices
    int *__restrict__ cell_ids
    )
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_samples)
    {
        return;
    }

    float ref_p[3] = { samples[3 * idx], samples[3 * idx + 1], samples[3 * idx + 2] };
    float data[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    float weights[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

    int size_feat = 21;
    int curr_tet = cell_ids[12*idx];
    if (curr_tet < 0) {
        geo_feat[size_feat * idx] = 20.0f;
        for (int i = 1; i < size_feat; i++)
            geo_feat[size_feat * idx + i] = 0.0f;
        return;
    }

    if ( neighbors[4 * curr_tet] == -1 || neighbors[4 * curr_tet + 1] == -1 || neighbors[4 * curr_tet + 2] == -1 || neighbors[4 * curr_tet + 3] == -1) {
        geo_feat[size_feat * idx] = 20.0f;
        for (int i = 1; i < size_feat; i++)
            geo_feat[size_feat * idx + i] = 0.0f;
        return;
    }

    float* curr_features = &geo_feat[size_feat * idx];

    int tet_ids[4] = {tets[4*curr_tet], tets[4*curr_tet + 1], tets[4*curr_tet + 2], 
                        tets[4*curr_tet] ^ tets[4*curr_tet+1] ^ tets[4*curr_tet+2] ^ tets[4*curr_tet+3]};

    // first feature is sdf at sample location
    curr_features[0] = get_sdf_cvt(weights, ref_p, vertices, sdf, tets, curr_tet, 0.0f);

    // 1-4 feature are sdf at summits of the tetrahedron
    int id_list[4] = { 0,1,2,3 };
    for (int i = 0; i < 4; i++) {
        int curr_id = int(round(curand_uniform(my_curandstate + (idx + i) % 256) * (3.0f - float(i))));
        int tmp = id_list[i];
        id_list[i] = id_list[i + curr_id];
        id_list[i + curr_id] = tmp;
        curr_features[1 + id_list[i]] = sdf[tet_ids[i]];
    }

    // 5-8 feature are sdf at summits of the adjacent tetrahedra
    // feat 5 + i corresponds to sdf on adjacent tet opposite to the ith summit, (feat[1+i])
    int n_tet_ids[4] {};
    int n_id[4] = {-1, -1, -1, -1};
    for (int i = 0; i < 4; i++) {
        int curr_id = int(round(curand_uniform(my_curandstate + (idx + 4 + i) % 256) * (3.0f - float(i))));
        int tmp = id_list[i];
        id_list[i] = id_list[i + curr_id];
        id_list[i + curr_id] = tmp;

        int n_tet = neighbors[4 * curr_tet + i];
        n_tet_ids[0] = tets[4*n_tet]; n_tet_ids[1] = tets[4*n_tet + 1]; n_tet_ids[2] = tets[4*n_tet + 2]; 
        n_tet_ids[3] = tets[4*n_tet] ^ tets[4*n_tet+1] ^ tets[4*n_tet+2] ^ tets[4*n_tet+3];
        //search summit that is not in current tetrahedron
        for (int j = 0; j < 4; j++) {
            if (n_tet_ids[j] != tet_ids[0] && n_tet_ids[j] != tet_ids[1] &&
                n_tet_ids[j] != tet_ids[2] && n_tet_ids[j] != tet_ids[3]) {
                curr_features[5 + id_list[i]] = sdf[n_tet_ids[j]];
                n_id[i] = n_tet_ids[j];
                break;
            }
        }
    }

    if (n_id[0] == -1 || n_id[1] == -1 || n_id[2] == -1 || n_id[3] == -1) {
        geo_feat[size_feat * idx] = 20.0f;
        for (int i = 1; i < size_feat; i++)
            geo_feat[size_feat * idx + i] = 0.0f;
        return;
    }

    // 9-21 feature are 3D vectors difference of sdf at opposite summits

    for (int i = 0; i < 4; i++) {
        int curr_id = int(round(curand_uniform(my_curandstate + (idx + 8 + i) % 256) * (3.0f - float(i))));
        int tmp = id_list[i];
        id_list[i] = id_list[i + curr_id];
        id_list[i + curr_id] = tmp;

        float vec[3] = { vertices[3 * n_id[i]] - vertices[3 * tet_ids[i]],
                        vertices[3 * n_id[i] + 1] - vertices[3 * tet_ids[i] + 1], 
                        vertices[3 * n_id[i] + 2] - vertices[3 * tet_ids[i] + 2]};
        float norm_2 = vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2];

        curr_features[9 + 3 * id_list[i]] = (curr_features[5 + i] - curr_features[1 + i]) * vec[0]/norm_2;
        curr_features[9 + 3 * id_list[i] + 1] = (curr_features[5 + i] - curr_features[1 + i]) * vec[1] / norm_2;
        curr_features[9 + 3 * id_list[i] + 2] = (curr_features[5 + i] - curr_features[1 + i]) * vec[2] / norm_2;
    }
}


__global__ void knn_interpolate_kernel(
    const size_t num_sites,                // number of rays
    const size_t num_knn,  
    float sigma,
    const size_t dim_sdf,
    double *__restrict__ vertices_src,     // [N_voxels, 4] for each voxel => it's vertices
    double *__restrict__ vertices_trg,     // [N_voxels, 4] for each voxel => it's vertices
    float *__restrict__ sdf,     // [N_voxels, 4] for each voxel => it's vertices
    int *__restrict__ neighbors,     // [N_voxels, 4] for each voxel => it's vertices)
    float *__restrict__ sdf_out
    )
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_sites)
    {
        return;
    }

    double total_weight = 0.0;
    double total_sdf = 0.0;

    double length_edge, length_feat, length_o = 0.0;
    int knn_id;

    double curr_point[3] {vertices_trg[3*idx], vertices_trg[3*idx + 1], vertices_trg[3*idx + 2]};
    double curr_edge[3] {};

    for (int i = 0; i < num_knn; i++) {
        knn_id = neighbors[num_knn*idx + i];
        if (knn_id == -1)
            continue;
        
        curr_edge[0] = (curr_point[0] - vertices_src[3*knn_id]);
        curr_edge[1] = (curr_point[1] - vertices_src[3*knn_id + 1]);
        curr_edge[2] = (curr_point[2] - vertices_src[3*knn_id + 2]);
        length_edge = curr_edge[0]*curr_edge[0] + curr_edge[1]*curr_edge[1] + curr_edge[2]*curr_edge[2];
        
        for (int i = 0; i < dim_sdf; i++) {
            total_sdf = total_sdf + exp(-length_edge/double(sigma*sigma)) * double(sdf[dim_sdf*knn_id + i]);
            total_weight = total_weight + exp(-length_edge/double(sigma*sigma));
        }
    }
    
    for (int i = 0; i < dim_sdf; i++) {
        sdf_out[dim_sdf*idx + i] = total_weight < 1.0e-12 ? sdf[dim_sdf*knn_id + i] : float(total_sdf / total_weight);
    }

    return;
}

__global__ void knn_smooth_kernel_o(
    const size_t num_sites,                // number of rays
    const size_t num_knn,  
    const size_t num_lvl,  
    float sigma,
    float *__restrict__ vertices,     // [N_voxels, 4] for each voxel => it's vertices
    int *__restrict__ activated,     // [N_voxels, 4] for each voxel => it's vertices
    float *__restrict__ sdf,     // [N_voxels, 4] for each voxel => it's vertices
    int *__restrict__ neighbors,     // [N_voxels, 4] for each voxel => it's vertices)
    float *__restrict__ sdf_smooth
    )
{
    const size_t idx = blockIdx.x;
    if (idx >= num_sites) // || threadIdx.x >= 32)
    {
        return;
    }

    if (activated[idx] == 0)
        return;

    float length_edge = 0.0f;
    int knn_id;

    float3 curr_point = make_float3(vertices[3*idx],vertices[3*idx+1],vertices[3*idx+2]); //vertices[idx]; //make_float3(vertices[3*idx],vertices[3*idx+1],vertices[3*idx+2]);
    float3 curr_edge = make_float3(0.0,0.0,0.0);
    float max_dist = -1.0f;

    __shared__ float2 smem[96];
    
    int lvl_curr = threadIdx.x / num_knn;
    int i_curr = threadIdx.x % num_knn;
    knn_id = neighbors[num_knn*num_lvl*idx + lvl_curr*num_knn + i_curr];
    if (knn_id != -1) { //} && threadIdx.x < 16) {
        if (activated[knn_id] == 2) {
            curr_edge = curr_point - make_float3(vertices[3*knn_id],vertices[3*knn_id+1],vertices[3*knn_id+2]); //vertices[knn_id]; //make_float3(vertices[3*knn_id],vertices[3*knn_id+1],vertices[3*knn_id+2]);
            length_edge = dot(curr_edge, curr_edge);
            if (length_edge >= max_dist) {
                if (length_edge > max_dist)
                    max_dist = length_edge;
                
                smem[num_knn*lvl_curr + i_curr] = make_float2(exp(-length_edge/(sigma*sigma)) * sdf[knn_id], exp(-length_edge/(sigma*sigma)));
            } 
        } else {
            smem[num_knn*lvl_curr + i_curr] = make_float2(0.0f, 0.0f);
        }        
    } else {
        smem[num_knn*lvl_curr + i_curr] = make_float2(0.0f, 0.0f);
    }

    __syncthreads();
    if (i_curr < 16 && i_curr + 16 < num_knn) {
        smem[num_knn*lvl_curr + i_curr] = smem[num_knn*lvl_curr + i_curr] + smem[num_knn*lvl_curr + i_curr + 16]; 
    }
    __syncthreads();
    if (i_curr < 8) {
        smem[num_knn*lvl_curr + i_curr] = smem[num_knn*lvl_curr + i_curr] + smem[num_knn*lvl_curr + i_curr + 8]; 
    }
    __syncthreads();
    if (i_curr < 4) {
        smem[num_knn*lvl_curr + i_curr] = smem[num_knn*lvl_curr + i_curr] + smem[num_knn*lvl_curr + i_curr + 4]; 
    }
    __syncthreads();
    if (i_curr < 2) {
        smem[num_knn*lvl_curr + i_curr] = smem[num_knn*lvl_curr + i_curr] + smem[num_knn*lvl_curr + i_curr + 2]; 
    }
    __syncthreads();
    
    if (threadIdx.x  == 0) {
        float2 total_sdf = smem[0] + smem[1];
        if (num_lvl > 1)
            total_sdf = total_sdf + smem[num_knn] + smem[num_knn + 1];
        if (num_lvl > 2)
            total_sdf = total_sdf + smem[2*num_knn] + smem[2*num_knn + 1]; 
        sdf_smooth[idx] = total_sdf.y == 0.0f ? 0.0f : total_sdf.x/total_sdf.y;
    }
    
    return;
}

__global__ void knn_smooth_kernel(
    const size_t num_sites,                // number of rays
    const size_t num_knn,  
    float sigma,
    float sigma_feat,
    const size_t dim_sdf,
    double3 *__restrict__ vertices,     // [N_voxels, 4] for each voxel => it's vertices
    int *__restrict__ activated,     // [N_voxels, 4] for each voxel => it's vertices
    float *__restrict__ grads,     // [N_voxels, 4] for each voxel => it's vertices
    float *__restrict__ sdf,     // [N_voxels, 4] for each voxel => it's vertices
    float *__restrict__ feat,     // [N_voxels, 4] for each voxel => it's vertices
    int *__restrict__ neighbors,     // [N_voxels, 4] for each voxel => it's vertices)
    float *__restrict__ sdf_smooth
    )
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_sites)
    {
        return;
    }

    if (activated[idx] == 0)
        return;

    double total_weight = 0.0;
    double total_sdf = 0.0;

    int nb_lvl = 2;//fmin(3, num_knn / 32); //num_knn / 32; //fmin(2, num_knn / 32);
    double length_edge = 0.0;
    int knn_id;

    double3 curr_point = vertices[idx]; 
    double3 curr_edge = make_double3(0.0,0.0,0.0);

    float radius = 2.0f*sigma;
    double max_dist = -1.0;
    for (int lvl_curr = 0; lvl_curr < nb_lvl; lvl_curr++) {
        for (int i = 0; i < 16; i++) {
            knn_id = neighbors[num_knn*idx + lvl_curr*16 + i];
            if (knn_id == -1)
                break;

            curr_edge = curr_point - vertices[knn_id];
            
            length_edge = dot(curr_edge, curr_edge);
            
            if (length_edge < max_dist)
                continue;

            if (length_edge > max_dist)
                max_dist = length_edge;
            
            for (int i = 0; i < dim_sdf; i++) {
                total_sdf = total_sdf + exp(-length_edge/double(sigma*sigma)) * double(sdf[dim_sdf*knn_id + i]);
                total_weight = total_weight + exp(-length_edge/double(sigma*sigma));
            }
        }
        radius = 2.0f*radius;
    }
    
    for (int i = 0; i < dim_sdf; i++) {
        sdf_smooth[dim_sdf*idx + i] = total_weight == 0.0 ? sdf[dim_sdf*idx + i] : float(total_sdf / total_weight);
    }

    return;
}

__global__ void bnn_smooth_kernel(
    const size_t num_sites,                // number of rays
    float sigma,
    const size_t dim_sdf,
    float *__restrict__ vertices,     // [N_voxels, 4] for each voxel => it's vertices
    float *__restrict__ sdf,     // [N_voxels, 4] for each voxel => it's vertices
    int *__restrict__ bnn_sites,     // [N_voxels, 4] for each voxel => it's vertices)
    int *__restrict__ bnn_offset,     // [N_voxels, 4] for each voxel => it's vertices)
    float *__restrict__ sdf_smooth
    )
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_sites)
    {
        return;
    }

    float total_weight = 0.0f;
    float total_sdf = 0.0f;

    float length_edge;
    int bnn_id;
    int start = bnn_offset[2*idx];
    int end = bnn_offset[2*idx+1];
    for (int i = start; i < start+end; i++) {
        bnn_id = bnn_sites[i];
        length_edge = (vertices[3*idx] - vertices[3*bnn_id])*(vertices[3*idx] - vertices[3*bnn_id]) +
                            (vertices[3*idx + 1] - vertices[3*bnn_id + 1])*(vertices[3*idx + 1] - vertices[3*bnn_id + 1]) +
                            (vertices[3*idx + 2] - vertices[3*bnn_id + 2])*(vertices[3*idx + 2] - vertices[3*bnn_id + 2]);
              
        for (int i = 0; i < dim_sdf; i++) {
            total_sdf = total_sdf + exp(-length_edge/(sigma*sigma)) * sdf[dim_sdf*bnn_id + i];
            total_weight = total_weight + exp(-length_edge/(sigma*sigma));
        }
    }
    
    for (int i = 0; i < dim_sdf; i++) {
        sdf_smooth[dim_sdf*idx + i] = total_weight == 0.0f ? 0.0f : total_sdf / total_weight;
    }

    return;
}


__global__ void smooth_kernel(
    const size_t num_edges,                // number of rays
    float sigma,
    const size_t dim_sdf,
    float *__restrict__ vertices,     // [N_voxels, 4] for each voxel => it's vertices
    float *__restrict__ sdf,     // [N_voxels, 4] for each voxel => it's vertices
    float *__restrict__ feat,     // [N_voxels, 4] for each voxel => it's vertices
    int *__restrict__ edges,     // [N_voxels, 4] for each voxel => it's vertices)
    float *__restrict__ sdf_smooth,    // [N_voxels, 4] for each voxel => it's vertices
    float *__restrict__ counter     // [N_voxels, 4] for each voxel => it's vertices
    )
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_edges)
    {
        return;
    }
    
    //float length_edge = (vertices[3*edges[2*idx]] - vertices[3*edges[2*idx+1]])*(vertices[3*edges[2*idx]] - vertices[3*edges[2*idx+1]]) +
    //                        (vertices[3*edges[2*idx] + 1] - vertices[3*edges[2*idx+1] + 1])*(vertices[3*edges[2*idx] + 1] - vertices[3*edges[2*idx+1] + 1]) +
    //                        (vertices[3*edges[2*idx] + 2] - vertices[3*edges[2*idx+1] + 2])*(vertices[3*edges[2*idx] + 2] - vertices[3*edges[2*idx+1] + 2]);

    float length_edge = 0.0f;
    // add bilateral smooth term with features
    float length_feat = 0.0f;
    /*if (dim_sdf > 1) {
        for (int i = 0; i < DIM_L_FEAT; i++) {
            length_feat = length_feat +  (feat[DIM_L_FEAT*edges[2*idx]+ i] - feat[DIM_L_FEAT*edges[2*idx+1]+ i])*
                                            (feat[DIM_L_FEAT*edges[2*idx]+ i] - feat[DIM_L_FEAT*edges[2*idx+1]+ i]);
        }
    }*/
              
    for (int i = 0; i < dim_sdf; i++) {
        if (sdf[dim_sdf*edges[2*idx + 1] + i] != 0.0f)
            atomicAdd(&sdf_smooth[dim_sdf*edges[2*idx] + i], exp(-length_edge/(sigma*sigma)) * (sdf[dim_sdf*edges[2*idx] + i] - sdf[dim_sdf*edges[2*idx + 1] + i])); //exp(-length_edge/(sigma*sigma) - length_feat/(0.05f)) * sdf[dim_sdf*edges[2*idx + 1] + i]);          
        
        if (sdf[dim_sdf*edges[2*idx] + i] != 0.0f)
            atomicAdd(&sdf_smooth[dim_sdf*edges[2*idx + 1] + i], -exp(-length_edge/(sigma*sigma)) * (sdf[dim_sdf*edges[2*idx] + i] - sdf[dim_sdf*edges[2*idx + 1] + i])); //exp(-length_edge/(sigma*sigma) - length_feat/(0.05f)) * sdf[dim_sdf*edges[2*idx] + i]);
    }

    if (sdf[dim_sdf*edges[2*idx + 1]] != 0.0f)
        atomicAdd(&counter[edges[2*idx]], exp(-length_edge/(sigma*sigma)));//exp(-length_edge/(sigma*sigma) - length_feat/(0.05f)));
    if (sdf[dim_sdf*edges[2*idx]] != 0.0f)
        atomicAdd(&counter[edges[2*idx + 1]], exp(-length_edge/(sigma*sigma)));// exp(-length_edge/(sigma*sigma) - length_feat/(0.05f)));

    return;
}

__global__ void normalize_kernel(
    const size_t num_sites,                // number of rays
    const size_t dim_sdf,
    float *__restrict__ sdf_smooth,     // [N_voxels, 4] for each voxel => it's vertices
    float *__restrict__ weights     // [N_voxels, 4] for each voxel => it's vertices
    ) 
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_sites)
    {
        return;
    }

    if (weights[idx] < 0.3f)
        return;

    for (int i = 0; i < dim_sdf; i++) 
        sdf_smooth[dim_sdf*idx + i] = sdf_smooth[dim_sdf*idx + i]/weights[idx];
}


__global__ void sdf_smooth_kernel(
    const size_t num_sites,   
    const size_t dim_sdf,
    int *__restrict__ activated,     // [N_voxels, 4] for each voxel => it's vertices
    float *__restrict__ sdf,     // [N_voxels, 4] for each voxel => it's vertices
    float *__restrict__ sdf_smooth,
    float *__restrict__ grads
    )
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_sites)
    {
        return;
    }

    if (activated[idx] == 0)
        return;
    
    for (int i = 0; i < dim_sdf; i++) {
        grads[dim_sdf*idx + i] = 2.0*(sdf[dim_sdf*idx + i] - sdf_smooth[dim_sdf*idx + i]);
    }

    return;
}



__global__ void space_reg_kernel(
    const size_t num_rays,                // number of rays
    const float *__restrict__ rays_d,       // [N_rays, 6]
    const float *__restrict__ grad_space,       // [N_rays, 6]
    const float *__restrict__ out_weights,       // [N_rays, 6]
    const float *__restrict__ out_z,       // [N_rays, 6]
    const float *__restrict__ out_sdf,       // [N_rays, 6]
    const float *__restrict__ out_feat,       // [N_rays, 6]
    const int *__restrict__ out_ids,     // [N_voxels, 4] for each voxel => it's vertices)
    const int *__restrict__ offset,     // [N_voxels, 4] for each voxel => it's vertices
    float *__restrict__ sdf_grad,    // [N_voxels, 4] for each voxel => it's vertices
    float *__restrict__ feat_grad
    )
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rays)
    {
        return;
    }

    float direction[3] {rays_d[idx * 3], rays_d[idx * 3 + 1], rays_d[idx * 3 + 2]};

    float grad_in[3] {0.0f, 0.0f, 0.0f};
    float grad_exit[3] {0.0f, 0.0f, 0.0f};
    float grad_curr[3] {0.0f, 0.0f, 0.0f};
    float vec_curr[3] {0.0f, 0.0f, 0.0f};

    float diff_sdf = 0.0f;
    float err_diff, weight_color;

    int start = offset[2*idx];
    int end = offset[2*idx+1];
    int s_id = 0;
    for (int i = start; i < start+end; i++) {
        for (int j = 0; j < 3; j++) {
            grad_in[j] = grad_space[3*out_ids[6*i] + j]*out_weights[7*i] + 
                            grad_space[3*out_ids[6*i+1] + j]*out_weights[7*i+1] + 
                            grad_space[3*out_ids[6*i+2] + j]*out_weights[7*i+2];
                            
            grad_exit[j] = grad_space[3*out_ids[6*i + 3] + j]*out_weights[7*i + 3] + 
                            grad_space[3*out_ids[6*i + 3 + 1] + j]*out_weights[7*i + 3 + 1] + 
                            grad_space[3*out_ids[6*i + 3 + 2] + j]*out_weights[7*i + 3 + 2];
            
            grad_curr[j] = (grad_in[j] + grad_exit[j]) / 2.0f;
        }

        float norm = sqrt(grad_curr[0]*grad_curr[0] + grad_curr[1]*grad_curr[1] + grad_curr[2]*grad_curr[2]);
        if (norm == 0.0f)
            continue;
        grad_curr[0] = grad_curr[0]/norm;  grad_curr[1] = grad_curr[1]/norm; grad_curr[2] = grad_curr[2]/norm; 

        vec_curr[0] = direction[0] * (out_z[2*i + 1] - out_z[2*i]);
        vec_curr[1] = direction[1] * (out_z[2*i + 1] - out_z[2*i]);
        vec_curr[2] = direction[2] * (out_z[2*i + 1] - out_z[2*i]);

        diff_sdf = grad_curr[0]*vec_curr[0] + grad_curr[1]*vec_curr[1] + grad_curr[2]*vec_curr[2];

        err_diff = ((out_sdf[2*i + 1] - out_sdf[2*i]) - diff_sdf) ;
        for (int j = 0; j < 3; j++) {            
            atomicAdd(&sdf_grad[out_ids[6*i + j]], out_weights[7*i + j] * err_diff);
            atomicAdd(&sdf_grad[out_ids[6*i + 3 + j]], -out_weights[7*i + 3 + j] * err_diff);
        }

        weight_color = fabs(grad_curr[0]*direction[0] + grad_curr[1]*direction[1] + grad_curr[2]*direction[2]);
    
        for (int j = 0; j < 3; j++) {            
            for (int k = 0; k < 6; k++) {
                atomicAdd(&feat_grad[6*out_ids[6*i + j] + k], out_weights[7*i + j] * weight_color * (out_feat[12*i + 6 + k] - out_feat[12*i + k]));
                atomicAdd(&feat_grad[6*out_ids[6*i + 3 + j] + k], -out_weights[7*i + 3 + j] * weight_color * (out_feat[12*i + 6 + k] - out_feat[12*i + k]));
            }
        }
    }


    return;
}

__global__ void activate_sites_kernel(
    const size_t num_rays,                // number of rays
    const size_t num_knn, 
    const int *__restrict__ cell_ids,
    const int *__restrict__ offsets,
    int *__restrict__ activated
    )
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rays)
    {
        return;
    }
    
    int knn_id, id_prev, id;
    int start = offsets[2 * idx];
    int end = offsets[2 * idx + 1];
    for (int t = start; t < start + end; t++) {      
        for (int i = 0; i < 6; i++) {
            id = cell_ids[6 * t + i];
            //activated[id] = 1;
            atomicExch(&activated[id], 1);
        }        
    }
    return;
}

__global__ void activate_knn_kernel(
    const size_t num_sites,                // number of rays
    const size_t num_knn, 
    const int *__restrict__ neighbors,     
    const int *__restrict__ activated_buff,     
    int *__restrict__ activated
    )
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_sites)
    {
        return;
    }

    if (activated_buff[idx] == 0)
        return;
    
    int knn_id, id_prev, id;

    for (int j = 0; j < num_knn; j++) {
        knn_id = neighbors[num_knn*idx + j];
        if (knn_id == -1)
            continue;
        activated[knn_id] = 1;
        //atomicExch(&activated[knn_id], 1);
    }    

    return;
}

__global__ void activate_knn_kernel_o(
    const size_t num_sites,                // number of rays
    const size_t num_knn, 
    const int *__restrict__ neighbors,     
    const int *__restrict__ activated_buff,     
    int *__restrict__ activated
    )
{
    const size_t id_thread = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t idx = id_thread / num_knn;
    const size_t idx_k = id_thread % num_knn;
    if (id_thread >= num_knn*num_sites) // || idx_k >= 16)
    {
        return;
    }

    if (activated_buff[idx] == 0)
        return;
    
    int knn_id = neighbors[num_knn*idx + idx_k];

    if (knn_id != -1)
        atomicExch(&activated[knn_id], 1);
        //activated[knn_id] = 1;

    return;
}


/** CPU functions **/
/** CPU functions **/
/** CPU functions **/

// *************************
void backprop_feat_cuda(
    size_t num_samples,
    size_t num_sites,
    size_t dim_feats,
    torch::Tensor sdf,
    torch::Tensor grad_feat,
    torch::Tensor counter,
    torch::Tensor grad_samples,
    torch::Tensor cell_ids,
    torch::Tensor cell_weights 
)
{
    const int threads = 1024;
    const int blocks = (num_samples + threads - 1) / threads; // ceil for example 8192 + 255 / 256 = 32
    AT_DISPATCH_FLOATING_TYPES( grad_feat.type(),"backprop_feat_kernel", ([&] {  
        backprop_feat_kernel CUDA_KERNEL(blocks,threads) (
            num_samples,
            dim_feats,
            sdf.data_ptr<float>(),
            grad_feat.data_ptr<float>(),
            counter.data_ptr<float>(),
            grad_samples.data_ptr<float>(),
            cell_ids.data_ptr<int>(),
            cell_weights.data_ptr<float>());
    }));
    cudaDeviceSynchronize();

    /*AT_DISPATCH_FLOATING_TYPES( grad_feat.type(),"backprop_feat_kernel_o", ([&] {  
        backprop_feat_kernel_o CUDA_KERNEL(blocks,threads) (
            num_samples,
            dim_feats,
            (float3*)thrust::raw_pointer_cast(sdf.data_ptr<float>()),
            (float4*)thrust::raw_pointer_cast(grad_feat.data_ptr<float>()),
            counter.data_ptr<float>(),
            (float4*)thrust::raw_pointer_cast(grad_samples.data_ptr<float>()),
            (int3*)thrust::raw_pointer_cast(cell_ids.data_ptr<int>()),
            (float3*)thrust::raw_pointer_cast(cell_weights.data_ptr<float>()));
    }));
    cudaDeviceSynchronize();*/
    
    /*const int threads2 = 512;
    const int blocks2 = (num_sites + threads2 - 1) / threads2; // ceil for example 8192 + 255 / 256 = 32
    AT_DISPATCH_FLOATING_TYPES( grad_feat.type(),"normalize_kernel", ([&] {  
        normalize_kernel CUDA_KERNEL(blocks2,threads2) (
            num_sites,
            dim_feats,
            grad_feat.data_ptr<float>(),
            counter.data_ptr<float>());
    }));
    cudaDeviceSynchronize();*/
}

// *************************
void backprop_grad_cuda(
    size_t num_samples,
    size_t num_sites,
    torch::Tensor sdf,
    torch::Tensor grad_sites,
    torch::Tensor grad_samples,
    torch::Tensor cell_ids,
    torch::Tensor cell_weights 
)
{
    const int threads = 1024;
    const int blocks = (num_samples + threads - 1) / threads; // ceil for example 8192 + 255 / 256 = 32
    AT_DISPATCH_FLOATING_TYPES( grad_sites.type(),"backprop_grad_kernel", ([&] {  
        backprop_grad_kernel CUDA_KERNEL(blocks,threads) (
            num_samples,
            sdf.data_ptr<float>(),
            grad_sites.data_ptr<float>(),
            grad_samples.data_ptr<float>(),
            cell_ids.data_ptr<int>(),
            cell_weights.data_ptr<float>());
    }));
    cudaDeviceSynchronize();

    /*AT_DISPATCH_FLOATING_TYPES( grad_sites.type(),"backprop_grad_kernel_o", ([&] {  
        backprop_grad_kernel_o CUDA_KERNEL(blocks,threads) (
            num_samples,
            (float3*)thrust::raw_pointer_cast(sdf.data_ptr<float>()),
            (float3*)thrust::raw_pointer_cast(grad_sites.data_ptr<float>()),
            (float3*)thrust::raw_pointer_cast(grad_samples.data_ptr<float>()),
            (int3*)thrust::raw_pointer_cast(cell_ids.data_ptr<int>()),
            (float3*)thrust::raw_pointer_cast(cell_weights.data_ptr<float>()));
    }));
    cudaDeviceSynchronize();*/
}


// *************************
void backprop_sdf_cuda(
    size_t num_samples,
    torch::Tensor sdf,
    torch::Tensor grad_sdf,
    torch::Tensor grad_sdf_samples,
    torch::Tensor cell_ids,
    torch::Tensor cell_weights 
)
{
    const int threads = 512;
    const int blocks = (num_samples + threads - 1) / threads; // ceil for example 8192 + 255 / 256 = 32
    AT_DISPATCH_FLOATING_TYPES( grad_sdf.type(),"backprop_sdf_kernel", ([&] {  
        backprop_sdf_kernel CUDA_KERNEL(blocks,threads) (
            num_samples,
            (float4*)thrust::raw_pointer_cast(sdf.data_ptr<float>()),
            grad_sdf.data_ptr<float>(),
            (float2*)thrust::raw_pointer_cast(grad_sdf_samples.data_ptr<float>()),
            (int3*)thrust::raw_pointer_cast(cell_ids.data_ptr<int>()),
            cell_weights.data_ptr<float>());
    }));
    cudaDeviceSynchronize();
}

void  backprop_norm_cuda(
    size_t num_tets,
    torch::Tensor tets,
    torch::Tensor sites,
    torch::Tensor vol,    
    torch::Tensor weights,  
    torch::Tensor weights_tot, 
    torch::Tensor grad_norm ,
    torch::Tensor grad_sdf,
    torch::Tensor activated 
)
{
    const int threads = 1024;
    const int blocks = (num_tets + threads - 1) / threads; // ceil for example 8192 + 255 / 256 = 32
    AT_DISPATCH_FLOATING_TYPES( grad_sdf.type(),"backprop_norm_kernel", ([&] {  
        backprop_norm_kernel CUDA_KERNEL(blocks,threads) (
            num_tets,
            tets.data_ptr<int>(),
            sites.data_ptr<float>(),
            vol.data_ptr<float>(),
            weights.data_ptr<float>(),
            weights_tot.data_ptr<float>(),
            grad_norm.data_ptr<float>(),
            grad_sdf.data_ptr<float>(),
            activated.data_ptr<int>());
    }));
    //cudaDeviceSynchronize();
}


void  backprop_unit_norm_cuda(
    size_t num_tets,
    torch::Tensor tets,
    torch::Tensor sites,
    torch::Tensor norm_grad,
    torch::Tensor grad_unormed,
    torch::Tensor vol,    
    torch::Tensor weights,  
    torch::Tensor weights_tot, 
    torch::Tensor grad_norm ,
    torch::Tensor grad_sdf,
    torch::Tensor activated 
)
{
    const int threads = 1024;
    const int blocks = (num_tets + threads - 1) / threads; // ceil for example 8192 + 255 / 256 = 32
    AT_DISPATCH_FLOATING_TYPES( grad_sdf.type(),"backprop_unit_norm_kernel", ([&] {  
        backprop_unit_norm_kernel CUDA_KERNEL(blocks,threads) (
            num_tets,
            tets.data_ptr<int>(),
            sites.data_ptr<float>(),
            norm_grad.data_ptr<float>(),
            grad_unormed.data_ptr<float>(),
            vol.data_ptr<float>(),
            weights.data_ptr<float>(),
            weights_tot.data_ptr<float>(),
            grad_norm.data_ptr<float>(),
            grad_sdf.data_ptr<float>(),
            activated.data_ptr<int>());
    }));
    cudaDeviceSynchronize();
}

// *************************
float eikonal_loss_cuda(
    size_t num_sites,
    size_t num_knn,
    torch::Tensor neighbors,
    torch::Tensor Weights,
    torch::Tensor grad,
    torch::Tensor grad_eik 
)
{
    float* loss;
    cudaMalloc((void**)&loss, sizeof(float));
    cudaMemset(loss, 0, sizeof(float));

    const int threads = 1024;
    const int blocks = (num_sites + threads - 1) / threads; // ceil for example 8192 + 255 / 256 = 32
    AT_DISPATCH_FLOATING_TYPES( grad.type(),"eikonal_loss_cuda", ([&] {  
        eikonal_loss_kernel CUDA_KERNEL(blocks,threads) (
            num_sites,
            num_knn,
            neighbors.data_ptr<int>(),
            Weights.data_ptr<float>(),
            grad.data_ptr<float>(),
            grad_eik.data_ptr<float>(),
            loss);
    }));
    
    //cudaDeviceSynchronize();
    float res = 0.0f;
    cudaMemcpy(&res, loss, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(loss);

    return res/float(num_sites);
}

// *************************
void smooth_cuda(
    size_t num_edges,
    float sigma,
    float dim,
    torch::Tensor vertices,
    torch::Tensor activated,
    torch::Tensor sdf,
    torch::Tensor feat,
    torch::Tensor edges,
    torch::Tensor sdf_grad,
    torch::Tensor feat_grad,
    torch::Tensor counter 
)
{
    const int threads = 1024;
    const int blocks = (num_edges + threads - 1) / threads; // ceil for example 8192 + 255 / 256 = 32
    AT_DISPATCH_FLOATING_TYPES( sdf.type(),"smooth_grad_n_kernel", ([&] {  
        smooth_grad_n_kernel CUDA_KERNEL(blocks,threads) (
            num_edges,
            sigma,
            dim,
            vertices.data_ptr<float>(),
            activated.data_ptr<int>(),
            sdf.data_ptr<float>(),
            feat.data_ptr<float>(),
            edges.data_ptr<int>(),
            sdf_grad.data_ptr<float>(),
            feat_grad.data_ptr<float>(),
            counter.data_ptr<float>());
    }));
    cudaDeviceSynchronize();
    
}

// *************************
void space_reg_cuda(
    size_t num_rays,
    torch::Tensor rays_d,
    torch::Tensor grad_space,
    torch::Tensor out_weights,
    torch::Tensor out_z,
    torch::Tensor out_sdf,
    torch::Tensor out_feat,
    torch::Tensor out_ids,
    torch::Tensor offset,
    torch::Tensor sdf_grad,
    torch::Tensor feat_grad 
)
{
    const int threads = 1024;
    const int blocks = (num_rays + threads - 1) / threads; // ceil for example 8192 + 255 / 256 = 32
    AT_DISPATCH_FLOATING_TYPES( out_sdf.type(),"space_reg_kernel", ([&] {  
        space_reg_kernel CUDA_KERNEL(blocks,threads) (
            num_rays,                // number of rays
            rays_d.data_ptr<float>(),       // [N_rays, 6]
            grad_space.data_ptr<float>(),       // [N_rays, 6]
            out_weights.data_ptr<float>(),       // [N_rays, 6]
            out_z.data_ptr<float>(),       // [N_rays, 6]
            out_sdf.data_ptr<float>(),       // [N_rays, 6]
            out_feat.data_ptr<float>(),       // [N_rays, 6]
            out_ids.data_ptr<int>(),     // [N_voxels, 4] for each voxel => it's vertices)
            offset.data_ptr<int>(),     // [N_voxels, 4] for each voxel => it's vertices)
            sdf_grad.data_ptr<float>(),    // [N_voxels, 4] for each voxel => it's vertices
            feat_grad.data_ptr<float>());
    }));
    
}

// *************************
void sdf_smooth_cuda(
    size_t num_sites,
    size_t dim_sdf,
    torch::Tensor activated,
    torch::Tensor sdf,
    torch::Tensor sdf_smooth,
    torch::Tensor grads 
)
{
    const int threads = 1024;
    const int blocks = (num_sites + threads - 1) / threads; // ceil for example 8192 + 255 / 256 = 32
    AT_DISPATCH_FLOATING_TYPES( sdf.type(),"sdf_smooth_kernel", ([&] {  
        sdf_smooth_kernel CUDA_KERNEL(blocks,threads) (
            num_sites,   
            dim_sdf,
            activated.data_ptr<int>(),     // [N_voxels, 4] for each voxel => it's vertices
            sdf.data_ptr<float>(),     // [N_voxels, 4] for each voxel => it's vertices
            sdf_smooth.data_ptr<float>(),
            grads.data_ptr<float>());
    }));
    cudaDeviceSynchronize();
}

// *************************
void smooth_sdf_cuda(
    size_t num_edges,
    size_t num_sites,
    float sigma,
    size_t dim_sdf,
    torch::Tensor vertices,
    torch::Tensor sdf,
    torch::Tensor feat,
    torch::Tensor edges,
    torch::Tensor sdf_smooth,
    torch::Tensor counter 
)
{
    const int threads = 1024;
    const int blocks = (num_edges + threads - 1) / threads; // ceil for example 8192 + 255 / 256 = 32
    AT_DISPATCH_FLOATING_TYPES( sdf.type(),"smooth_kernel", ([&] {  
        smooth_kernel CUDA_KERNEL(blocks,threads) (
            num_edges,                // number of rays
            sigma,
            dim_sdf,
            vertices.data_ptr<float>(),       // [N_rays, 6]
            sdf.data_ptr<float>(),       // [N_rays, 6]
            feat.data_ptr<float>(),       // [N_rays, 6]
            edges.data_ptr<int>(),     // [N_voxels, 4] for each voxel => it's vertices)
            sdf_smooth.data_ptr<float>(),     // [N_voxels, 4] for each voxel => it's vertices)
            counter.data_ptr<float>());
    }));
    cudaDeviceSynchronize();

    const int threads2 = 1024;
    const int blocks2 = (num_sites + threads2 - 1) / threads2; // ceil for example 8192 + 255 / 256 = 32
    AT_DISPATCH_FLOATING_TYPES( sdf.type(),"normalize_kernel", ([&] {  
        normalize_kernel CUDA_KERNEL(blocks2,threads2) (
            num_sites,                
            dim_sdf,
            sdf_smooth.data_ptr<float>(), 
            counter.data_ptr<float>());
    }));
    
    cudaDeviceSynchronize();
}

void bnn_smooth_sdf_cuda(
    size_t num_sites,
    float sigma,
    size_t dim_sdf,
    torch::Tensor vertices,
    torch::Tensor sdf,
    torch::Tensor bnn_sites,
    torch::Tensor bnn_offset,
    torch::Tensor sdf_smooth
)
{
    const int threads = 1024;
    const int blocks = (num_sites + threads - 1) / threads; // ceil for example 8192 + 255 / 256 = 32
    AT_DISPATCH_FLOATING_TYPES( sdf.type(),"bnn_smooth_kernel", ([&] {  
        bnn_smooth_kernel CUDA_KERNEL(blocks,threads) (
            num_sites,                // number of rays
            sigma,
            dim_sdf,
            vertices.data_ptr<float>(),       // [N_rays, 6]
            sdf.data_ptr<float>(),       // [N_rays, 6]
            bnn_sites.data_ptr<int>(),     // [N_voxels, 4] for each voxel => it's vertices)
            bnn_offset.data_ptr<int>(),     // [N_voxels, 4] for each voxel => it's vertices)
            sdf_smooth.data_ptr<float>());
    }));
}

void knn_interpolate_cuda(
    size_t num_sites,
    size_t num_knn,
    float sigma,
    size_t dim_sdf,
    torch::Tensor vertices_src,
    torch::Tensor vertices_trg, 
    torch::Tensor sdf,
    torch::Tensor neighbors,
    torch::Tensor sdf_out
)
{
    const int threads = 512;
    const int blocks = (num_sites + threads - 1) / threads; // ceil for example 8192 + 255 / 256 = 32
    AT_DISPATCH_FLOATING_TYPES( sdf.type(),"knn_interpolate_kernel", ([&] {  
        knn_interpolate_kernel CUDA_KERNEL(blocks,threads) (
            num_sites,                // number of rays
            num_knn,
            sigma,
            dim_sdf,
            vertices_src.data_ptr<double>(),       // [N_rays, 6]
            vertices_trg.data_ptr<double>(),       // [N_rays, 6]
            sdf.data_ptr<float>(),       // [N_rays, 6]
            neighbors.data_ptr<int>(),     // [N_voxels, 4] for each voxel => it's vertices)
            sdf_out.data_ptr<float>());
    }));
    cudaDeviceSynchronize();
}

void knn_smooth_sdf_cuda(
    size_t num_sites,
    size_t num_knn,
    size_t num_lvl,
    float sigma,
    float sigma_feat,
    size_t dim_sdf,
    torch::Tensor vertices,
    torch::Tensor activated, 
    torch::Tensor grads,
    torch::Tensor sdf,
    torch::Tensor feat,
    torch::Tensor neighbors,
    torch::Tensor sdf_smooth
)
{
    const int threads = 512;//1024;
    const int blocks = (num_sites + threads - 1) / threads; // ceil for example 8192 + 255 / 256 = 32
    AT_DISPATCH_FLOATING_TYPES( sdf.type(),"knn_smooth_kernel", ([&] {  
        knn_smooth_kernel CUDA_KERNEL(blocks,threads) (
            num_sites,                // number of rays
            num_knn*num_lvl,
            sigma,
            sigma_feat,
            dim_sdf,
            (double3*)thrust::raw_pointer_cast(vertices.data_ptr<double>()),       // [N_rays, 6]
            activated.data_ptr<int>(),       // [N_rays, 6]
            grads.data_ptr<float>(),       // [N_rays, 6]
            sdf.data_ptr<float>(),       // [N_rays, 6]
            feat.data_ptr<float>(),       // [N_rays, 6]
            neighbors.data_ptr<int>(),     // [N_voxels, 4] for each voxel => it's vertices)
            sdf_smooth.data_ptr<float>());
    }));
    cudaDeviceSynchronize();

    /*const int threads = num_knn*num_lvl;
    const int blocks = num_sites; // ceil for example 8192 + 255 / 256 = 32
    AT_DISPATCH_FLOATING_TYPES( sdf.type(),"knn_smooth_kernel_o", ([&] {  
        knn_smooth_kernel_o CUDA_KERNEL(blocks,threads) (
            num_sites,                // number of rays
            num_knn,
            num_lvl,
            sigma,
            vertices.data_ptr<float>(),  //(float3*)thrust::raw_pointer_cast(vertices.data_ptr<float>()),       // [N_rays, 6]
            activated.data_ptr<int>(),       // [N_rays, 6]
            sdf.data_ptr<float>(),       // [N_rays, 6]
            neighbors.data_ptr<int>(),     // [N_voxels, 4] for each voxel => it's vertices)
            sdf_smooth.data_ptr<float>());
    }));
    cudaDeviceSynchronize();*/
}


void geo_feat_cuda(
    size_t num_samples,
    size_t num_knn,
    float sigma,
    torch::Tensor samples,
    torch::Tensor vertices, 
    torch::Tensor grads,
    torch::Tensor sdf,
    torch::Tensor geo_feat,
    torch::Tensor tets,
    torch::Tensor neighbors,
    torch::Tensor cell_ids
)
{
    /*cudaStream_t computeStream;
    curandState* d_state;
    cudaMalloc(&d_state, 256 * sizeof(curandState));
    setup_kernel << < 1, 256, 0, computeStream >> > (d_state, time(NULL));
    cudaDeviceSynchronize();

    const int threads = 512;
    const int blocks = (num_samples + threads - 1) / threads; // ceil for example 8192 + 255 / 256 = 32
    AT_DISPATCH_FLOATING_TYPES( sdf.type(),"geo_feat_kernel", ([&] {  
        geo_feat_kernel CUDA_KERNEL(blocks,threads) (
            num_samples,                // number of rays
            num_knn,
            sigma,
            samples.data_ptr<float>(),       // [N_rays, 6]
            vertices.data_ptr<float>(),       // [N_rays, 6]
            grads.data_ptr<float>(),       // [N_rays, 6]
            sdf.data_ptr<float>(),       // [N_rays, 6]
            geo_feat.data_ptr<float>(),       // [N_rays, 6]
            d_state,
            neighbors.data_ptr<int>(),       // [N_rays, 6]
            cell_ids.data_ptr<int>());
    }));
    cudaDeviceSynchronize();

    cudaFree(d_state);*/
}


void activate_sites_cuda(
    size_t num_rays,
    size_t num_sites,
    size_t num_knn,
    torch::Tensor cell_ids,
    torch::Tensor offsets,
    torch::Tensor neighbors,
    torch::Tensor activated_buff,
    torch::Tensor activated
)
{
    /*const int threads = 1024;
    const int blocks = (num_rays + threads - 1) / threads; // ceil for example 8192 + 255 / 256 = 32
    AT_DISPATCH_ALL_TYPES( cell_ids.type(),"activate_sites_kernel", ([&] {  
        activate_sites_kernel CUDA_KERNEL(blocks,threads) (
            num_rays,    
            num_knn,
            cell_ids.data_ptr<int>(),     
            offsets.data_ptr<int>(),    
            activated_buff.data_ptr<int>());
    }));

    cudaDeviceSynchronize();*/

    /*const int threads = 1024;
    const int blocks2 = (num_sites + threads - 1) / threads; // ceil for example 8192 + 255 / 256 = 32
    AT_DISPATCH_ALL_TYPES( cell_ids.type(),"activate_knn_kernel", ([&] {  
        activate_knn_kernel CUDA_KERNEL(blocks2,threads) (
            num_sites,    
            num_knn,
            neighbors.data_ptr<int>(),      
            activated_buff.data_ptr<int>(),
            activated.data_ptr<int>());
    }));
    cudaDeviceSynchronize();*/

    const int threads = 1024;
    const int blocks = (num_knn*num_sites + threads - 1) / threads;
    AT_DISPATCH_ALL_TYPES( cell_ids.type(),"activate_knn_kernel_o", ([&] {  
        activate_knn_kernel_o CUDA_KERNEL(blocks,threads) (
            num_sites,    
            num_knn,
            neighbors.data_ptr<int>(),      
            activated_buff.data_ptr<int>(),
            activated.data_ptr<int>());
    }));
    cudaDeviceSynchronize();
}

void backprop_multi_cuda(
    size_t num_sites, 
    size_t num_knn,  
    size_t dim_feat,
    torch::Tensor vertices,     // [N_voxels, 4] for each voxel => it's vertices
    torch::Tensor activated,     // [N_voxels, 4] for each voxel => it's vertices
    torch::Tensor grad_norm,     // [N_voxels, 4] for each voxel => it's vertices
    torch::Tensor grad_norm_feat,     // [N_voxels, 4] for each voxel => it's vertices
    torch::Tensor grad_feat,     // [N_voxels, 4] for each voxel => it's vertices
    torch::Tensor neighbors)
{
    const int threads = 1024;
    const int blocks = (num_sites + threads - 1) / threads; // ceil for example 8192 + 255 / 256 = 32
    AT_DISPATCH_ALL_TYPES( grad_norm.type(),"backprop_multi_kernel", ([&] {  
        backprop_multi_kernel CUDA_KERNEL(blocks,threads) (
            num_sites,    
            num_knn,
            dim_feat,
            vertices.data_ptr<float>(),      
            activated.data_ptr<int>(),
            grad_norm.data_ptr<float>(),
            grad_norm_feat.data_ptr<float>(),
            grad_feat.data_ptr<float>(),
            neighbors.data_ptr<int>());
    }));
    cudaDeviceSynchronize();
}
