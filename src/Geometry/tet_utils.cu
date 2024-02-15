#include <torch/extension.h>

#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

#include <device_launch_parameters.h>

#ifdef __INTELLISENSE__
#define CUDA_KERNEL(...)
#define FAKEINIT = {0}
#else
#define CUDA_KERNEL(...) <<< __VA_ARGS__ >>>
#define FAKEINIT
#endif

#define DIM_ADJ 64
#define DIM_L_FEAT 12

/** Device functions **/
/** Device functions **/
/** Device functions **/
__global__ void upsample_counter_kernel(
    const size_t nb_edges,
    const int *__restrict__ edge,
    const float *__restrict__ sites,
    const float *__restrict__ sdf,
    int *__restrict__ counter)
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nb_edges)
    {
        return;
    }

    float edge_length = sqrt((sites[3*edge[2*idx]] - sites[3*edge[2*idx+1]]) * (sites[3*edge[2*idx]] - sites[3*edge[2*idx+1]]) + 
                               (sites[3*edge[2*idx]+1] - sites[3*edge[2*idx+1]+1]) * (sites[3*edge[2*idx]+1] - sites[3*edge[2*idx+1]+1]) + 
                               (sites[3*edge[2*idx]+2] - sites[3*edge[2*idx+1]+2]) * (sites[3*edge[2*idx]+2] - sites[3*edge[2*idx+1]+2])); 

    if (sdf[edge[2*idx]]*sdf[edge[2*idx+1]] <= 0.0f || fmin(fabs(sdf[edge[2*idx]]), fabs(sdf[edge[2*idx+1]])) < edge_length)
        atomicAdd(counter, 1);

}


__global__ void upsample_kernel(
    const size_t nb_edges,
    const int *__restrict__ edge,
    const float *__restrict__ sites,
    const float *__restrict__ sdf,
    const float *__restrict__ feats,
    float *__restrict__ new_sites,
    float *__restrict__ new_sdf,
    float *__restrict__ new_feats,
    int *__restrict__ counter)
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nb_edges)
    {
        return;
    }

    float edge_length = sqrt((sites[3*edge[2*idx]] - sites[3*edge[2*idx+1]]) * (sites[3*edge[2*idx]] - sites[3*edge[2*idx+1]]) + 
                               (sites[3*edge[2*idx]+1] - sites[3*edge[2*idx+1]+1]) * (sites[3*edge[2*idx]+1] - sites[3*edge[2*idx+1]+1]) + 
                               (sites[3*edge[2*idx]+2] - sites[3*edge[2*idx+1]+2]) * (sites[3*edge[2*idx]+2] - sites[3*edge[2*idx+1]+2])); 

    if (sdf[edge[2*idx]]*sdf[edge[2*idx+1]] <= 0.0f || fmin(fabs(sdf[edge[2*idx]]), fabs(sdf[edge[2*idx+1]])) < edge_length) {
        int new_idx = atomicAdd(counter, 1);
        new_sites[3*new_idx] = (sites[3*edge[2*idx]] + sites[3*edge[2*idx+1]]) / 2.0f;
        new_sites[3*new_idx + 1] = (sites[3*edge[2*idx] + 1] + sites[3*edge[2*idx+1] + 1]) / 2.0f;
        new_sites[3*new_idx + 2] = (sites[3*edge[2*idx] + 2] + sites[3*edge[2*idx+1] + 2]) / 2.0f;
        
        new_sdf[3*new_idx] = (sdf[edge[2*idx]] + sdf[edge[2*idx+1]]) / 2.0f;

        for (int l = 0; l < DIM_L_FEAT; l++) {
            new_feats[DIM_L_FEAT*new_idx + l] = (feats[DIM_L_FEAT*edge[2*idx] + l] + feats[DIM_L_FEAT*edge[2*idx+1] + l]) / 2.0f;
        }
    }
}


__global__ void vertex_adjacencies_kernel(
    const size_t nb_tets,
    const int *__restrict__ tetras,
    int *__restrict__ summits,
    int *__restrict__ adjacencies)
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nb_tets)
    {
        return;
    }
    
    // make last summit index as xor
    summits[4*idx] = tetras[4*idx]; summits[4*idx+1] = tetras[4*idx+1]; summits[4*idx+2] = tetras[4*idx+2]; 
    summits[4*idx+3] = tetras[4*idx] ^ tetras[4*idx+1] ^ tetras[4*idx+2] ^ tetras[4*idx+3];

    int id0 = tetras[4*idx]; int id1 = tetras[4*idx + 1]; int id2 = tetras[4*idx + 2]; int id3 = tetras[4*idx + 3];
    int id_a = atomicAdd(&adjacencies[DIM_ADJ*id0], 1);
    if (id_a < DIM_ADJ-1)
        adjacencies[DIM_ADJ*id0 + id_a + 1] = idx;

    id_a = atomicAdd(&adjacencies[DIM_ADJ*id1], 1);
    if (id_a < DIM_ADJ-1)
        adjacencies[DIM_ADJ*id1 + id_a + 1] = idx;

    id_a = atomicAdd(&adjacencies[DIM_ADJ*id2], 1);
    if (id_a < DIM_ADJ-1)
        adjacencies[DIM_ADJ*id2 + id_a + 1] = idx;

    id_a = atomicAdd(&adjacencies[DIM_ADJ*id3], 1);
    if (id_a < DIM_ADJ-1)
        adjacencies[DIM_ADJ*id3 + id_a + 1] = idx;
}

__global__ void make_adjacencies_kernel(
    const size_t nb_tets,
    const int *__restrict__ tetras,
    const int *__restrict__ adjacencies,
    int *__restrict__ neighbors)
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nb_tets)
    {
        return;
    }

    int id0 = tetras[4*idx]; int id1 = tetras[4*idx + 1]; int id2 = tetras[4*idx + 2]; int id3 = tetras[4*idx + 3];

    int nb_n0 = adjacencies[DIM_ADJ*id0];
    int nb_n1 = adjacencies[DIM_ADJ*id1];
    int nb_n2 = adjacencies[DIM_ADJ*id2];
    int nb_n3 = adjacencies[DIM_ADJ*id3];

    // Face 1
    {
        int id_s = -1;
        if (nb_n0 < DIM_ADJ-1)
            id_s = id0;
        if (nb_n1 < nb_n0 && nb_n1 < nb_n2 && nb_n1 < nb_n3 && nb_n1 < DIM_ADJ-1)
            id_s = id1;
        if (nb_n2 < nb_n0 && nb_n2 < nb_n1 && nb_n2 < nb_n3 && nb_n2 < DIM_ADJ-1)
            id_s = id2;

        if (id_s != -1) {
            int nb_s = adjacencies[DIM_ADJ*id_s];
            int adj_curr;
            int counter = 0;
            for (int i = 0; i < nb_s; i++) {
                adj_curr = adjacencies[DIM_ADJ*id_s + i + 1];
                if (adj_curr == idx)
                    continue;

                if ((id0 == tetras[4*adj_curr] || id0 == tetras[4*adj_curr + 1] || id0 == tetras[4*adj_curr + 2] || id0 == tetras[4*adj_curr + 3]) &&
                    (id1 == tetras[4*adj_curr] || id1 == tetras[4*adj_curr + 1] || id1 == tetras[4*adj_curr + 2] || id1 == tetras[4*adj_curr + 3]) &&
                    (id2 == tetras[4*adj_curr] || id2 == tetras[4*adj_curr + 1] || id2 == tetras[4*adj_curr + 2] || id2 == tetras[4*adj_curr + 3])) {
                    
                    neighbors[4*idx+3] = adj_curr;
                    break;
                    
                }
            }
        }
    }

    // Face 2
    {
        int id_s = -1;
        if (nb_n3 < DIM_ADJ-1)
            id_s = id3;
        if (nb_n1 < nb_n0 && nb_n1 < nb_n2 && nb_n1 < nb_n3 && nb_n1 < DIM_ADJ-1)
            id_s = id1;
        if (nb_n2 < nb_n0 && nb_n2 < nb_n1 && nb_n2 < nb_n3 && nb_n2 < DIM_ADJ-1)
            id_s = id2;

        if (id_s != -1) {
            int nb_s = adjacencies[DIM_ADJ*id_s];
            int adj_curr;
            int counter = 0;
            for (int i = 0; i < nb_s; i++) {
                adj_curr = adjacencies[DIM_ADJ*id_s + i + 1];
                if (adj_curr == idx)
                    continue;
                if ((id3 == tetras[4*adj_curr] || id3 == tetras[4*adj_curr + 1] || id3 == tetras[4*adj_curr + 2] || id3 == tetras[4*adj_curr + 3]) &&
                    (id1 == tetras[4*adj_curr] || id1 == tetras[4*adj_curr + 1] || id1 == tetras[4*adj_curr + 2] || id1 == tetras[4*adj_curr + 3]) &&
                    (id2 == tetras[4*adj_curr] || id2 == tetras[4*adj_curr + 1] || id2 == tetras[4*adj_curr + 2] || id2 == tetras[4*adj_curr + 3])) {
                    
                    neighbors[4*idx] = adj_curr;
                    break;
                    
                }
            }
        }
    }
    
    // Face 3
    {
        int id_s = -1;
        if (nb_n0 < DIM_ADJ-131)
            id_s = id0;
        if (nb_n1 < nb_n0 && nb_n1 < nb_n2 && nb_n1 < nb_n3 && nb_n1 < DIM_ADJ-1)
            id_s = id1;
        if (nb_n3 < nb_n0 && nb_n3 < nb_n1 && nb_n3 < nb_n2 && nb_n3 < DIM_ADJ-1)
            id_s = id3;

        if (id_s != -1) {
            int nb_s = adjacencies[DIM_ADJ*id_s];
            int adj_curr;
            int counter = 0;
            for (int i = 0; i < nb_s; i++) {
                adj_curr = adjacencies[DIM_ADJ*id_s + i + 1];
                if (adj_curr == idx)
                    continue;
                if ((id3 == tetras[4*adj_curr] || id3 == tetras[4*adj_curr + 1] || id3 == tetras[4*adj_curr + 2] || id3 == tetras[4*adj_curr + 3]) &&
                    (id1 == tetras[4*adj_curr] || id1 == tetras[4*adj_curr + 1] || id1 == tetras[4*adj_curr + 2] || id1 == tetras[4*adj_curr + 3]) &&
                    (id0 == tetras[4*adj_curr] || id0 == tetras[4*adj_curr + 1] || id0 == tetras[4*adj_curr + 2] || id0 == tetras[4*adj_curr + 3])) {
                    
                    neighbors[4*idx + 2] = adj_curr;
                    break;
                    
                }
            }
        }
    }

    // Face 4
    {
        int id_s = -1;
        if (nb_n0 < DIM_ADJ-1)
            id_s = id0;
        if (nb_n2 < nb_n0 && nb_n2 < nb_n1 && nb_n2 < nb_n3 && nb_n2 < DIM_ADJ-1)
            id_s = id2;
        if (nb_n3 < nb_n0 && nb_n3 < nb_n1 && nb_n3 < nb_n2 && nb_n3 < DIM_ADJ-1)
            id_s = id3;

        if (id_s != -1) {
            int nb_s = adjacencies[DIM_ADJ*id_s];
            int adj_curr;
            int counter = 0;
            for (int i = 0; i < nb_s; i++) {
                adj_curr = adjacencies[DIM_ADJ*id_s + i + 1];
                if (adj_curr == idx)
                    continue;
                if ((id3 == tetras[4*adj_curr] || id3 == tetras[4*adj_curr + 1] || id3 == tetras[4*adj_curr + 2] || id3 == tetras[4*adj_curr + 3]) &&
                    (id2 == tetras[4*adj_curr] || id2 == tetras[4*adj_curr + 1] || id2 == tetras[4*adj_curr + 2] || id2 == tetras[4*adj_curr + 3]) &&
                    (id0 == tetras[4*adj_curr] || id0 == tetras[4*adj_curr + 1] || id0 == tetras[4*adj_curr + 2] || id0 == tetras[4*adj_curr + 3])) {
                    
                    neighbors[4*idx + 1] = adj_curr;
                    break;
                    
                }
            }
        }
    }
}



/** CPU functions **/
/** CPU functions **/
/** CPU functions **/


// 
int upsample_counter_cuda(
    size_t nb_edges,
    torch::Tensor edges,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor sites,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor sdf
)   {
        int* counter;
        cudaMalloc((void**)&counter, sizeof(int));
        cudaMemset(counter, 0, sizeof(int));
        
        const int threads = 1024;
        const int blocks = (nb_edges + threads - 1) / threads; // ceil for example 8192 + 255 / 256 = 32
        AT_DISPATCH_FLOATING_TYPES( sdf.type(),"upsample_counter_kernel", ([&] {  
            upsample_counter_kernel CUDA_KERNEL(blocks,threads) (
                nb_edges,
                edges.data_ptr<int>(),
                sites.data_ptr<float>(),
                sdf.data_ptr<float>(),
                counter); 
        }));

        cudaDeviceSynchronize();

        int res = 0.;
        cudaMemcpy(&res, counter, sizeof(int), cudaMemcpyDeviceToHost);
        cudaFree(counter);

        return res;
}


void upsample_cuda(
    size_t nb_edges,
    torch::Tensor edges,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor sites,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor sdf,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor feats,
    torch::Tensor new_sites,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor new_sdf,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor new_feats
)   {
        int* counter;
        cudaMalloc((void**)&counter, sizeof(int));
        cudaMemset(counter, 0, sizeof(int));
        
        const int threads = 1024;
        const int blocks = (nb_edges + threads - 1) / threads; // ceil for example 8192 + 255 / 256 = 32
        AT_DISPATCH_FLOATING_TYPES( sdf.type(),"upsample_kernel", ([&] {  
            upsample_kernel CUDA_KERNEL(blocks,threads) (
                nb_edges,
                edges.data_ptr<int>(),
                sites.data_ptr<float>(),
                sdf.data_ptr<float>(),
                feats.data_ptr<float>(),
                new_sites.data_ptr<float>(),
                new_sdf.data_ptr<float>(),
                new_feats.data_ptr<float>(),
                counter); 
        }));

        cudaDeviceSynchronize();
        cudaFree(counter);
}

void vertex_adjacencies_cuda(
    size_t nb_tets,
    torch::Tensor tetras,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor summits,
    torch::Tensor adjacencies)
{
        
        const int threads = 1024;
        const int blocks = (nb_tets + threads - 1) / threads; // ceil for example 8192 + 255 / 256 = 32
        AT_DISPATCH_ALL_TYPES( tetras.type(),"vertex_adjacencies_kernel", ([&] {  
            vertex_adjacencies_kernel CUDA_KERNEL(blocks,threads) (
                nb_tets,
                tetras.data_ptr<int>(),
                summits.data_ptr<int>(),
                adjacencies.data_ptr<int>()); 
        }));
}

void make_adjacencies_cuda(
    size_t nb_tets,
    torch::Tensor tetras,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor adjacencies,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor neighbors)
{
        
        const int threads = 1024;
        const int blocks = (nb_tets + threads - 1) / threads; // ceil for example 8192 + 255 / 256 = 32
        AT_DISPATCH_ALL_TYPES( tetras.type(),"make_adjacencies_kernel", ([&] {  
            make_adjacencies_kernel CUDA_KERNEL(blocks,threads) (
                nb_tets,
                tetras.data_ptr<int>(),
                adjacencies.data_ptr<int>(),
                neighbors.data_ptr<int>()); 
        }));
}