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

#define DIM_ADJ 128
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

    if (sdf[edge[2*idx]]*sdf[edge[2*idx+1]] <= 0.0f || fmin(fabs(sdf[edge[2*idx]]), fabs(sdf[edge[2*idx+1]])) < 3.0*edge_length)
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

    if (sdf[edge[2*idx]]*sdf[edge[2*idx+1]] <= 0.0f || 
            fmin(fabs(sdf[edge[2*idx]]), fabs(sdf[edge[2*idx+1]])) < 3.0*edge_length) {
        int new_idx = atomicAdd(counter, 1);
        new_sites[3*new_idx] = (sites[3*edge[2*idx]] + sites[3*edge[2*idx+1]]) / 2.0f;
        new_sites[3*new_idx + 1] = (sites[3*edge[2*idx] + 1] + sites[3*edge[2*idx+1] + 1]) / 2.0f;
        new_sites[3*new_idx + 2] = (sites[3*edge[2*idx] + 2] + sites[3*edge[2*idx+1] + 2]) / 2.0f;
        
        new_sdf[new_idx] = (sdf[edge[2*idx]] + sdf[edge[2*idx+1]]) / 2.0f;

        for (int l = 0; l < DIM_L_FEAT; l++) {
            new_feats[DIM_L_FEAT*new_idx + l] = (feats[DIM_L_FEAT*edge[2*idx] + l] + feats[DIM_L_FEAT*edge[2*idx+1] + l]) / 2.0f;
        }
    }
}

__global__ void count_adjacencies_kernel(
    const size_t nb_tets,
    const int *__restrict__ tetras,
    int *__restrict__ counter)
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nb_tets)
    {
        return;
    }
    
    for (int i = 0; i < 4; i++)
        atomicAdd(&counter[tetras[4*idx + i]], 1);

}

__global__ void vertex_adjacencies_kernel(
    const size_t nb_tets,
    int * offset,
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
    int id_a = atomicAdd(&adjacencies[offset[id0]], 1);
    adjacencies[offset[id0] + id_a + 1] = idx;

    id_a = atomicAdd(&adjacencies[offset[id1]], 1);
    adjacencies[offset[id1] + id_a + 1] = idx;

    id_a = atomicAdd(&adjacencies[offset[id2]], 1);
    adjacencies[offset[id2] + id_a + 1] = idx;

    id_a = atomicAdd(&adjacencies[offset[id3]], 1);
    adjacencies[offset[id3] + id_a + 1] = idx;
}

__global__ void make_adjacencies_kernel(
    const size_t nb_tets,
    const int *__restrict__ tetras,
    const int * offset,
    const int *__restrict__ adjacencies,
    int *__restrict__ neighbors)
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nb_tets)
    {
        return;
    }
    
    int id0 = tetras[4*idx]; int id1 = tetras[4*idx + 1]; int id2 = tetras[4*idx + 2]; int id3 = tetras[4*idx + 3];

    int nb_n0 = adjacencies[offset[id0]];
    int nb_n1 = adjacencies[offset[id1]];
    int nb_n2 = adjacencies[offset[id2]];
    int nb_n3 = adjacencies[offset[id3]];


    // Face 1
    {
        int id_s = id0;
        if (nb_n1 < nb_n0 && nb_n1 < nb_n2)
            id_s = id1;
        if (nb_n2 < nb_n0 && nb_n2 < nb_n1)
            id_s = id2;
            
        if (id_s != -1) {
            int start_s = offset[id_s];
            int nb_s = adjacencies[start_s];
            int adj_curr;
            int counter = 0;
            for (int i = 0; i < nb_s; i++) {
                adj_curr = adjacencies[start_s + i + 1];
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
        int id_s = id3;
        if (nb_n1 < nb_n2 && nb_n1 < nb_n3)
            id_s = id1;
        if (nb_n2 < nb_n1 && nb_n2 < nb_n3)
            id_s = id2;

        if (id_s != -1) {
            int nb_s = adjacencies[offset[id_s]];
            int adj_curr;
            int counter = 0;
            for (int i = 0; i < nb_s; i++) {
                adj_curr = adjacencies[offset[id_s] + i + 1];
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
        int id_s = id0;
        if (nb_n1 < nb_n0 && nb_n1 < nb_n3)
            id_s = id1;
        if (nb_n3 < nb_n0 && nb_n3 < nb_n1)
            id_s = id3;

        if (id_s != -1) {
            int nb_s = adjacencies[offset[id_s]];
            int adj_curr;
            int counter = 0;
            for (int i = 0; i < nb_s; i++) {
                adj_curr = adjacencies[offset[id_s] + i + 1];
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
        int id_s = id0;
        if (nb_n2 < nb_n0 && nb_n2 < nb_n3)
            id_s = id2;
        if (nb_n3 < nb_n0 && nb_n3 < nb_n2)
            id_s = id3;

        if (id_s != -1) {
            int nb_s = adjacencies[offset[id_s]];
            int adj_curr;
            int counter = 0;
            for (int i = 0; i < nb_s; i++) {
                adj_curr = adjacencies[offset[id_s] + i + 1];
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


__global__ void count_cam_adjacencies_kernel(
    const size_t nb_tets,
    const size_t nb_cams,
    const int *__restrict__ tetras,
    const int *__restrict__ cam_ids,
    int *__restrict__ counter)
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nb_tets)
    {
        return;
    }

    int id0 = tetras[4*idx]; int id1 = tetras[4*idx + 1]; int id2 = tetras[4*idx + 2]; int id3 = tetras[4*idx + 3];

    for (int c_id = 0; c_id < nb_cams; c_id++) {
        if (id0 == cam_ids[c_id] || id1 == cam_ids[c_id] ||
            id2 == cam_ids[c_id] || id3 == cam_ids[c_id]) {
            atomicAdd(&counter[c_id], 1);
        }
    }
}

__global__ void make_cam_adjacencies_kernel(
    const size_t nb_tets,
    const size_t nb_cams,
    int * offset,
    const int *__restrict__ tetras,
    const int *__restrict__ cam_ids,
    int *__restrict__ adjacencies)
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nb_tets)
    {
        return;
    }

    int id0 = tetras[4*idx]; int id1 = tetras[4*idx + 1]; int id2 = tetras[4*idx + 2]; int id3 = tetras[4*idx + 3];

    int id_a;
    int start_s;
    for (int c_id = 0; c_id < nb_cams; c_id++) {
        if (id0 == cam_ids[c_id] || id1 == cam_ids[c_id] ||
            id2 == cam_ids[c_id] || id3 == cam_ids[c_id]) {
            start_s = c_id == 0 ? 0 : offset[c_id-1];
            id_a =  atomicAdd(&adjacencies[start_s], 1);
            adjacencies[start_s + id_a + 1] = idx;
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
        AT_DISPATCH_ALL_TYPES( sdf.type(),"upsample_kernel", ([&] {  
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
    size_t nb_sites,
    torch::Tensor tetras,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor summits,
    int ** adjacencies,
    int ** offset)
{
    
        int* counter;
        cudaMalloc((void**)&counter, nb_sites*sizeof(int));
        cudaMemset(counter, 0, nb_sites*sizeof(int));
    
        cudaMalloc((void**)offset, nb_sites*sizeof(int));
        cudaMemset((*offset), 0, nb_sites*sizeof(int));

        const int threads = 1024;
        const int blocks = (nb_tets + threads - 1) / threads; // ceil for example 8192 + 255 / 256 = 32
        AT_DISPATCH_ALL_TYPES( tetras.type(),"count_adjacencies_kernel", ([&] {  
            count_adjacencies_kernel CUDA_KERNEL(blocks,threads) (
                nb_tets,
                tetras.data_ptr<int>(),
                counter); 
        }));
        cudaDeviceSynchronize();

        int* counter_cpu = (int *) malloc(nb_sites*sizeof(int));
        cudaMemcpy(counter_cpu, counter, nb_sites*sizeof(int), cudaMemcpyDeviceToHost);

        int *offset_cpu = (int *) malloc(nb_sites*sizeof(int));
        int tot_size = 0;
        for (int i = 0; i < nb_sites; i++) {
            offset_cpu[i] = tot_size;
            tot_size += counter_cpu[i] + 1;
        }
        cudaMemcpy(*offset, offset_cpu, nb_sites*sizeof(int), cudaMemcpyHostToDevice);

        cudaMalloc((void**)adjacencies, tot_size*sizeof(int));
        cudaMemset((*adjacencies), 0, tot_size*sizeof(int));
        
        AT_DISPATCH_ALL_TYPES( tetras.type(),"vertex_adjacencies_kernel", ([&] {  
            vertex_adjacencies_kernel CUDA_KERNEL(blocks,threads) (
                nb_tets,
                *offset,
                tetras.data_ptr<int>(),
                summits.data_ptr<int>(),
                *adjacencies); 
        }));
        cudaDeviceSynchronize();

        
        /*int* adj_cpu = (int *) malloc(tot_size*sizeof(int));
        cudaMemcpy(adj_cpu, *adjacencies, tot_size*sizeof(int), cudaMemcpyDeviceToHost);

        for (int i = 0; i < nb_sites; i++) {
            //std::cout << adj_cpu[offset_cpu[i]] << " ==> " << std::endl;
            for (int j = 0; j < adj_cpu[offset_cpu[i]]; j++) {
                if (adj_cpu[offset_cpu[i] +j + 1] > nb_tets - 1 ||
                    adj_cpu[offset_cpu[i] +j + 1] < 0)
                    std::cout << adj_cpu[offset_cpu[i] +j + 1] << ", " << std::endl;
            }
        }*/

        cudaFree(counter);
        free(counter_cpu);
        free(offset_cpu);
}

void make_adjacencies_cuda(
    size_t nb_tets,
    torch::Tensor tetras,    // [N_sites, 3] for each voxel => it's vertices
    int * offset,
    int * adjacencies,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor neighbors)
{
        
    const int threads = 1024;
    const int blocks = (nb_tets + threads - 1) / threads; // ceil for example 8192 + 255 / 256 = 32
    AT_DISPATCH_ALL_TYPES( tetras.type(),"make_adjacencies_kernel", ([&] {  
        make_adjacencies_kernel CUDA_KERNEL(blocks,threads) (
            nb_tets,
            tetras.data_ptr<int>(),
            offset,
            adjacencies,
            neighbors.data_ptr<int>()); 
    }));
    cudaDeviceSynchronize();
        
    cudaFree(offset);
    cudaFree(adjacencies);
}


int count_cam_neighbors_cuda(
    size_t nb_tets,
    size_t nb_cams,
    torch::Tensor tetras,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor cam_ids,
    torch::Tensor offset)
{
    
        int* counter;
        cudaMalloc((void**)&counter, nb_cams*sizeof(int));
        cudaMemset(counter, 0, nb_cams*sizeof(int));

        const int threads = 1024;
        const int blocks = (nb_tets + threads - 1) / threads; // ceil for example 8192 + 255 / 256 = 32
        AT_DISPATCH_ALL_TYPES( tetras.type(),"count_cam_adjacencies_kernel", ([&] {  
            count_cam_adjacencies_kernel CUDA_KERNEL(blocks,threads) (
                nb_tets,
                nb_cams,
                tetras.data_ptr<int>(),
                cam_ids.data_ptr<int>(),
                counter); 
        }));
        cudaDeviceSynchronize();

        int* counter_cpu = (int *) malloc(nb_cams*sizeof(int));
        cudaMemcpy(counter_cpu, counter, nb_cams*sizeof(int), cudaMemcpyDeviceToHost);

        int *offset_cpu = (int *) malloc(nb_cams*sizeof(int));
        int tot_size = 0;
        for (int i = 0; i < nb_cams; i++) {
            tot_size += counter_cpu[i]+1;
            offset_cpu[i] = tot_size;
        }
        cudaMemcpy(offset.data_ptr<int>(), offset_cpu, nb_cams*sizeof(int), cudaMemcpyHostToDevice);

        cudaFree(counter);
        free(counter_cpu);
        free(offset_cpu);

        return tot_size;
}



void cameras_adjacencies_cuda(
    size_t nb_tets,
    size_t nb_cams,
    torch::Tensor tetras,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor cam_ids,
    torch::Tensor adjacencies,
    torch::Tensor offset)
{
       
    const int threads = 1024;
    const int blocks = (nb_tets + threads - 1) / threads;  
    AT_DISPATCH_ALL_TYPES( tetras.type(),"make_cam_adjacencies_kernel", ([&] {  
        make_cam_adjacencies_kernel CUDA_KERNEL(blocks,threads) (
            nb_tets,
            nb_cams,
            offset.data_ptr<int>(),
            tetras.data_ptr<int>(),
            cam_ids.data_ptr<int>(),
            adjacencies.data_ptr<int>()); 
    }));
}

