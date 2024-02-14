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