#include <torch/extension.h>

#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

#include <device_launch_parameters.h>
#include "cudaType.cuh"

#ifdef __INTELLISENSE__
#define CUDA_KERNEL(...)
#define FAKEINIT = {0}
#else
#define CUDA_KERNEL(...) <<< __VA_ARGS__ >>>
#define FAKEINIT
#endif

#define DIM_L_FEAT 6

/** Device functions **/
/** Device functions **/
/** Device functions **/

__global__ void backprop_feat_kernel(
    const size_t num_samples,
    float *__restrict__ grad_feat,
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
    for (int i = 0; i < 3; i++) {
            id_prev = cell_ids[6 * idx + i];
            id = cell_ids[6 * idx + 3 + i];

            if (id_prev < 0 || id < 0)
                return;
            
            for (int k = 0; k < DIM_L_FEAT; k++) {  
                atomicAdd(&grad_feat[DIM_L_FEAT * id_prev + k], cell_weights[6*idx + i] * 0.5 * grad_samples[DIM_L_FEAT * idx + k]);              
                atomicAdd(&grad_feat[DIM_L_FEAT * id + k], cell_weights[6*idx + 3 + i] * 0.5 * grad_samples[DIM_L_FEAT * idx + k]);
            }
        }

    return;
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

__global__ void smooth_kernel(
    const size_t num_edges,                // number of rays
    float *__restrict__ vertices,     // [N_voxels, 4] for each voxel => it's vertices
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
    
    float sigma = 0.01f;
    float length_edge = (vertices[3*edges[2*idx]] - vertices[3*edges[2*idx+1]])*(vertices[3*edges[2*idx]] - vertices[3*edges[2*idx+1]]) +
                            (vertices[3*edges[2*idx] + 1] - vertices[3*edges[2*idx+1] + 1])*(vertices[3*edges[2*idx] + 1] - vertices[3*edges[2*idx+1] + 1]) +
                            (vertices[3*edges[2*idx] + 2] - vertices[3*edges[2*idx+1] + 2])*(vertices[3*edges[2*idx] + 2] - vertices[3*edges[2*idx+1] + 2]);
              
    atomicAdd(&sdf_grad[edges[2*idx]], exp(-length_edge/sigma) * (sdf[edges[2*idx]] - sdf[edges[2*idx+1]]));          
    atomicAdd(&sdf_grad[edges[2*idx + 1]], -exp(-length_edge/sigma) * (sdf[edges[2*idx]] - sdf[edges[2*idx+1]]));

    for (int i = 0; i < DIM_L_FEAT; i++) {
        atomicAdd(&feat_grad[DIM_L_FEAT*edges[2*idx] + i], exp(-length_edge/sigma) * (feat[DIM_L_FEAT*edges[2*idx] + i] - feat[DIM_L_FEAT*edges[2*idx+1] + i]));          
        atomicAdd(&feat_grad[DIM_L_FEAT*edges[2*idx + 1] + i], -exp(-length_edge/sigma) * (feat[DIM_L_FEAT*edges[2*idx] + i] - feat[DIM_L_FEAT*edges[2*idx+1] + i]));
    }
    
    atomicAdd(&counter[edges[2*idx]], exp(-length_edge/sigma));
    atomicAdd(&counter[edges[2*idx + 1]], exp(-length_edge/sigma));

    return;
}



/** CPU functions **/
/** CPU functions **/
/** CPU functions **/

// *************************
void backprop_feat_cuda(
    size_t num_samples,
    torch::Tensor grad_feat,
    torch::Tensor grad_samples,
    torch::Tensor cell_ids,
    torch::Tensor cell_weights 
)
{
    const int threads = 1024;
    const int blocks = (num_samples + threads - 1) / threads; // ceil for example 8192 + 255 / 256 = 32
    AT_DISPATCH_FLOATING_TYPES( grad_feat.type(),"render_cuda", ([&] {  
        backprop_feat_kernel CUDA_KERNEL(blocks,threads) (
            num_samples,
            grad_feat.data_ptr<float>(),
            grad_samples.data_ptr<float>(),
            cell_ids.data_ptr<int>(),
            cell_weights.data_ptr<float>());
    }));
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
    
    cudaDeviceSynchronize();
    float res = 0.0f;
    cudaMemcpy(&res, loss, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(loss);

    return res/float(num_sites);
}

// *************************
void smooth_cuda(
    size_t num_edges,
    torch::Tensor vertices,
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
    AT_DISPATCH_FLOATING_TYPES( sdf.type(),"smooth_kernel", ([&] {  
        smooth_kernel CUDA_KERNEL(blocks,threads) (
            num_edges,
            vertices.data_ptr<float>(),
            sdf.data_ptr<float>(),
            feat.data_ptr<float>(),
            edges.data_ptr<int>(),
            sdf_grad.data_ptr<float>(),
            feat_grad.data_ptr<float>(),
            counter.data_ptr<float>());
    }));
    
}
