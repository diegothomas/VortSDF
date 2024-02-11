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
    float *__restrict__ grad_sdf,
    const float *__restrict__ grad_sdf_samples,
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
    ////////////////////////Linear interpolation//////////////////////////
    //////////////////////////////////////////////////////////////
    float lamda = cell_weights[13*idx + 12] ;
    for (int i = 0; i < 6; i++) {
        id_prev = cell_ids[12 * idx + i];
        id = cell_ids[12 * idx + 6 + i];

        for (int k = 0; k < DIM_L_FEAT; k++) {  
            atomicAdd(&grad_feat[DIM_L_FEAT * id_prev + k], cell_weights[13*idx + i] * lamda * grad_samples[DIM_L_FEAT * idx + k]);              
            atomicAdd(&grad_feat[DIM_L_FEAT * id + k], cell_weights[13*idx + 6 + i] * (1.0f - lamda) * grad_samples[DIM_L_FEAT * idx + k]);
        }
    }
    ////////////////////////Network interpolation//////////////////////////
    //////////////////////////////////////////////////////////////
    /*for (int i = 0; i < 3; i++) {
        id_prev = cell_ids[6 * idx + i];
        id = cell_ids[6 * idx + 3 + i];
        //atomicAdd(&grad_sdf[id_prev], grad_sdf_samples[6*idx + i]);       
        //atomicAdd(&grad_sdf[id], grad_sdf_samples[6*idx + 3 + i]);  
        for (int k = 0; k < DIM_L_FEAT; k++) {  
            atomicAdd(&grad_feat[DIM_L_FEAT * id_prev + k], grad_samples[6*DIM_L_FEAT * idx + DIM_L_FEAT * i  + k]);       
            atomicAdd(&grad_feat[DIM_L_FEAT * id + k], grad_samples[6*DIM_L_FEAT * idx + 18 + DIM_L_FEAT * i  + k]);       
        }
    }*/

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

__global__ void smooth_grad_kernel(
    const size_t num_edges,                // number of rays
    float sigma,
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
    
    float length_edge = (vertices[3*edges[2*idx]] - vertices[3*edges[2*idx+1]])*(vertices[3*edges[2*idx]] - vertices[3*edges[2*idx+1]]) +
                            (vertices[3*edges[2*idx] + 1] - vertices[3*edges[2*idx+1] + 1])*(vertices[3*edges[2*idx] + 1] - vertices[3*edges[2*idx+1] + 1]) +
                            (vertices[3*edges[2*idx] + 2] - vertices[3*edges[2*idx+1] + 2])*(vertices[3*edges[2*idx] + 2] - vertices[3*edges[2*idx+1] + 2]);
              
    atomicAdd(&sdf_grad[edges[2*idx]], exp(-length_edge/(sigma*sigma)) * (sdf[edges[2*idx]] - sdf[edges[2*idx+1]]));          
    atomicAdd(&sdf_grad[edges[2*idx + 1]], -exp(-length_edge/(sigma*sigma)) * (sdf[edges[2*idx]] - sdf[edges[2*idx+1]]));

    for (int i = 0; i < DIM_L_FEAT; i++) {
        atomicAdd(&feat_grad[DIM_L_FEAT*edges[2*idx] + i], exp(-length_edge/(sigma*sigma)) * (feat[DIM_L_FEAT*edges[2*idx] + i] - feat[DIM_L_FEAT*edges[2*idx+1] + i]));          
        atomicAdd(&feat_grad[DIM_L_FEAT*edges[2*idx + 1] + i], -exp(-length_edge/(sigma*sigma)) * (feat[DIM_L_FEAT*edges[2*idx] + i] - feat[DIM_L_FEAT*edges[2*idx+1] + i]));
    }
    
    atomicAdd(&counter[edges[2*idx]], exp(-length_edge/(sigma*sigma)));
    atomicAdd(&counter[edges[2*idx + 1]], exp(-length_edge/(sigma*sigma)));

    return;
}


__global__ void smooth_kernel(
    const size_t num_edges,                // number of rays
    float sigma,
    const size_t dim_sdf,
    float *__restrict__ vertices,     // [N_voxels, 4] for each voxel => it's vertices
    float *__restrict__ sdf,     // [N_voxels, 4] for each voxel => it's vertices
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
    
    float length_edge = (vertices[3*edges[2*idx]] - vertices[3*edges[2*idx+1]])*(vertices[3*edges[2*idx]] - vertices[3*edges[2*idx+1]]) +
                            (vertices[3*edges[2*idx] + 1] - vertices[3*edges[2*idx+1] + 1])*(vertices[3*edges[2*idx] + 1] - vertices[3*edges[2*idx+1] + 1]) +
                            (vertices[3*edges[2*idx] + 2] - vertices[3*edges[2*idx+1] + 2])*(vertices[3*edges[2*idx] + 2] - vertices[3*edges[2*idx+1] + 2]);
              
    for (int i = 0; i < dim_sdf; i++) {
        if (sdf[dim_sdf*edges[2*idx + 1] + i] != 0.0f)
            atomicAdd(&sdf_smooth[dim_sdf*edges[2*idx] + i], exp(-length_edge/(sigma*sigma)) * sdf[dim_sdf*edges[2*idx + 1] + i]);          
        
        if (sdf[dim_sdf*edges[2*idx] + i] != 0.0f)
            atomicAdd(&sdf_smooth[dim_sdf*edges[2*idx + 1] + i], exp(-length_edge/(sigma*sigma)) * sdf[dim_sdf*edges[2*idx] + i]);
    }

    if (sdf[dim_sdf*edges[2*idx + 1]] != 0.0f)
        atomicAdd(&counter[edges[2*idx]], exp(-length_edge/(sigma*sigma)));
    if (sdf[dim_sdf*edges[2*idx]] != 0.0f)
        atomicAdd(&counter[edges[2*idx + 1]], exp(-length_edge/(sigma*sigma)));

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

    if (weights[idx] == 0.0f)
        return;

    for (int i = 0; i < dim_sdf; i++) 
        sdf_smooth[dim_sdf*idx + i] = sdf_smooth[dim_sdf*idx + i]/weights[idx];
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



/** CPU functions **/
/** CPU functions **/
/** CPU functions **/

// *************************
void backprop_feat_cuda(
    size_t num_samples,
    torch::Tensor grad_sdf,
    torch::Tensor grad_sdf_samples,
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
            grad_sdf.data_ptr<float>(),
            grad_sdf_samples.data_ptr<float>(),
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
    float sigma,
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
    AT_DISPATCH_FLOATING_TYPES( sdf.type(),"smooth_grad_kernel", ([&] {  
        smooth_grad_kernel CUDA_KERNEL(blocks,threads) (
            num_edges,
            sigma,
            vertices.data_ptr<float>(),
            sdf.data_ptr<float>(),
            feat.data_ptr<float>(),
            edges.data_ptr<int>(),
            sdf_grad.data_ptr<float>(),
            feat_grad.data_ptr<float>(),
            counter.data_ptr<float>());
    }));
    
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
void smooth_sdf_cuda(
    size_t num_edges,
    size_t num_sites,
    float sigma,
    size_t dim_sdf,
    torch::Tensor vertices,
    torch::Tensor sdf,
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
            edges.data_ptr<int>(),     // [N_voxels, 4] for each voxel => it's vertices)
            sdf_smooth.data_ptr<float>(),     // [N_voxels, 4] for each voxel => it's vertices)
            counter.data_ptr<float>());
    }));

    const int threads2 = 1024;
    const int blocks2 = (num_sites + threads2 - 1) / threads2; // ceil for example 8192 + 255 / 256 = 32
    AT_DISPATCH_FLOATING_TYPES( sdf.type(),"normalize_kernel", ([&] {  
        normalize_kernel CUDA_KERNEL(blocks2,threads2) (
            num_sites,                
            dim_sdf,
            sdf_smooth.data_ptr<float>(), 
            counter.data_ptr<float>());
    }));
    
}
