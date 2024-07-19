#include <torch/extension.h>

#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>

#ifdef __INTELLISENSE__
#define CUDA_KERNEL(...)
#define FAKEINIT = {0}
#else
#define CUDA_KERNEL(...) <<< __VA_ARGS__ >>>
#define FAKEINIT
#endif

#define divUp(x,y) (x%y) ? ((x+y-1)/y) : (x/y)

#define THREAD_SIZE_X 8
#define THREAD_SIZE_Y 8

// This is left side multiplication!! L is symmetric so this is OK for the Laplacian operator
__global__ void MaskLaplacianKernel(int* mask_sites, float* L_values, int* L_inner_indices, int* L_outer_starts, int L_nnZ, int L_outerSize, int L_cols) {
    const size_t n = blockIdx.x * blockDim.x + threadIdx.x;

	if (n > L_cols - 1)
		return;

	int nb_nnz = n < L_cols - 1 ? L_outer_starts[n + 1] - L_outer_starts[n] : L_nnZ - L_outer_starts[n];
	int start = L_outer_starts[n];
	for (int i = 0; i < nb_nnz; i++) {
		if (mask_sites[L_inner_indices[start + i]] == 1)
			L_values[start + i] = 0.0f;
	} 
}


// This is left side multiplication!! L is symmetric so this is OK for the Laplacian operator
__global__ void SparseMulKernel(float* div, float* sdf, int dim, int* active_sites, float* M_values, float* L_values, int* L_inner_indices, int* L_outer_starts, int L_nnZ, int L_outerSize, int L_cols) {
	/*unsigned int threadsPerBlock = blockDim.x * blockDim.y;
	unsigned int threadNumInBlock = threadIdx.x + blockDim.x * threadIdx.y;
	unsigned int blockNumInGrid = blockIdx.x + gridDim.x * blockIdx.y;
	int n = int(blockNumInGrid * threadsPerBlock + threadNumInBlock);*/
    const size_t n = blockIdx.x * blockDim.x + threadIdx.x;

	if (n > L_cols - 1)
		return;

	//if (active_sites[n] == 0)
	//	return;

	int nb_nnz = n < L_cols - 1 ? L_outer_starts[n + 1] - L_outer_starts[n] : L_nnZ - L_outer_starts[n];
	int start = L_outer_starts[n];
	int id_curr = 0;
	for (int dim_it = 0; dim_it < dim; dim_it++) {
		float res = 0.0f;
		float w_tot = 0.0f;
		for (int i = 0; i < nb_nnz; i++) {
			id_curr = L_inner_indices[start + i];
			if (L_inner_indices[start + i] == n) {
				continue;
				//res += sdf[dim*L_inner_indices[start + i] + dim_it];
				//w_tot +=  1.0f;
			} else {
				//if (L_values[start + i] > 0.0f && M_values[id_curr] > 0.0f ) {
				//	atomicAdd(&div[dim*n + dim_it],  (L_values[start + i] / M_values[id_curr]) * (sdf[dim*n + dim_it] - sdf[dim*id_curr+ dim_it])) ;   
				//	atomicAdd(&div[dim*id_curr + dim_it],  -(L_values[start + i] / M_values[id_curr]) * (sdf[dim*n + dim_it] - sdf[dim*id_curr + dim_it])) ;  


					res += L_values[start + i] < 0.0f || M_values[L_inner_indices[start + i]] == 0.0f ? 0.0f : L_values[start + i] * (sdf[dim*n + dim_it] - sdf[dim*L_inner_indices[start + i] + dim_it]) / M_values[L_inner_indices[start + i]];
					//res += L_values[start + i] < 0.0f || M_values[L_inner_indices[start + i]] == 0.0f ? 0.0f : L_values[start + i] * sdf[dim*L_inner_indices[start + i] + dim_it] / M_values[L_inner_indices[start + i]];
					w_tot +=  L_values[start + i] < 0.0f || M_values[L_inner_indices[start + i]] == 0.0f ? 0.0f : L_values[start + i] / M_values[L_inner_indices[start + i]];
					//res += M_values[L_inner_indices[start + i]] == 0.0f ? 0.0f : fabs(L_values[start + i]) * sdf[dim*L_inner_indices[start + i] + dim_it] / M_values[L_inner_indices[start + i]];
					//w_tot +=  M_values[L_inner_indices[start + i]] == 0.0f ? 0.0f :fabs(L_values[start + i]) / M_values[L_inner_indices[start + i]];
					//res += L_values[start + i] * sdf[dim*L_inner_indices[start + i] + dim_it];
				//}
			}
		} 
		div[dim*n + dim_it] = w_tot == 0.0 ? 0.0f : res/w_tot; //0.5*(res/w_tot + sdf[dim*n + dim_it]);///float(nb_nnz);
	}
}

// This is left side multiplication!! L is symmetric so this is OK for the Laplacian operator
__global__ void TVNormKernel(float* div, float* weights, float* norm, int* active_sites, float* M_values, float* L_values, int* L_inner_indices, int* L_outer_starts, int L_nnZ, int L_outerSize, int L_cols) {
    const size_t n = blockIdx.x * blockDim.x + threadIdx.x;

	if (n > L_cols - 1)
		return;

	//if (active_sites[n] == 0)
	//	return;

	int nb_nnz = n < L_cols - 1 ? L_outer_starts[n + 1] - L_outer_starts[n] : L_nnZ - L_outer_starts[n];
	int start = L_outer_starts[n];

	float norm_grad = sqrt(norm[3*n]*norm[3*n] + norm[3*n + 1]*norm[3*n + 1] + norm[3*n + 2]*norm[3*n + 2]);

	if (abs(norm_grad) < 1.0e-8f)
		return;

	int id_curr = 0;
	for (int i = 0; i < nb_nnz; i++) {
		id_curr = L_inner_indices[start + i];
		if (id_curr == n) {
			continue;
		} else {
			float norm_grad_curr = sqrt(norm[3*id_curr]*norm[3*id_curr] + norm[3*id_curr + 1]*norm[3*id_curr + 1] + norm[3*id_curr + 2]*norm[3*id_curr + 2]);
			if (abs(norm_grad_curr) < 1.0e-8f)
				continue;

			float dot_prod = norm[3*n]*norm[3*id_curr] + norm[3*n + 1]*norm[3*id_curr + 1] + norm[3*n + 2]*norm[3*id_curr + 2];

			if (L_values[start + i] > 0.0f && M_values[id_curr] > 0.0f ) {
				for (int dim_it = 0; dim_it < 3; dim_it++) {
					/*atomicAdd(&div[3*n + dim_it],  L_values[start + i]  * 
													(norm[3*id_curr+ dim_it] / norm_grad_curr) * (1.0f + norm[3*n + dim_it]*norm[3*n + dim_it]) / norm_grad) ;    													
					atomicAdd(&div[3*id_curr + dim_it], L_values[start + i] * 
													(norm[3*n + dim_it] / norm_grad) * (1.0f + norm[3*id_curr + dim_it]*norm[3*id_curr + dim_it]) / norm_grad_curr) ;    */

					atomicAdd(&div[3*n + dim_it],  (L_values[start + i] / M_values[id_curr]) * 
													(1.0f / norm_grad_curr) * (dot_prod * norm[3*n + dim_it] / (norm_grad*norm_grad*norm_grad) - norm[3*id_curr + dim_it] / norm_grad)) ;    
					atomicAdd(&div[3*id_curr + dim_it],  (L_values[start + i] / M_values[id_curr]) * 
													(1.0f / norm_grad) * (dot_prod * norm[3*id_curr + dim_it] / (norm_grad_curr*norm_grad_curr*norm_grad_curr) - norm[3*n + dim_it] / norm_grad_curr)) ;     
				}
				//atomicAdd(&weights[n],  (L_values[start + i] / M_values[id_curr])) ; 
				//atomicAdd(&weights[id_curr],  (L_values[start + i] / M_values[id_curr])) ;    												
			}
		}
	} 
}


void SparseMul_gpu(torch::Tensor div, torch::Tensor sdf, int dim, torch::Tensor active_sites, torch::Tensor M_values, torch::Tensor L_values, torch::Tensor L_outer_start, torch::Tensor L_nonZeros,  size_t L_nnZ, size_t L_outerSize, size_t L_cols) {
    /*dim3 dimBlock(THREAD_SIZE_X, THREAD_SIZE_Y, 1);
	int thread_size = int(round(sqrt(L_cols)) + 1);
	dim3 dimGrid(1, 1, 1);
	dimGrid.x = divUp(thread_size, dimBlock.x); // #cols
	dimGrid.y = divUp(thread_size, dimBlock.y); // # rows*/

	const int threads = 1024;
	const int blocks = (L_cols + threads - 1) / threads;
	AT_DISPATCH_ALL_TYPES( div.type(),"SparseMulKernel", ([&] {  
            SparseMulKernel CUDA_KERNEL(blocks,threads) (
                div.data_ptr<float>(),
                sdf.data_ptr<float>(),
				dim, 
                active_sites.data_ptr<int>(),
                M_values.data_ptr<float>(),
                L_values.data_ptr<float>(),
                L_nonZeros.data_ptr<int>(),
				L_outer_start.data_ptr<int>(), 
				L_nnZ, 
				L_outerSize, 
				L_cols); 
	}));

	cudaDeviceSynchronize();
}

void TVNorm_gpu(torch::Tensor div, torch::Tensor weights, torch::Tensor norm, torch::Tensor active_sites, torch::Tensor M_values, torch::Tensor L_values, torch::Tensor L_outer_start, torch::Tensor L_nonZeros,  size_t L_nnZ, size_t L_outerSize, size_t L_cols) {
    const int threads = 1024;
	const int blocks = (L_cols + threads - 1) / threads;
	AT_DISPATCH_ALL_TYPES( div.type(),"TVNormKernel", ([&] {  
            TVNormKernel CUDA_KERNEL(blocks,threads) (
                div.data_ptr<float>(),
				weights.data_ptr<float>(),
                norm.data_ptr<float>(),
                active_sites.data_ptr<int>(),
                M_values.data_ptr<float>(),
                L_values.data_ptr<float>(),
                L_nonZeros.data_ptr<int>(),
				L_outer_start.data_ptr<int>(), 
				L_nnZ, 
				L_outerSize, 
				L_cols); 
	}));

	cudaDeviceSynchronize();
}

void MaskLaplacian_gpu(torch::Tensor mask_sites, torch::Tensor L_values, torch::Tensor L_outer_start, torch::Tensor L_nonZeros,  size_t L_nnZ, size_t L_outerSize, size_t L_cols) {

	const int threads = 1024;
	const int blocks = (L_cols + threads - 1) / threads;
	AT_DISPATCH_ALL_TYPES( L_values.type(),"MaskLaplacianKernel", ([&] {  
            MaskLaplacianKernel CUDA_KERNEL(blocks,threads) (
                mask_sites.data_ptr<int>(),
                L_values.data_ptr<float>(),
                L_nonZeros.data_ptr<int>(),
				L_outer_start.data_ptr<int>(), 
				L_nnZ, 
				L_outerSize, 
				L_cols); 
	}));

	cudaDeviceSynchronize();
}