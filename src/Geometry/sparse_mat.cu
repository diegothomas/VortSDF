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

#define THREAD_SIZE_X 8
#define THREAD_SIZE_Y 8

// This is left side multiplication!! L is symmetric so this is OK for the Laplacian operator
__global__ void SparseMulKernel(float* div, float* sdf, int* active_sites, float* L_values, int* L_inner_indices, int* L_outer_starts, int L_nnZ, int L_outerSize, int L_cols) {
	unsigned int threadsPerBlock = blockDim.x * blockDim.y;
	unsigned int threadNumInBlock = threadIdx.x + blockDim.x * threadIdx.y;
	unsigned int blockNumInGrid = blockIdx.x + gridDim.x * blockIdx.y;
	int n = int(blockNumInGrid * threadsPerBlock + threadNumInBlock);

	if (n > L_cols - 1)
		return;

	//if (active_sites[n] == 0)
	//	return;

	if (fabs(sdf[n]) > 0.1)
		return;

	int nb_nnz = n < L_cols - 1 ? L_outer_starts[n + 1] - L_outer_starts[n] : L_nnZ - L_outer_starts[n];
	int start = L_outer_starts[n];
	float res = 0.0f;
	for (int i = 0; i < nb_nnz; i++) {
		res += L_values[start + i] * sdf[L_inner_indices[start + i]];
	} 
	div[n] = res;
}



void SparseMul_gpu(torch::Tensor div, torch::Tensor sdf, torch::Tensor active_sites, torch::Tensor L_values, torch::Tensor L_outer_start, torch::Tensor L_nonZeros,  int L_nnZ, size_t L_outerSize, size_t L_cols) {
    dim3 dimBlock(THREAD_SIZE_X, THREAD_SIZE_Y, 1);
	int thread_size = int(round(sqrt(L_cols)) + 1);
	dim3 dimGrid(1, 1, 1);
	dimGrid.x = divUp(thread_size, dimBlock.x); // #cols
	dimGrid.y = divUp(thread_size, dimBlock.y); // # rows

	SparseMulKernel << < dimGrid, dimBlock >> > (div.data_ptr<float>(), sdf.data_ptr<float>(), active_sites.data_ptr<int>(), 
                                                L_values.data_ptr<float>(), L_inner_indices.data_ptr<int>(), L_outer_starts.data_ptr<int>(), L_nnZ, L_outerSize, L_cols);

	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
}