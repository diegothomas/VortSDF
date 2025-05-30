#include <torch/extension.h>

#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include <device_launch_parameters.h>
#include "../Models/cudaType.cuh"

#ifdef __INTELLISENSE__
#define CUDA_KERNEL(...)
#define FAKEINIT = {0}
#else
#define CUDA_KERNEL(...) <<< __VA_ARGS__ >>>
#define FAKEINIT
#endif


#define PI 3.141592653589793238462643383279502884197
#define _MAX_K_NN 24
#define _BUFF_SIZE 2048
#define DIM_L_FEAT 16

/** Device functions **/
/** Device functions **/
/** Device functions **/


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

__device__ double volume_tetrahedron_64(double a[3], double b[3], double c[3], double d[3]) {
	double ad[3] = { a[0] - d[0], a[1] - d[1], a[2] - d[2] };
	double bd[3] = { b[0] - d[0], b[1] - d[1], b[2] - d[2] };
	double cd[3] = { c[0] - d[0], c[1] - d[1], c[2] - d[2] };

	double n[3] = { bd[1] * cd[2] - bd[2] * cd[1],
					-(bd[0] * cd[2] - bd[2] * cd[0]),
					bd[0] * cd[1] - bd[1] * cd[0] };

	double res = abs(dot3D_gpu_d(ad, n)) / 6.0;
	return res;
}

//---------------------------------------------------
//	calculate minor of matrix OR build new matrix : k-had = minor
__device__ void Minor(float* minorMatrix, int colMatrix,  float* newMinorMatrix, int sizeMatrix){
    int col, row,
        row2 = 0,
        col2 = 0;
    for(row=1; row < sizeMatrix; row++){
        for( col=0; col < sizeMatrix; col++){
            if(col == colMatrix){
                continue;
            }
            newMinorMatrix[row2*(sizeMatrix - 1) + col2] = minorMatrix[row*sizeMatrix + col];
            col2++;
            if(col2 == (sizeMatrix - 1)){
                row2++;
                col2 = 0;
            }
        }
    }
    //return;
}

//---------------------------------------------------
//	calculate determinte of matrix
__device__ float Determinante(float* minorMatrix, int sizeMatrix, float* BUFF, int offset_in){
    if ((offset_in + (sizeMatrix - 1)*(sizeMatrix - 1)) >= _BUFF_SIZE)
        return 0.0f;

    int col;
    float sum = 0;
    //float newMinorMatrix[_MAX_K_NN*_MAX_K_NN];
    float* newMinorMatrix = &BUFF[offset_in];
    if (sizeMatrix == 1){
        return minorMatrix[0];
    }
    else if(sizeMatrix == 2){
        return (minorMatrix[0] * minorMatrix[3] - minorMatrix[1] * minorMatrix[2]);
    }
    else {
        for(col=0; col < sizeMatrix; col++){
            Minor(minorMatrix, col, newMinorMatrix, sizeMatrix);	// function
            sum += (float) (minorMatrix[col] * pow(-1, col) * Determinante(newMinorMatrix, (sizeMatrix - 1), BUFF, offset_in + (sizeMatrix - 1)*(sizeMatrix - 1)));	// function
        }
    }
    return (float) sum;
}// end function


//---------------------------------------------------
//	calculate transpose of matrix
__device__ void Transpose(float* A_inv, float* cofactorMatrix, int sizeMatrix, float determinante){
    int row, col;
    for (row=0; row < sizeMatrix; row++){
        for (col=0; col < sizeMatrix; col++){
            //transposeMatrix[row*sizeMatrix + col] = cofactorMatrix[col*sizeMatrix + row];
            A_inv[row*sizeMatrix + col] = cofactorMatrix[col*sizeMatrix + row] / determinante; // adjoint method
        }
    }
    //return;
}// end function


//---------------------------------------------------
//	calculate cofactor of matrix
__device__ void Cofactor(float* A, float* A_inv, int sizeMatrix, float det, float* BUFF, int offset_in){
    //float minorMatrix[_MAX_K_NN*_MAX_K_NN],
    //    cofactorMatrix[_MAX_K_NN*_MAX_K_NN];
    float* minorMatrix = &BUFF[offset_in];
    float* cofactorMatrix = &BUFF[_BUFF_SIZE + offset_in];
    int col3, row3, row2, col2, row, col;
    for (row3=0; row3 < sizeMatrix; row3++){
        for (col3=0; col3 < sizeMatrix; col3++){
            row2 =0;
            col2 = 0;
            for (row=0; row < sizeMatrix; row++){
                for (col=0; col < sizeMatrix; col++){
                    if (row != row3 && col != col3){
                        //minorMatrix[row2*(sizeMatrix - 1) + col2] = 1.0f;// A[row*sizeMatrix + col];
                        if (col2 < (sizeMatrix - 2)){
                            col2++;
                        }
                        else {
                            col2 = 0;
                            row2++;
                        }
                    }
                }
            }
            //cofactorMatrix[row3*sizeMatrix + col3] = pow(-1, (row3 + col3)) * Determinante(minorMatrix, (sizeMatrix - 1), BUFF, offset_in + (sizeMatrix - 1)*(sizeMatrix - 1));
        }
    }
    //Transpose(A_inv, cofactorMatrix, sizeMatrix, det);
}// end function*/

__device__ bool inverse(float* Buff, float* A, float* A_inv, int sizeMatrix) {
    //	calculate determinante of matrix
    float det = Determinante(A, sizeMatrix, Buff, 0); 

    if (det == 0.0f) // || isinf(det) || isnan (det) )
        return false;

    /*if(sizeMatrix == 1){
        A_inv[0] = 1.0f;
    }
    else {
        Cofactor(A, A_inv, sizeMatrix, det, Buff, 0);	
    }*/
    
    return true;    
}

__global__ void test_inverse_kernel(size_t sizeMatrix, float *__restrict__ Buff, float *__restrict__ A, float *__restrict__ A_inv) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > 0)
        return;
    
    inverse(Buff, A, A_inv, sizeMatrix);
}


__global__ void concat_feat_kernel(
    const size_t num_sites,                // number of rays
    const size_t num_knn,  
    const size_t dim_feat,
    float *__restrict__ vertices,     // [N_voxels, 4] for each voxel => it's vertices
    int *__restrict__ activated,     // [N_voxels, 4] for each voxel => it's vertices
    float *__restrict__ grads,     // [N_voxels, 4] for each voxel => it's vertices
    float *__restrict__ feat,     // [N_voxels, 4] for each voxel => it's vertices
    int *__restrict__ neighbors
    )
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
            if (knn_id == -1) {
                for (int last_lvl = lvl_curr; last_lvl < nb_lvl; last_lvl++) {
                    /*for (int i = 0; i < dim_feat; i++) {
                        feat[dim_feat*(4*idx+last_lvl+1) + i] = feat[4*dim_feat*idx + i];
                    }  */
                    for (int i = 0; i < 3; i++) {
                        grads[3*(4*idx+last_lvl+1) + i] = grads[3*4*idx + i];
                    } 
                }
                return;
            }
            
            /*for (int i = 0; i < dim_feat; i++) {
                feat[dim_feat*(4*idx+lvl_curr+1) + i] = feat[dim_feat*(4*idx+lvl_curr+1) + i] + feat[4*dim_feat*knn_id + i]/32.0f;
            }  */
            for (int i = 0; i < 3; i++) {
                grads[3*(4*idx+lvl_curr+1) + i] = grads[3*(4*idx+lvl_curr+1) + i] + grads[3*4*knn_id + i]/32.0f;
            }                       
        }
    }

    return;
}



__global__ void concat_feat_kernel_o(
    const size_t num_sites,                // number of rays
    const size_t num_knn,  
    const size_t dim_feat,
    float *__restrict__ vertices,     // [N_voxels, 4] for each voxel => it's vertices
    int *__restrict__ activated,     // [N_voxels, 4] for each voxel => it's vertices
    float3 *__restrict__ grads,     // [N_voxels, 4] for each voxel => it's vertices
    float *__restrict__ feat,     // [N_voxels, 4] for each voxel => it's vertices
    int *__restrict__ neighbors
    )
{
    const size_t idx = blockIdx.x;
    if (idx >= num_sites)
    {
        return;
    }

    if (activated[idx] != 2)
        return;

    int nb_lvl = num_knn / 32;
    int knn_id; 
    
    __shared__ float3 smem[96];
    
    int lvl_curr = threadIdx.x / 32;
    int i_curr = threadIdx.x % 32;
    knn_id = neighbors[num_knn*idx + lvl_curr*32 + i_curr];
    if (knn_id != -1) {
        smem[32*lvl_curr + i_curr] = grads[4*knn_id]/32.0f;
    } else {
        smem[32*lvl_curr + i_curr] = grads[4*idx]/32.0f;
    }

    __syncthreads();
    if (threadIdx.x < 16) {
        smem[32*lvl_curr + threadIdx.x] = smem[32*lvl_curr + threadIdx.x] + smem[32*lvl_curr + threadIdx.x + 16]; 
    }
    __syncthreads();
    if (threadIdx.x < 8) {
        smem[32*lvl_curr + threadIdx.x] = smem[32*lvl_curr + threadIdx.x] + smem[32*lvl_curr + threadIdx.x + 8]; 
    }
    __syncthreads();
    if (threadIdx.x < 4) {
        smem[32*lvl_curr + threadIdx.x] = smem[32*lvl_curr + threadIdx.x] + smem[32*lvl_curr + threadIdx.x + 4]; 
    }
    __syncthreads();
    if (threadIdx.x < 2) {
        smem[32*lvl_curr + threadIdx.x] = smem[32*lvl_curr + threadIdx.x] + smem[32*lvl_curr + threadIdx.x + 2]; 
    }
    __syncthreads();

    if (i_curr == 0)
        grads[(4*idx+lvl_curr+1)] = smem[32*lvl_curr] + smem[32*lvl_curr + 1]; 

    return;
}













// compute the gradient for each tetrahedra
// gradient is fixed inside each tetrahedron
__global__ void knn_sdf_space_grad_cuda_kernel(
    const size_t num_sites,                // number of rays
    const size_t num_knn,                // number of rays   
    const int *__restrict__ neighbors,  // [N_voxels, 4] for each voxel => it's neighbors
    const float *__restrict__ sites,     // [N_voxels, 4] for each voxel => it's vertices
    const int *__restrict__ activated,     // [N_voxels, 4] for each voxel => it's vertices
    const float *__restrict__ SDF,     // [N_voxels, 4] for each voxel => it's vertices
    const float *__restrict__ Feat,     // [N_voxels, 4] for each voxel => it's vertices
    float *__restrict__ Weights,     // [N_voxels, 4] for each voxel => it's vertices
    float *__restrict__ grad,     // [N_voxels, 4] for each voxel => it's vertices
    float *__restrict__ grad_feat     // [N_voxels, 4] for each voxel => it's vertices
)
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_sites)
    {
        return;
    }

    if (activated[idx] == 0)
        return;

    float curr_site[3] {sites[3*idx], sites[3*idx + 1], sites[3*idx + 2]};    
    float curr_n[3] {0.0, 0.0, 0.0};
    float dX[3*_MAX_K_NN] {};
    float G[9] {0.0, 0.0, 0.0, 
                0.0, 0.0, 0.0, 
                0.0, 0.0, 0.0}; //dXT dX
    float G_inv[9] {0.0, 0.0, 0.0, 
                0.0, 0.0, 0.0, 
                0.0, 0.0, 0.0};

    int knn_id = -1;

    float *Weights_curr = &Weights[3*num_knn*idx];    

    for (int r_id = 0; r_id < _MAX_K_NN; r_id++) {
        knn_id = neighbors[num_knn*idx + r_id];

        // Coords of k-Nearest Neighbors
        curr_n[0] = sites[3*knn_id];
        curr_n[1] = sites[3*knn_id + 1];
        curr_n[2] = sites[3*knn_id + 2];

        // Calculate coefficients
        dX[3*r_id] = curr_n[0] - curr_site[0];
        dX[3*r_id + 1] = curr_n[1] - curr_site[1];
        dX[3*r_id + 2] = curr_n[2] - curr_site[2];
        
        G[0] = G[0] + dX[3*r_id]*dX[3*r_id]; G[1] = G[1] + dX[3*r_id]*dX[3*r_id+1];  G[2] = G[2] + dX[3*r_id]*dX[3*r_id+2];
        G[3] = G[3] + dX[3*r_id+1]*dX[3*r_id]; G[4] = G[4] + dX[3*r_id+1]*dX[3*r_id + 1]; G[5] = G[5] + dX[3*r_id+1]*dX[3*r_id + 2];
        G[6] = G[6] + dX[3*r_id+2]*dX[3*r_id]; G[7] = G[7] + dX[3*r_id+2]*dX[3*r_id + 1]; G[8] = G[8] + dX[3*r_id+2]*dX[3*r_id + 2];
    }


    // Compute inverse of G
    // det = a11 (a22 a33 – a23 a32) – a12 (a21 a33 – a23 a31) + a13 (a21 a32 – a22 a31)
    float det = G[0] * (G[4]*G[8] - G[5]*G[7]) - G[1]*(G[3]*G[8] - G[5]*G[6]) + G[2] * (G[3]*G[7] - G[4] * G[6]);
    if (det == 0.0f) { //!inverse(&Buff[(_BUFF_SIZE+64)*idx], G, G_inv, num_knn)) {
        grad[3*idx] = 0.0f;
        grad[3*idx + 1] = 0.0f;
        grad[3*idx + 2] = 0.0f;
        return;
    }
    
    G_inv[0] = (G[4]*G[8] - G[5]*G[7])/det; 
    G_inv[3] = -(G[3]*G[8] - G[5]*G[6])/det; 
    G_inv[6] = (G[3]*G[7] - G[4]*G[6])/det;
    G_inv[1] = -(G[1]*G[8] - G[2]*G[7])/det; 
    G_inv[4] = (G[0]*G[8] - G[2]*G[6])/det; 
    G_inv[7] = -(G[0]*G[7] - G[1]*G[6])/det; 
    G_inv[2] = (G[1]*G[5] - G[2]*G[4])/det; 
    G_inv[5] = -(G[0]*G[5] - G[2]*G[3])/det; 
    G_inv[8] = (G[0]*G[4] - G[1]*G[3])/det; 

    // Matrix multiplication
    for (int i = 0; i < num_knn; i++) {
        Weights_curr[3*i] = G_inv[0] * dX[3*i] + G_inv[1] * dX[3*i + 1] + G_inv[2] * dX[3*i + 2];
        Weights_curr[3*i + 1] = G_inv[3] * dX[3*i] + G_inv[4] * dX[3*i + 1] + G_inv[5] * dX[3*i + 2];
        Weights_curr[3*i + 2] = G_inv[6] * dX[3*i] + G_inv[7] * dX[3*i + 1] + G_inv[8] * dX[3*i + 2];
    }
    
    // Matrix multiplication
    float elem_0 = 0.0f;
    float elem_1 = 0.0f;
    float elem_2 = 0.0f;
    float feat_0[DIM_L_FEAT] {};//= {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    float feat_1[DIM_L_FEAT] {};//={0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    float feat_2[DIM_L_FEAT] {};//={0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    for (int i = 0; i < _MAX_K_NN; i++) {
        knn_id = neighbors[num_knn*idx + i];
        elem_0 += (SDF[knn_id] - SDF[idx]) * Weights_curr[3*i];
        elem_1 += (SDF[knn_id] - SDF[idx]) * Weights_curr[3*i + 1];
        elem_2 += (SDF[knn_id] - SDF[idx]) * Weights_curr[3*i + 2];
        for (int f_id = 0; f_id < DIM_L_FEAT; f_id++) {
            feat_0[f_id] = feat_0[f_id] + (Feat[DIM_L_FEAT*knn_id + f_id] - Feat[DIM_L_FEAT*idx + f_id]) * Weights_curr[3*i];
            feat_1[f_id] = feat_1[f_id] + (Feat[DIM_L_FEAT*knn_id + f_id] - Feat[DIM_L_FEAT*idx + f_id]) * Weights_curr[3*i + 1];
            feat_2[f_id] = feat_2[f_id] + (Feat[DIM_L_FEAT*knn_id + f_id] - Feat[DIM_L_FEAT*idx + f_id]) * Weights_curr[3*i + 2];
        }
    }
    
    grad[3*idx] = elem_0;
    grad[3*idx + 1] = elem_1;    
    grad[3*idx + 2] = elem_2;   
    
    for (int f_id = 0; f_id < DIM_L_FEAT; f_id++) {
        grad_feat[3*DIM_L_FEAT*idx + f_id] = feat_0[f_id];
        grad_feat[3*DIM_L_FEAT*idx + DIM_L_FEAT + f_id] = feat_1[f_id];  
        grad_feat[3*DIM_L_FEAT*idx + 2*DIM_L_FEAT + f_id] = feat_2[f_id];  
    }

    return;
}

__global__ void sdf_space_grad_cuda_kernel(
    const size_t num_tets,                // number of rays
    const int *__restrict__ tets,  // [N_voxels, 4] for each voxel => it's neighbors
    float *__restrict__ sites,     // [N_voxels, 4] for each voxel => it's vertices
    float *__restrict__ sdf,     // [N_voxels, 4] for each voxel => it's vertices
    float *__restrict__ feat,     // [N_voxels, 4] for each voxel => it's vertices
    float *__restrict__ grad_sdf,     // [N_voxels, 4] for each voxel => it's vertices)
    float *__restrict__ grad_feat,     // [N_voxels, 4] for each voxel => it's vertices
    float *__restrict__ weights_tot     // [N_voxels, 4] for each voxel => it's vertices
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

    float center_point[3] {0.0, 0.0, 0.0};
    for (int i = 0; i < 3; i++) {
        center_point[i] = (sites[3*ids[0] + i] + sites[3*ids[1] + i] + sites[3*ids[2] + i] + sites[3*ids[3] + i])/4.0f;
    }
    float center_sdf = (sdf[ids[0]] + sdf[ids[1]] + sdf[ids[2]] + sdf[ids[3]])/4.0f;
    float center_feat[DIM_L_FEAT] = { };
    for (int i = 0; i < DIM_L_FEAT; i++) {
        center_feat[i] = (feat[DIM_L_FEAT*ids[0] + i] + feat[DIM_L_FEAT*ids[1] + i] + feat[DIM_L_FEAT*ids[2] + i] + feat[DIM_L_FEAT*ids[3] + i])/4.0f;
    }

    float volume_tet = volume_tetrahedron_32(&sites[3*ids[0]], &sites[3*ids[1]], &sites[3*ids[2]], &sites[3*ids[3]]);
    
    float curr_n[3] {0.0, 0.0, 0.0};
    float dX[12];
    float G[9] {0.0, 0.0, 0.0, 
                0.0, 0.0, 0.0, 
                0.0, 0.0, 0.0}; //dXT dX
    float G_inv[9] {0.0, 0.0, 0.0, 
                0.0, 0.0, 0.0, 
                0.0, 0.0, 0.0};

    float Weights_curr[12] {0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0, 0.0};   

    for (int r_id = 0; r_id < 4; r_id++) {
        curr_n[0] = sites[3*ids[r_id]];
        curr_n[1] = sites[3*ids[r_id] + 1];
        curr_n[2] = sites[3*ids[r_id] + 2];

        // Calculate coefficients
        dX[3*r_id] = curr_n[0] - center_point[0];
        dX[3*r_id + 1] = curr_n[1] - center_point[1];
        dX[3*r_id + 2] = curr_n[2] - center_point[2];
        
        G[0] = G[0] + dX[3*r_id]*dX[3*r_id]; G[1] = G[1] + dX[3*r_id]*dX[3*r_id+1];  G[2] = G[2] + dX[3*r_id]*dX[3*r_id+2];
        G[3] = G[3] + dX[3*r_id+1]*dX[3*r_id]; G[4] = G[4] + dX[3*r_id+1]*dX[3*r_id + 1]; G[5] = G[5] + dX[3*r_id+1]*dX[3*r_id + 2];
        G[6] = G[6] + dX[3*r_id+2]*dX[3*r_id]; G[7] = G[7] + dX[3*r_id+2]*dX[3*r_id + 1]; G[8] = G[8] + dX[3*r_id+2]*dX[3*r_id + 2];
    }

    // Compute inverse of G
    // det = a11 (a22 a33 – a23 a32) – a12 (a21 a33 – a23 a31) + a13 (a21 a32 – a22 a31)
    float det = G[0] * (G[4]*G[8] - G[5]*G[7]) - G[1]*(G[3]*G[8] - G[5]*G[6]) + G[2] * (G[3]*G[7] - G[4] * G[6]);
    if (det == 0.0f) { 
        return;
    }
    
    G_inv[0] = (G[4]*G[8] - G[5]*G[7])/det; 
    G_inv[3] = -(G[3]*G[8] - G[5]*G[6])/det; 
    G_inv[6] = (G[3]*G[7] - G[4]*G[6])/det;
    G_inv[1] = -(G[1]*G[8] - G[2]*G[7])/det; 
    G_inv[4] = (G[0]*G[8] - G[2]*G[6])/det; 
    G_inv[7] = -(G[0]*G[7] - G[1]*G[6])/det; 
    G_inv[2] = (G[1]*G[5] - G[2]*G[4])/det; 
    G_inv[5] = -(G[0]*G[5] - G[2]*G[3])/det; 
    G_inv[8] = (G[0]*G[4] - G[1]*G[3])/det; 

    // Matrix multiplication
    for (int i = 0; i < 4; i++) {
        Weights_curr[3*i] = G_inv[0] * dX[3*i] + G_inv[1] * dX[3*i + 1] + G_inv[2] * dX[3*i + 2];
        Weights_curr[3*i + 1] = G_inv[3] * dX[3*i] + G_inv[4] * dX[3*i + 1] + G_inv[5] * dX[3*i + 2];
        Weights_curr[3*i + 2] = G_inv[6] * dX[3*i] + G_inv[7] * dX[3*i + 1] + G_inv[8] * dX[3*i + 2];
    }
    
    // Matrix multiplication
    float elem_0 = 0.0f;
    float elem_1 = 0.0f;
    float elem_2 = 0.0f;
    float feat_0[DIM_L_FEAT] = { };
    float feat_1[DIM_L_FEAT] = { };
    float feat_2[DIM_L_FEAT] = { };
    for (int i = 0; i < 4; i++) {
        elem_0 += (sdf[ids[i]] - center_sdf) * Weights_curr[3*i];
        elem_1 += (sdf[ids[i]] - center_sdf) * Weights_curr[3*i + 1];
        elem_2 += (sdf[ids[i]] - center_sdf) * Weights_curr[3*i + 2];
        for (int f_id = 0; f_id < DIM_L_FEAT; f_id++) {
            feat_0[f_id] = feat_0[f_id] + (feat[DIM_L_FEAT*ids[i] + f_id] - center_feat[f_id]) * Weights_curr[3*i];
            feat_1[f_id] = feat_1[f_id] + (feat[DIM_L_FEAT*ids[i] + f_id] - center_feat[f_id]) * Weights_curr[3*i + 1];
            feat_2[f_id] = feat_2[f_id] + (feat[DIM_L_FEAT*ids[i] + f_id] - center_feat[f_id]) * Weights_curr[3*i + 2];
        }
    }
    
    float norm_grad = elem_0*elem_0 + elem_1*elem_1 + elem_2*elem_2;

    float diff_loss[3]  {2.0*elem_0, 2.0*elem_1, 2.0*elem_2};
    if (norm_grad < 1.0f) {
        diff_loss[0] = -diff_loss[0];
        diff_loss[1] = -diff_loss[1];
        diff_loss[2] = -diff_loss[2];
    }

    for (int i = 0; i < 4; i++) {
        atomicAdd(&grad_sdf[3*ids[i]], elem_0*volume_tet);
        atomicAdd(&grad_sdf[3*ids[i] + 1], elem_1*volume_tet);
        atomicAdd(&grad_sdf[3*ids[i] + 2], elem_2*volume_tet);
        
        for (int f_id = 0; f_id < DIM_L_FEAT; f_id++) {
            atomicAdd(&grad_feat[3*DIM_L_FEAT*ids[i] + f_id], feat_0[f_id]*volume_tet);
            atomicAdd(&grad_feat[3*DIM_L_FEAT*ids[i] + DIM_L_FEAT + f_id], feat_1[f_id]*volume_tet);
            atomicAdd(&grad_feat[3*DIM_L_FEAT*ids[i] + 2*DIM_L_FEAT + f_id], feat_2[f_id]*volume_tet);
        }

        atomicAdd(&weights_tot[ids[i]], volume_tet);
    }


    return;
}


__global__ void normalize_grad_sdf_feat_kernel(
    const size_t num_sites,                // number of rays
    float *__restrict__ grad_sdf,     // [N_voxels, 4] for each voxel => it's vertices)
    float *__restrict__ grad_feat,     // [N_voxels, 4] for each voxel => it's vertices
    float *__restrict__ weights_tot
    )
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_sites)
    {
        return;
    }

    if (weights_tot[idx] == 0.0f)
        return;

    for (int i = 0; i < 3; i++)
        grad_sdf[3*idx + i] = grad_sdf[3*idx + i]/weights_tot[idx];

    for (int i = 0; i < DIM_L_FEAT; i++)
        grad_feat[DIM_L_FEAT*idx + i] = grad_feat[DIM_L_FEAT*idx + i]/weights_tot[idx];
}



__global__ void sdf_laplacian_cuda_kernel(
    const size_t num_tets,                // number of rays
    const int *__restrict__ tets,  // [N_voxels, 4] for each voxel => it's neighbors
    float *__restrict__ sites,     // [N_voxels, 4] for each voxel => it's vertices
    float *__restrict__ grad,     // [N_voxels, 4] for each voxel => it's vertices
    float *__restrict__ grad_lapl,     // [N_voxels, 4] for each voxel => it's vertices)
    float *__restrict__ weights_tot     // [N_voxels, 4] for each voxel => it's vertices
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

    float center_point[3] {0.0, 0.0, 0.0};
    for (int i = 0; i < 3; i++) {
        center_point[i] = (sites[3*ids[0] + i] + sites[3*ids[1] + i] + sites[3*ids[2] + i] + sites[3*ids[3] + i])/4.0f;
    }
    float center_grad[3] { (grad[3*ids[0]] + grad[3*ids[1]] + grad[3*ids[2]] + grad[3*ids[3]])/4.0f,
            (grad[3*ids[0] + 1] + grad[3*ids[1] + 1] + grad[3*ids[2] + 1] + grad[3*ids[3] + 1])/4.0f,
            (grad[3*ids[0] + 2] + grad[3*ids[1] + 2] + grad[3*ids[2] + 2] + grad[3*ids[3] + 2])/4.0f};

    float volume_tet = volume_tetrahedron_32(&sites[3*ids[0]], &sites[3*ids[1]], &sites[3*ids[2]], &sites[3*ids[3]]);
    
    float curr_n[3] {0.0, 0.0, 0.0};
    float dX[12];
    float G[9] {0.0, 0.0, 0.0, 
                0.0, 0.0, 0.0, 
                0.0, 0.0, 0.0}; //dXT dX
    float G_inv[9] {0.0, 0.0, 0.0, 
                0.0, 0.0, 0.0, 
                0.0, 0.0, 0.0};

    float Weights_curr[12] {0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0, 0.0};   

    for (int r_id = 0; r_id < 4; r_id++) {
        curr_n[0] = sites[3*ids[r_id]];
        curr_n[1] = sites[3*ids[r_id] + 1];
        curr_n[2] = sites[3*ids[r_id] + 2];

        // Calculate coefficients
        dX[3*r_id] = curr_n[0] - center_point[0];
        dX[3*r_id + 1] = curr_n[1] - center_point[1];
        dX[3*r_id + 2] = curr_n[2] - center_point[2];
        
        G[0] = G[0] + dX[3*r_id]*dX[3*r_id]; G[1] = G[1] + dX[3*r_id]*dX[3*r_id+1];  G[2] = G[2] + dX[3*r_id]*dX[3*r_id+2];
        G[3] = G[3] + dX[3*r_id+1]*dX[3*r_id]; G[4] = G[4] + dX[3*r_id+1]*dX[3*r_id + 1]; G[5] = G[5] + dX[3*r_id+1]*dX[3*r_id + 2];
        G[6] = G[6] + dX[3*r_id+2]*dX[3*r_id]; G[7] = G[7] + dX[3*r_id+2]*dX[3*r_id + 1]; G[8] = G[8] + dX[3*r_id+2]*dX[3*r_id + 2];
    }

    // Compute inverse of G
    // det = a11 (a22 a33 – a23 a32) – a12 (a21 a33 – a23 a31) + a13 (a21 a32 – a22 a31)
    float det = G[0] * (G[4]*G[8] - G[5]*G[7]) - G[1]*(G[3]*G[8] - G[5]*G[6]) + G[2] * (G[3]*G[7] - G[4] * G[6]);
    if (det == 0.0f) { 
        return;
    }
    
    G_inv[0] = (G[4]*G[8] - G[5]*G[7])/det; 
    G_inv[3] = -(G[3]*G[8] - G[5]*G[6])/det; 
    G_inv[6] = (G[3]*G[7] - G[4]*G[6])/det;
    G_inv[1] = -(G[1]*G[8] - G[2]*G[7])/det; 
    G_inv[4] = (G[0]*G[8] - G[2]*G[6])/det; 
    G_inv[7] = -(G[0]*G[7] - G[1]*G[6])/det; 
    G_inv[2] = (G[1]*G[5] - G[2]*G[4])/det; 
    G_inv[5] = -(G[0]*G[5] - G[2]*G[3])/det; 
    G_inv[8] = (G[0]*G[4] - G[1]*G[3])/det; 

    // Matrix multiplication
    for (int i = 0; i < 4; i++) {
        Weights_curr[3*i] = G_inv[0] * dX[3*i] + G_inv[1] * dX[3*i + 1] + G_inv[2] * dX[3*i + 2];
        Weights_curr[3*i + 1] = G_inv[3] * dX[3*i] + G_inv[4] * dX[3*i + 1] + G_inv[5] * dX[3*i + 2];
        Weights_curr[3*i + 2] = G_inv[6] * dX[3*i] + G_inv[7] * dX[3*i + 1] + G_inv[8] * dX[3*i + 2];
    }
    
    // Matrix multiplication
    float elem_0 = 0.0f;
    float elem_1 = 0.0f;
    float elem_2 = 0.0f;
    for (int i = 0; i < 4; i++) {
        elem_0 += (grad[3*ids[i]] - center_grad[0]) * Weights_curr[3*i];
        elem_1 += (grad[3*ids[i] + 1] - center_grad[1]) * Weights_curr[3*i + 1];
        elem_2 += (grad[3*ids[i] + 2] - center_grad[2]) * Weights_curr[3*i + 2];
    }
    
    float norm_lapl = elem_0*elem_0 + elem_1*elem_1 + elem_2*elem_2;

    for (int i = 0; i < 4; i++) {
        atomicAdd(&grad_lapl[ids[i]], (elem_0 * Weights_curr[3*i] * Weights_curr[3*i] + 
                                        elem_1 * Weights_curr[3*i + 1] * Weights_curr[3*i + 1] + 
                                        elem_2 * Weights_curr[3*i + 2]* Weights_curr[3*i + 2])*volume_tet);
        /*atomicAdd(&grad_lapl[3*ids[i]], elem_0*volume_tet * Weights_curr[3*i] );
        atomicAdd(&grad_lapl[3*ids[i] + 1], elem_1*volume_tet * Weights_curr[3*i + 1]);
        atomicAdd(&grad_lapl[3*ids[i] + 2], elem_2*volume_tet * Weights_curr[3*i + 2]);*/

        atomicAdd(&weights_tot[ids[i]], volume_tet);
    }


    return;
}


__global__ void normalize_laplacian_kernel(
    const size_t num_sites,                // number of rays
    float *__restrict__ grad_lapl,     // [N_voxels, 4] for each voxel => it's vertices)
    float *__restrict__ weights_tot
    )
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_sites)
    {
        return;
    }

    if (weights_tot[idx] == 0.0f)
        return;

    grad_lapl[idx] = grad_lapl[idx]/weights_tot[idx];
    //for (int i = 0; i < 3; i++)
    //    grad_lapl[3*idx + i] = grad_lapl[3*idx + i]/weights_tot[idx];

}




__global__ void cvt_grad_cuda_kernel(
    const size_t num_sites,                // number of rays
    const size_t num_knn,                // number of rays
    const float sigma,                // number of rays
    const double *__restrict__ thetas, 
    const double *__restrict__ phis, 
    const double *__restrict__ gammas, 
    const int *__restrict__ neighbors,  // [N_voxels, 4] for each voxel => it's neighbors
    const double *__restrict__ sites,     // [N_voxels, 4] for each voxel => it's vertices
    const int *__restrict__ freeze,     // [N_voxels, 4] for each voxel => it's vertices
    const float *__restrict__ sdf,     // [N_voxels, 4] for each voxel => it's vertices
    double *__restrict__ grad_sites,     // [N_voxels, 4] for each voxel => it's vertices
    float* loss
)
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_sites)
    {
        return;
    }

    if (freeze[idx] == 1)
        return;

    double ray[9] {1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0};

    double curr_site[3] {sites[3*idx], sites[3*idx + 1], sites[3*idx + 2]};
    
    double theta = thetas[idx] * 2.0f * PI;
    double phi = phis[idx] * 2.0f * PI;
    double gamma = gammas[idx] * 2.0f * PI;
    double rot[9] {cosf(phi)*cosf(gamma), sinf(theta)*sinf(phi)*cosf(gamma) - cosf(theta)*sinf(gamma), cosf(theta)*sinf(phi)*cosf(gamma) + sinf(theta)*sinf(gamma),
                    cosf(phi)*sinf(gamma), sinf(theta)*sinf(phi)*sinf(gamma) + cosf(theta)*cosf(gamma), cosf(theta)*sinf(phi)*sinf(gamma) - sinf(theta)*cosf(gamma),
                    -sinf(phi), sinf(theta)*cosf(phi), cosf(theta)*cosf(phi)}; 

    double curr_ray[3] {0.0, 0.0, 0.0};
    double nmle[3] {0.0, 0.0, 0.0};
    double b_point[3] {0.0, 0.0, 0.0};
    int knn_id = -1;

    double denom = 0.0f;
    double denom1 = 0.0f;
    double denom2 = 0.0f;
    double num1 = 0.0f;
    double num2 = 0.0f;
    double weight_cvt = 1.0f;

    double nmle_length, alpha, alpha_min, alpha_max;
    double d_min_dist[3] {0.0, 0.0, 0.0};
    double d_max_dist[3] {0.0, 0.0, 0.0};

    for (int r_id = 0; r_id < 3; r_id++) {
        
        curr_ray[0] = rot[0]*ray[3*r_id] + rot[1]*ray[3*r_id+1] + rot[2]*ray[3*r_id+2];
        curr_ray[1] = rot[3]*ray[3*r_id] + rot[4]*ray[3*r_id+1] + rot[5]*ray[3*r_id+2];
        curr_ray[2] = rot[6]*ray[3*r_id] + rot[7]*ray[3*r_id+1] + rot[8]*ray[3*r_id+2];

        double min_dist = 1.0e32;
        int min_id = -1;
        double max_dist = -1.0e32;
        int max_id = -1;
        double dist = 0.0f;
        alpha_min = 0.5f; alpha_max = 0.5f;

        for (int i = 0; i < num_knn; i++) {
            knn_id = neighbors[num_knn*idx + i];
            if (knn_id == -1)
                break;

            // Compute bisector normal vector
            nmle[0] = (sites[3*knn_id] - curr_site[0]);
            nmle[1] = (sites[3*knn_id + 1] - curr_site[1]);
            nmle[2] = (sites[3*knn_id + 2] - curr_site[2]);
            nmle_length = sqrt(nmle[0]*nmle[0] + nmle[1]*nmle[1] + nmle[2]*nmle[2]);
            if (nmle_length == 0.0f)
                continue;
            nmle[0] = nmle[0] / nmle_length;
            nmle[1] = nmle[1] / nmle_length;
            nmle[2] = nmle[2] / nmle_length;
            
            alpha = 0.5f;
            if (sdf[idx]*sdf[knn_id] < 0.0f) {
                alpha = double(fabs(sdf[idx])/(fabs(sdf[idx]) + fabs(sdf[knn_id])));
            } else if ((sdf[idx]-2.0f*sigma)*(sdf[knn_id]-2.0f*sigma) < 0.0f) {
                alpha = double(fabs((sdf[idx]-2.0f*sigma))/(fabs((sdf[idx]-2.0f*sigma)) + fabs(sdf[knn_id]-2.0f*sigma)));
            }else if ((sdf[idx]+2.0f*sigma)*(sdf[knn_id]+2.0f*sigma) < 0.0f) {
                alpha = double(fabs((sdf[idx]+2.0f*sigma))/(fabs((sdf[idx]+2.0f*sigma)) + fabs(sdf[knn_id]+2.0f*sigma)));
            }

            // Compute middle point
            b_point[0] = (alpha*sites[3*knn_id] + (1.0f-alpha)*curr_site[0]);
            b_point[1] = (alpha*sites[3*knn_id + 1] + (1.0f-alpha)*curr_site[1]);
            b_point[2] = (alpha*sites[3*knn_id + 2] + (1.0f-alpha)*curr_site[2]);
            
            // Compute ray - plane intersection point
            denom = nmle[0] * curr_ray[0] + nmle[1] * curr_ray[1] + nmle[2] * curr_ray[2];
            if (abs(denom) > 1.0e-6) {
                dist = (nmle[0] * (b_point[0] - curr_site[0]) + nmle[1] * (b_point[1] - curr_site[1]) + nmle[2] * (b_point[2] - curr_site[2])) / denom;
                if (dist >= 0.0 && dist < min_dist) {
                    min_dist = dist;
                    min_id = knn_id;
                    alpha_min = alpha;
                }
                if (dist <= 0.0 && dist > max_dist) {
                    max_dist = dist;
                    max_id = knn_id;
                    alpha_max = alpha;
                }
            }
        }                    

        if (min_id == -1 || max_id == -1)
            break;

        denom1 = curr_ray[0] * (sites[3*min_id] - curr_site[0]) + 
                curr_ray[1] * (sites[3*min_id+1] - curr_site[1]) + 
                curr_ray[2] * (sites[3*min_id+2] - curr_site[2]);     
        num1 = (sites[3*min_id] - curr_site[0]) * (sites[3*min_id] - curr_site[0]) + 
                (sites[3*min_id+1] - curr_site[1]) * (sites[3*min_id+1] - curr_site[1])+ 
                (sites[3*min_id+2] - curr_site[2]) * (sites[3*min_id+2] - curr_site[2]);
                
        denom2 = curr_ray[0] * (sites[3*max_id] - curr_site[0]) + 
                curr_ray[1] * (sites[3*max_id+1] - curr_site[1])+ 
                curr_ray[2] * (sites[3*max_id+2] - curr_site[2]);    
        num2 = (sites[3*max_id] - curr_site[0]) * (sites[3*max_id] - curr_site[0]) + 
                (sites[3*max_id+1] - curr_site[1]) * (sites[3*max_id+1] - curr_site[1])+ 
                (sites[3*max_id+2] - curr_site[2]) * (sites[3*max_id+2] - curr_site[2]);

        d_min_dist[0] = alpha_min * (-(sites[3*min_id] - curr_site[0])/denom1 + num1 * curr_ray[0]/(denom1*denom1));
        d_min_dist[1] = alpha_min * (-(sites[3*min_id+1] - curr_site[1])/denom1 + num1 * curr_ray[1]/(denom1*denom1));
        d_min_dist[2] = alpha_min * (-(sites[3*min_id+2] - curr_site[2])/denom1 + num1 * curr_ray[2]/(denom1*denom1));

        d_max_dist[0] = alpha_max * (-(sites[3*max_id] - curr_site[0])/denom2 + num2 * curr_ray[0]/(denom2*denom2));
        d_max_dist[1] = alpha_max * (-(sites[3*max_id+1] - curr_site[1])/denom2 + num2 * curr_ray[1]/(denom2*denom2));   
        d_max_dist[2] = alpha_max * (-(sites[3*max_id+2] - curr_site[2])/denom2 + num2 * curr_ray[2]/(denom2*denom2));       

        // gradients related to CVT loss
        grad_sites[3*idx] = grad_sites[3*idx] - (min_dist + max_dist) * (d_min_dist[0] + d_max_dist[0]);  
        grad_sites[3*idx+1] = grad_sites[3*idx+1] - (min_dist + max_dist) * (d_min_dist[1] + d_max_dist[1]);    
        grad_sites[3*idx+2] = grad_sites[3*idx+2] - (min_dist + max_dist) * (d_min_dist[2] + d_max_dist[2]);   
        /*grad_sites[3*idx] = grad_sites[3*idx] - (min_dist * d_min_dist[0] + max_dist * d_max_dist[0]);  
        grad_sites[3*idx+1] = grad_sites[3*idx+1] - (min_dist * d_min_dist[1] + max_dist * d_max_dist[1]);    
        grad_sites[3*idx+2] = grad_sites[3*idx+2] - (min_dist * d_min_dist[2] + max_dist * d_max_dist[2]);   */
        /*grad_sites[3*idx] = grad_sites[3*idx] - (min_dist+max_dist) * curr_ray[0];  
        grad_sites[3*idx+1] = grad_sites[3*idx+1] - (min_dist+max_dist) * curr_ray[1];    
        grad_sites[3*idx+2] = grad_sites[3*idx+2] - (min_dist+max_dist) * curr_ray[2];*/ 
        atomicAdd(loss, fabs(float(min_dist+max_dist)));
    }

    return;
}

__global__ void sdf_grad_cuda_kernel(
    const size_t num_sites,                // number of rays
    const size_t num_knn,                // number of rays
    const int *__restrict__ neighbors,  // [N_voxels, 4] for each voxel => it's neighbors
    const float *__restrict__ sites,     // [N_voxels, 4] for each voxel => it's vertices
    const float *__restrict__ sdf,     // [N_voxels, 4] for each voxel => it's vertices
    float *__restrict__ grad_sites     // [N_voxels, 4] for each voxel => it's vertices
)
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_sites)
    {
        return;
    }

    float curr_site[3] {sites[3*idx], sites[3*idx + 1], sites[3*idx + 2]};
    
    float curr_ray[3] {0.0, 0.0, 0.0};
    float nmle[3] {0.0, 0.0, 0.0};
    float b_point[3] {0.0, 0.0, 0.0};
    int knn_id = -1;

    float denom = 0.0f;

    float nmle_length;
    float sdf_mid, dot_prod_g;

    for (int r_id = 0; r_id < num_knn; r_id++) {
        knn_id = neighbors[num_knn*idx + r_id];
        if (sdf[idx]*sdf[knn_id] >= 0.0f)
            continue;
        
        // Check if k-nn neighbor is adjacent in the CVT 
        curr_ray[0] = sites[3*knn_id] - sites[3*idx];
        curr_ray[1] = sites[3*knn_id + 1] - sites[3*idx + 1];
        curr_ray[2] = sites[3*knn_id + 2] - sites[3*idx + 2];
        nmle_length = sqrt(curr_ray[0]*curr_ray[0] + curr_ray[1]*curr_ray[1] + curr_ray[2]*curr_ray[2]);
        if (nmle_length == 0.0f)
            continue;
        curr_ray[0] = curr_ray[0] / nmle_length;
        curr_ray[1] = curr_ray[1] / nmle_length;
        curr_ray[2] = curr_ray[2] / nmle_length;

        float min_dist = 1.0e32;
        int min_id = -1;
        float dist = 0.0f;
        int kk_nn_id = -1;

        for (int i = 0; i < num_knn; i++) {
            kk_nn_id  = neighbors[num_knn*idx + i];
            if (kk_nn_id == -1)
                break;

            // Compute bisector normal vector
            nmle[0] = (sites[3*kk_nn_id] - curr_site[0]);
            nmle[1] = (sites[3*kk_nn_id + 1] - curr_site[1]);
            nmle[2] = (sites[3*kk_nn_id + 2] - curr_site[2]);
            nmle_length = sqrt(nmle[0]*nmle[0] + nmle[1]*nmle[1] + nmle[2]*nmle[2]);
            if (nmle_length == 0.0f)
                continue;
            nmle[0] = nmle[0] / nmle_length;
            nmle[1] = nmle[1] / nmle_length;
            nmle[2] = nmle[2] / nmle_length;
            
            // Compute middle point
            b_point[0] = (sites[3*kk_nn_id] + curr_site[0]) / 2.0f;
            b_point[1] = (sites[3*kk_nn_id + 1] + curr_site[1]) / 2.0f;
            b_point[2] = (sites[3*kk_nn_id + 2] + curr_site[2]) / 2.0f;
            
            // Compute ray - plane intersection point
            denom = nmle[0] * curr_ray[0] + nmle[1] * curr_ray[1] + nmle[2] * curr_ray[2];
            if (abs(denom) > 1.0e-6) {
                dist = (nmle[0] * (b_point[0] - curr_site[0]) + nmle[1] * (b_point[1] - curr_site[1]) + nmle[2] * (b_point[2] - curr_site[2])) / denom;
                if (dist >= 0.0 && dist < min_dist) {
                    min_dist = dist;
                    min_id = kk_nn_id;
                }
            }
        }                    

        if (knn_id != min_id) {
            // curr neighbor is not adjacent
            continue;
        }

        sdf_mid = (sdf[idx] + sdf[knn_id])/2.0f; 
        denom = sqrt((sites[3*knn_id] - curr_site[0]) * (sites[3*knn_id] - curr_site[0]) + 
                    (sites[3*knn_id+1] - curr_site[1]) * (sites[3*knn_id+1] - curr_site[1])+ 
                    (sites[3*knn_id+2] - curr_site[2]) * (sites[3*knn_id+2] - curr_site[2]));  
        
        if (sdf_mid * sdf[idx] > 0.0f) {
            atomicAdd(&grad_sites[3*idx], -fabs(sdf_mid)*(sites[3*knn_id] - curr_site[0])/denom);
            atomicAdd(&grad_sites[3*idx + 1], -fabs(sdf_mid)*(sites[3*knn_id+1] - curr_site[1])/denom);
            atomicAdd(&grad_sites[3*idx + 2], -fabs(sdf_mid)*(sites[3*knn_id+2] - curr_site[2])/denom);
            
            atomicAdd(&grad_sites[3*knn_id], -fabs(sdf_mid)*(sites[3*knn_id] - curr_site[0])/denom);
            atomicAdd(&grad_sites[3*knn_id + 1], -fabs(sdf_mid)*(sites[3*knn_id+1] - curr_site[1])/denom);
            atomicAdd(&grad_sites[3*knn_id + 2], -fabs(sdf_mid)*(sites[3*knn_id+2] - curr_site[2])/denom);
        } else {
            atomicAdd(&grad_sites[3*idx], fabs(sdf_mid)*(sites[3*knn_id] - curr_site[0])/denom);
            atomicAdd(&grad_sites[3*idx + 1], fabs(sdf_mid)*(sites[3*knn_id+1] - curr_site[1])/denom);
            atomicAdd(&grad_sites[3*idx + 2], fabs(sdf_mid)*(sites[3*knn_id+2] - curr_site[2])/denom);
            
            atomicAdd(&grad_sites[3*knn_id], fabs(sdf_mid)*(sites[3*knn_id] - curr_site[0])/denom);
            atomicAdd(&grad_sites[3*knn_id + 1], fabs(sdf_mid)*(sites[3*knn_id+1] - curr_site[1])/denom);
            atomicAdd(&grad_sites[3*knn_id + 2], fabs(sdf_mid)*(sites[3*knn_id+2] - curr_site[2])/denom);
        } 
    }

    return;
}


__global__ void update_sdf_cuda_kernel(
    const size_t num_sites,                // number of rays
    const size_t num_knn,                // number of rays    
    const int *__restrict__ neighbors,  // [N_voxels, 4] for each voxel => it's neighbors
    const float *__restrict__ sites,     // [N_voxels, 4] for each voxel => it's vertices
    const float *__restrict__ sdf,     // [N_voxels, 4] for each voxel => it's vertices
    const float *__restrict__ feat,     // [N_voxels, 4] for each voxel => it's vertices
    const float *__restrict__ grad_sites,     // [N_voxels, 4] for each voxel => it's vertices
    float *__restrict__ sdf_diff,     // [N_voxels, 4] for each voxel => it's vertices
    float *__restrict__ feat_diff  
)
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_sites)
    {
        return;
    }

    float curr_site[2] {sites[2*idx], sites[2*idx + 1]};
    float curr_ray[2] {grad_sites[2*idx], grad_sites[2*idx + 1]};
    if (grad_sites[2*idx] == 0.0f && grad_sites[2*idx + 1] == 0.0f)
        return;
    float norm_grad = sqrt(grad_sites[2*idx]*grad_sites[2*idx] + grad_sites[2*idx + 1]*grad_sites[2*idx + 1]);
    curr_ray[0] = curr_ray[0]/norm_grad;
    curr_ray[1] = curr_ray[1]/norm_grad;

    float nmle[2] {0.0f, 0.0f};
    float b_point[2] {0.0f, 0.0f};

    float min_dist = 1.0e32;
    int min_id = -1;
    float dist = 0.0f;
    float nmle_length, denom;
    int knn_id;

    for (int i = 0; i < num_knn; i++) {
        knn_id = neighbors[num_knn*(idx+1) + i];
        if (knn_id == -1)
            break;

        // Compute bisector normal vector
        nmle[0] = (sites[2*knn_id] - curr_site[0]);
        nmle[1] = (sites[2*knn_id + 1] - curr_site[1]);
        nmle_length = sqrt(nmle[0]*nmle[0] + nmle[1]*nmle[1]);
        if (nmle_length == 0.0f)
            continue;
        nmle[0] = nmle[0] / nmle_length;
        nmle[1] = nmle[1] / nmle_length;
        
        // Compute middle point
        b_point[0] = (sites[2*knn_id] + curr_site[0]) / 2.0f;
        b_point[1] = (sites[2*knn_id + 1] + curr_site[1]) / 2.0f;
        
        // Compute ray - plane intersection point
        denom = nmle[0] * curr_ray[0] + nmle[1] * curr_ray[1];
        if (abs(denom) > 1.0e-6) {
            dist = (nmle[0] * (b_point[0] - curr_site[0]) + nmle[1] * (b_point[1] - curr_site[1])) / denom;
            if (dist >= 0.0 && dist < min_dist) {
                min_dist = dist;
                min_id = knn_id;
            }
        }
    }
    
    if (min_id == -1 || min_id == idx)
        return;

    float sdf_diff_curr = (sdf[min_id]-sdf[idx]);
    float dist_sites =  sqrt((sites[2*min_id] - curr_site[0])*(sites[2*min_id] - curr_site[0]) + (sites[2*min_id + 1] - curr_site[1])*(sites[2*min_id + 1] - curr_site[1]));
    if (dist_sites > 0.0f) {
        sdf_diff[idx] = sdf_diff_curr*norm_grad/dist_sites;
        for (int f_d = 0; f_d < 6; f_d++) {
            feat_diff[6*idx + f_d] = (feat[6*min_id+f_d]-feat[6*idx+f_d])*norm_grad/dist_sites;
        }
    }
}


__global__ void diff_tensor_kernel(
    const size_t num_tets,                // number of rays
    const int *__restrict__ tets,  // [N_voxels, 4] for each voxel => it's neighbors
    double *__restrict__ sites,     // [N_voxels, 4] for each voxel => it's vertices
    float *__restrict__ vol,     // [N_voxels, 4] for each voxel => it's vertices
    float *__restrict__ weights,     // [N_voxels, 4] for each voxel => it's vertices
    float *__restrict__ weights_tot
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

    double center_point[3] {0.0, 0.0, 0.0};
    for (int i = 0; i < 3; i++) {
        center_point[i] = (sites[3*ids[0] + i] + sites[3*ids[1] + i] + sites[3*ids[2] + i] + sites[3*ids[3] + i])/4.0f;
    }

    vol[idx] = volume_tetrahedron_64(&sites[3*ids[0]], &sites[3*ids[1]], &sites[3*ids[2]], &sites[3*ids[3]]);
    
    double curr_n[3] {0.0, 0.0, 0.0};
    double dX[12];
    double G[9] {0.0, 0.0, 0.0, 
                0.0, 0.0, 0.0, 
                0.0, 0.0, 0.0}; //dXT dX
    double G_inv[9] {0.0, 0.0, 0.0, 
                0.0, 0.0, 0.0, 
                0.0, 0.0, 0.0};

    for (int r_id = 0; r_id < 4; r_id++) {
        curr_n[0] = sites[3*ids[r_id]];
        curr_n[1] = sites[3*ids[r_id] + 1];
        curr_n[2] = sites[3*ids[r_id] + 2];

        // Calculate coefficients
        dX[3*r_id] = curr_n[0] - center_point[0];
        dX[3*r_id + 1] = curr_n[1] - center_point[1];
        dX[3*r_id + 2] = curr_n[2] - center_point[2];
        
        G[0] = G[0] + dX[3*r_id]*dX[3*r_id]; G[1] = G[1] + dX[3*r_id]*dX[3*r_id+1];  G[2] = G[2] + dX[3*r_id]*dX[3*r_id+2];
        G[3] = G[3] + dX[3*r_id+1]*dX[3*r_id]; G[4] = G[4] + dX[3*r_id+1]*dX[3*r_id + 1]; G[5] = G[5] + dX[3*r_id+1]*dX[3*r_id + 2];
        G[6] = G[6] + dX[3*r_id+2]*dX[3*r_id]; G[7] = G[7] + dX[3*r_id+2]*dX[3*r_id + 1]; G[8] = G[8] + dX[3*r_id+2]*dX[3*r_id + 2];
    }

    // Compute inverse of G
    // det = a11 (a22 a33 – a23 a32) – a12 (a21 a33 – a23 a31) + a13 (a21 a32 – a22 a31)
    double det = G[0] * (G[4]*G[8] - G[5]*G[7]) - G[1]*(G[3]*G[8] - G[5]*G[6]) + G[2] * (G[3]*G[7] - G[4] * G[6]);
    if (det == 0.0f) { 
        return;
    }
    
    G_inv[0] = (G[4]*G[8] - G[5]*G[7])/det; 
    G_inv[3] = -(G[3]*G[8] - G[5]*G[6])/det; 
    G_inv[6] = (G[3]*G[7] - G[4]*G[6])/det;
    G_inv[1] = -(G[1]*G[8] - G[2]*G[7])/det; 
    G_inv[4] = (G[0]*G[8] - G[2]*G[6])/det; 
    G_inv[7] = -(G[0]*G[7] - G[1]*G[6])/det; 
    G_inv[2] = (G[1]*G[5] - G[2]*G[4])/det; 
    G_inv[5] = -(G[0]*G[5] - G[2]*G[3])/det; 
    G_inv[8] = (G[0]*G[4] - G[1]*G[3])/det; 

    // Matrix multiplication
    for (int i = 0; i < 4; i++) {
        weights[12*idx + 3*i] = float(G_inv[0] * dX[3*i] + G_inv[1] * dX[3*i + 1] + G_inv[2] * dX[3*i + 2]);
        weights[12*idx + 3*i + 1] = float(G_inv[3] * dX[3*i] + G_inv[4] * dX[3*i + 1] + G_inv[5] * dX[3*i + 2]);
        weights[12*idx + 3*i + 2] = float(G_inv[6] * dX[3*i] + G_inv[7] * dX[3*i + 1] + G_inv[8] * dX[3*i + 2]);
    }   
    
    
    for (int i = 0; i < 4; i++) {
        atomicAdd(&weights_tot[ids[i]], vol[idx]);
    }

    return;
}


__global__ void eikonal_grad_kernel(
    const size_t num_tets,                // number of rays
    const int *__restrict__ tets,  // [N_voxels, 4] for each voxel => it's neighbors
    const int *__restrict__ valid_tets,  // [N_voxels, 4] for each voxel => it's neighbors
    float *__restrict__ sites,     // [N_voxels, 4] for each voxel => it's vertices
    int *__restrict__ activated,     // [N_voxels, 4] for each voxel => it's vertices
    float *__restrict__ sdf,     // [N_voxels, 4] for each voxel => it's vertices
    float *__restrict__ sdf_smooth,     // [N_voxels, 4] for each voxel => it's vertices
    float *__restrict__ feat,     // [N_voxels, 4] for each voxel => it's vertices
    float *__restrict__ grad_eik,     // [N_voxels, 4] for each voxel => it's vertices
    float *__restrict__ grad_smooth,     // [N_voxels, 4] for each voxel => it's vertices)
    float *__restrict__ grad_sdf,     // [N_voxels, 4] for each voxel => it's vertices)
    float *__restrict__ vol,     // [N_voxels, 4] for each voxel => it's vertices
    float *__restrict__ weights,     // [N_voxels, 4] for each voxel => it's vertices
    float *__restrict__ weights_tot,     // [N_voxels, 4] for each voxel => it's vertices
    float *__restrict__ Loss
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

    /*if (activated[ids[0]] == -1 && 
        activated[ids[1]] == -1 && 
        activated[ids[2]] == -1 && 
        activated[ids[3]] == -1)
        return;*/

    if (valid_tets[idx] == 0)
        return;

    float center_sdf = (sdf[ids[0]] + sdf[ids[1]] + sdf[ids[2]] + sdf[ids[3]])/4.0f;
    float center_sdf_smooth = (sdf_smooth[ids[0]] + sdf_smooth[ids[1]] + sdf_smooth[ids[2]] + sdf_smooth[ids[3]])/4.0f;

    float volume_tet = vol[idx];
    
    float *Weights_curr = &weights[12*idx];
    
    // Matrix multiplication
    float elem_0 = 0.0f;
    float elem_1 = 0.0f;
    float elem_2 = 0.0f;
    float elem_smooth_0 = 0.0f;
    float elem_smooth_1 = 0.0f;
    float elem_smooth_2 = 0.0f;
    for (int i = 0; i < 4; i++) {
        elem_0 += (sdf[ids[i]] - center_sdf) * Weights_curr[3*i];
        elem_1 += (sdf[ids[i]] - center_sdf) * Weights_curr[3*i + 1];
        elem_2 += (sdf[ids[i]] - center_sdf) * Weights_curr[3*i + 2];
        
        elem_smooth_0 += (sdf_smooth[ids[i]] - center_sdf_smooth) * Weights_curr[3*i];
        elem_smooth_1 += (sdf_smooth[ids[i]] - center_sdf_smooth) * Weights_curr[3*i + 1];
        elem_smooth_2 += (sdf_smooth[ids[i]] - center_sdf_smooth) * Weights_curr[3*i + 2];
    }
    
    float norm_grad = sqrt(elem_0*elem_0 + elem_1*elem_1 + elem_2*elem_2);
    float norm_grad_smooth = sqrt(elem_smooth_0*elem_smooth_0 + elem_smooth_1*elem_smooth_1 + elem_smooth_2*elem_smooth_2);

    float diff_loss[3]  {};
    if (norm_grad > 0.0f) {
        diff_loss[0] = 2.0f*(norm_grad-1.0f) * elem_0 / norm_grad;
        diff_loss[1] = 2.0f*(norm_grad-1.0f) * elem_1 / norm_grad;
        diff_loss[2] = 2.0f*(norm_grad-1.0f) * elem_2 / norm_grad;
        /*diff_loss[0] = 2.0f*(norm_grad-1.0f) * elem_smooth_0 / norm_grad;
        diff_loss[1] = 2.0f*(norm_grad-1.0f) * elem_smooth_1 / norm_grad;
        diff_loss[2] = 2.0f*(norm_grad-1.0f) * elem_smooth_2 / norm_grad;*/
    }

    /*float diff_loss[3]  {2.0*elem_0, 2.0*elem_1, 2.0*elem_2};
    if (norm_grad < 1.0f) {
        diff_loss[0] = -diff_loss[0];
        diff_loss[1] = -diff_loss[1];
        diff_loss[2] = -diff_loss[2];
    }*/

    for (int i = 0; i < 4; i++) {
        if (weights_tot[ids[i]] == 0.0f) 
            continue;
        
        
        atomicAdd(&grad_eik[ids[i]], (diff_loss[0] * (3.0f*Weights_curr[3*i] - Weights_curr[3*((i+1)%4)] - Weights_curr[3*((i+2)%4)] - Weights_curr[3*((i+3)%4)]) / 4.0f +
                                        diff_loss[1] * (3.0f*Weights_curr[3*i + 1] - Weights_curr[3*((i+1)%4) + 1] - Weights_curr[3*((i+2)%4) + 1] - Weights_curr[3*((i+3)%4) + 1]) / 4.0f + 
                                        diff_loss[2] * (3.0f*Weights_curr[3*i + 2] - Weights_curr[3*((i+1)%4) + 2] - Weights_curr[3*((i+2)%4) + 2] - Weights_curr[3*((i+3)%4) + 2]) / 4.0f) * volume_tet / weights_tot[ids[i]]);
                                        
        if (norm_grad > 1.0e-8 && norm_grad_smooth > 1.0e-8) {
            float dot_prod = elem_0 * elem_smooth_0 + elem_1 * elem_smooth_1 + elem_2 * elem_smooth_2;
            atomicAdd(&grad_smooth[ids[i]], ((1.0f / norm_grad_smooth) * (dot_prod * elem_0 / (norm_grad*norm_grad*norm_grad) - elem_smooth_0 / norm_grad) * (3.0f*Weights_curr[3*i] - Weights_curr[3*((i+1)%4)] - Weights_curr[3*((i+2)%4)] - Weights_curr[3*((i+3)%4)]) / 4.0f  + 
                                            (1.0f / norm_grad_smooth) * (dot_prod * elem_1 / (norm_grad*norm_grad*norm_grad) - elem_smooth_1 / norm_grad)* (3.0f*Weights_curr[3*i + 1] - Weights_curr[3*((i+1)%4) + 1] - Weights_curr[3*((i+2)%4) + 1] - Weights_curr[3*((i+3)%4) + 1]) / 4.0f + 
                                            (1.0f / norm_grad_smooth) * (dot_prod * elem_2 / (norm_grad*norm_grad*norm_grad) - elem_smooth_2 / norm_grad) * (3.0f*Weights_curr[3*i + 2] - Weights_curr[3*((i+1)%4) + 2] - Weights_curr[3*((i+2)%4) + 2] - Weights_curr[3*((i+3)%4) + 2]) / 4.0f) * volume_tet / weights_tot[ids[i]]);

            /*atomicAdd(&grad_smooth[ids[i]], ((elem_0 - elem_smooth_0) * (3.0f*Weights_curr[3*i] - Weights_curr[3*((i+1)%4)] - Weights_curr[3*((i+2)%4)] - Weights_curr[3*((i+3)%4)]) / 4.0f  + 
                                            (elem_1 - elem_smooth_1) * (3.0f*Weights_curr[3*i + 1] - Weights_curr[3*((i+1)%4) + 1] - Weights_curr[3*((i+2)%4) + 1] - Weights_curr[3*((i+3)%4) + 1]) / 4.0f + 
                                            (elem_2 - elem_smooth_2)* (3.0f*Weights_curr[3*i + 2] - Weights_curr[3*((i+1)%4) + 2] - Weights_curr[3*((i+2)%4) + 2] - Weights_curr[3*((i+3)%4) + 2]) / 4.0f)*volume_tet / weights_tot[ids[i]]);*/

        }

        /*atomicAdd(&grad_smooth[ids[i]], ((elem_0 - elem_smooth_0) * Weights_curr[3*i] + 
                                        (elem_1 - elem_smooth_1) * Weights_curr[3*i + 1] + 
                                        (elem_2 - elem_smooth_2) * Weights_curr[3*i + 2])*volume_tet / (2.0f*weights_tot[ids[i]]));*/

        atomicAdd(&grad_sdf[3*ids[i]], elem_0*volume_tet / weights_tot[ids[i]]);
        atomicAdd(&grad_sdf[3*ids[i] + 1], elem_1*volume_tet / weights_tot[ids[i]]);
        atomicAdd(&grad_sdf[3*ids[i] + 2], elem_2*volume_tet / weights_tot[ids[i]]);

        atomicAdd(&Loss[ids[i]], abs(sqrt(norm_grad)-1)*volume_tet / weights_tot[ids[i]]);
    }


    return;
}

__global__ void normalize_grad_kernel(
    const size_t num_sites,                // number of rays
    float *__restrict__ grad_eik,     // [N_voxels, 4] for each voxel => it's vertices
    float *__restrict__ grad_smooth,     // [N_voxels, 4] for each voxel => it's vertices)
    float *__restrict__ grad_sdf,     // [N_voxels, 4] for each voxel => it's vertices)
    float *__restrict__ grad_feat,     // [N_voxels, 4] for each voxel => it's vertices
    float *__restrict__ weights_tot,     // [N_voxels, 4] for each voxel => it's vertices
    float *__restrict__ Loss
    )
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_sites)
    {
        return;
    }
    
    if (weights_tot[idx] == 0.0f) {
        return;
    }

    grad_eik[idx] = grad_eik[idx]/weights_tot[idx];
    grad_smooth[idx] = grad_smooth[idx]/weights_tot[idx];


    for (int i = 0; i < 3; i++)
        grad_sdf[3*idx + i] = grad_sdf[3*idx + i]/weights_tot[idx];

    for (int i = 0; i < 6; i++)
        grad_feat[6*idx + i] = grad_feat[6*idx + i]/weights_tot[idx];

    Loss[idx] = Loss[idx]/weights_tot[idx];
}

__global__ void backprop_norm_grad_kernel(
    const size_t num_tets,                // number of rays
    const int *__restrict__ tets,  // [N_voxels, 4] for each voxel => it's neighbors
    float *__restrict__ sites,     // [N_voxels, 4] for each voxel => it's vertices
    int *__restrict__ activated,     // [N_voxels, 4] for each voxel => it's vertices
    float *__restrict__ grad_sdf,     // [N_voxels, 4] for each voxel => it's vertices)
    float *__restrict__ grad_norm,     // [N_voxels, 4] for each voxel => it's vertices)
    float *__restrict__ vol,     // [N_voxels, 4] for each voxel => it's vertices
    float *__restrict__ weights,     // [N_voxels, 4] for each voxel => it's vertices
    float *__restrict__ weights_tot     // [N_voxels, 4] for each voxel => it's vertices
    )
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_tets)
    {
        return;
    }

    int ids[4] = {0, 0, 0, 0};
    ids[0] = tets[4*idx];  ids[1] = tets[4*idx + 1];  ids[2] = tets[4*idx + 2];
    ids[3] = ids[0] ^ ids[1] ^ ids[2] ^ tets[4*idx + 3];

    float volume_tet = vol[idx];
    
    float *Weights_curr = &weights[12*idx];
    
    for (int i = 0; i < 4; i++) {
        if (weights_tot[ids[i]] == 0.0f) 
            continue;

        /*atomicAdd(&grad_sdf[ids[i]], (grad_norm[3*ids[i]] * Weights_curr[3*i] + 
                                        grad_norm[3*ids[i] + 1] * Weights_curr[3*i + 1] + 
                                        grad_norm[3*ids[i] + 2] * Weights_curr[3*i + 2])*volume_tet / (2.0f*weights_tot[ids[i]]));*/

        atomicAdd(&grad_sdf[ids[i]], (grad_norm[3*ids[i]] * (3.0f*Weights_curr[3*i] - Weights_curr[3*((i+1)%4)] - Weights_curr[3*((i+2)%4)] - Weights_curr[3*((i+3)%4)]) / 4.0f +
                                        grad_norm[3*ids[i] + 1] * (3.0f*Weights_curr[3*i + 1] - Weights_curr[3*((i+1)%4) + 1] - Weights_curr[3*((i+2)%4) + 1] - Weights_curr[3*((i+3)%4) + 1]) / 4.0f + 
                                        grad_norm[3*ids[i] + 2] * (3.0f*Weights_curr[3*i + 2] - Weights_curr[3*((i+1)%4) + 2] - Weights_curr[3*((i+2)%4) + 2] - Weights_curr[3*((i+3)%4) + 2]) / 4.0f) * volume_tet / weights_tot[ids[i]]);
    }


    return;
}


/** CPU functions **/
/** CPU functions **/
/** CPU functions **/


// 
void test_inverse_cuda(
    size_t sizeMatrix,
    torch::Tensor Buff,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor A,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor A_inv    // [N_sites, 3] for each voxel => it's vertices
)   {
        const int threads = 1;
        const int blocks = 1;
        AT_DISPATCH_FLOATING_TYPES( A.type(),"test_inverse_cuda", ([&] {  
            test_inverse_kernel CUDA_KERNEL(blocks,threads) (
                sizeMatrix,
                Buff.data_ptr<float>(),
                A.data_ptr<float>(),
                A_inv.data_ptr<float>()); 
    }));
}

void knn_sdf_space_grad_cuda(
    size_t num_sites,                // number of rays
    size_t num_knn,                // number of rays
    torch::Tensor  neighbors,  // [N_voxels, 4] for each voxel => it's neighbors
    torch::Tensor  sites,  // [N_voxels, 4] for each voxel => it's neighbors
    torch::Tensor  activated,  // [N_voxels, 4] for each voxel => it's neighbors
    torch::Tensor  sdf,  // [N_voxels, 4] for each voxel => it's neighbors
    torch::Tensor  feat,  // [N_voxels, 4] for each voxel => it's neighbors
    torch::Tensor  grad_sdf,     // [N_voxels, 4] for each voxel => it's vertices
    torch::Tensor  grad_feat,     // [N_voxels, 4] for each voxel => it's vertices
    torch::Tensor  weights_tot
)   {
        const int threads = 512;
        const int blocks = (num_sites + threads - 1) / threads; // ceil for example 8192 + 255 / 256 = 32
        AT_DISPATCH_FLOATING_TYPES( sites.type(),"knn_sdf_space_grad_cuda_kernel", ([&] {  
            knn_sdf_space_grad_cuda_kernel CUDA_KERNEL(blocks,threads) (
                num_sites,            
                num_knn,            
                neighbors.data_ptr<int>(),
                sites.data_ptr<float>(),
                activated.data_ptr<int>(),
                sdf.data_ptr<float>(),
                feat.data_ptr<float>(),
                weights_tot.data_ptr<float>(),
                grad_sdf.data_ptr<float>(),
                grad_feat.data_ptr<float>()); 
        }));
}

// 
void sdf_space_grad_cuda(
    size_t num_tets,                // number of rays
    size_t num_sites,                // number of rays
    torch::Tensor  tets,  // [N_voxels, 4] for each voxel => it's neighbors
    torch::Tensor  sites,  // [N_voxels, 4] for each voxel => it's neighbors
    torch::Tensor  sdf,  // [N_voxels, 4] for each voxel => it's neighbors
    torch::Tensor  feat,  // [N_voxels, 4] for each voxel => it's neighbors
    torch::Tensor  grad_sdf,     // [N_voxels, 4] for each voxel => it's vertices
    torch::Tensor  grad_feat,     // [N_voxels, 4] for each voxel => it's vertices
    torch::Tensor  weights_tot
)   {
        const int threads = 512;
        const int blocks = (num_tets + threads - 1) / threads; // ceil for example 8192 + 255 / 256 = 32
        AT_DISPATCH_FLOATING_TYPES( sites.type(),"sdf_space_grad_cuda_kernel", ([&] {  
            sdf_space_grad_cuda_kernel CUDA_KERNEL(blocks,threads) (
                num_tets,            
                tets.data_ptr<int>(),
                sites.data_ptr<float>(),
                sdf.data_ptr<float>(),
                feat.data_ptr<float>(),
                grad_sdf.data_ptr<float>(),
                grad_feat.data_ptr<float>(),
                weights_tot.data_ptr<float>()); 
        }));
    
        const int threads2 = 1024;
        const int blocks2 = (num_sites + threads2 - 1) / threads2; // ceil for example 8192 + 255 / 256 = 32
        AT_DISPATCH_FLOATING_TYPES( sites.type(),"normalize_grad_sdf_feat_kernel", ([&] {  
            normalize_grad_sdf_feat_kernel CUDA_KERNEL(blocks2,threads2) (
                num_sites,
                grad_sdf.data_ptr<float>(),
                grad_feat.data_ptr<float>(),
                weights_tot.data_ptr<float>()); 
        }));
}


// 
void Laplace_grad_cuda(
    size_t num_tets,                // number of rays
    size_t num_sites,                // number of rays
    torch::Tensor  tets,  // [N_voxels, 4] for each voxel => it's neighbors
    torch::Tensor  sites,  // [N_voxels, 4] for each voxel => it's neighbors
    torch::Tensor  grad,  // [N_voxels, 4] for each voxel => it's neighbors
    torch::Tensor  grad_lapl,     // [N_voxels, 4] for each voxel => it's vertices
    torch::Tensor  weights_tot
)   {
        const int threads = 512;
        const int blocks = (num_tets + threads - 1) / threads; // ceil for example 8192 + 255 / 256 = 32
        AT_DISPATCH_FLOATING_TYPES( sites.type(),"sdf_laplacian_cuda_kernel", ([&] {  
            sdf_laplacian_cuda_kernel CUDA_KERNEL(blocks,threads) (
                num_tets,            
                tets.data_ptr<int>(),
                sites.data_ptr<float>(),
                grad.data_ptr<float>(),
                grad_lapl.data_ptr<float>(),
                weights_tot.data_ptr<float>()); 
        }));
    
        const int threads2 = 1024;
        const int blocks2 = (num_sites + threads2 - 1) / threads2; // ceil for example 8192 + 255 / 256 = 32
        AT_DISPATCH_FLOATING_TYPES( sites.type(),"normalize_laplacian_kernel", ([&] {  
            normalize_laplacian_kernel CUDA_KERNEL(blocks2,threads2) (
                num_sites,
                grad_lapl.data_ptr<float>(),
                weights_tot.data_ptr<float>()); 
        }));
}


// 
float cvt_grad_cuda(
    size_t num_sites,
    size_t num_knn,
    float sigma,                // number of rays
    torch::Tensor thetas, 
    torch::Tensor phis, 
    torch::Tensor gammas, 
    torch::Tensor neighbors,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor sites,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor freeze,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor sdf,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor grad_sites    // [N_sites, 3] for each voxel => it's vertices
)   {
    
        float* loss;
        cudaMalloc((void**)&loss, sizeof(float));
        cudaMemset(loss, 0, sizeof(float));


        const int threads = 512;//1024;
        const int blocks = (num_sites + threads - 1) / threads; // ceil for example 8192 + 255 / 256 = 32
        AT_DISPATCH_FLOATING_TYPES( sites.type(),"cvt_grad_cuda", ([&] {  
            cvt_grad_cuda_kernel CUDA_KERNEL(blocks,threads) (
                num_sites,
                num_knn,
                sigma,
                thetas.data_ptr<double>(),
                phis.data_ptr<double>(),
                gammas.data_ptr<double>(),
                neighbors.data_ptr<int>(),
                sites.data_ptr<double>(),
                freeze.data_ptr<int>(),
                sdf.data_ptr<float>(),
                grad_sites.data_ptr<double>(),
                loss); 
    }));

    
		cudaDeviceSynchronize();
		float res = 0;
		cudaMemcpy(&res, loss, sizeof(float), cudaMemcpyDeviceToHost);
		cudaFree(loss);

		return res;
}

void sdf_grad_cuda(
    size_t num_sites,
    size_t num_knn,
    torch::Tensor neighbors,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor sites,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor sdf,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor grad_sites    // [N_sites, 3] for each voxel => it's vertices
)   {
        const int threads = 1024;
        const int blocks = (num_sites + threads - 1) / threads; // ceil for example 8192 + 255 / 256 = 32
        AT_DISPATCH_FLOATING_TYPES( sites.type(),"sdf_grad_cuda", ([&] {  
            sdf_grad_cuda_kernel CUDA_KERNEL(blocks,threads) (
                num_sites,
                num_knn,
                neighbors.data_ptr<int>(),
                sites.data_ptr<float>(),
                sdf.data_ptr<float>(),
                grad_sites.data_ptr<float>()); 
    }));
}

void update_sdf_cuda(
    size_t num_sites,
    size_t num_knn,
    torch::Tensor neighbors,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor sites,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor sdf,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor feat,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor grad_sites,    // [N_sites, 3] for each voxel => it's vertices,    
    torch::Tensor sdf_diff,    // [N_sites, 3] for each voxel => it's vertices,    
    torch::Tensor feat_diff    // [N_sites, 3] for each voxel => it's vertices
)   {
        const int threads = 1024;
        const int blocks = (num_sites + threads - 1) / threads; // ceil for example 8192 + 255 / 256 = 32
        AT_DISPATCH_FLOATING_TYPES( sites.type(),"update_sdf_cuda", ([&] {  
            update_sdf_cuda_kernel CUDA_KERNEL(blocks,threads) (
                num_sites,
                num_knn,
                neighbors.data_ptr<int>(),
                sites.data_ptr<float>(),
                sdf.data_ptr<float>(),
                feat.data_ptr<float>(),
                grad_sites.data_ptr<float>(),
                sdf_diff.data_ptr<float>(),
                feat_diff.data_ptr<float>()); 
    }));
}

void diff_tensor_cuda(
    size_t num_tets,                // number of rays
    torch::Tensor  tets,  // [N_voxels, 4] for each voxel => it's neighbors
    torch::Tensor  sites,  // [N_voxels, 4] for each voxel => it's neighbors
    torch::Tensor  vol,     // [N_voxels, 4] for each voxel => it's vertices
    torch::Tensor  weights,     // [N_voxels, 4] for each voxel => it's vertices)
    torch::Tensor  weights_tot
)   {
    const int threads = 256;
    const int blocks = (num_tets + threads - 1) / threads; // ceil for example 8192 + 255 / 256 = 32
    AT_DISPATCH_FLOATING_TYPES( sites.type(),"diff_tensor_kernel", ([&] {  
        diff_tensor_kernel CUDA_KERNEL(blocks,threads) (
            num_tets,
            tets.data_ptr<int>(),
            sites.data_ptr<double>(),
            vol.data_ptr<float>(),
            weights.data_ptr<float>(),
            weights_tot.data_ptr<float>()); 
    }));
}

// 
void eikonal_grad_cuda(
    size_t num_tets,                // number of rays
    size_t num_sites,                // number of rays
    torch::Tensor  tets,  // [N_voxels, 4] for each voxel => it's neighbors
    torch::Tensor  valid_tets,  // [N_voxels, 4] for each voxel => it's neighbors
    torch::Tensor  sites,  // [N_voxels, 4] for each voxel => it's neighbors
    torch::Tensor  activated,  // [N_voxels, 4] for each voxel => it's neighbors
    torch::Tensor  sdf,  // [N_voxels, 4] for each voxel => it's neighbors
    torch::Tensor  sdf_smooth,  // [N_voxels, 4] for each voxel => it's neighbors
    torch::Tensor  feat,  // [N_voxels, 4] for each voxel => it's neighbors
    torch::Tensor  grad_eik,     // [N_voxels, 4] for each voxel => it's vertices
    torch::Tensor  grad_smooth,     // [N_voxels, 4] for each voxel => it's vertices
    torch::Tensor  grad_sdf,     // [N_voxels, 4] for each voxel => it's vertices)
    torch::Tensor  vol,     // [N_voxels, 4] for each voxel => it's vertices
    torch::Tensor  weights,     // [N_voxels, 4] for each voxel => it's vertices
    torch::Tensor  weights_tot,     // [N_voxels, 4] for each voxel => it's vertices
    torch::Tensor  eik_loss     // [N_voxels, 4] for each voxel => it's vertices
)   {
        const int threads = 1024;
        const int blocks = (num_tets + threads - 1) / threads; // ceil for example 8192 + 255 / 256 = 32
        AT_DISPATCH_FLOATING_TYPES( sites.type(),"eikonal_grad_kernel", ([&] {  
            eikonal_grad_kernel CUDA_KERNEL(blocks,threads) (
                num_tets,
                tets.data_ptr<int>(),
                valid_tets.data_ptr<int>(),
                sites.data_ptr<float>(),
                activated.data_ptr<int>(),
                sdf.data_ptr<float>(),
                sdf_smooth.data_ptr<float>(),
                feat.data_ptr<float>(),
                grad_eik.data_ptr<float>(),
                grad_smooth.data_ptr<float>(),
                grad_sdf.data_ptr<float>(),
                vol.data_ptr<float>(),
                weights.data_ptr<float>(),
                weights_tot.data_ptr<float>(),
                eik_loss.data_ptr<float>()); 
        }));
		cudaDeviceSynchronize();

        /*const int threads2 = 1024;
        const int blocks2 = (num_sites + threads2 - 1) / threads2; // ceil for example 8192 + 255 / 256 = 32
        AT_DISPATCH_FLOATING_TYPES( sites.type(),"normalize_grad_kernel", ([&] {  
            normalize_grad_kernel CUDA_KERNEL(blocks2,threads2) (
                num_sites,
                grad_eik.data_ptr<float>(),
                grad_smooth.data_ptr<float>(),
                grad_sdf.data_ptr<float>(),
                grad_feat.data_ptr<float>(),
                weights_tot.data_ptr<float>(),
                eik_loss.data_ptr<float>()); 
        }));*/
    
}

// 
void backprop_norm_grad_cuda(
    size_t num_tets,                // number of rays
    torch::Tensor  tets,  // [N_voxels, 4] for each voxel => it's neighbors
    torch::Tensor  sites,  // [N_voxels, 4] for each voxel => it's neighbors
    torch::Tensor  activated,  // [N_voxels, 4] for each voxel => it's neighbors
    torch::Tensor  grad_sdf,     // [N_voxels, 4] for each voxel => it's vertices
    torch::Tensor  grad_norm,     // [N_voxels, 4] for each voxel => it's vertices)
    torch::Tensor  vol,     // [N_voxels, 4] for each voxel => it's vertices
    torch::Tensor  weights,     // [N_voxels, 4] for each voxel => it's vertices
    torch::Tensor  weights_tot    // [N_voxels, 4] for each voxel => it's vertices
)   {
        const int threads = 1024;
        const int blocks = (num_tets + threads - 1) / threads; // ceil for example 8192 + 255 / 256 = 32
        AT_DISPATCH_FLOATING_TYPES( sites.type(),"backprop_norm_grad_kernel", ([&] {  
            backprop_norm_grad_kernel CUDA_KERNEL(blocks,threads) (
                num_tets,
                tets.data_ptr<int>(),
                sites.data_ptr<float>(),
                activated.data_ptr<int>(),
                grad_sdf.data_ptr<float>(),
                grad_norm.data_ptr<float>(),
                vol.data_ptr<float>(),
                weights.data_ptr<float>(),
                weights_tot.data_ptr<float>()); 
        }));    
		cudaDeviceSynchronize();
}


void concat_feat_cuda(
    size_t num_sites,                // number of rays
    size_t num_knn,  
    size_t dim_feat,
    torch::Tensor  vertices,     // [N_voxels, 4] for each voxel => it's vertices
    torch::Tensor  activated,     // [N_voxels, 4] for each voxel => it's vertices
    torch::Tensor  grads,     // [N_voxels, 4] for each voxel => it's vertices
    torch::Tensor  feat,     // [N_voxels, 4] for each voxel => it's vertices
    torch::Tensor  neighbors
    ) {
        /*const int threads = 1024;
        const int blocks = (num_sites + threads - 1) / threads; // ceil for example 8192 + 255 / 256 = 32
        AT_DISPATCH_FLOATING_TYPES( grads.type(),"concat_feat_kernel", ([&] {  
            concat_feat_kernel CUDA_KERNEL(blocks,threads) (
                num_sites,
                num_knn,
                dim_feat,
                vertices.data_ptr<float>(),
                activated.data_ptr<int>(),
                grads.data_ptr<float>(),
                feat.data_ptr<float>(),
                neighbors.data_ptr<int>()); 
        }));*/

        const int threads = 96;
        const int blocks = num_sites; 
        AT_DISPATCH_FLOATING_TYPES( grads.type(),"concat_feat_kernel_o", ([&] {  
            concat_feat_kernel_o CUDA_KERNEL(blocks,threads) (
                num_sites,
                num_knn,
                dim_feat,
                vertices.data_ptr<float>(),
                activated.data_ptr<int>(),
                (float3*)thrust::raw_pointer_cast(grads.data_ptr<float>()),
                feat.data_ptr<float>(),
                neighbors.data_ptr<int>()); 
        }));
    }
