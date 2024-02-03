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


#define PI 3.141592653589793238462643383279502884197
#define _MAX_K_NN 24
#define _BUFF_SIZE 2048

/** Device functions **/
/** Device functions **/
/** Device functions **/

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

// compute the gradient for each tetrahedra
// gradient is fixed inside each tetrahedron
/*__global__ void Gradients_tet_kernel(float* gradients, float* weights, float* sdf, float* bbox, float* sites, int* active_sites, int* tets, int nb_tets) {
    unsigned int threadsPerBlock = blockDim.x * blockDim.y;
    unsigned int threadNumInBlock = threadIdx.x + blockDim.x * threadIdx.y;
    unsigned int blockNumInGrid = blockIdx.x + gridDim.x * blockIdx.y;
    int n = int(blockNumInGrid * threadsPerBlock + threadNumInBlock);

    if (n > nb_tets - 1) // n corresponds to a vertex in the voronoi diagram
        return;

    gradients[3 * n] = 0.0f;
    gradients[3 * n + 1] = 0.0f;
    gradients[3 * n + 2] = 0.0f;
    for (int k = 0; k < 4; k++) {
        weights[3 * (4 * n + k)] = 0.0f;
        weights[3 * (4 * n + k) + 1] = 0.0f;
        weights[3 * (4 * n + k) + 2] = 0.0f;
    }

    float p0[3] = { sites[3 * tets[4 * n]] ,
                    sites[3 * tets[4 * n] + 1],
                    sites[3 * tets[4 * n] + 2] };

    float p1[3] = { sites[3 * tets[4 * n+1]],
                    sites[3 * tets[4 * n+1] + 1],
                    sites[3 * tets[4 * n+1] + 2]};

    float p2[3] = { sites[3 * tets[4 * n+2]],
                    sites[3 * tets[4 * n+2] + 1],
                    sites[3 * tets[4 * n+2] + 2]};

    float p3[3] = { sites[3 * tets[4 * n + 3]],
                    sites[3 * tets[4 * n + 3] + 1],
                    sites[3 * tets[4 * n + 3] + 2]};
    
    float e0[3] = { sites[3 * tets[4 * n]] - sites[3 * tets[4 * n + 1]],
                    sites[3 * tets[4 * n] + 1] - sites[3 * tets[4 * n + 1] + 1],
                    sites[3 * tets[4 * n] + 2] - sites[3 * tets[4 * n + 1] + 2]};

    float e1[3] = { sites[3 * tets[4 * n]] - sites[3 * tets[4 * n + 2]],
                    sites[3 * tets[4 * n] + 1] - sites[3 * tets[4 * n + 2] + 1],
                    sites[3 * tets[4 * n] + 2] +- sites[3 * tets[4 * n + 2] + 2]};

    float e2[3] = { sites[3 * tets[4 * n]] - sites[3 * tets[4 * n + 3]],
                    sites[3 * tets[4 * n] + 1] - sites[3 * tets[4 * n + 3] + 1],
                    sites[3 * tets[4 * n] + 2] - sites[3 * tets[4 * n + 3] + 2]};

    float e3[3] = { sites[3 * tets[4 * n + 1]] - sites[3 * tets[4 * n + 2]],
                    sites[3 * tets[4 * n + 1] + 1] - sites[3 * tets[4 * n + 2] + 1],
                    sites[3 * tets[4 * n + 1] + 2] - sites[3 * tets[4 * n + 2] + 2]};

    float e4[3] = { sites[3 * tets[4 * n + 1]] - sites[3 * tets[4 * n + 3]],
                    sites[3 * tets[4 * n + 1] + 1] - sites[3 * tets[4 * n + 3] + 1],
                    sites[3 * tets[4 * n + 1] + 2] - sites[3 * tets[4 * n + 3] + 2]};

    float e5[3] = { sites[3 * tets[4 * n + 2]] - sites[3 * tets[4 * n + 3]],
                    sites[3 * tets[4 * n + 2] + 1] - sites[3 * tets[4 * n + 3] + 1],
                    sites[3 * tets[4 * n + 2] + 2] - sites[3 * tets[4 * n + 3] + 2]};

    float max_edge_length = max(squared_length_f(e1), squared_length_f(e2));
    max_edge_length = max(max_edge_length, squared_length_f(e3));
    max_edge_length = max(max_edge_length, squared_length_f(e4));
    max_edge_length = max(max_edge_length, squared_length_f(e5));

    if (max_edge_length > 0.1f) {
        gradients[3 * n] = 0.0f;
        gradients[3 * n + 1] = 0.0f;
        gradients[3 * n + 2] = 0.0f;
        for (int k = 0; k < 4; k++) {
            weights[3 * (4 * n + k)] = 0.0f;
            weights[3 * (4 * n + k) + 1] = 0.0f;
            weights[3 * (4 * n + k) + 2] = 0.0f;
        }
        return;
    }

    //1. Compute center of tetrahedra
    float pt[3] = { (sites[3 * tets[4 * n]] + sites[3 * tets[4 * n + 1]] + sites[3 * tets[4 * n + 2]] + sites[3 * tets[4 * n + 3]]) / 4.0f,
                    (sites[3 * tets[4 * n] + 1] + sites[3 * tets[4 * n + 1] + 1] + sites[3 * tets[4 * n + 2] + 1] + sites[3 * tets[4 * n + 3] + 1]) / 4.0f, 
                    (sites[3 * tets[4 * n] + 2] + sites[3 * tets[4 * n + 1] + 2] + sites[3 * tets[4 * n + 2] + 2] + sites[3 * tets[4 * n + 3] + 2]) / 4.0f};

    //2. Compute the baricentric gradients
    float ray[3] = { 0.0f, 0.0f, 0.0f };
    float n_tri[3] = { 0.0f, 0.0f, 0.0f };
    float p_curr[3] = { 0.0f, 0.0f, 0.0f };
    float result[3] = { 0.0f, 0.0f, 0.0f };
    int face_curr[3] = { 0,0,0 };
    float norm_n = 0.0f;
    float dist = 0.0f;

    gradients[3 * n] = 0.0f;
    gradients[3 * n + 1] = 0.0f;
    gradients[3 * n + 2] = 0.0f;

    for (int j = 0; j < 4; j++) {
        p_curr[0] = sites[3 * tets[4 * n + j]] - pt[0];
        p_curr[1] = sites[3 * tets[4 * n + j] + 1] - pt[1];
        p_curr[2] = sites[3 * tets[4 * n + j] + 2] - pt[2];

        face_curr[0] = tets[4 * n + tet_faces[3 * j]];
        face_curr[1] = tets[4 * n + tet_faces[3 * j + 1]];
        face_curr[2] = tets[4 * n + tet_faces[3 * j + 2]];

        //a. Compute normal of opposite face
        get_normal_f(&sites[3 * face_curr[0]], &sites[3 * face_curr[1]], &sites[3 * face_curr[2]], n_tri);
        norm_n = squared_length_f(n_tri);
        if (norm_n == 0.0f) {
            gradients[3 * n] = 0.0f;
            gradients[3 * n + 1] = 0.0f;
            gradients[3 * n + 2] = 0.0f;
            for (int k = 0; k < 4; k++) {
                weights[3 * (4 * n + k)] = 0.0f;
                weights[3 * (4 * n + k) + 1] = 0.0f;
                weights[3 * (4 * n + k) + 2] = 0.0f;
            }
            return;
        }
        else {
            n_tri[0] = n_tri[0] / norm_n;
            n_tri[1] = n_tri[1] / norm_n;
            n_tri[2] = n_tri[2] / norm_n;
        }

        //b. Orient normal towards current summit
        if (dot_prod_f(p_curr, n_tri) < 0.0f) {
            n_tri[0] = -n_tri[0];
            n_tri[1] = -n_tri[1];
            n_tri[2] = -n_tri[2];
        }

        //c. Compute distance point to triangle
        p_curr[0] = sites[3 * tets[4 * n + j]] ;
        p_curr[1] = sites[3 * tets[4 * n + j] + 1];
        p_curr[2] = sites[3 * tets[4 * n + j] + 2];
        IntersectionRayTriangle3D_gpu_f_cvt(result, n_tri, p_curr, &sites[3 * face_curr[0]], &sites[3 * face_curr[1]], &sites[3 * face_curr[2]], n_tri);
        dist = sqrt((result[0] - p_curr[0]) * (result[0] - p_curr[0]) +
            (result[1] - p_curr[1]) * (result[1] - p_curr[1]) +
            (result[2] - p_curr[2]) * (result[2] - p_curr[2]));

        if (dist / max_edge_length < 1.0e-1f) {
            gradients[3 * n] = 0.0f;
            gradients[3 * n + 1] = 0.0f;
            gradients[3 * n + 2] = 0.0f;
            for (int k = 0; k < 4; k++) {
                weights[3 * (4 * n + k)] = 0.0f;
                weights[3 * (4 * n + k) + 1] = 0.0f;
                weights[3 * (4 * n + k) + 2] = 0.0f;
            }
            return;
        }

        weights[3 * (4 * n + j)] = n_tri[0] / dist;
        weights[3 * (4 * n + j) + 1] = n_tri[1] / dist;
        weights[3 * (4 * n + j) + 2] = n_tri[2] / dist;

        gradients[3 * n] = gradients[3 * n] + sdf[tets[4 * n + j]] * n_tri[0] / dist;
        gradients[3 * n + 1] = gradients[3 * n + 1] + sdf[tets[4 * n + j]] * n_tri[1] / dist;
        gradients[3 * n + 2] = gradients[3 * n + 2] + sdf[tets[4 * n + j]] * n_tri[2] / dist;
    }
}*/

__global__ void sdf_space_grad_cuda_kernel(
    const size_t num_sites,                // number of rays
    const size_t num_knn,                // number of rays   
    const int *__restrict__ neighbors,  // [N_voxels, 4] for each voxel => it's neighbors
    const float *__restrict__ sites,     // [N_voxels, 4] for each voxel => it's vertices
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

    float curr_site[3] {sites[3*idx], sites[3*idx + 1], sites[3*idx + 2]};    
    float curr_n[3] {0.0, 0.0, 0.0};
    float dX[3*_MAX_K_NN];
    float G[9] {0.0, 0.0, 0.0, 
                0.0, 0.0, 0.0, 
                0.0, 0.0, 0.0}; //dXT dX
    float G_inv[9] {0.0, 0.0, 0.0, 
                0.0, 0.0, 0.0, 
                0.0, 0.0, 0.0};

    int knn_id = -1;

    float *Weights_curr = &Weights[3*num_knn*idx];    

    for (int r_id = 0; r_id < num_knn; r_id++) {
        knn_id = neighbors[num_knn*(idx+1) + r_id];

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
    float feat_0[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    float feat_1[6] ={0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    float feat_2[6] ={0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    for (int i = 0; i < num_knn; i++) {
        knn_id = neighbors[num_knn*(idx+1) + i];
        elem_0 += (SDF[knn_id] - SDF[idx]) * Weights_curr[3*i];
        elem_1 += (SDF[knn_id] - SDF[idx]) * Weights_curr[3*i + 1];
        elem_2 += (SDF[knn_id] - SDF[idx]) * Weights_curr[3*i + 2];
        for (int f_id = 0; f_id < 6; f_id++) {
            feat_0[f_id] = feat_0[f_id] + (Feat[6*knn_id + f_id] - Feat[6*idx + f_id]) * Weights_curr[3*i];
            feat_1[f_id] = feat_1[f_id] + (Feat[6*knn_id + f_id] - Feat[6*idx + f_id]) * Weights_curr[3*i + 1];
            feat_2[f_id] = feat_2[f_id] + (Feat[6*knn_id + f_id] - Feat[6*idx + f_id]) * Weights_curr[3*i + 2];
        }
    }
    
    grad[3*idx] = elem_0;
    grad[3*idx + 1] = elem_1;    
    grad[3*idx + 2] = elem_2;   
    
    for (int f_id = 0; f_id < 6; f_id++) {
        grad_feat[3*6*idx + f_id] = feat_0[f_id];
        grad_feat[3*6*idx + 6 + f_id] = feat_1[f_id];  
        grad_feat[3*6*idx + 12 + f_id] = feat_2[f_id];  
    }

    return;
}



__global__ void cvt_grad_cuda_kernel(
    const size_t num_sites,                // number of rays
    const size_t num_knn,                // number of rays
    const float *__restrict__ thetas, 
    const float *__restrict__ phis, 
    const float *__restrict__ gammas, 
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

    float ray[9] {1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0};

    float curr_site[3] {sites[3*idx], sites[3*idx + 1], sites[3*idx + 2]};
    
    float theta = thetas[idx] * 2.0f * PI;
    float phi = phis[idx] * 2.0f * PI;
    float gamma = gammas[idx] * 2.0f * PI;
    float rot[9] {cosf(phi)*cosf(gamma), sinf(theta)*sinf(phi)*cosf(gamma) - cosf(theta)*sinf(gamma), cosf(theta)*sinf(phi)*cosf(gamma) + sinf(theta)*sinf(gamma),
                    cosf(phi)*sinf(gamma), sinf(theta)*sinf(phi)*sinf(gamma) + cosf(theta)*cosf(gamma), cosf(theta)*sinf(phi)*sinf(gamma) - sinf(theta)*cosf(gamma),
                    -sinf(phi), sinf(theta)*cosf(phi), cosf(theta)*cosf(phi)}; 

    float curr_ray[3] {0.0, 0.0, 0.0};
    float nmle[3] {0.0, 0.0, 0.0};
    float b_point[3] {0.0, 0.0, 0.0};
    int knn_id = -1;

    float denom = 0.0f;
    float denom1 = 0.0f;
    float denom2 = 0.0f;
    float num1 = 0.0f;
    float num2 = 0.0f;
    float weight_cvt = 1.0f;

    float nmle_length;
    float d_min_dist[3] {0.0, 0.0, 0.0};
    float d_max_dist[3] {0.0, 0.0, 0.0};

    for (int r_id = 0; r_id < 3; r_id++) {
        
        curr_ray[0] = rot[0]*ray[3*r_id] + rot[1]*ray[3*r_id+1] + rot[2]*ray[3*r_id+2];
        curr_ray[1] = rot[3]*ray[3*r_id] + rot[4]*ray[3*r_id+1] + rot[5]*ray[3*r_id+2];
        curr_ray[3] = rot[6]*ray[3*r_id] + rot[7]*ray[3*r_id+1] + rot[8]*ray[3*r_id+2];

        float min_dist = 1.0e32;
        int min_id = -1;
        float max_dist = -1.0e32;
        int max_id = -1;
        float dist = 0.0f;

        for (int i = 0; i < num_knn; i++) {
            knn_id = neighbors[num_knn*(idx+1) + i];
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
            
            // Compute middle point
            b_point[0] = (sites[3*knn_id] + curr_site[0]) / 2.0f;
            b_point[1] = (sites[3*knn_id + 1] + curr_site[1]) / 2.0f;
            b_point[2] = (sites[3*knn_id + 2] + curr_site[2]) / 2.0f;
            
            // Compute ray - plane intersection point
            denom = nmle[0] * curr_ray[0] + nmle[1] * curr_ray[1] + nmle[2] * curr_ray[2];
            if (abs(denom) > 1.0e-6) {
                dist = (nmle[0] * (b_point[0] - curr_site[0]) + nmle[1] * (b_point[1] - curr_site[1]) + nmle[2] * (b_point[21] - curr_site[2])) / denom;
                if (dist >= 0.0 && dist < min_dist) {
                    min_dist = dist;
                    min_id = knn_id;
                }
                if (dist <= 0.0 && dist > max_dist) {
                    max_dist = dist;
                    max_id = knn_id;
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

        d_min_dist[0] = 0.5 * (-(sites[3*min_id] - curr_site[0])/denom1 + num1 * curr_ray[0]/(denom1*denom1));
        d_min_dist[1] = 0.5 * (-(sites[3*min_id+1] - curr_site[1])/denom1 + num1 * curr_ray[1]/(denom1*denom1));
        d_min_dist[2] = 0.5 * (-(sites[3*min_id+2] - curr_site[2])/denom1 + num1 * curr_ray[21]/(denom1*denom1));

        d_max_dist[0] = 0.5 * (-(sites[3*max_id] - curr_site[0])/denom2 + num2 * curr_ray[0]/(denom2*denom2));
        d_max_dist[1] = 0.5 * (-(sites[3*max_id+1] - curr_site[1])/denom2 + num2 * curr_ray[1]/(denom2*denom2));   
        d_max_dist[2] = 0.5 * (-(sites[3*max_id+2] - curr_site[2])/denom2 + num2 * curr_ray[2]/(denom2*denom2));       

        // gradients related to CVT loss
        grad_sites[3*idx] = grad_sites[3*idx] - (min_dist * d_min_dist[0] + max_dist * d_max_dist[0]);  
        grad_sites[3*idx+1] = grad_sites[3*idx+1] - (min_dist * d_min_dist[1] + max_dist * d_max_dist[1]);    
        grad_sites[3*idx+2] = grad_sites[3*idx+2] - (min_dist * d_min_dist[2] + max_dist * d_max_dist[2]);    
    }

    return;
}

__global__ void sdf_grad_cuda_kernel(
    const size_t num_sites,                // number of rays
    const size_t num_knn,                // number of rays
    float weight_sdf,
    const float *__restrict__ thetas, 
    const float *__restrict__ grad_sdf_space,       
    const float *__restrict__ grad_grad,       
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

    float curr_site[2] {sites[2*idx], sites[2*idx + 1]};
    
    float curr_ray[2] {0.0, 0.0};
    float nmle[2] {0.0, 0.0};
    float b_point[2] {0.0, 0.0};
    int knn_id = -1;

    float denom = 0.0f;

    float nmle_length;
    float sdf_mid, dot_prod_g;

    for (int r_id = 0; r_id < num_knn; r_id++) {
        knn_id = neighbors[num_knn*(idx+1) + r_id];
        if (sdf[idx]*sdf[knn_id] >= 0.0f)
            continue;
        
        // Check if k-nn neighbor is adjacent in the CVT 
        curr_ray[0] = sites[2*knn_id] - sites[2*idx];
        curr_ray[1] = sites[2*knn_id + 1] - sites[2*idx + 1];
        nmle_length = sqrt(curr_ray[0]*curr_ray[0] + curr_ray[1]*curr_ray[1]);
        if (nmle_length == 0.0f)
            continue;
        curr_ray[0] = curr_ray[0] / nmle_length;
        curr_ray[1] = curr_ray[1] / nmle_length;

        float min_dist = 1.0e32;
        int min_id = -1;
        float dist = 0.0f;
        int kk_nn_id = -1;

        for (int i = 0; i < num_knn; i++) {
            kk_nn_id  = neighbors[num_knn*(idx+1) + i];
            if (kk_nn_id == -1)
                break;

            // Compute bisector normal vector
            nmle[0] = (sites[2*kk_nn_id] - curr_site[0]);
            nmle[1] = (sites[2*kk_nn_id + 1] - curr_site[1]);
            nmle_length = sqrt(nmle[0]*nmle[0] + nmle[1]*nmle[1]);
            if (nmle_length == 0.0f)
                continue;
            nmle[0] = nmle[0] / nmle_length;
            nmle[1] = nmle[1] / nmle_length;
            
            // Compute middle point
            b_point[0] = (sites[2*kk_nn_id] + curr_site[0]) / 2.0f;
            b_point[1] = (sites[2*kk_nn_id + 1] + curr_site[1]) / 2.0f;
            
            // Compute ray - plane intersection point
            denom = nmle[0] * curr_ray[0] + nmle[1] * curr_ray[1];
            if (abs(denom) > 1.0e-6) {
                dist = (nmle[0] * (b_point[0] - curr_site[0]) + nmle[1] * (b_point[1] - curr_site[1])) / denom;
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
        denom = sqrt((sites[2*knn_id] - curr_site[0]) * (sites[2*knn_id] - curr_site[0]) + (sites[2*knn_id+1] - curr_site[1])* (sites[2*knn_id+1] - curr_site[1]));  
        
        if (sdf_mid * sdf[idx] > 0.0f) {
            atomicAdd(&grad_sites[2*idx], -weight_sdf*fabs(sdf_mid)*(sites[2*knn_id] - curr_site[0])/denom);
            atomicAdd(&grad_sites[2*idx + 1], -weight_sdf*fabs(sdf_mid)*(sites[2*knn_id+1] - curr_site[1])/denom);
            
            atomicAdd(&grad_sites[2*knn_id], -weight_sdf*fabs(sdf_mid)*(sites[2*knn_id] - curr_site[0])/denom);
            atomicAdd(&grad_sites[2*knn_id + 1], -weight_sdf*fabs(sdf_mid)*(sites[2*knn_id+1] - curr_site[1])/denom);
        } else {
            atomicAdd(&grad_sites[2*idx], weight_sdf*fabs(sdf_mid)*(sites[2*knn_id] - curr_site[0])/denom);
            atomicAdd(&grad_sites[2*idx + 1], weight_sdf*fabs(sdf_mid)*(sites[2*knn_id+1] - curr_site[1])/denom);
            
            atomicAdd(&grad_sites[2*knn_id], weight_sdf*fabs(sdf_mid)*(sites[2*knn_id] - curr_site[0])/denom);
            atomicAdd(&grad_sites[2*knn_id + 1], weight_sdf*fabs(sdf_mid)*(sites[2*knn_id+1] - curr_site[1])/denom);
        } 

        /*if (sdf_mid * sdf[idx] > 0.0f) {
            grad_sites[2*idx] = grad_sites[2*idx] - weight_sdf*fabs(sdf_mid)*(sites[2*knn_id] - curr_site[0])/denom;
            grad_sites[2*idx+1] = grad_sites[2*idx+1] - weight_sdf*fabs(sdf_mid)*(sites[2*knn_id+1] - curr_site[1])/denom;
        }/* else {
            grad_sites[2*knn_id] = grad_sites[2*knn_id] - weight_sdf*sdf_mid*(sites[2*knn_id] - curr_site[0])/denom;
            grad_sites[2*knn_id+1] = grad_sites[2*knn_id+1] - weight_sdf*sdf_mid*(sites[2*knn_id+1] - curr_site[1])/denom;
        }  */
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


// 
void sdf_space_grad_cuda(
    size_t num_sites,
    size_t num_knn,
    torch::Tensor neighbors,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor sites,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor sdf,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor feat,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor Weights,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor grad_sites,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor grad_feat_sites    // [N_sites, 3] for each voxel => it's vertices
)   {
        const int threads = 512;
        const int blocks = (num_sites + threads - 1) / threads; // ceil for example 8192 + 255 / 256 = 32
        AT_DISPATCH_FLOATING_TYPES( sites.type(),"sdf_space_grad_cuda", ([&] {  
            sdf_space_grad_cuda_kernel CUDA_KERNEL(blocks,threads) (
                num_sites,
                num_knn,
                neighbors.data_ptr<int>(),
                sites.data_ptr<float>(),
                sdf.data_ptr<float>(),
                feat.data_ptr<float>(),
                Weights.data_ptr<float>(),
                grad_sites.data_ptr<float>(),
                grad_feat_sites.data_ptr<float>()); 
    }));
}


// Ray marching in 3D
void cvt_grad_cuda(
    size_t num_sites,
    size_t num_knn,
    torch::Tensor thetas, 
    torch::Tensor phis, 
    torch::Tensor gammas, 
    torch::Tensor neighbors,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor sites,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor sdf,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor grad_sites    // [N_sites, 3] for each voxel => it's vertices
)   {
        const int threads = 1024;
        const int blocks = (num_sites + threads - 1) / threads; // ceil for example 8192 + 255 / 256 = 32
        AT_DISPATCH_FLOATING_TYPES( sites.type(),"cvt_grad_cuda", ([&] {  
            cvt_grad_cuda_kernel CUDA_KERNEL(blocks,threads) (
                num_sites,
                num_knn,
                thetas.data_ptr<float>(),
                phis.data_ptr<float>(),
                gammas.data_ptr<float>(),
                neighbors.data_ptr<int>(),
                sites.data_ptr<float>(),
                sdf.data_ptr<float>(),
                grad_sites.data_ptr<float>()); 
    }));
}

void sdf_grad_cuda(
    size_t num_sites,
    size_t num_knn,
    float weight_sdf,
    torch::Tensor thetas, 
    torch::Tensor grad_sdf_space,      // [N_rays, 6]
    torch::Tensor grad_grad, // [N_voxels, 26] for each voxel => it's neighbors
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
                weight_sdf,
                thetas.data_ptr<float>(),
                grad_sdf_space.data_ptr<float>(),
                grad_grad.data_ptr<float>(),
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

