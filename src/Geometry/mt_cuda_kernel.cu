#include <torch/extension.h>

#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include <device_launch_parameters.h>
#include "../Models/cudaType.cuh"

#ifdef __INTELLISENSE__
#define CUDA_KERNEL(...)
#define FAKEINIT = {0}
#else
#define CUDA_KERNEL(...) <<< __VA_ARGS__ >>>
#define FAKEINIT
#endif

#define TRUNCATE 0.4f

using namespace std;

//#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__, true); }

__global__ void  GenerateFaces(int* Faces, float* TSDF, int* Edges_row_ptr, int* Edges_columns,
    int* Tets, float m_iso, int nb_tets) {
    unsigned int threadsPerBlock = blockDim.x * blockDim.y;
    unsigned int threadNumInBlock = threadIdx.x + blockDim.x * threadIdx.y;
    unsigned int blockNumInGrid = blockIdx.x + gridDim.x * blockIdx.y;
    int voxid = int(blockNumInGrid * threadsPerBlock + threadNumInBlock);

    // assuming x and y inputs are same length
    if (voxid >= nb_tets - 1)
        return;
    float a, b, c, d; //4 summits if the tetrahedra voxel
    int Tetra[4] = { Tets[voxid * 4], Tets[voxid * 4+1], Tets[voxid * 4+2],
                    Tets[voxid * 4]^Tets[voxid * 4+1]^Tets[voxid * 4+2]^Tets[voxid * 4 + 3] };
    //Value of the TSDF
    a = TSDF[Tetra[0]];
    b = TSDF[Tetra[1]];
    c = TSDF[Tetra[2]];
    d = TSDF[Tetra[3]];
    int count = 0;
    if (a >= m_iso)
        count += 1;
    if (b >= m_iso)
        count += 1;
    if (c >= m_iso)
        count += 1;
    if (d >= m_iso)
        count += 1;
    if (count == 0 || count == 4) //return;
    {
        Faces[6 * (voxid)+0] = 0;
        Faces[6 * (voxid)+1] = 0;
        Faces[6 * (voxid)+2] = 0;
        Faces[6 * (voxid)+3] = 0;
        Faces[6 * (voxid)+4] = 0;
        Faces[6 * (voxid)+5] = 0;
    }
    //! Three vertices are inside the volume
    else if (count == 3) {
        int2 list[6] = { make_int2(0,1), make_int2(0,2), make_int2(0,3),
        make_int2(1,2), make_int2(1,3), make_int2(2,3) };
        //! Make sure that fourth value lies outside
        if (d < m_iso)
        {
        }
        else if (c < m_iso)
        {
            list[0] = make_int2(0, 3);
            list[1] = make_int2(0, 1);
            list[2] = make_int2(0, 2);
            list[3] = make_int2(1, 3);
            list[4] = make_int2(2, 3);
            list[5] = make_int2(1, 2);
        }
        else if (b < m_iso)
        {
            list[0] = make_int2(0, 2);
            list[1] = make_int2(0, 3);
            list[2] = make_int2(0, 1);
            list[3] = make_int2(2, 3);
            list[4] = make_int2(1, 2);
            list[5] = make_int2(1, 3);
        }
        else
        {
            list[0] = make_int2(1, 3);
            list[1] = make_int2(1, 2);
            list[2] = make_int2(0, 1);
            list[3] = make_int2(2, 3);
            list[4] = make_int2(0, 3);
            list[5] = make_int2(0, 2);
        }
        //ad
        int sum1 = Tetra[list[2].x] < Tetra[list[2].y] ? Tetra[list[2].x] : Tetra[list[2].y];
        int sum2 = Tetra[list[2].x] < Tetra[list[2].y] ? Tetra[list[2].y] : Tetra[list[2].x];
        int idx_ad = 0;
        for (int k = Edges_row_ptr[sum1]; k < Edges_row_ptr[sum1 + 1]; k++) {
            if (Edges_columns[k] == sum2) {
                idx_ad = k;
                break;
            }
        }
        //bd
        sum1 = Tetra[list[4].x] < Tetra[list[4].y] ? Tetra[list[4].x] : Tetra[list[4].y];
        sum2 = Tetra[list[4].x] < Tetra[list[4].y] ? Tetra[list[4].y] : Tetra[list[4].x];
        int idx_bd = 0;
        for (int k = Edges_row_ptr[sum1]; k < Edges_row_ptr[sum1 + 1]; k++) {
            if (Edges_columns[k] == sum2) {
                idx_bd = k;
                break;
            }
        }
        //cd
        sum1 = Tetra[list[5].x] < Tetra[list[5].y] ? Tetra[list[5].x] : Tetra[list[5].y];
        sum2 = Tetra[list[5].x] < Tetra[list[5].y] ? Tetra[list[5].y] : Tetra[list[5].x];
        int idx_cd = 0;
        for (int k = Edges_row_ptr[sum1]; k < Edges_row_ptr[sum1 + 1]; k++) {
            if (Edges_columns[k] == sum2) {
                idx_cd = k;
                break;
            }
        }
        Faces[6 * (voxid)+0] = idx_ad;
        Faces[6 * (voxid)+1] = idx_cd;
        Faces[6 * (voxid)+2] = idx_bd;
        Faces[6 * (voxid)+3] = 0;
        Faces[6 * (voxid)+4] = 0;
        Faces[6 * (voxid)+5] = 0;
    }
    //! Two vertices are inside the volume
    else if (count == 2) {
        //! Make sure that the last two points lie outside
        int2 list[6] = { make_int2(0,1), make_int2(0,2), make_int2(0,3),
        make_int2(1,2), make_int2(1,3), make_int2(2,3) };
        if (a >= m_iso && b >= m_iso)
        {
        }
        else if (a >= m_iso && c >= m_iso)
        {
            list[0] = make_int2(0, 2);
            list[1] = make_int2(0, 3);
            list[2] = make_int2(0, 1);
            list[3] = make_int2(2, 3);
            list[4] = make_int2(1, 2);
            list[5] = make_int2(1, 3);
        }
        else if (a >= m_iso && d >= m_iso)
        {
            list[0] = make_int2(0, 3);
            list[1] = make_int2(0, 1);
            list[2] = make_int2(0, 2);
            list[3] = make_int2(1, 3);
            list[4] = make_int2(2, 3);
            list[5] = make_int2(1, 2);
        }
        else if (b >= m_iso && c >= m_iso)
        {
            list[0] = make_int2(1, 2);
            list[1] = make_int2(0, 1);
            list[2] = make_int2(1, 3);
            list[3] = make_int2(0, 2);
            list[4] = make_int2(2, 3);
            list[5] = make_int2(0, 3);
        }
        else if (b >= m_iso && d >= m_iso)
        {
            list[0] = make_int2(1, 3);
            list[1] = make_int2(1, 2);
            list[2] = make_int2(0, 1);
            list[3] = make_int2(2, 3);
            list[4] = make_int2(0, 3);
            list[5] = make_int2(0, 2);
        }
        else //c && d > m_iso
        {
            list[0] = make_int2(2, 3);
            list[1] = make_int2(0, 2);
            list[2] = make_int2(1, 2);
            list[3] = make_int2(0, 3);
            list[4] = make_int2(1, 3);
            list[5] = make_int2(0, 1);
        }
        //ac
        int sum1 = Tetra[list[1].x] < Tetra[list[1].y] ? Tetra[list[1].x] : Tetra[list[1].y];
        int sum2 = Tetra[list[1].x] < Tetra[list[1].y] ? Tetra[list[1].y] : Tetra[list[1].x];
        int idx_ac = 0;
        for (int k = Edges_row_ptr[sum1]; k < Edges_row_ptr[sum1 + 1]; k++) {
            if (Edges_columns[k] == sum2) {
                idx_ac = k;
                break;
            }
        }
        //ad
        sum1 = Tetra[list[2].x] < Tetra[list[2].y] ? Tetra[list[2].x] : Tetra[list[2].y];
        sum2 = Tetra[list[2].x] < Tetra[list[2].y] ? Tetra[list[2].y] : Tetra[list[2].x];
        int idx_ad = 0;
        for (int k = Edges_row_ptr[sum1]; k < Edges_row_ptr[sum1 + 1]; k++) {
            if (Edges_columns[k] == sum2) {
                idx_ad = k;
                break;
            }
        }
        //bc
        sum1 = Tetra[list[3].x] < Tetra[list[3].y] ? Tetra[list[3].x] : Tetra[list[3].y];
        sum2 = Tetra[list[3].x] < Tetra[list[3].y] ? Tetra[list[3].y] : Tetra[list[3].x];
        int idx_bc = 0;
        for (int k = Edges_row_ptr[sum1]; k < Edges_row_ptr[sum1 + 1]; k++) {
            if (Edges_columns[k] == sum2) {
                idx_bc = k;
                break;
            }
        }
        //bd
        sum1 = Tetra[list[4].x] < Tetra[list[4].y] ? Tetra[list[4].x] : Tetra[list[4].y];
        sum2 = Tetra[list[4].x] < Tetra[list[4].y] ? Tetra[list[4].y] : Tetra[list[4].x];
        int idx_bd = 0;
        for (int k = Edges_row_ptr[sum1]; k < Edges_row_ptr[sum1 + 1]; k++) {
            if (Edges_columns[k] == sum2) {
                idx_bd = k;
                break;
            }
        }
        // storeTriangle(ac,bc,ad);
        Faces[6 * (voxid)+0] = idx_ac;
        Faces[6 * (voxid)+1] = idx_bc;
        Faces[6 * (voxid)+2] = idx_ad;
        //storeTriangle(bc,bd,ad);
        Faces[6 * (voxid)+3] = idx_bc;
        Faces[6 * (voxid)+4] = idx_bd;
        Faces[6 * (voxid)+5] = idx_ad;
    }
    //! One vertex is inside the volume
    else if (count == 1) {
        //! Make sure that the last three points lie outside
        int2 list[6] = { make_int2(0,1), make_int2(0,2), make_int2(0,3),
            make_int2(1,2), make_int2(1,3), make_int2(2,3) };
        if (a >= m_iso)
        {
        }
        else if (b >= m_iso)
        {
            list[0] = make_int2(1, 2);
            list[1] = make_int2(0, 1);
            list[2] = make_int2(1, 3);
            list[3] = make_int2(0, 2);
            list[4] = make_int2(2, 3);
            list[5] = make_int2(0, 3);
        }
        else if (c >= m_iso)
        {
            list[0] = make_int2(0, 2);
            list[1] = make_int2(1, 2);
            list[2] = make_int2(2, 3);
            list[3] = make_int2(0, 1);
            list[4] = make_int2(0, 3);
            list[5] = make_int2(1, 3);
        }
        else // d > m_iso
        {
            list[0] = make_int2(2, 3);
            list[1] = make_int2(1, 3);
            list[2] = make_int2(0, 3);
            list[3] = make_int2(1, 2);
            list[4] = make_int2(0, 2);
            list[5] = make_int2(0, 1);
        }
        //ab
        int sum1 = Tetra[list[0].x] < Tetra[list[0].y] ? Tetra[list[0].x] : Tetra[list[0].y];
        int sum2 = Tetra[list[0].x] < Tetra[list[0].y] ? Tetra[list[0].y] : Tetra[list[0].x];
        int idx_ab = 0;
        for (int k = Edges_row_ptr[sum1]; k < Edges_row_ptr[sum1 + 1]; k++) {
            if (Edges_columns[k] == sum2) {
                idx_ab = k;
                break;
            }
        }
        //ac
        sum1 = Tetra[list[1].x] < Tetra[list[1].y] ? Tetra[list[1].x] : Tetra[list[1].y];
        sum2 = Tetra[list[1].x] < Tetra[list[1].y] ? Tetra[list[1].y] : Tetra[list[1].x];
        int idx_ac = 0;
        for (int k = Edges_row_ptr[sum1]; k < Edges_row_ptr[sum1 + 1]; k++) {
            if (Edges_columns[k] == sum2) {
                idx_ac = k;
                break;
            }
        }
        //ad
        sum1 = Tetra[list[2].x] < Tetra[list[2].y] ? Tetra[list[2].x] : Tetra[list[2].y];
        sum2 = Tetra[list[2].x] < Tetra[list[2].y] ? Tetra[list[2].y] : Tetra[list[2].x];
        int idx_ad = 0;
        for (int k = Edges_row_ptr[sum1]; k < Edges_row_ptr[sum1 + 1]; k++) {
            if (Edges_columns[k] == sum2) {
                idx_ad = k;
                break;
            }
        }
        //storeTriangle(ab,ad,ac);
        Faces[6 * (voxid)+0] = idx_ab;
        Faces[6 * (voxid)+1] = idx_ad;
        Faces[6 * (voxid)+2] = idx_ac;
        Faces[6 * (voxid)+3] = 0;
        Faces[6 * (voxid)+4] = 0;
        Faces[6 * (voxid)+5] = 0;
    }
}


__global__ void  ComputeVertices(float* vertices, float* nodes, float* sdf, int* edges, float m_iso, int nb_edges) {
    unsigned int threadsPerBlock = blockDim.x * blockDim.y;
    unsigned int threadNumInBlock = threadIdx.x + blockDim.x * threadIdx.y;
    unsigned int blockNumInGrid = blockIdx.x + gridDim.x * blockIdx.y;
    int idx = int(blockNumInGrid * threadsPerBlock + threadNumInBlock);

    // assuming x and y inputs are same length
    if (idx >= nb_edges)
        return;

    int sum_1 = edges[2 * idx];
    int sum_2 = edges[2 * idx + 1];


    float sdf_1 = sdf[sum_1] - m_iso;
    float sdf_2 = sdf[sum_2] - m_iso;
    float eps = 1e-8;
    if (abs(sdf_1) < eps) {
        if (signbit(sdf_1))
            sdf_1 = -1e-8;
        else
            sdf_1 = 1e-8;
    }
    if (abs(sdf_2) < eps) {
        if (signbit(sdf_2))
            sdf_2 = -1e-8;
        else
            sdf_2 = 1e-8;
    }
    if (sdf_1 * sdf_2 < 0.0f) {
        //float weight_1 = -sdf_2 / (sdf_1 - sdf_2 + eps);
        //float weight_2 = sdf_1 / (sdf_1 - sdf_2 + eps);
        float weight_1 = 1.0f / (eps + fabs(sdf_1));
        float weight_2 = 1.0f / (eps + fabs(sdf_2));

        vertices[3 * idx] = (weight_1 * nodes[3 * sum_1] + weight_2 * nodes[3 * sum_2])/(weight_1 + weight_2);
        vertices[3 * idx + 1] = (weight_1 * nodes[3 * sum_1 + 1] + weight_2 * nodes[3 * sum_2 + 1]) / (weight_1 + weight_2);
        vertices[3 * idx + 2] = (weight_1 * nodes[3 * sum_1 + 2] + weight_2 * nodes[3 * sum_2 + 2]) / (weight_1 + weight_2);
    }
}

__global__ void  ComputeNormals(float* vertices, float* normals, int* faces, int nb_faces) {
    unsigned int threadsPerBlock = blockDim.x * blockDim.y;
    unsigned int threadNumInBlock = threadIdx.x + blockDim.x * threadIdx.y;
    unsigned int blockNumInGrid = blockIdx.x + gridDim.x * blockIdx.y;
    int n = int(blockNumInGrid * threadsPerBlock + threadNumInBlock);

    // assuming x and y inputs are same length
    if (n > nb_faces-1)
        return;


    float edge_length0 = sqrt((vertices[3 * faces[3 * n]] - vertices[3 * faces[3 * n + 1]]) * (vertices[3 * faces[3 * n]] - vertices[3 * faces[3 * n + 1]]) +
        (vertices[3 * faces[3 * n] + 1] - vertices[3 * faces[3 * n + 1] + 1]) * (vertices[3 * faces[3 * n] + 1] - vertices[3 * faces[3 * n + 1] + 1]) +
        (vertices[3 * faces[3 * n] + 2] - vertices[3 * faces[3 * n + 1] + 2]) * (vertices[3 * faces[3 * n] + 2] - vertices[3 * faces[3 * n + 1] + 2]));

    float edge_length1 = sqrt((vertices[3 * faces[3 * n]] - vertices[3 * faces[3 * n + 2]]) * (vertices[3 * faces[3 * n]] - vertices[3 * faces[3 * n + 2]]) +
        (vertices[3 * faces[3 * n] + 1] - vertices[3 * faces[3 * n + 2] + 1]) * (vertices[3 * faces[3 * n] + 1] - vertices[3 * faces[3 * n + 2] + 1]) +
        (vertices[3 * faces[3 * n] + 2] - vertices[3 * faces[3 * n + 2] + 2]) * (vertices[3 * faces[3 * n] + 2] - vertices[3 * faces[3 * n + 2] + 2]));

    float edge_length2 = sqrt((vertices[3 * faces[3 * n + 1]] - vertices[3 * faces[3 * n + 2]]) * (vertices[3 * faces[3 * n + 1]] - vertices[3 * faces[3 * n + 2]]) +
        (vertices[3 * faces[3 * n + 1] + 1] - vertices[3 * faces[3 * n + 2] + 1]) * (vertices[3 * faces[3 * n + 1] + 1] - vertices[3 * faces[3 * n + 2] + 1]) +
        (vertices[3 * faces[3 * n + 1] + 2] - vertices[3 * faces[3 * n + 2] + 2]) * (vertices[3 * faces[3 * n + 1] + 2] - vertices[3 * faces[3 * n + 2] + 2]));

    if (edge_length0 > 0.3f || edge_length1 > 0.3f || edge_length2 > 0.3f) {
        faces[3 * n] = 0;
        faces[3 * n + 1] = 1;
        faces[3 * n + 2] = 2;
        return;
    }


    float n_tri[3] = { 0.0f, 0.0f, 0.0f };
    get_normal_f(&vertices[3 * faces[3*n]], &vertices[3 * faces[3 * n + 1]], &vertices[3 * faces[3 * n + 2]], n_tri);
    float norm_n = squared_length_f(n_tri);
    if (norm_n > 0.0f){
        normals[3 * n] = n_tri[0] / norm_n;
        normals[3 * n + 1] = n_tri[1] / norm_n;
        normals[3 * n + 2] = n_tri[2] / norm_n;
    }
    else {
        normals[3 * n] = 0.0f;
        normals[3 * n + 1] = 0.0f;
        normals[3 * n + 2] = 0.0f;
    }
}

__global__ void  ComputeNormalsVertices(float* count, float* normals, int* faces, float* normals_faces, int nb_faces) {
    unsigned int threadsPerBlock = blockDim.x * blockDim.y;
    unsigned int threadNumInBlock = threadIdx.x + blockDim.x * threadIdx.y;
    unsigned int blockNumInGrid = blockIdx.x + gridDim.x * blockIdx.y;
    int n = int(blockNumInGrid * threadsPerBlock + threadNumInBlock);

    // assuming x and y inputs are same length
    if (n > nb_faces - 1)
        return;

    atomicAdd(&normals[3 * faces[3 * n]], normals_faces[3*n]);
    atomicAdd(&normals[3 * faces[3 * n] + 1], normals_faces[3 * n + 1]);
    atomicAdd(&normals[3 * faces[3 * n] + 2], normals_faces[3 * n + 2]);
    atomicAdd(&count[3 * faces[3 * n]], 1.0f);

    atomicAdd(&normals[3 * faces[3 * n+1]], normals_faces[3 * n]);
    atomicAdd(&normals[3 * faces[3 * n+1] + 1], normals_faces[3 * n + 1]);
    atomicAdd(&normals[3 * faces[3 * n+1] + 2], normals_faces[3 * n + 2]);
    atomicAdd(&count[3 * faces[3 * n+1]], 1.0f);

    atomicAdd(&normals[3 * faces[3 * n+2]], normals_faces[3 * n]);
    atomicAdd(&normals[3 * faces[3 * n+2] + 1], normals_faces[3 * n + 1]);
    atomicAdd(&normals[3 * faces[3 * n+2] + 2], normals_faces[3 * n + 2]);
    atomicAdd(&count[3 * faces[3 * n+2]], 1.0f);

}

__global__ void  NormalizeNormals(float* count, float* normals, int nb_points) {
    unsigned int threadsPerBlock = blockDim.x * blockDim.y;
    unsigned int threadNumInBlock = threadIdx.x + blockDim.x * threadIdx.y;
    unsigned int blockNumInGrid = blockIdx.x + gridDim.x * blockIdx.y;
    int n = int(blockNumInGrid * threadsPerBlock + threadNumInBlock);

    // assuming x and y inputs are same length
    if (n > nb_points - 1)
        return;

    float n_tri[3] = { normals[3 * n], normals[3 * n + 1], normals[3 * n + 2] };
    float norm_n = squared_length_f(n_tri);
    if (norm_n > 0.0f) {
        normals[3 * n] = n_tri[0] / norm_n;
        normals[3 * n + 1] = n_tri[1] / norm_n;
        normals[3 * n + 2] = n_tri[2] / norm_n;
    }
    else {
        normals[3 * n] = 0.0f;
        normals[3 * n + 1] = 0.0f;
        normals[3 * n + 2] = 0.0f;
    }
}


void marching_tets_cuda(size_t nb_nodes,
    size_t nb_edges,
    size_t nb_tets,
    float m_iso,
    torch::Tensor Faces,       // [N_rays, 6]
    torch::Tensor vertices,       // [N_rays, 6]
    torch::Tensor normals,       // [N_rays, 6]
    torch::Tensor nodes,       // [N_rays, 6]
    torch::Tensor sdf,       // [N_rays, 6]
    torch::Tensor edges,       // [N_rays, 6]
    torch::Tensor tetra       // [N_rays, 6]
    ) {
    
    //0. Create adjacency matrix
    int* edges_cpu = new int[2 * nb_edges];
    cudaMemcpy(edges_cpu, edges.data_ptr<int>(), 2 * nb_edges * sizeof(int), cudaMemcpyDeviceToHost);

    int* Edges_row_ptr_cpu = new int[nb_nodes+1];
    int* Edges_columns_cpu = new int[nb_edges];
    vector<vector<int>> adjacency;
    for (int p = 0; p < nb_nodes; p++) {
        vector<int> tmp;
        adjacency.push_back(tmp);
    }

    for (int e = 0; e < nb_edges; e++) {
        int s_0 = edges_cpu[2 * e] < edges_cpu[2 * e + 1] ? edges_cpu[2 * e] : edges_cpu[2 * e + 1];
        int s_1 = s_0 == edges_cpu[2 * e + 1] ? edges_cpu[2 * e] : edges_cpu[2 * e + 1];
        adjacency[s_0].push_back(s_1);
    }

    int offset = 0;
    Edges_row_ptr_cpu[0] = 0;
    for (int p = 0; p < nb_nodes; p++) {
        int nb_curr_edges = adjacency[p].size();
        for (int i = 0; i < nb_curr_edges; i++) {
            Edges_columns_cpu[offset + i] = adjacency[p][i];
            edges_cpu[2 * (offset + i)] = p;
            edges_cpu[2 * (offset + i) + 1] = adjacency[p][i];
        }
        offset += nb_curr_edges;
        Edges_row_ptr_cpu[p + 1] = offset;
    }

    int* Edges_row_ptr;
    int* Edges_columns;
    cudaMalloc((void**) &Edges_row_ptr, (nb_nodes + 1) * sizeof(int));
    cudaMemcpy(Edges_row_ptr, Edges_row_ptr_cpu, (nb_nodes + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc((void**) &Edges_columns, nb_edges * sizeof(int));
    cudaMemcpy(Edges_columns, Edges_columns_cpu, nb_edges * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(edges.data_ptr<int>(), edges_cpu, 2 * nb_edges * sizeof(int), cudaMemcpyHostToDevice);


    //1. Compute vertices
    const int threads = 1024;
    int blocks = (nb_edges + threads - 1) / threads; 
    AT_DISPATCH_FLOATING_TYPES( sdf.type(),"ComputeVertices", ([&] {  
        ComputeVertices CUDA_KERNEL(blocks,threads) (
            vertices.data_ptr<float>(),
            nodes.data_ptr<float>(),
            sdf.data_ptr<float>(),
            edges.data_ptr<int>(),
            m_iso,  
            nb_edges); 
    }));

    //2. Create triangular faces
    blocks = (nb_tets + threads - 1) / threads; 
    AT_DISPATCH_FLOATING_TYPES( sdf.type(),"GenerateFaces", ([&] {  
        GenerateFaces CUDA_KERNEL(blocks,threads) (
            Faces.data_ptr<int>(),
            sdf.data_ptr<float>(),
            Edges_row_ptr,
            Edges_columns,
            tetra.data_ptr<int>(),
            m_iso,  
            nb_tets); 
    }));


    //3. Compute normals of triangular faces
    int nb_faces = 2 * nb_tets;
    blocks = (nb_faces + threads - 1) / threads;  
    AT_DISPATCH_FLOATING_TYPES( vertices.type(),"ComputeNormals", ([&] {  
        ComputeNormals CUDA_KERNEL(blocks,threads) (
            vertices.data_ptr<float>(),
            normals.data_ptr<float>(),
            Faces.data_ptr<int>(),
            nb_faces); 
    }));

    cudaFree(Edges_row_ptr);
    cudaFree(Edges_columns);

    delete[] edges_cpu;
    delete[] Edges_row_ptr_cpu;
    delete[] Edges_columns_cpu;
    adjacency.clear();
}


/*void compute_normals_mesh(float* normals_v, int* Faces, float* normals, int nb_faces, int nb_points) {


    float* count;
    gpuErrchk(cudaMalloc((void**)&count, nb_points * sizeof(float)));
    gpuErrchk(cudaMemset(count, 0, nb_points * sizeof(float)));

    //1. Compute vertices
    dim3 dimBlock(THREAD_SIZE_X, THREAD_SIZE_Y, 1);
    int thread_size_1 = int(round(sqrt(nb_faces)) + 1);
    dim3 dimGrid_1(1, 1, 1);
    dimGrid_1.x = divUp(thread_size_1, dimBlock.x); // #cols
    dimGrid_1.y = divUp(thread_size_1, dimBlock.y); // # rows

    ComputeNormalsVertices << < dimGrid_1, dimBlock >> > (count, normals_v, Faces, normals, nb_faces);
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaPeekAtLastError());


    int thread_size_2 = int(round(sqrt(nb_points)) + 1);
    dim3 dimGrid_2(1, 1, 1);
    dimGrid_2.x = divUp(thread_size_2, dimBlock.x); // #cols
    dimGrid_2.y = divUp(thread_size_2, dimBlock.y); // # rows

    NormalizeNormals << < dimGrid_2, dimGrid_2 >> > (count, normals_v, nb_points);
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaPeekAtLastError());

    gpuErrchk(cudaFree(count));
}*/

