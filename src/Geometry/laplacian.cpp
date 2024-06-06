#include <torch/extension.h>
#include <vector>
#include <Eigen/Core>
#include <igl/grad.h>
#include <igl/cotmatrix.h>
#include <igl/list_to_matrix.h>
#include <igl/massmatrix.h>
#include <igl/hessian_energy.h>
#include <igl/curved_hessian_energy.h>
#include <Eigen/SparseCholesky>
#include <iostream>

//#include "../Models/cudaType.cuh"

void SparseMul_gpu(torch::Tensor div, torch::Tensor sdf, torch::Tensor active_sites, torch::Tensor L_values, torch::Tensor L_outer_start, torch::Tensor L_nonZeros, size_t L_nnZ, size_t L_outerSize, size_t L_cols);

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


std::vector<at::Tensor> MakeLaplacian(size_t num_vertices, size_t num_tets, torch::Tensor vertices_T, torch::Tensor tets_T) {
        std::cout << "Load data to compute laplacian" << std::endl;
        Eigen::MatrixXd V, U;
        Eigen::MatrixXi F;
        float* vertices = (float*) vertices_T.data_ptr<float>();
        int* tets = (int*) tets_T.data_ptr<int>();

        // Load tetrahedral structure onto the mesh
        std::vector<std::vector<double> > vV;
        std::vector<std::vector<int> > vF;
        vV.resize(num_vertices);
        for (int i = 0; i < num_vertices; i++) {
            std::vector<double > vertex;
            vertex.resize(3);
            vertex[0] = double(vertices[3*i]);
            vertex[1] = double(vertices[3 * i + 1]);
            vertex[2] = double(vertices[3 * i + 2]);
            vV[i] = vertex;
        }

        vF.resize(num_tets);
        for (int i = 0; i < num_tets; i++) {
            std::vector<int > face;
            face.resize(4);
            face[0] = tets[4 * i];
            face[1] = tets[4 * i + 1];
            face[2] = tets[4 * i + 2];
            face[3] = tets[4 * i]^ tets[4 * i+1]^ tets[4 * i+2]^tets[4 * i + 3];
            vF[i] = face;
        }

        std::cout << "data loaded computed" << std::endl;
        bool V_rect = igl::list_to_matrix(vV, V);
        if (!V_rect)
        {
            std::cout << "ERROR loading Vertices mesh!!" << std::endl;
            return {torch::zeros({0}, torch::TensorOptions().dtype(torch::kFloat64)), 
                    torch::zeros({0}, torch::TensorOptions().dtype(torch::kInt32)), 
                    torch::zeros({0}, torch::TensorOptions().dtype(torch::kInt32))};
        }
        bool F_rect = igl::list_to_matrix(vF, F);
        if (!F_rect)
        {
            std::cout << "ERROR loading Tetrahedra mesh!!" << std::endl;
            return {torch::zeros({0}, torch::TensorOptions().dtype(torch::kFloat64)), 
                    torch::zeros({0}, torch::TensorOptions().dtype(torch::kInt32)), 
                    torch::zeros({0}, torch::TensorOptions().dtype(torch::kInt32))};
        }

        std::cout << "Matrix computed" << std::endl;
        // Compute Laplace-Beltrami operator: #V by #V
        //Eigen::SparseMatrix<double> QL;
        Eigen::SparseMatrix<double> L;
        igl::cotmatrix(V, F, L);

        std::cout << "Laplacian computed" << std::endl;
        // Load Laplacian on GPU and store in tet structure
        //std::vector<int64_t> size_vec;
        //size_vec.push_back(L.nonZeros());
        auto d_values = torch::zeros({L.nonZeros()}, torch::TensorOptions().dtype(torch::kFloat64));
        memcpy((void*)d_values.data_ptr<double>(), (void*)L.valuePtr(), L.nonZeros() * sizeof(double));

        //double* values; // stores the coefficient values of the non-zeros.
        //gpuErrchk(cudaMalloc(&values, L.nonZeros() * sizeof(double)));
        //gpuErrchk(cudaMemcpy((void*)d_values.data_ptr<double>(), (void*)L.valuePtr(), L.nonZeros() * sizeof(double), cudaMemcpyHostToDevice));
        //gpuErrchk(cudaDeviceSynchronize());

        std::cout << "values computed" << std::endl;
        auto d_nonZeros= torch::zeros({L.nonZeros()}, torch::TensorOptions().dtype(torch::kInt32));
        memcpy((void*)d_nonZeros.data_ptr<int>(), (void*)L.innerIndexPtr(), L.nonZeros() * sizeof(int));

        //int* indx; //  stores the row (resp. column) indices of the non-zeros.
        //gpuErrchk(cudaMalloc(&indx, L.nonZeros() * sizeof(int)));
        //gpuErrchk(cudaMemcpy((void*)d_indx.data_ptr<int>(), (void*)L.innerIndexPtr(), L.nonZeros() * sizeof(int), cudaMemcpyHostToDevice));
        //gpuErrchk(cudaDeviceSynchronize());

        std::cout << "indx computed" << std::endl;
        //size_vec[0] = L.outerSize();
        auto d_outer_start= torch::zeros({L.outerSize()}, torch::TensorOptions().dtype(torch::kInt32));
        memcpy((void*)d_outer_start.data_ptr<int>(), (void*)L.outerIndexPtr(), L.outerSize() * sizeof(int));

        //int* outer_start; // stores for each column (resp. row) the index of the first non-zero in the previous two arrays.
        //gpuErrchk(cudaMalloc(&outer_start, L.outerSize() * sizeof(int)));
        //gpuErrchk(cudaMemcpy((void*)d_outer_start.data_ptr<int>(), (void*)L.outerIndexPtr(), L.outerSize() * sizeof(int), cudaMemcpyHostToDevice));
        //gpuErrchk(cudaDeviceSynchronize());
        
        auto d_size= torch::zeros({3}, torch::TensorOptions().dtype(torch::kInt32));
        int sizes_cpu[3] = {int(L.nonZeros()), int(L.outerSize()), int(L.cols())};
        memcpy((void*)d_size.data_ptr<int>(), (void*)sizes_cpu, 3 * sizeof(int));

        return {d_values, d_nonZeros, d_outer_start, d_size};
    }


void MeanCurve(torch::Tensor div, torch::Tensor sdf, torch::Tensor active_sites, torch::Tensor L_values, torch::Tensor L_outer_start, torch::Tensor L_nonZeros, size_t L_nnZ, size_t L_outerSize, size_t L_cols) {
    SparseMul_gpu(div, sdf, active_sites, L_values, L_outer_start, L_nonZeros, L_nnZ, L_outerSize, L_cols);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("MakeLaplacian", &MakeLaplacian, "MakeLaplacian (CPP)");
    m.def("MeanCurve", &MeanCurve, "MeanCurve (CPP)");
}