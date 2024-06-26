#include <torch/extension.h>
#include <vector>
#include <Eigen/Core>
#include <igl/grad.h>
#include <igl/cotmatrix.h>
#include <igl/list_to_matrix.h>
#include <igl/massmatrix.h>
#include <igl/hessian_energy.h>
#include <igl/curved_hessian_energy.h>
#include <igl/invert_diag.h>
#include <Eigen/SparseCholesky>
#include <iostream>

//#include "../Models/cudaType.cuh"

void MaskLaplacian_gpu(torch::Tensor mask_sites, torch::Tensor L_values, torch::Tensor L_outer_start, torch::Tensor L_nonZeros,  size_t L_nnZ, size_t L_outerSize, size_t L_cols);

void SparseMul_gpu(torch::Tensor div, torch::Tensor sdf, int dim, torch::Tensor active_sites, torch::Tensor M_values, torch::Tensor L_values, torch::Tensor L_outer_start, torch::Tensor L_nonZeros, size_t L_nnZ, size_t L_outerSize, size_t L_cols);

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


std::vector<at::Tensor> MakeLaplacian(size_t num_vertices, size_t num_tets, torch::Tensor vertices_T, torch::Tensor tets_T, torch::Tensor tets_valid) {
        std::cout << "Load data to compute laplacian" << std::endl;
        Eigen::MatrixXd V, U;
        Eigen::MatrixXi F;
        double* vertices = (double*) vertices_T.data_ptr<double>();
        int* tets = (int*) tets_T.data_ptr<int>();
        int* valid = (int*) tets_valid.data_ptr<int>();

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

        int nb_valid_tets = 0;
        for (int i = 0; i < num_tets; i++) {
            int id_3 = tets[4 * i]^ tets[4 * i+1]^ tets[4 * i+2]^tets[4 * i + 3];
            double edge_1[3] = {vertices[3*tets[4 * i]] - vertices[3*tets[4 * i + 1]],
                                vertices[3*tets[4 * i] + 1] - vertices[3*tets[4 * i + 1] + 1],
                                vertices[3*tets[4 * i] + 2] - vertices[3*tets[4 * i + 1] + 2]};
            double edge_2[3] = {vertices[3*tets[4 * i]] - vertices[3*tets[4 * i + 2]],
                                vertices[3*tets[4 * i] + 1] - vertices[3*tets[4 * i + 2] + 1],
                                vertices[3*tets[4 * i] + 2] - vertices[3*tets[4 * i + 2] + 2]};
            double edge_3[3] = {vertices[3*tets[4 * i]] - vertices[3*id_3],
                                vertices[3*tets[4 * i] + 1] - vertices[3*id_3 + 1],
                                vertices[3*tets[4 * i] + 2] - vertices[3*id_3 + 2]};
            double edge_4[3] = {vertices[3*tets[4 * i + 1]] - vertices[3*tets[4 * i + 2]],
                                vertices[3*tets[4 * i + 1] + 1] - vertices[3*tets[4 * i + 2] + 1],
                                vertices[3*tets[4 * i + 1] + 2] - vertices[3*tets[4 * i + 2] + 2]};
            double edge_5[3] = {vertices[3*tets[4 * i + 1]] - vertices[3*id_3],
                                vertices[3*tets[4 * i + 1] + 1] - vertices[3*id_3 + 1],
                                vertices[3*tets[4 * i + 1] + 2] - vertices[3*id_3 + 2]};
            double edge_6[3] = {vertices[3*tets[4 * i + 2]] - vertices[3*id_3],
                                vertices[3*tets[4 * i + 2] + 1] - vertices[3*id_3 + 1],
                                vertices[3*tets[4 * i + 2] + 2] - vertices[3*id_3 + 2]};

            double length_1 = sqrt(edge_1[0]*edge_1[0] + edge_1[1]*edge_1[1] + edge_1[2]*edge_1[2]);
            double length_2 = sqrt(edge_2[0]*edge_2[0] + edge_2[1]*edge_2[1] + edge_2[2]*edge_2[2]);
            double length_3 = sqrt(edge_3[0]*edge_3[0] + edge_3[1]*edge_3[1] + edge_3[2]*edge_3[2]);
            double length_4 = sqrt(edge_4[0]*edge_4[0] + edge_4[1]*edge_4[1] + edge_4[2]*edge_4[2]);
            double length_5 = sqrt(edge_5[0]*edge_5[0] + edge_5[1]*edge_5[1] + edge_5[2]*edge_5[2]);
            double length_6 = sqrt(edge_6[0]*edge_6[0] + edge_6[1]*edge_6[1] + edge_6[2]*edge_6[2]);

            if (length_1 + length_2 <= length_3 || length_1 + length_3 <= length_2 || length_2 + length_3 <= length_1 ||
                  length_1 + length_2 <= length_4 || length_1 + length_4 <= length_2 || length_2 + length_4 <= length_1 || 
                  length_1 + length_4 <= length_3 || length_1 + length_3 <= length_4 || length_4 + length_3 <= length_1 ||
                  length_4 + length_2 <= length_3 || length_4 + length_3 <= length_2 || length_2 + length_3 <= length_4)
                  valid[i] = 0;

            if (valid[i] == 1) 
                nb_valid_tets++;
        }

        std::cout << "total nb tet: " << num_tets << std::endl;
        std::cout << "total valid tet: " << nb_valid_tets << std::endl;
        vF.resize(nb_valid_tets);
        int id_tet = 0;
        for (int i = 0; i < num_tets; i++) {
            if (valid[i] == 0) 
                continue;
            std::vector<int > face;
            face.resize(4);
            face[0] = tets[4 * i];
            face[1] = tets[4 * i + 1];
            face[2] = tets[4 * i + 2];
            face[3] = tets[4 * i]^ tets[4 * i+1]^ tets[4 * i+2]^tets[4 * i + 3];
            vF[id_tet] = face;
            id_tet++;
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

        std::cout << "Laplace beltrami operator computed" << std::endl;
        Eigen::SparseMatrix<double> L;
        igl::cotmatrix(V, F, L);

        /*std::cout << "Mass matrix computed" << std::endl;
        Eigen::SparseMatrix<double> M;
        igl::massmatrix(V,F,igl::MASSMATRIX_TYPE_BARYCENTRIC,M);
        Eigen::SparseMatrix<double> M_inv;
        igl::invert_diag(M, M_inv);
        std::cout << "M size " << M.nonZeros() << std::endl;
        std::cout << "num_vertices " << num_vertices << std::endl;        
        L = M_inv*L;*/

        std::cout << "Cotangent computed" << std::endl;
        auto d_values = torch::zeros({L.nonZeros()}, torch::TensorOptions().dtype(torch::kFloat64));
        memcpy((void*)d_values.data_ptr<double>(), (void*)L.valuePtr(), L.nonZeros() * sizeof(double));
        
        // Attention -> Hack!!
        auto d_values_M = torch::zeros({L.nonZeros()}, torch::TensorOptions().dtype(torch::kFloat64));
        memcpy((void*)d_values_M.data_ptr<double>(), (void*)L.valuePtr(), L.nonZeros() * sizeof(double));
        //auto d_values_M = torch::zeros({M.nonZeros()}, torch::TensorOptions().dtype(torch::kFloat64));
        //memcpy((void*)d_values_M.data_ptr<double>(), (void*)M.valuePtr(), M.nonZeros() * sizeof(double));

        std::cout << "values computed" << std::endl;
        auto d_nonZeros= torch::zeros({L.nonZeros()}, torch::TensorOptions().dtype(torch::kInt32));
        memcpy((void*)d_nonZeros.data_ptr<int>(), (void*)L.innerIndexPtr(), L.nonZeros() * sizeof(int));
        
        //auto d_nonZeros_M= torch::zeros({M.nonZeros()}, torch::TensorOptions().dtype(torch::kInt32));
        //memcpy((void*)d_nonZeros_M.data_ptr<int>(), (void*)M.innerIndexPtr(), M.nonZeros() * sizeof(int));

        std::cout << "indx computed" << std::endl;
        auto d_outer_start= torch::zeros({L.outerSize()}, torch::TensorOptions().dtype(torch::kInt32));
        memcpy((void*)d_outer_start.data_ptr<int>(), (void*)L.outerIndexPtr(), L.outerSize() * sizeof(int));
        
        ///auto d_outer_start_M= torch::zeros({M.outerSize()}, torch::TensorOptions().dtype(torch::kInt32));
        //memcpy((void*)d_outer_start_M.data_ptr<int>(), (void*)M.outerIndexPtr(), M.outerSize() * sizeof(int));

        auto d_size= torch::zeros({3}, torch::TensorOptions().dtype(torch::kInt32));
        int sizes_cpu[3] = {int(L.nonZeros()), int(L.outerSize()), int(L.cols())};
        memcpy((void*)d_size.data_ptr<int>(), (void*)sizes_cpu, 3 * sizeof(int));
        
        //auto d_size_M= torch::zeros({3}, torch::TensorOptions().dtype(torch::kInt32));
        //int sizes_cpu_M[3] = {int(M.nonZeros()), int(M.outerSize()), int(M.cols())};
        //memcpy((void*)d_size_M.data_ptr<int>(), (void*)sizes_cpu_M, 3 * sizeof(int));

        return {d_values_M, d_values, d_nonZeros, d_outer_start, d_size};
    }


void MeanCurve(torch::Tensor div, torch::Tensor sdf, int dim, torch::Tensor active_sites, torch::Tensor M_values, torch::Tensor L_values, torch::Tensor L_outer_start, torch::Tensor L_nonZeros, size_t L_nnZ, size_t L_outerSize, size_t L_cols) {
    SparseMul_gpu(div, sdf, dim, active_sites, M_values, L_values, L_outer_start, L_nonZeros, L_nnZ, L_outerSize, L_cols);
}

void MaskLaplacian(torch::Tensor mask_sites, torch::Tensor L_values, torch::Tensor L_outer_start, torch::Tensor L_nonZeros,  size_t L_nnZ, size_t L_outerSize, size_t L_cols){
    MaskLaplacian_gpu(mask_sites, L_values, L_outer_start, L_nonZeros, L_nnZ, L_outerSize, L_cols);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("MakeLaplacian", &MakeLaplacian, "MakeLaplacian (CPP)");
    m.def("MeanCurve", &MeanCurve, "MeanCurve (CPP)");
    m.def("MaskLaplacian", &MaskLaplacian, "MaskLaplacian (CPP)");
}