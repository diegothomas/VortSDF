
#include <torch/extension.h>
#include <vector>

// *************************
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
);

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


// ***************************
void marching_tets(size_t nb_nodes,
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
    //std::cout << "Backprop feature gradients" << std::endl; 
    marching_tets_cuda(nb_nodes,
        nb_edges,
        nb_tets,
        m_iso,
        Faces,       // [N_rays, 6]
        vertices,       // [N_rays, 6]
        normals,       // [N_rays, 6]
        nodes,       // [N_rays, 6]
        sdf,       // [N_rays, 6]
        edges,       // [N_rays, 6]
        tetra       // [N_rays, 6]);
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("marching_tets", &marching_tets, "eikonal_marching_tetsloss (CPP)");
}