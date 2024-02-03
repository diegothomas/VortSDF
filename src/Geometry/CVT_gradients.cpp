#include <torch/extension.h>
#include <vector>


// Definition of cuda functions
void test_inverse_cuda(
    size_t sizeMatrix,
    torch::Tensor Buff, 
    torch::Tensor A,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor A_inv    // [N_sites, 3] for each voxel => it's vertices
);


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
);


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
);

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
);


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
);

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void test_inverse(
    size_t sizeMatrix,
    torch::Tensor Buff, 
    torch::Tensor A,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor A_inv    // [N_sites, 3] for each voxel => it's vertices
) {
    test_inverse_cuda(
        sizeMatrix,
        Buff, 
        A,
        A_inv );

}

void sdf_space_grad(
    size_t num_sites,
    size_t num_knn,
    torch::Tensor neighbors,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor sites,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor sdf,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor feat,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor Weights,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor grad_sites,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor grad_feat_sites    // [N_sites, 3] for each voxel => it's vertices
)  {
    //std::cout << "March through implicit cvt" << std::endl; 
    /*CHECK_INPUT(rays);
    CHECK_INPUT(neighbors);
    CHECK_INPUT(sites);
    CHECK_INPUT(samples);*/

    sdf_space_grad_cuda(
        num_sites,
        num_knn,
        neighbors,   
        sites, 
        sdf,
        feat,
        Weights,
        grad_sites,
        grad_feat_sites );

}

void cvt_grad(
    size_t num_sites,
    size_t num_knn,
    torch::Tensor thetas, 
    torch::Tensor phis, 
    torch::Tensor gammas, 
    torch::Tensor neighbors,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor sites,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor sdf,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor grad_sites    // [N_sites, 3] for each voxel => it's vertices
) {
    //std::cout << "March through implicit cvt" << std::endl; 
    /*CHECK_INPUT(rays);
    CHECK_INPUT(neighbors);
    CHECK_INPUT(sites);
    CHECK_INPUT(samples);*/

    cvt_grad_cuda(
        num_sites,
        num_knn,
        thetas,
        phis,
        gammas, 
        neighbors,   
        sites, 
        sdf,
        grad_sites );

}

void sdf_grad(
    size_t num_sites,
    size_t num_knn,
    float weight_sdf,
    torch::Tensor thetas,      // [N_rays, 6]
    torch::Tensor grad_sdf_space,      // [N_rays, 6]
    torch::Tensor grad_grad, // [N_voxels, 26] for each voxel => it's neighbors
    torch::Tensor neighbors,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor sites,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor sdf,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor grad_sites    // [N_sites, 3] for each voxel => it's vertices
) {
    //std::cout << "March through implicit cvt" << std::endl; 
    /*CHECK_INPUT(rays);
    CHECK_INPUT(neighbors);
    CHECK_INPUT(sites);
    CHECK_INPUT(samples);*/

    sdf_grad_cuda(
        num_sites,
        num_knn,
        weight_sdf,
        thetas,
        grad_sdf_space,   
        grad_grad, 
        neighbors,   
        sites, 
        sdf,
        grad_sites );

}

void update_sdf(
    size_t num_sites,
    size_t num_knn,
    torch::Tensor neighbors,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor sites,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor sdf,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor feat,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor grad_sites,    // [N_sites, 3] for each voxel => it's vertices,    
    torch::Tensor sdf_diff,    // [N_sites, 3] for each voxel => it's vertices,    
    torch::Tensor feat_diff    // [N_sites, 3] for each voxel => it's vertices
) {
    update_sdf_cuda(
        num_sites,
        num_knn,
        neighbors,   
        sites, 
        sdf,
        feat,
        grad_sites,
        sdf_diff,
        feat_diff  );

}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("test_inverse", &test_inverse, "test_inverse (CPP)");
    m.def("sdf_space_grad", &sdf_space_grad, "sdf_space_grad (CPP)");
    m.def("cvt_grad", &cvt_grad, "cvt_grad (CPP)");
    m.def("sdf_grad", &sdf_grad, "sdf_grad (CPP)");
    m.def("update_sdf", &update_sdf, "update_sdf (CPP)");
}