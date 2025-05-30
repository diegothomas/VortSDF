#include <torch/extension.h>
#include <vector>


// Definition of cuda functions
void test_inverse_cuda(
    size_t sizeMatrix,
    torch::Tensor Buff, 
    torch::Tensor A,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor A_inv    // [N_sites, 3] for each voxel => it's vertices
);

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
);

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
);

void Laplace_grad_cuda(
    size_t num_tets,                // number of rays
    size_t num_sites,                // number of rays
    torch::Tensor  tets,  // [N_voxels, 4] for each voxel => it's neighbors
    torch::Tensor  sites,  // [N_voxels, 4] for each voxel => it's neighbors
    torch::Tensor  grad,  // [N_voxels, 4] for each voxel => it's neighbors
    torch::Tensor  grad_lapl,     // [N_voxels, 4] for each voxel => it's vertices
    torch::Tensor  weights_tot
);


float cvt_grad_cuda(
    size_t num_sites,
    size_t num_knn,
    const float sigma,                // number of rays
    torch::Tensor thetas, 
    torch::Tensor phis, 
    torch::Tensor gammas, 
    torch::Tensor neighbors,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor sites,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor freeze,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor sdf,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor grad_sites    // [N_sites, 3] for each voxel => it's vertices
);

void sdf_grad_cuda(
    size_t num_sites,
    size_t num_knn,
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

void diff_tensor_cuda(
    size_t num_tets,                // number of rays
    torch::Tensor  tets,  // [N_voxels, 4] for each voxel => it's neighbors
    torch::Tensor  sites,  // [N_voxels, 4] for each voxel => it's neighbors
    torch::Tensor  vol,     // [N_voxels, 4] for each voxel => it's vertices
    torch::Tensor  weights,     // [N_voxels, 4] for each voxel => it's vertices)
    torch::Tensor  weights_tot
);

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
    torch::Tensor  grad_smooth,     // [N_voxels, 4] for each voxel => it's vertices)
    torch::Tensor  grad_sdf,     // [N_voxels, 4] for each voxel => it's vertices)
    torch::Tensor  vol,     // [N_voxels, 4] for each voxel => it's vertices
    torch::Tensor  weights,     // [N_voxels, 4] for each voxel => it's vertices
    torch::Tensor  weights_tot,     // [N_voxels, 4] for each voxel => it's vertices
    torch::Tensor  eik_loss     // [N_voxels, 4] for each voxel => it's vertices
);

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
);

void concat_feat_cuda(
    size_t num_sites,                // number of rays
    size_t num_knn,  
    size_t dim_feat,
    torch::Tensor  vertices,     // [N_voxels, 4] for each voxel => it's vertices
    torch::Tensor  activated,     // [N_voxels, 4] for each voxel => it's vertices
    torch::Tensor  grads,     // [N_voxels, 4] for each voxel => it's vertices
    torch::Tensor  feat,     // [N_voxels, 4] for each voxel => it's vertices
    torch::Tensor  neighbors
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

void knn_sdf_space_grad(
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
){
    knn_sdf_space_grad_cuda(
        num_sites,                // number of rays
        num_knn,                // number of rays
        neighbors,  // [N_voxels, 4] for each voxel => it's neighbors
        sites,  // [N_voxels, 4] for each voxel => it's neighbors
        activated,  // [N_voxels, 4] for each voxel => it's neighbors
        sdf,  // [N_voxels, 4] for each voxel => it's neighbors
        feat,  // [N_voxels, 4] for each voxel => it's neighbors
        grad_sdf,     // [N_voxels, 4] for each voxel => it's vertices
        grad_feat,     // [N_voxels, 4] for each voxel => it's vertices
        weights_tot
    );

}

void sdf_space_grad(
    size_t num_tets,                // number of rays
    size_t num_sites,                // number of rays
    torch::Tensor  tets,  // [N_voxels, 4] for each voxel => it's neighbors
    torch::Tensor  sites,  // [N_voxels, 4] for each voxel => it's neighbors
    torch::Tensor  sdf,  // [N_voxels, 4] for each voxel => it's neighbors
    torch::Tensor  feat,  // [N_voxels, 4] for each voxel => it's neighbors
    torch::Tensor  grad_sdf,     // [N_voxels, 4] for each voxel => it's vertices
    torch::Tensor  grad_feat,     // [N_voxels, 4] for each voxel => it's vertices
    torch::Tensor  weights_tot
)  {
    //std::cout << "March through implicit cvt" << std::endl; 
    /*CHECK_INPUT(rays);
    CHECK_INPUT(neighbors);
    CHECK_INPUT(sites);
    CHECK_INPUT(samples);*/

    sdf_space_grad_cuda(
        num_tets,                // number of rays
        num_sites,                // number of rays
        tets,  // [N_voxels, 4] for each voxel => it's neighbors
        sites,  // [N_voxels, 4] for each voxel => it's neighbors
        sdf,  // [N_voxels, 4] for each voxel => it's neighbors
        feat,  // [N_voxels, 4] for each voxel => it's neighbors
        grad_sdf,     // [N_voxels, 4] for each voxel => it's vertices
        grad_feat,     // [N_voxels, 4] for each voxel => it's vertices
        weights_tot
    );

}

void sdf_laplace_grad(
    size_t num_tets,                // number of rays
    size_t num_sites,                // number of rays
    torch::Tensor  tets,  // [N_voxels, 4] for each voxel => it's neighbors
    torch::Tensor  sites,  // [N_voxels, 4] for each voxel => it's neighbors
    torch::Tensor  grad,  // [N_voxels, 4] for each voxel => it's neighbors
    torch::Tensor  grad_lapl,     // [N_voxels, 4] for each voxel => it's vertices
    torch::Tensor  weights_tot
)  {

    Laplace_grad_cuda(
        num_tets,                // number of rays
        num_sites,                // number of rays
        tets,  // [N_voxels, 4] for each voxel => it's neighbors
        sites,  // [N_voxels, 4] for each voxel => it's neighbors
        grad,  // [N_voxels, 4] for each voxel => it's neighbors
        grad_lapl,     // [N_voxels, 4] for each voxel => it's vertices
        weights_tot
    );

}


float cvt_grad(
    size_t num_sites,
    size_t num_knn,
    const float sigma,                // number of rays
    torch::Tensor thetas, 
    torch::Tensor phis, 
    torch::Tensor gammas, 
    torch::Tensor neighbors,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor sites,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor freeze,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor sdf,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor grad_sites    // [N_sites, 3] for each voxel => it's vertices
) {
    //std::cout << "March through implicit cvt" << std::endl; 
    /*CHECK_INPUT(rays);
    CHECK_INPUT(neighbors);
    CHECK_INPUT(sites);
    CHECK_INPUT(samples);*/

    return cvt_grad_cuda(
        num_sites,
        num_knn,
        sigma,
        thetas,
        phis,
        gammas, 
        neighbors,   
        sites, 
        freeze, 
        sdf, 
        grad_sites );

}

void sdf_grad(
    size_t num_sites,
    size_t num_knn,
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

void diff_tensor(
    size_t num_tets,                // number of rays
    torch::Tensor  tets,  // [N_voxels, 4] for each voxel => it's neighbors
    torch::Tensor  sites,  // [N_voxels, 4] for each voxel => it's neighbors
    torch::Tensor  vol,     // [N_voxels, 4] for each voxel => it's vertices
    torch::Tensor  weights,     // [N_voxels, 4] for each voxel => it's vertices)
    torch::Tensor  weights_tot
) {
    diff_tensor_cuda(
        num_tets,
        tets,
        sites, 
        vol,
        weights, 
        weights_tot);
}

void eikonal_grad(
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
) {
    eikonal_grad_cuda(
        num_tets,
        num_sites,
        tets,
        valid_tets, 
        sites, 
        activated,
        sdf, 
        sdf_smooth, 
        feat, 
        grad_eik,   
        grad_smooth,   
        grad_sdf, 
        vol,     // [N_voxels, 4] for each voxel => it's vertices
        weights,     // [N_voxels, 4] for each voxel => it's vertices
        weights_tot,     // [N_voxels, 4] for each voxel => it's vertices
        eik_loss);

}

void backprop_norm_grad(
    size_t num_tets,                // number of rays
    torch::Tensor  tets,  // [N_voxels, 4] for each voxel => it's neighbors
    torch::Tensor  sites,  // [N_voxels, 4] for each voxel => it's neighbors
    torch::Tensor  activated,  // [N_voxels, 4] for each voxel => it's neighbors
    torch::Tensor  grad_sdf,     // [N_voxels, 4] for each voxel => it's vertices
    torch::Tensor  grad_norm,     // [N_voxels, 4] for each voxel => it's vertices)
    torch::Tensor  vol,     // [N_voxels, 4] for each voxel => it's vertices
    torch::Tensor  weights,     // [N_voxels, 4] for each voxel => it's vertices
    torch::Tensor  weights_tot    // [N_voxels, 4] for each voxel => it's vertices
) {
    backprop_norm_grad_cuda(
        num_tets,
        tets,
        sites, 
        activated,
        grad_sdf,   
        grad_norm, 
        vol,     // [N_voxels, 4] for each voxel => it's vertices
        weights,     // [N_voxels, 4] for each voxel => it's vertices
        weights_tot);
}

void concat_feat(
    size_t num_sites,                // number of rays
    size_t num_knn,  
    size_t dim_feat,
    torch::Tensor  vertices,     // [N_voxels, 4] for each voxel => it's vertices
    torch::Tensor  activated,     // [N_voxels, 4] for each voxel => it's vertices
    torch::Tensor  grads,     // [N_voxels, 4] for each voxel => it's vertices
    torch::Tensor  feat,     // [N_voxels, 4] for each voxel => it's vertices
    torch::Tensor  neighbors
    ) {
        concat_feat_cuda(
            num_sites,                // number of rays
            num_knn,  
            dim_feat,
            vertices,     // [N_voxels, 4] for each voxel => it's vertices
            activated,     // [N_voxels, 4] for each voxel => it's vertices
            grads,     // [N_voxels, 4] for each voxel => it's vertices
            feat,     // [N_voxels, 4] for each voxel => it's vertices
            neighbors
        );
    }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("test_inverse", &test_inverse, "test_inverse (CPP)");
    m.def("knn_sdf_space_grad", &knn_sdf_space_grad, "knn_sdf_space_grad (CPP)");
    m.def("sdf_space_grad", &sdf_space_grad, "sdf_space_grad (CPP)");
    m.def("sdf_laplace_grad", &sdf_laplace_grad, "sdf_laplace_grad (CPP)");
    m.def("cvt_grad", &cvt_grad, "cvt_grad (CPP)");
    m.def("sdf_grad", &sdf_grad, "sdf_grad (CPP)");
    m.def("update_sdf", &update_sdf, "update_sdf (CPP)");
    m.def("eikonal_grad", &eikonal_grad, "eikonal_grad (CPP)");
    m.def("concat_feat", &concat_feat, "concat_feat (CPP)");
    m.def("diff_tensor", &diff_tensor, "diff_tensor (CPP)");
    m.def("backprop_norm_grad", &backprop_norm_grad, "backprop_norm_grad (CPP)");
}