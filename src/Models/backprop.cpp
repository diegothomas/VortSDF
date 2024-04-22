#include <torch/extension.h>
#include <vector>

// *************************
void backprop_feat_cuda(
    size_t num_samples,
    size_t num_sites,
    size_t dim_feats,
    torch::Tensor sdf,
    torch::Tensor grad_feat,
    torch::Tensor counter,
    torch::Tensor grad_samples,
    torch::Tensor cell_ids,
    torch::Tensor cell_weights 
); 

void backprop_sdf_cuda(
    size_t num_samples,
    torch::Tensor grad_sdf,
    torch::Tensor grad_sdf_samples,
    torch::Tensor cell_ids,
    torch::Tensor cell_weights 
);

void backprop_norm_cuda(
    size_t num_tets,
    torch::Tensor tets,
    torch::Tensor sites,
    torch::Tensor vol,    
    torch::Tensor weights,  
    torch::Tensor weights_tot, 
    torch::Tensor grad_norm,
    torch::Tensor grad_sdf,
    torch::Tensor activated 
);

void  backprop_unit_norm_cuda(
    size_t num_tets,
    torch::Tensor tets,
    torch::Tensor sites,
    torch::Tensor norm_grad,
    torch::Tensor grad_unormed,
    torch::Tensor vol,    
    torch::Tensor weights,  
    torch::Tensor weights_tot, 
    torch::Tensor grad_norm ,
    torch::Tensor grad_sdf,
    torch::Tensor activated 
);

float eikonal_loss_cuda(
    size_t num_sites,
    size_t num_knn,
    torch::Tensor neighbors,
    torch::Tensor Weights,
    torch::Tensor grad,
    torch::Tensor grad_eik 
);

void smooth_cuda(
    size_t num_edges,
    float sigma,
    torch::Tensor vertices,
    torch::Tensor activated,
    torch::Tensor sdf,
    torch::Tensor feat,
    torch::Tensor edges,
    torch::Tensor sdf_grad,
    torch::Tensor feat_grad,
    torch::Tensor counter 
);

void space_reg_cuda(
    size_t num_rays,
    torch::Tensor rays_d,
    torch::Tensor grad_space,
    torch::Tensor out_weights,
    torch::Tensor out_z,
    torch::Tensor out_sdf,
    torch::Tensor out_feat,
    torch::Tensor out_ids,
    torch::Tensor offset,
    torch::Tensor sdf_grad,
    torch::Tensor feat_grad 
);

void smooth_sdf_cuda(
    size_t num_edges,
    size_t num_sites,
    float sigma,
    size_t dim_sdf,
    torch::Tensor vertices,
    torch::Tensor sdf,
    torch::Tensor feat,
    torch::Tensor edges,
    torch::Tensor sdf_smooth,
    torch::Tensor counter 
);

void bnn_smooth_sdf_cuda(
    size_t num_sites,
    float sigma,
    size_t dim_sdf,
    torch::Tensor vertices,
    torch::Tensor sdf,
    torch::Tensor bnn_sites,
    torch::Tensor bnn_offset,
    torch::Tensor sdf_smooth
);

void knn_smooth_sdf_cuda(
    size_t num_sites,
    size_t num_knn,
    float sigma,
    float sigma_feat,
    size_t dim_sdf,
    torch::Tensor vertices,
    torch::Tensor activated, 
    torch::Tensor grads,
    torch::Tensor sdf,
    torch::Tensor feat,
    torch::Tensor neighbors,
    torch::Tensor sdf_smooth
);

void knn_interpolate_cuda(
    size_t num_sites,
    size_t num_knn,
    float sigma,
    size_t dim_sdf,
    torch::Tensor vertices_src,
    torch::Tensor vertices_trg, 
    torch::Tensor sdf,
    torch::Tensor neighbors,
    torch::Tensor sdf_out
);

void geo_feat_cuda(
    size_t num_samples,
    size_t num_knn,
    float sigma,
    torch::Tensor samples,
    torch::Tensor vertices, 
    torch::Tensor grads,
    torch::Tensor sdf,
    torch::Tensor geo_feat,
    torch::Tensor tets,
    torch::Tensor neighbors,
    torch::Tensor cell_ids
);

void activate_sites_cuda(
    size_t num_rays,
    size_t num_sites,
    size_t num_knn,
    torch::Tensor cell_ids,
    torch::Tensor offsets,
    torch::Tensor neighbors,
    torch::Tensor activated_buff,
    torch::Tensor activated
);

void backprop_multi_cuda(
    size_t num_sites, 
    size_t num_knn,  
    size_t dim_feat,
    torch::Tensor vertices,     // [N_voxels, 4] for each voxel => it's vertices
    torch::Tensor activated,     // [N_voxels, 4] for each voxel => it's vertices
    torch::Tensor grad_norm,     // [N_voxels, 4] for each voxel => it's vertices
    torch::Tensor grad_norm_feat,     // [N_voxels, 4] for each voxel => it's vertices
    torch::Tensor grad_feat,     // [N_voxels, 4] for each voxel => it's vertices
    torch::Tensor neighbors
);

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


// ***************************
void backprop_feat(
    size_t num_samples,
    size_t num_sites,
    size_t dim_feats,
    torch::Tensor sdf,
    torch::Tensor grad_feat,
    torch::Tensor counter,
    torch::Tensor grad_samples,
    torch::Tensor cell_ids,
    torch::Tensor cell_weights 
) {
    //std::cout << "Backprop feature gradients" << std::endl; 
    backprop_feat_cuda(num_samples,
    num_sites,
    dim_feats,
    sdf,
    grad_feat,
    counter,
    grad_samples,
    cell_ids,
    cell_weights);
}


void backprop_sdf(
    size_t num_samples,
    torch::Tensor grad_sdf,
    torch::Tensor grad_sdf_samples,
    torch::Tensor cell_ids,
    torch::Tensor cell_weights 
) {
    //std::cout << "Backprop feature gradients" << std::endl; 
    backprop_sdf_cuda(num_samples,
    grad_sdf,
    grad_sdf_samples,
    cell_ids,
    cell_weights);
}

void backprop_norm(
    size_t num_tets,
    torch::Tensor tets,
    torch::Tensor sites,
    torch::Tensor vol,    
    torch::Tensor weights,  
    torch::Tensor weights_tot, 
    torch::Tensor grad_norm,
    torch::Tensor grad_sdf,
    torch::Tensor activated
){
    //std::cout << "Backprop feature gradients" << std::endl; 
    backprop_norm_cuda(num_tets,
    tets,
    sites,
    vol,    
    weights,  
    weights_tot, 
    grad_norm,
    grad_sdf,
    activated);
}

void  backprop_unit_norm(
    size_t num_tets,
    torch::Tensor tets,
    torch::Tensor sites,
    torch::Tensor norm_grad,
    torch::Tensor grad_unormed,
    torch::Tensor vol,    
    torch::Tensor weights,  
    torch::Tensor weights_tot, 
    torch::Tensor grad_norm ,
    torch::Tensor grad_sdf,
    torch::Tensor activated 
){
    //std::cout << "Backprop feature gradients" << std::endl; 
    backprop_unit_norm_cuda(num_tets,
    tets,
    sites,
    norm_grad,
    grad_unormed,
    vol,    
    weights,  
    weights_tot, 
    grad_norm,
    grad_sdf,
    activated);
}

float eikonal_loss(
    size_t num_sites,
    size_t num_knn,
    torch::Tensor neighbors,
    torch::Tensor Weights,
    torch::Tensor grad,
    torch::Tensor grad_eik 
){
    //std::cout << "Backprop feature gradients" << std::endl; 
    return eikonal_loss_cuda(num_sites,
    num_knn,
    neighbors,
    Weights,
    grad,
    grad_eik);
}

void smooth_sdf(
    size_t num_edges,
    float sigma,
    torch::Tensor vertices,
    torch::Tensor activated,
    torch::Tensor sdf,
    torch::Tensor feat,
    torch::Tensor edges,
    torch::Tensor sdf_grad,
    torch::Tensor feat_grad,
    torch::Tensor counter 
){
    //std::cout << "Backprop feature gradients" << std::endl; 
    smooth_cuda(num_edges,
    sigma,
    vertices,
    activated,
    sdf,
    feat,
    edges,
    sdf_grad, 
    feat_grad,
    counter);
}

void space_reg(
    size_t num_rays,
    torch::Tensor rays_d,
    torch::Tensor grad_space,
    torch::Tensor out_weights,
    torch::Tensor out_z,
    torch::Tensor out_sdf,
    torch::Tensor out_feat,
    torch::Tensor out_ids,
    torch::Tensor offset,
    torch::Tensor sdf_grad,
    torch::Tensor feat_grad 
){
    //std::cout << "Backprop feature gradients" << std::endl; 
    space_reg_cuda(num_rays,
    rays_d,
    grad_space,
    out_weights,
    out_z,
    out_sdf,
    out_feat,
    out_ids,
    offset,
    sdf_grad,
    feat_grad );
}

void smooth(
    size_t num_edges,
    size_t num_sites,
    float sigma,
    size_t dim_sdf,
    torch::Tensor vertices,
    torch::Tensor sdf,
    torch::Tensor feat,
    torch::Tensor edges,
    torch::Tensor sdf_smooth,
    torch::Tensor counter 
){
    //std::cout << "Backprop feature gradients" << std::endl; 
    smooth_sdf_cuda(
    num_edges,
    num_sites,
    sigma,
    dim_sdf,
    vertices,
    sdf,
    feat,
    edges,
    sdf_smooth,
    counter);
}

void bnn_smooth(
    size_t num_sites,
    float sigma,
    size_t dim_sdf,
    torch::Tensor vertices,
    torch::Tensor sdf,
    torch::Tensor bnn_sites,
    torch::Tensor bnn_offset,
    torch::Tensor sdf_smooth
) {
    bnn_smooth_sdf_cuda(
        num_sites,
        sigma,
        dim_sdf,
        vertices,
        sdf,
        bnn_sites,
        bnn_offset,
        sdf_smooth);
}


void knn_smooth(
    size_t num_sites,
    size_t num_knn,
    float sigma,
    float sigma_feat,
    size_t dim_sdf,
    torch::Tensor vertices,
    torch::Tensor activated, 
    torch::Tensor grads,
    torch::Tensor sdf,
    torch::Tensor feat,
    torch::Tensor neighbors,
    torch::Tensor sdf_smooth
) {
    knn_smooth_sdf_cuda(
        num_sites,
        num_knn,
        sigma,
        sigma_feat,
        dim_sdf,
        vertices,
        activated, 
        grads,
        sdf,
        feat,
        neighbors,
        sdf_smooth);
}

void knn_interpolate(
    size_t num_sites,
    size_t num_knn,
    float sigma,
    size_t dim_sdf,
    torch::Tensor vertices_src,
    torch::Tensor vertices_trg, 
    torch::Tensor sdf,
    torch::Tensor neighbors,
    torch::Tensor sdf_out
) {
    knn_interpolate_cuda(
        num_sites,
        num_knn,
        sigma,
        dim_sdf,
        vertices_src,
        vertices_trg, 
        sdf,
        neighbors,
        sdf_out);
}

void geo_feat(
    size_t num_samples,
    size_t num_knn,
    float sigma,
    torch::Tensor samples,
    torch::Tensor vertices, 
    torch::Tensor grads,
    torch::Tensor sdf,
    torch::Tensor geo_feat,
    torch::Tensor tets,
    torch::Tensor neighbors,
    torch::Tensor cell_ids
) {
    geo_feat_cuda(
        num_samples,
        num_knn,
        sigma,
        samples,
        vertices, 
        grads,
        sdf,
        geo_feat,
        tets,
        neighbors,
        cell_ids);
}


void activate_sites(
    size_t num_rays,
    size_t num_sites,
    size_t num_knn,
    torch::Tensor cell_ids,
    torch::Tensor offsets,
    torch::Tensor neighbors,
    torch::Tensor activated_buff,
    torch::Tensor activated
) {
    activate_sites_cuda(
        num_rays,
        num_sites,
        num_knn,
        cell_ids,
        offsets,
        neighbors,
        activated_buff,
        activated);
}

void backprop_multi(
    size_t num_sites, 
    size_t num_knn,  
    size_t dim_feat,
    torch::Tensor vertices,     // [N_voxels, 4] for each voxel => it's vertices
    torch::Tensor activated,     // [N_voxels, 4] for each voxel => it's vertices
    torch::Tensor grad_norm,     // [N_voxels, 4] for each voxel => it's vertices
    torch::Tensor grad_norm_feat,     // [N_voxels, 4] for each voxel => it's vertices
    torch::Tensor grad_feat,     // [N_voxels, 4] for each voxel => it's vertices
    torch::Tensor neighbors
) {
    backprop_multi_cuda(
    num_sites, 
    num_knn,  
    dim_feat,
    vertices,     // [N_voxels, 4] for each voxel => it's vertices
    activated,     // [N_voxels, 4] for each voxel => it's vertices
    grad_norm,     // [N_voxels, 4] for each voxel => it's vertices
    grad_norm_feat,     // [N_voxels, 4] for each voxel => it's vertices
    grad_feat,     // [N_voxels, 4] for each voxel => it's vertices
    neighbors
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("eikonal_loss", &eikonal_loss, "eikonal_loss (CPP)");
    m.def("smooth_sdf", &smooth_sdf, "smooth_sdf (CPP)");
    m.def("backprop_feat", &backprop_feat, "backprop_feat (CPP)");
    m.def("backprop_sdf", &backprop_sdf, "backprop_sdf (CPP)");
    m.def("backprop_norm", &backprop_norm, "backprop_norm (CPP)");
    m.def("backprop_unit_norm", &backprop_unit_norm, "backprop_unit_norm (CPP)");
    m.def("space_reg", &space_reg, "space_reg (CPP)");
    m.def("smooth", &smooth, "smooth (CPP)");
    m.def("bnn_smooth", &bnn_smooth, "bnn_smooth (CPP)");
    m.def("knn_smooth", &knn_smooth, "knn_smooth (CPP)");
    m.def("knn_interpolate", &knn_interpolate, "knn_interpolate (CPP)");
    m.def("geo_feat", &geo_feat, "geo_feat (CPP)");
    m.def("activate_sites", &activate_sites, "activate_sites (CPP)");
    m.def("backprop_multi", &backprop_multi, "backprop_multi (CPP)");
}