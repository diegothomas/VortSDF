#include <torch/extension.h>
#include <vector>


// Definition of cuda functions
int tet32_march_cuda(
	float inv_s,
    size_t num_rays,
    size_t num_samples,
    size_t cam_id,
    torch::Tensor rays,      // [N_rays, 6]
    torch::Tensor vertices, // [N_voxels, 26] for each voxel => it's neighbors
    torch::Tensor sdf, // [N_voxels, 26] for each voxel => it's neighbors
    torch::Tensor tets, // [N_voxels, 26] for each voxel => it's neighbors
    torch::Tensor nei_tets, // [N_voxels, 26] for each voxel => it's neighbors
    torch::Tensor cam_ids,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor offsets_cam,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor cam_tets,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor weights,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor z_vals,     // [N_voxels, 4] for each voxel => it's vertices
    torch::Tensor z_sdfs,     // [N_voxels, 4] for each voxel => it's vertices
    torch::Tensor z_ids,     // [N_voxels, 4] for each voxel => it's vertices
    torch::Tensor activate,     // [N_voxels, 4] for each voxel => it's vertices
    torch::Tensor offset     // [N_voxels, 4] for each voxel => it's vertices
);

// Ray marching in 2Deorganize samples in contiguous array
void fill_samples_cuda(
    size_t num_rays,
    size_t num_samples,
    torch::Tensor rays_o,       // [N_rays, 6]
    torch::Tensor rays_d,       // [N_rays, 6]
    torch::Tensor sites,       // [N_rays, 6]
    torch::Tensor in_z,       // [N_rays, 6]
    torch::Tensor in_sdf,       // [N_rays, 6]
    torch::Tensor in_feat,       // [N_rays, 6]
    torch::Tensor in_weights,       // [N_rays, 6]
    torch::Tensor in_grads,       // [N_rays, 6]
    torch::Tensor in_ids,       // [N_rays, 6]
    torch::Tensor out_z,     // [N_voxels, 4] for each voxel => it's vertices
    torch::Tensor out_sdf,     // [N_voxels, 4] for each voxel => it's vertices
    torch::Tensor out_feat,     // [N_voxels, 4] for each voxel => it's vertices
    torch::Tensor out_weights,     // [N_voxels, 4] for each voxel => it's vertices
    torch::Tensor out_grads,       // [N_rays, 6]
    torch::Tensor out_ids,     // [N_voxels, 4] for each voxel => it's vertices
    torch::Tensor offset,     // [N_voxels, 4] for each voxel => it's vertices
    torch::Tensor samples,     // [N_voxels, 4] for each voxel => it's vertices
    torch::Tensor sample_rays     // [N_voxels, 4] for each voxel => it's vertices
); 

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


int tet32_march(
	float inv_s,
    size_t num_rays,
    size_t num_samples,
    size_t cam_id,
    torch::Tensor rays,      // [N_rays, 6]
    torch::Tensor vertices, // [N_voxels, 26] for each voxel => it's neighbors
    torch::Tensor sdf, // [N_voxels, 26] for each voxel => it's neighbors
    torch::Tensor tets, // [N_voxels, 26] for each voxel => it's neighbors
    torch::Tensor nei_tets, // [N_voxels, 26] for each voxel => it's neighbors
    torch::Tensor cam_ids,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor offsets_cam,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor cam_tets,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor weights,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor z_vals,     // [N_voxels, 4] for each voxel => it's vertices
    torch::Tensor z_sdfs,     // [N_voxels, 4] for each voxel => it's vertices
    torch::Tensor z_ids,     // [N_voxels, 4] for each voxel => it's vertices
    torch::Tensor activate,     // [N_voxels, 4] for each voxel => it's vertices
    torch::Tensor offset     // [N_voxels, 4] for each voxel => it's vertices
) {
    //std::cout << "March through Tet32" << std::endl; 

    return tet32_march_cuda(
	    inv_s,
        num_rays,
        num_samples,
        cam_id,
        rays,
        vertices,
        sdf,
        tets,
        nei_tets, 
        cam_ids,
        offsets_cam,
        cam_tets,
        weights,    // [N_sites, 3] for each voxel => it's vertices
        z_vals,     // [N_voxels, 4] for each voxel => it's vertices
        z_sdfs,     // [N_voxels, 4] for each voxel => it's vertices
        z_ids,     // [N_voxels, 4] for each voxel => it's vertices
        activate,     // [N_voxels, 4] for each voxel => it's vertices
        offset     // [N_voxels, 4] for each voxel => it's vertices
        );

}

// Ray marching in 2Deorganize samples in contiguous array
void fill_samples(
    size_t num_rays,
    size_t num_samples,
    torch::Tensor rays_o,       // [N_rays, 6]
    torch::Tensor rays_d,       // [N_rays, 6]
    torch::Tensor sites,       // [N_rays, 6]
    torch::Tensor in_z,       // [N_rays, 6]
    torch::Tensor in_sdf,       // [N_rays, 6]
    torch::Tensor in_feat,       // [N_rays, 6]
    torch::Tensor in_weights,       // [N_rays, 6]
    torch::Tensor in_grads,       // [N_rays, 6]
    torch::Tensor in_ids,       // [N_rays, 6]
    torch::Tensor out_z,     // [N_voxels, 4] for each voxel => it's vertices
    torch::Tensor out_sdf,     // [N_voxels, 4] for each voxel => it's vertices
    torch::Tensor out_feat,     // [N_voxels, 4] for each voxel => it's vertices
    torch::Tensor out_weights,     // [N_voxels, 4] for each voxel => it's vertices
    torch::Tensor out_grads,     // [N_voxels, 4] for each voxel => it's vertices
    torch::Tensor out_ids,     // [N_voxels, 4] for each voxel => it's vertices
    torch::Tensor offset,     // [N_voxels, 4] for each voxel => it's vertices
    torch::Tensor samples,     // [N_voxels, 4] for each voxel => it's vertices
    torch::Tensor sample_rays     // [N_voxels, 4] for each voxel => it's vertices
) {
    //std::cout << "Fill samples" << std::endl; 

    fill_samples_cuda(
        num_rays,
        num_samples,
        rays_o,      
        rays_d, 
        sites, 
        in_z,       
        in_sdf,  
        in_feat,  
        in_weights,    
        in_grads,    
        in_ids,       
        out_z,     
        out_sdf,    
        out_feat, 
        out_weights,     
        out_grads,     
        out_ids,     
        offset,     
        samples,    
        sample_rays);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tet32_march", &tet32_march, "tet32_march (CPP)");
    m.def("fill_samples", &fill_samples, "fill_samples (CPP)");
}