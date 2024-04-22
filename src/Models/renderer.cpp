#include <torch/extension.h>
#include <vector>

// *************************
void render_cuda(
    size_t num_rays,
    float inv_s,
    float mask_reg,
    torch::Tensor sdf_seg,
    torch::Tensor weights_seg,
    torch::Tensor color_samples,
    torch::Tensor true_color,
    torch::Tensor mask,
    torch::Tensor cell_ids,
    torch::Tensor offsets,
    torch::Tensor grad_space,
    torch::Tensor rays,
    torch::Tensor grads_sdf,
    torch::Tensor grads_color,
    torch::Tensor color_loss,
    torch::Tensor mask_loss); 

void normalize_grads_cuda(
    size_t num_sites,
    torch::Tensor grads_sdf,
    torch::Tensor counter);

void render_no_sdf_cuda(
    size_t num_rays,
    float inv_s,
    float mask_reg,
    torch::Tensor sdf_seg,
    torch::Tensor neighbors,
    torch::Tensor weights_seg,
    torch::Tensor color_samples,
    torch::Tensor true_color,
    torch::Tensor mask,
    torch::Tensor cell_ids,
    torch::Tensor offsets,
    torch::Tensor grads_sdf,
    torch::Tensor grads_color,
    torch::Tensor color_loss,
    torch::Tensor mask_loss
); 


void render_no_grad_cuda(
    size_t num_rays,
    float inv_s,
    torch::Tensor sdf_seg,
    torch::Tensor color_samples,
    torch::Tensor offsets,
    torch::Tensor color_out,
    torch::Tensor mask_out
); 

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// ***************************
void render(
    size_t num_rays,
    float inv_s,
    float mask_reg,
    torch::Tensor sdf_seg,
    torch::Tensor neighbors,
    torch::Tensor weights_seg,
    torch::Tensor color_samples,
    torch::Tensor true_color,
    torch::Tensor mask,
    torch::Tensor cell_ids,
    torch::Tensor offsets,
    torch::Tensor grad_space,
    torch::Tensor rays,
    torch::Tensor grads_sdf,
    torch::Tensor grads_color,
    torch::Tensor grads_sdf_net,
    torch::Tensor counter,
    torch::Tensor color_loss,
    torch::Tensor mask_loss        //***************
) {
    //std::cout << "Render image" << std::endl; 
    render_cuda(num_rays,
    inv_s,
    mask_reg,
    sdf_seg,
    weights_seg,
    color_samples,
    true_color,
    mask,
    cell_ids,
    offsets,
    grad_space,
    rays,
    grads_sdf,
    grads_color,
    color_loss,
    mask_loss  );
}

void normalize_grads(
    size_t num_sites,
    torch::Tensor grads_sdf,
    torch::Tensor counter
) {
    
    normalize_grads_cuda(num_sites, grads_sdf, counter);
}

void render_no_sdf(
    size_t num_rays,
    float inv_s,
    float mask_reg,
    torch::Tensor sdf_seg,
    torch::Tensor neighbors,
    torch::Tensor weights_seg,
    torch::Tensor color_samples,
    torch::Tensor true_color,
    torch::Tensor mask,
    torch::Tensor cell_ids,
    torch::Tensor offsets,
    torch::Tensor grads_sdf,
    torch::Tensor grads_color,
    torch::Tensor color_loss,
    torch::Tensor mask_loss        //***************
) {
    //std::cout << "Render image" << std::endl; 
    render_no_sdf_cuda(num_rays,
    inv_s,
    mask_reg,
    sdf_seg,
    neighbors,
    weights_seg,
    color_samples,
    true_color,
    mask,
    cell_ids,
    offsets,
    grads_sdf,
    grads_color,
    color_loss,
    mask_loss  );
}


void render_no_grad(
    size_t num_rays,
    float inv_s,
    torch::Tensor sdf_seg,
    torch::Tensor color_samples,
    torch::Tensor offsets,
    torch::Tensor color_out,
    torch::Tensor mask_out     //***************
) {
    //std::cout << "Render image" << std::endl; 
    render_no_grad_cuda(num_rays,
    inv_s,
    sdf_seg,
    color_samples,
    offsets,
    color_out,
    mask_out );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("render", &render, "render (CPP)");
    m.def("render_no_sdf", &render_no_sdf, "render_no_sdf (CPP)");
    m.def("render_no_grad", &render_no_grad, "render_no_grad (CPP)");
    m.def("normalize_grads", &normalize_grads, "normalize_grads (CPP)");
}