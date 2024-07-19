#include <torch/extension.h>

#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include <device_launch_parameters.h>
#include "cudaType.cuh"

#ifdef __INTELLISENSE__
#define CUDA_KERNEL(...)
#define FAKEINIT = {0}
#else
#define CUDA_KERNEL(...) <<< __VA_ARGS__ >>>
#define FAKEINIT
#endif

#define STOP_TRANS 1.0e-12
#define CLIP_ALPHA 60.0
#define BACK_R 0.5f
#define BACK_G 0.5f
#define BACK_B 0.5f 
#define PI 3.141592653589793238462643383279502884197


/** Device functions **/
/** Device functions **/
/** Device functions **/
__device__ const float HUBER_EPS = 1.5f / 255.0f;

__device__ float huber_loss(float3 x) 
{
    //float3 absx = abs(x);
    float3 r = x * x * 0.5f; //mix(x * x * 0.5f, (absx - 0.5f * HUBER_EPS) * HUBER_EPS, greaterThan(abs(x), make_float3(HUBER_EPS, HUBER_EPS, HUBER_EPS)));
    return r.x + r.y + r.z;
}

__device__ float3 huber_grad(float3 x) 
{
    return x; //mix(x, sign(x) * HUBER_EPS, greaterThan(abs(x), make_float3(HUBER_EPS, HUBER_EPS, HUBER_EPS)));
}

__device__ float mid_loss(float3 x, float3 y) 
{
    float3 ra = x*x;
    float3 rb = y*y;
    float3 r = (x-y)*(x-y);
    return 0.5f * (r.x + r.y + r.z);///(min(sqrt(ra.x + ra.y + ra.z), sqrt(rb.x + rb.y + rb.z)) + 0.005f);
}

__device__ float3 mid_grad(float3 x, float3 y) 
{
    /*float3 ra = x*x;
    float3 rb = y*y;
    return (x-y)/(min(sqrt(ra.x + ra.y + ra.z), sqrt(rb.x + rb.y + rb.z)) + 0.005f);*/
    float3 r = x-y;
    /*r.x = r.x/(fabs(y.x) + 0.1f);
    r.y = r.y/(fabs(y.y) + 0.1f);
    r.z = r.z/(fabs(y.z) + 0.1f);
    if (y.x == 1.0f || y.y == 1.0f || y.z == 1.0f)
        return 0.1*r;*/
    return r;
    //return (x-y);
}


__device__ float sdf2Alpha(float sdf, float sdf_prev, float inv_s) 
{
    if (sdf_prev > sdf) {
        double sdf_prev_clamp = fmin(CLIP_ALPHA, fmax(double(sdf_prev * inv_s), -CLIP_ALPHA));
        double sdf_clamp = fmin(CLIP_ALPHA, fmax(double(sdf * inv_s), -CLIP_ALPHA));
        return min(1.0f, __double2float_rn((1.0 + exp(-sdf_prev_clamp)) / (1.0 + exp(-sdf_clamp))));
    }
    return 1.0f;
}

__device__ float4 trace_ray(const float4* sdf_seg, const float3* color_samples, const int* offsets, float *Wpartial, float inv_s, int n) 
{
    float Tpartial = 1.0f;
    float3 Cpartial = make_float3(0.0f, 0.0f, 0.0f);
    float3 color;
    float alpha = 0.0f;

    float4 sdf;

    int start = offsets[2 * n];
    int end = offsets[2 * n + 1];
    for (int t = start; t < start + end; t++) {
        sdf = sdf_seg[t];
        color = color_samples[t];

        alpha = sdf2Alpha(sdf.y, sdf.x, inv_s);

        Cpartial = Cpartial + color * (1.0f - alpha) * Tpartial;
        *Wpartial = (*Wpartial) + (1.0f - alpha) * Tpartial;
        Tpartial *= alpha;

        if (Tpartial < STOP_TRANS) {// stop if the transmittance is low
            break;
        }

    }

    // return the total color as well as the final transmittance. The background color will be added after.
    return make_float4(Cpartial, Tpartial);
}

__device__ float4 trace_ray_no(const float* sdf_seg, const float* color_samples, const int* offsets, float *Wpartial, float inv_s, int n) 
{
    float Tpartial = 1.0f;
    float3 Cpartial = make_float3(0.0f, 0.0f, 0.0f);
    float3 color;
    float alpha = 0.0f;

    int start = offsets[2 * n];
    int end = offsets[2 * n + 1];
    for (int t = start; t < start + end; t++) {
        color.x = color_samples[3*t];
        color.y = color_samples[3*t+1];
        color.z = color_samples[3*t+2];

        alpha = sdf2Alpha(sdf_seg[4*t + 1], sdf_seg[4*t], inv_s);

        Cpartial = Cpartial + color * (1.0f - alpha) * Tpartial;
        *Wpartial = (*Wpartial) + (1.0f - alpha) * Tpartial;
        Tpartial *= alpha;

        if (Tpartial < STOP_TRANS) {// stop if the transmittance is low
            break;
        }

    }

    // return the total color as well as the final transmittance. The background color will be added after.
    return make_float4(Cpartial, Tpartial);
}

__device__ void backward_no_sdf(float3 Ctotal, float Wtotal, float3 TrueColor, float3 grad_color_diff, float Mask, int n,
                        float* grads_color, float* grads_sdf, const float* sdf_seg, const int *neighbors, const float* weights_seg, const float* color_samples, const int* offsets, const int* cell_ids,
                        float inv_s, float MaskReg = 0.0f, float colorDiscrepancyReg = 0.0f, float BackgroundEntropyReg = 0.0f, int NoColorSpilling = 0) 
{

    float Tpartial = 1.0f;
    float Wpartial = 0.0f;
    float3 Cpartial = make_float3(0.0f, 0.0f, 0.0f);
    float3 color = make_float3(0.0f, 0.0f, 0.0f);
    float3 dc = make_float3(0.0f, 0.0f, 0.0f);
    float3 dCtotal_dalpha = make_float3(0.0f, 0.0f, 0.0f);
    float3 sample_color_diff = make_float3(0.0f, 0.0f, 0.0f);
    float3 A = make_float3(0.0f, 0.0f, 0.0f);

    float sdf_prev = 20.0f; // some large value
    float sdf = 20.0f; // some large value

    int num_knn = 8;

    float alpha, dalpha_dsdf_p, dalpha_dsdf_n, dalpha, contrib;
    double sdf_prev_clamp, sdf_clamp, inv_clipped_p, inv_clipped;
    int id, id_prev;

    int start = offsets[2 * n];
    int end = offsets[2 * n + 1];
    for (int t = start; t < start + end; t++) {        
        
        sdf_prev = sdf_seg[2*t];
        sdf = sdf_seg[2*t+1];

        color.x = color_samples[3 * t];
        color.y = color_samples[3 * t + 1];
        color.z = color_samples[3 * t + 2];

        alpha = 1.0f;
        sdf_prev_clamp = fmin(CLIP_ALPHA, fmax(double(sdf_prev * inv_s), -CLIP_ALPHA));
        sdf_clamp = fmin(CLIP_ALPHA, fmax(double(sdf * inv_s), -CLIP_ALPHA));
        inv_clipped_p = (fabs(sdf_prev) < CLIP_ALPHA / inv_s) ? double(inv_s) : sdf_prev_clamp / double(sdf_prev);
        inv_clipped = (fabs(sdf) < CLIP_ALPHA / inv_s) ? double(inv_s) : sdf_clamp / double(sdf);

        alpha = min(1.0f, __double2float_rn((1.0 + exp(-sdf_prev_clamp)) / (1.0 + exp(-sdf_clamp))));
        if (sdf_prev > sdf && alpha < 1.0f) { // && sdf_prev > 0.0f        
        //if (sdf_prev*sdf <= 0.0f || 
        //        (sdf_prev > sdf && (fabs(sdf)*inv_s < CLIP_ALPHA || fabs(sdf_prev)*inv_s < CLIP_ALPHA))){
            //alpha = min(1.0f, __double2float_rn((1.0 + exp(-sdf_prev_clamp)) / (1.0 + exp(-sdf_clamp))));
            dalpha_dsdf_p = (fabs(sdf_prev * inv_s) > CLIP_ALPHA) ? 0.0f : __double2float_rn(-inv_clipped_p * exp(-sdf_prev_clamp) / (1.0 + exp(-sdf_clamp)));
            //dalpha_dsdf_p = __double2float_rn(-inv_clipped_p * exp(-sdf_prev_clamp) / (1.0 + exp(-sdf_clamp)));
            dalpha_dsdf_n = (fabs(sdf * inv_s) > CLIP_ALPHA) ? 0.0f : __double2float_rn((1.0 + exp(-sdf_prev_clamp)) * ((inv_clipped * exp(-sdf_clamp)) / ((1.0 + exp(-sdf_clamp)) * (1.0 + exp(-sdf_clamp)))));
            //dalpha_dsdf_n = __double2float_rn((1.0 + exp(-sdf_prev_clamp)) * ((inv_clipped * exp(-sdf_clamp)) / ((1.0 + exp(-sdf_clamp)) * (1.0 + exp(-sdf_clamp)))));
        }
        else {
            alpha = 1.0f;
            dalpha_dsdf_p = 0.0f;
            dalpha_dsdf_n = 0.0f;
            continue;
        }
        
        contrib = Tpartial * (1.0f - alpha);
        Cpartial = Cpartial + color * contrib;
        Wpartial = Wpartial + contrib;
        dCtotal_dalpha = ((Ctotal - Cpartial) / alpha) - (color * Tpartial);  // equation 13 from the supplemental
        
        
        ///////////////////////////////////////////////////////// Photometric loss
        dalpha = dot(grad_color_diff, dCtotal_dalpha);
        dc = grad_color_diff * contrib;

        sample_color_diff = Ctotal - color;
        A = huber_grad(sample_color_diff) * contrib;
        
        ///////////////////////////////////////////////////////// Color discrepancy loss
        dalpha += colorDiscrepancyReg * (-huber_loss(sample_color_diff) * Tpartial + dot(A, dCtotal_dalpha));
        dc = dc + A * (contrib - 1.0f) * colorDiscrepancyReg;

        ///////////////////////////////////////////////////////// Background entropy loss
        dalpha += alpha == 0.0f? 0.0f: (1.0f - 2.0f * Wtotal) * (Wtotal / alpha) * BackgroundEntropyReg;

        ///////////////////////////////////////////////////////// Mask regularization
        dalpha += 2.0f * (Wtotal - Mask) * ((Wtotal - Wpartial) / alpha - Tpartial) * MaskReg;

        if (NoColorSpilling != 0) {
            dc = dc * Wtotal * Wtotal;
        }

        float lamda = weights_seg[13*t + 12];
        for (int i = 0; i < 6; i++) {
            id_prev = cell_ids[12 * t + i];
            id = cell_ids[12 * t + 6 + i];

            //atomicAdd(&grads_sdf[id_prev], weights_seg[6*t + i] * dalpha * dalpha_dsdf_p);
            //atomicAdd(&grads_sdf[id], weights_seg[6*t + 3 + i] * dalpha * dalpha_dsdf_n);
            atomicAdd(&grads_color[3 * id_prev], weights_seg[13*t + i] * lamda * dc.x);
            atomicAdd(&grads_color[3 * id_prev+1], weights_seg[13*t + i] * lamda * dc.y);
            atomicAdd(&grads_color[3 * id_prev+2], weights_seg[13*t + i] * lamda * dc.z);
            
            atomicAdd(&grads_color[3 * id], weights_seg[13*t + 6 + i] * (1.0f-lamda) * dc.x);
            atomicAdd(&grads_color[3 * id+1], weights_seg[13*t + 6 + i] * (1.0f-lamda) * dc.y);
            atomicAdd(&grads_color[3 * id+2], weights_seg[13*t + 6 + i] * (1.0f-lamda) * dc.z);
        }

        Tpartial = Tpartial * alpha;

        if (Tpartial < STOP_TRANS) { // stop if the transmittance is low
            break;
        }
    }

    return;
}

__global__ void render_no_sdf_kernel(
    const size_t num_rays,
    const float inv_s,
    const float mask_reg,
    const float *__restrict__ sdf_seg,
    const int *__restrict__ neighbors, 
    const float *__restrict__ weights_seg,
    const float *__restrict__ color_samples,
    const float *__restrict__ true_color,
    const float *__restrict__ mask,
    const int *__restrict__ cell_ids,
    const int *__restrict__ offsets,
    float *__restrict__ grads_sdf,
    float *__restrict__ grads_color,
    float *__restrict__ color_loss,
    float *__restrict__ mask_loss)
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rays)
    {
        return;
    }
    
    float Wtotal = 0.0f;
    float4 color;// = trace_ray(sdf_seg, color_samples, offsets, &Wtotal, inv_s, idx);

    //if (color.w < 1.0f) {
        float msk = mask[idx] > 0.5f ? 1.0f : 0.0f;
        float3 integrated_color = make_float3(color.x + color.w * BACK_R, color.y + color.w * BACK_G, color.z + color.w * BACK_B);
        float3 in_color = make_float3(true_color[3 * idx], true_color[3 * idx + 1], true_color[3 * idx + 2]);
        float3 grad_color_diff = huber_grad(integrated_color - in_color);

        backward_no_sdf(integrated_color, Wtotal, in_color, grad_color_diff, msk,
            idx, grads_color, grads_sdf, sdf_seg, neighbors, weights_seg, color_samples,
            offsets, cell_ids, inv_s, mask_reg, 0.0f, 0.0f, 1);

        //color_loss[3*idx] = integrated_color.x;
        //color_loss[3*idx + 1] = integrated_color.y;
        //color_loss[3*idx + 2] = integrated_color.z;
        color_loss[idx] = msk*huber_loss(integrated_color - in_color);      
        //color_loss[idx] = msk*(fabs(grad_color_diff.x) + fabs(grad_color_diff.y) + fabs(grad_color_diff.z));    
        mask_loss[idx] = (msk - Wtotal)*(msk - Wtotal); //-(msk * logf(Wtotal) + (1.0f - msk) * logf(1.0f-Wtotal));
    //}
    return;
}


__device__ void backward(float3 Ctotal, float Wtotal, float3 TrueColor, float3 grad_color_diff, float Mask, int n,
                        float3* grads_color, float* grads_sdf, const float4* sdf_seg, 
                        const float * weights_seg, const float3* color_samples, const int* offsets, const int3* cell_ids,
                        const float3 * grad_space, const float3 *rays,
                        float inv_s, float MaskReg = 0.0f, float colorDiscrepancyReg = 0.0f, float BackgroundEntropyReg = 0.0f, int NoColorSpilling = 0) 
{

    float Tpartial = 1.0f;
    float Wpartial = 0.0f;
    float3 Cpartial = make_float3(0.0f, 0.0f, 0.0f);
    float3 color = make_float3(0.0f, 0.0f, 0.0f);
    float3 color_prev = make_float3(0.0f, 0.0f, 0.0f);
    float3 dc = make_float3(0.0f, 0.0f, 0.0f);
    float3 dCtotal_dalpha = make_float3(0.0f, 0.0f, 0.0f);
    float3 sample_color_diff = make_float3(0.0f, 0.0f, 0.0f);
    float3 A = make_float3(0.0f, 0.0f, 0.0f);

    float4 sdf; 

    int num_knn = 8;

    float alpha, dalpha_dsdf_p, dalpha_dsdf_n, dalpha, contrib;
    double sdf_prev_clamp, sdf_clamp, inv_clipped_p, inv_clipped;
    int3 id, id_prev;

    float w_color = 1.0f;//  exp(-norm_2(Ctotal - TrueColor)/0.01f);

    int start = offsets[2 * n];
    int end = offsets[2 * n + 1];
    for (int t = start; t < start + end; t++) {        
        sdf = sdf_seg[t];
        color = color_samples[t];

        alpha = 1.0f;
        sdf_prev_clamp = fmin(CLIP_ALPHA, fmax(double(sdf.x * inv_s), -CLIP_ALPHA));
        sdf_clamp = fmin(CLIP_ALPHA, fmax(double(sdf.y * inv_s), -CLIP_ALPHA));
        inv_clipped_p = (fabs(sdf.x) < CLIP_ALPHA / inv_s) ? double(inv_s) : sdf_prev_clamp / double(sdf.x);
        inv_clipped = (fabs(sdf.y) < CLIP_ALPHA / inv_s) ? double(inv_s) : sdf_clamp / double(sdf.y);

        alpha = min(1.0f, __double2float_rn((1.0 + exp(-sdf_prev_clamp)) / (1.0 + exp(-sdf_clamp))));
        if (sdf.x > sdf.y) {
            dalpha_dsdf_p = __double2float_rn(-inv_clipped_p * exp(-sdf_prev_clamp) / (1.0 + exp(-sdf_clamp)));
            dalpha_dsdf_n = __double2float_rn((1.0 + exp(-sdf_prev_clamp)) * ((inv_clipped * exp(-sdf_clamp)) / ((1.0 + exp(-sdf_clamp)) * (1.0 + exp(-sdf_clamp)))));
        }
        else {
            alpha = 1.0f;
            dalpha_dsdf_p = 0.0f;
            dalpha_dsdf_n = 0.0f;
            continue;
        }
        
        contrib = Tpartial * (1.0f - alpha);
        Cpartial = Cpartial + color * contrib;
        Wpartial = Wpartial + contrib;
        dCtotal_dalpha = ((Ctotal - Cpartial) / alpha) - (color * Tpartial);  // equation 13 from the supplemental
        
        
        ///////////////////////////////////////////////////////// Photometric loss
        dalpha = Mask == 0.0f ? 0.0f : dot(grad_color_diff, dCtotal_dalpha);
        //dalpha = dot(grad_color_diff, dCtotal_dalpha);
        dc = grad_color_diff * contrib;

        sample_color_diff = Ctotal - color;
        A = huber_grad(sample_color_diff) * contrib;
        
        ///////////////////////////////////////////////////////// Color discrepancy loss
        //dalpha += colorDiscrepancyReg * (-huber_loss(sample_color_diff) * Tpartial + dot(A, dCtotal_dalpha));
        //dc = dc + A * (contrib - 1.0f) * colorDiscrepancyReg;

        ///////////////////////////////////////////////////////// Background entropy loss
        //dalpha += alpha == 0.0f? 0.0f: (1.0f - 2.0f * Wtotal) * (Wtotal / alpha) * BackgroundEntropyReg;

        ///////////////////////////////////////////////////////// Mask regularization
        /*if (1.0f - Wtotal > 0.5f) {
            //dalpha += 2.0f * (1.0f - Mask) * ((Wtotal - Wpartial) / alpha - Tpartial) * MaskReg;
            dalpha += -2.0f * (1.0f - Mask) * (Wtotal / alpha) * MaskReg;
        } else {
            //dalpha += 2.0f * (Wtotal - Mask) * ((Wtotal - Wpartial) / alpha - Tpartial) * MaskReg;
            dalpha += -2.0f * (1.0f - Wtotal - Mask) * (Wtotal / alpha) * MaskReg;
        }*/
        dalpha += -2.0f * (1.0f - Wtotal - Mask) * (Wtotal / alpha) * MaskReg;

        if (NoColorSpilling != 0) {
            dc = dc * (1.0f - Wtotal) * (1.0f - Wtotal);
        }

        //float w_photo = fabs(dot(grad_space[t], rays[n])); //(grad_space[12 * t]*rays[3*n] + grad_space[12 * t + 1]*rays[3*n+1] + grad_space[12 * t + 2]*rays[3*n+2]);
        //dalpha = dalpha*w_photo;

        //if (MaskReg < 0.01f)
        //    dalpha = dalpha*w_color;

        //////////////////////////////////////////////////////////////
        float lambda1 = sdf.z;
        float lambda2 = sdf.w;
        float fact1 = lambda1 == 1.0? 1.0f : 1.0f;///3.0f;
        id_prev = cell_ids[2 * t];
        id = cell_ids[2 * t + 1];
        atomicAdd(&grads_sdf[id_prev.x], weights_seg[7 * t] * fact1 * (lambda1 * dalpha * dalpha_dsdf_p + (1.0-lambda1)*dalpha * dalpha_dsdf_n));
        atomicAdd(&grads_sdf[id.x],  weights_seg[7 * t + 3] * fact1 * (lambda2 * dalpha * dalpha_dsdf_p + (1.0-lambda2)*dalpha * dalpha_dsdf_n));
        
        atomicAdd(&grads_sdf[id_prev.y], weights_seg[7 * t + 1] * fact1 * (lambda1 * dalpha * dalpha_dsdf_p + (1.0-lambda1)*dalpha * dalpha_dsdf_n));
        atomicAdd(&grads_sdf[id.y],  weights_seg[7 * t + 4]  * fact1 * (lambda2 * dalpha * dalpha_dsdf_p + (1.0-lambda2)*dalpha * dalpha_dsdf_n));
        
        atomicAdd(&grads_sdf[id_prev.z], weights_seg[7 * t + 2] * fact1 * (lambda1 * dalpha * dalpha_dsdf_p + (1.0-lambda1)*dalpha * dalpha_dsdf_n));
        atomicAdd(&grads_sdf[id.z],  weights_seg[7 * t + 5]* fact1 * (lambda2 * dalpha * dalpha_dsdf_p + (1.0-lambda2)*dalpha * dalpha_dsdf_n));
        /*if (lambda < 0.5f) {
            atomicAdd(&grads_sdf[id_prev.x], weights_seg_prev.x * 2.0f * lambda * dalpha * dalpha_dsdf_p);
            atomicAdd(&grads_sdf[id.x],  weights_seg_next.x * ((1.0f-2.0f*lambda) * dalpha * dalpha_dsdf_p + dalpha * dalpha_dsdf_n));
            
            atomicAdd(&grads_sdf[id_prev.y], weights_seg_prev.y * 2.0f * lambda * dalpha * dalpha_dsdf_p);
            atomicAdd(&grads_sdf[id.y],  weights_seg_next.y * ((1.0f-2.0f*lambda) * dalpha * dalpha_dsdf_p + dalpha * dalpha_dsdf_n));
            
            atomicAdd(&grads_sdf[id_prev.z], weights_seg_prev.z * 2.0f * lambda * dalpha * dalpha_dsdf_p);
            atomicAdd(&grads_sdf[id.z],  weights_seg_next.z * ((1.0f-2.0f*lambda) * dalpha * dalpha_dsdf_p + dalpha * dalpha_dsdf_n));
        } else {
            atomicAdd(&grads_sdf[id_prev.x], weights_seg_prev.x * (2.0f*lambda * dalpha * dalpha_dsdf_p + (1.0-2.0f*lambda)*dalpha * dalpha_dsdf_n));
            atomicAdd(&grads_sdf[id.x],  weights_seg_next.x * (1.0f-(1.0-2.0f*lambda)) * dalpha * dalpha_dsdf_n);
            
            atomicAdd(&grads_sdf[id_prev.y], weights_seg_prev.y * (2.0f*lambda * dalpha * dalpha_dsdf_p + (1.0-2.0f*lambda)*dalpha * dalpha_dsdf_n));
            atomicAdd(&grads_sdf[id.y],  weights_seg_next.y * (1.0f-(1.0-2.0f*lambda)) * dalpha * dalpha_dsdf_n);
            
            atomicAdd(&grads_sdf[id_prev.z], weights_seg_prev.z * (2.0f*lambda * dalpha * dalpha_dsdf_p + (1.0-2.0f*lambda)*dalpha * dalpha_dsdf_n));
            atomicAdd(&grads_sdf[id.z],  weights_seg_next.z * (1.0f-(1.0-2.0f*lambda)) * dalpha * dalpha_dsdf_n);
        }*/

        //if (Mask > 0.0f) {
            grads_color[t] = dc;
        //}  

        Tpartial = Tpartial * alpha;

        if (Tpartial < STOP_TRANS) { // stop if the transmittance is low
            break;
        }
    }

    return;
}

__global__ void render_kernel(
    const size_t num_rays,
    const float inv_s,
    const float mask_reg,
    const float4 *__restrict__ sdf_seg,
    const float *__restrict__ weights_seg,
    const float3 *__restrict__ color_samples,
    const float3 *__restrict__ true_color,
    const float *__restrict__ mask,
    const int3 *__restrict__ cell_ids,
    const int *__restrict__ offsets, 
    const float3 *__restrict__ grad_space, 
    const float3 *__restrict__ rays,
    float *__restrict__ grads_sdf,
    float3 *__restrict__ grads_color,
    float *__restrict__ color_loss,
    float *__restrict__ mask_loss)
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rays)
    {
        return;
    }
    
    float Wtotal = 0.0f;
    float4 color = trace_ray(sdf_seg, color_samples, offsets, &Wtotal, inv_s, idx);

    //if (color.w < 1.0f) {
        float msk = mask[idx] > 0.5f ? 1.0f : 0.0f;
        float3 integrated_color = make_float3(color.x + color.w * BACK_R, color.y + color.w * BACK_G, color.z + color.w * BACK_B);
        //float3 grad_color_diff = huber_grad(integrated_color - true_color[idx]);
        float3 grad_color_diff = mid_grad(integrated_color, true_color[idx]);

        //Wtotal
        float mask_weight = msk > 0.5f ? mask_reg : max(mask_reg, 0.01f);
        backward(integrated_color, color.w, true_color[idx], grad_color_diff, msk,
            idx, grads_color, grads_sdf, sdf_seg, weights_seg, color_samples,
            offsets, cell_ids, grad_space, rays, inv_s, mask_weight, 0.0f, 0.0f, 1);

        //color_loss[3*idx] = integrated_color.x;
        //color_loss[3*idx + 1] = integrated_color.y;
        //color_loss[3*idx + 2] = integrated_color.z;
        //color_loss[idx] = msk*huber_loss(integrated_color - true_color[idx]);      
        color_loss[idx] = msk*mid_loss(integrated_color, true_color[idx]);  
        //color_loss[idx] = msk*(fabs(grad_color_diff.x) + fabs(grad_color_diff.y) + fabs(grad_color_diff.z));    
        mask_loss[idx] = (msk - Wtotal)*(msk - Wtotal); //-(msk * logf(Wtotal) + (1.0f - msk) * logf(1.0f-Wtotal));
    //}
    return;
}


__global__ void normalize_grads_kernel(
    const size_t num_sites,
    float *__restrict__ grads_sdf,
    const float *__restrict__ counter)
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_sites)
    {
        return;
    }
    
    grads_sdf[idx] = counter[idx] == 0.0f ? 0.0f : grads_sdf[idx] / counter[idx];

    return;
}



__global__ void render_no_grad_kernel(
    const size_t num_rays,
    const float inv_s,
    const float *__restrict__ sdf_seg,
    const float *__restrict__ color_samples,
    const int *__restrict__ offsets,
    float *__restrict__ color_out,
    float *__restrict__ mask_out)
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rays)
    {
        return;
    }
    
    float Wtotal = 0.0f;
    float4 color = trace_ray_no(sdf_seg, color_samples, offsets, &Wtotal, inv_s, idx);

    color_out[3*idx] = color.x + color.w * 0.0f;
    color_out[3*idx + 1] = color.y + color.w * 0.0f;
    color_out[3*idx + 2] = color.z + color.w * 0.0f;

    mask_out[idx] = color.w;

    return;
}


/** CPU functions **/
/** CPU functions **/
/** CPU functions **/


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
    torch::Tensor mask_loss)
{
        const int threads = 512;
        const int blocks = (num_rays + threads - 1) / threads; // ceil for example 8192 + 255 / 256 = 32
        AT_DISPATCH_FLOATING_TYPES( color_samples.type(),"render_cuda", ([&] {  
            render_kernel CUDA_KERNEL(blocks,threads) (
                num_rays,
                inv_s,
                mask_reg,
                (float4*)thrust::raw_pointer_cast(sdf_seg.data_ptr<float>()),
                weights_seg.data_ptr<float>(),
                (float3*)thrust::raw_pointer_cast(color_samples.data_ptr<float>()),
                (float3*)thrust::raw_pointer_cast(true_color.data_ptr<float>()),
                mask.data_ptr<float>(),
                (int3*)thrust::raw_pointer_cast(cell_ids.data_ptr<int>()),
                offsets.data_ptr<int>(),
                (float3*)thrust::raw_pointer_cast(grad_space.data_ptr<float>()),
                (float3*)thrust::raw_pointer_cast(rays.data_ptr<float>()),
                grads_sdf.data_ptr<float>(),
                (float3*)thrust::raw_pointer_cast(grads_color.data_ptr<float>()),
                color_loss.data_ptr<float>(),
                mask_loss.data_ptr<float>());
        }));

        // Need normalization ??
        /*const int threads_n = 1024;
        const int blocks_n = (num_sites + threads_n - 1) / threads_n; 
        AT_DISPATCH_FLOATING_TYPES( grads_sdf.type(),"normalize_grads_kernel", ([&] {  
            normalize_grads_kernel CUDA_KERNEL(blocks_n,threads_n) (
                num_sites,
                grads_sdf.data_ptr<float>(),
                counter.data_ptr<float>());
        }));*/
    cudaDeviceSynchronize();
}

void normalize_grads_cuda(
    size_t num_sites,
    torch::Tensor grads_sdf,
    torch::Tensor counter)
{
        const int threads = 1024;
        const int blocks = (num_sites + threads - 1) / threads; // ceil for example 8192 + 255 / 256 = 32
        AT_DISPATCH_FLOATING_TYPES( grads_sdf.type(),"normalize_grads_kernel", ([&] {  
            normalize_grads_kernel CUDA_KERNEL(blocks,threads) (
                num_sites,
                grads_sdf.data_ptr<float>(),
                counter.data_ptr<float>());
        }));
}

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
    torch::Tensor mask_loss)
{
        const int threads = 512;
        const int blocks = (num_rays + threads - 1) / threads; // ceil for example 8192 + 255 / 256 = 32
        AT_DISPATCH_FLOATING_TYPES( color_samples.type(),"render_no_sdf_cuda", ([&] {  
            render_no_sdf_kernel CUDA_KERNEL(blocks,threads) (
                num_rays,
                inv_s,
                mask_reg,
                sdf_seg.data_ptr<float>(),
                neighbors.data_ptr<int>(),
                weights_seg.data_ptr<float>(),
                color_samples.data_ptr<float>(),
                true_color.data_ptr<float>(),
                mask.data_ptr<float>(),
                cell_ids.data_ptr<int>(),
                offsets.data_ptr<int>(),
                grads_sdf.data_ptr<float>(),
                grads_color.data_ptr<float>(),
                color_loss.data_ptr<float>(),
                mask_loss.data_ptr<float>());
        }));

        // Need normalization ??
}



void render_no_grad_cuda(
    size_t num_rays,
    float inv_s,
    torch::Tensor sdf_seg,
    torch::Tensor color_samples,
    torch::Tensor offsets,
    torch::Tensor color_out,
    torch::Tensor mask_out)
{
        const int threads = 1024;
        const int blocks = (num_rays + threads - 1) / threads; // ceil for example 8192 + 255 / 256 = 32
        AT_DISPATCH_FLOATING_TYPES( color_samples.type(),"render_no_grad_cuda", ([&] {  
            render_no_grad_kernel CUDA_KERNEL(blocks,threads) (
                num_rays,
                inv_s,
                sdf_seg.data_ptr<float>(),
                color_samples.data_ptr<float>(),
                offsets.data_ptr<int>(),
                color_out.data_ptr<float>(),
                mask_out.data_ptr<float>());
        }));

        // Need normalization ??
}