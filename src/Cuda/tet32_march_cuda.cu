#include <torch/extension.h>

#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include <device_launch_parameters.h>
#include "../Models/cudaType.cuh"

#ifdef __INTELLISENSE__
#define CUDA_KERNEL(...)
#define FAKEINIT = {0}
#else
#define CUDA_KERNEL(...) <<< __VA_ARGS__ >>>
#define FAKEINIT
#endif

#define STOP_TRANS 1.0e-10
#define DIM_L_FEAT 32
#define CLIP_ALPHA 60.0

/** Device functions **/
/** Device functions **/
/** Device functions **/

struct Ray
{
    float origin[3];
    float direction[3];
};

__device__ float sdf2Alpha(float sdf, float sdf_prev, float inv_s) 
{
    if (sdf_prev > sdf) {
        double sdf_prev_clamp = fmin(CLIP_ALPHA, fmax(double(sdf_prev * inv_s), -CLIP_ALPHA));
        double sdf_clamp = fmin(CLIP_ALPHA, fmax(double(sdf * inv_s), -CLIP_ALPHA));
        return min(1.0f, __double2float_rn((1.0 + exp(-sdf_prev_clamp)) / (1.0 + exp(-sdf_clamp))));
    }
    return 1.0f;
}

inline __device__ double dot3D_gpu_d(double a[3], double b[3]) {
	return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

inline __device__ float dot3D_gpu(float a[3], float b[3]) {
	return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

__device__ float dist_tri(float ray_o[3], float ray_d[3], float v0[3], float v1[3], float v2[3], float n[3]) {
	float den = dot3D_gpu(ray_d, n);
	if (den == 0.0f) {
		return 0.0f;
	}
	float tmp[3] = { v0[0] - ray_o[0], v0[1] - ray_o[1], v0[2] - ray_o[2] };
	float fact = (dot3D_gpu(tmp, n) / den);

	return fact;
}

__device__ float dist_tri3(float3 ray_o, float3 ray_d, float3 v0, float3 v1, float3 v2, float3 n) {
	float den = dot(ray_d, n);
	if (den == 0.0f) {
		return 0.0f;
	}
	float3 tmp = v0-ray_o;
	float fact = (dot(tmp, n) / den);

	return fact;
}


__device__ bool OriginInTriangle_gpu(float v1[2], float v2[2], float v3[2])
{
	float d1, d2, d3;
	bool has_neg, has_pos;

	d1 = v1[0] * v2[1] - v1[1] * v2[0]; // sign(pt, v1, v2);
	d2 = v2[0] * v3[1] - v2[1] * v3[0]; //  //sign(pt, v2, v3);
	d3 = v3[0] * v1[1] - v3[1] * v1[0]; // //sign(pt, v3, v1);

	has_neg = (d1 < 0.0f) || (d2 < 0.0f) || (d3 < 0.0f);
	has_pos = (d1 > 0.0f) || (d2 > 0.0f) || (d3 > 0.0f);

	return !(has_neg && has_pos);
}

__device__ int GetExitFaceBis(float p0[2], float p1[2], float p2[2], float p3[2]) {
	int exit_face = 0;

	//test if point is in face 1
	if (OriginInTriangle_gpu(p0, p2, p3)) {
		exit_face = 1;
	}
	else if (OriginInTriangle_gpu(p0, p1, p3)) {
		exit_face = 2;
	}

	return exit_face;
}	

__device__ int GetNextTet(int* tets, int* nei_tets, int tet_id, int id_exit_face) {
	int next_tet_id = nei_tets[4 * tet_id + 3];
	for (int i = 0; i < 3; i++) {
		if (id_exit_face == tets[4 * tet_id + i]) {
			next_tet_id = nei_tets[4 * tet_id + i];
		}
	}

	return next_tet_id;
}

__device__ int GetNextTet3(int4* tets, int4* nei_tets, int tet_id, int id_exit_face) {
	int4 curr_tet = tets[tet_id];
	int4 neis = nei_tets[tet_id];
	int next_tet_id = neis.w;
	
	if (id_exit_face == curr_tet.x) {
		next_tet_id = neis.x;
	}
	
	if (id_exit_face == curr_tet.y) {
		next_tet_id = neis.y;
	}
	
	if (id_exit_face == curr_tet.z) {
		next_tet_id = neis.z;
	}

	return next_tet_id;
}

__device__ double volume_tetrahedron_32(float a[3], float b[3], float c[3], float d[3]) {
	double ad[3] = { a[0] - d[0], a[1] - d[1], a[2] - d[2] };
	double bd[3] = { b[0] - d[0], b[1] - d[1], b[2] - d[2] };
	double cd[3] = { c[0] - d[0], c[1] - d[1], c[2] - d[2] };

	double n[3] = { bd[1] * cd[2] - bd[2] * cd[1],
					-(bd[0] * cd[2] - bd[2] * cd[0]),
					bd[0] * cd[1] - bd[1] * cd[0] };

	double res = abs(dot3D_gpu_d(ad, n)) / 6.0;
	return res;
}

__device__ float volume_tetrahedron_3(float3 a, float3 b, float3 c, float3 d) {
	float3 ad = a-d;
	float3 bd = b-d;
	float3 cd = c-d; 
	float3 n = cross(bd, cd); 
	return fabs(dot(ad, n)) / 6.0f;
}


__device__ float Triangle_Area_3D(float a[3], float b[3], float c[3])
{
	float ab[3] = { b[0] - a[0], b[1] - a[1], b[2] - a[2] };
	float ac[3] = { c[0] - a[0], c[1] - a[1], c[2] - a[2] };
	float cross[3] = { ab[1] * ac[2] - ab[2] * ac[1],
						ab[2] * ac[0] - ab[0] * ac[2],
						ab[0] * ac[1] - ab[1] * ac[0] };
	return sqrt(cross[0]*cross[0] + cross[1]*cross[1] + cross[2]*cross[2]) / 2.0f;
}


__device__ float Triangle_Area_3(float3 a, float3 b, float3 c)
{
	return norm_2(cross(b - a, c - a)) / 2.0f;
}


__device__ float get_sdf32(float p[3], float* weights, float* sites, float* sdf, int* tets, int tet_id) {
	int id0 = tets[4 * tet_id];
	int id1 = tets[4 * tet_id + 1];
	int id2 = tets[4 * tet_id + 2];
	int id3 = id0 ^ id1 ^ id2 ^ tets[4 * tet_id + 3];
	float tot_vol = __double2float_rn(volume_tetrahedron_32(&sites[3 * id0], &sites[3 * id1],
		&sites[3 * id2], &sites[3 * id3]));

    weights[0] = tot_vol == 0.0f ? 0.25f : __double2float_rn(volume_tetrahedron_32(p, &sites[3 * id1],
		&sites[3 * id2], &sites[3 * id3])) / tot_vol;
    weights[1] = tot_vol == 0.0f ? 0.25f : __double2float_rn(volume_tetrahedron_32(p, &sites[3 * id0],
		&sites[3 * id2], &sites[3 * id3])) / tot_vol;
    weights[2] = tot_vol == 0.0f ? 0.25f : __double2float_rn(volume_tetrahedron_32(p, &sites[3 * id0],
		&sites[3 * id1], &sites[3 * id3])) / tot_vol;
    weights[3] = tot_vol == 0.0f ? 0.25f : __double2float_rn(volume_tetrahedron_32(p, &sites[3 * id0],
		&sites[3 * id1], &sites[3 * id2])) / tot_vol;

    float sum_weights = weights[0] + weights[1] + weights[2] + weights[3];
	if (sum_weights > 0.0f) {
		weights[0] = weights[0] / sum_weights;
		weights[1] = weights[1] / sum_weights;
		weights[2] = weights[2] / sum_weights;
		weights[3] = weights[3] / sum_weights;
	}
	else {
		weights[0] = 0.25f;
		weights[1] = 0.25f;
		weights[2] = 0.25f;
		weights[3] = 0.25f;
	}

	return sdf[id0] * weights[0] + sdf[id1] * weights[1] +
            sdf[id2] * weights[2] + sdf[id3] * weights[3];
}

__device__ float get_sdf3(float3 p, float* weights, float3* sites, float* sdf, int4* tets, int tet_id) {
	int4 id = tets[tet_id];
	id.w = id.x ^ id.y ^ id.z ^ id.w;

	float tot_vol = volume_tetrahedron_3(sites[id.x], sites[id.y], sites[id.z], sites[id.w]);

    weights[0] = tot_vol == 0.0f ? 0.25f : volume_tetrahedron_3(p, sites[id.y],
		sites[id.z], sites[id.w]) / tot_vol;
    weights[1] = tot_vol == 0.0f ? 0.25f : volume_tetrahedron_3(p, sites[id.x],
		sites[id.z], sites[id.w]) / tot_vol;
    weights[2] = tot_vol == 0.0f ? 0.25f : volume_tetrahedron_3(p, sites[id.x],
		sites[id.y], sites[id.w]) / tot_vol;
    weights[3] = tot_vol == 0.0f ? 0.25f : volume_tetrahedron_3(p, sites[id.x],
		sites[id.y], sites[id.z]) / tot_vol;

    float sum_weights = weights[0] + weights[1] + weights[2] + weights[3];
	if (sum_weights > 0.0f) {
		weights[0] = weights[0] / sum_weights;
		weights[1] = weights[1] / sum_weights;
		weights[2] = weights[2] / sum_weights;
		weights[3] = weights[3] / sum_weights;
	}
	else {
		weights[0] = 0.25f;
		weights[1] = 0.25f;
		weights[2] = 0.25f;
		weights[3] = 0.25f;
	}

	return sdf[id.x] * weights[0] + sdf[id.y] * weights[1] +
            sdf[id.z] * weights[2] + sdf[id.w] * weights[3];
}

__device__ void get_feat32(float *feat, float weights[4], float* vol_feat, int* tets, int tet_id) {
	int id0 = tets[4 * tet_id];
	int id1 = tets[4 * tet_id + 1];
	int id2 = tets[4 * tet_id + 2];
	int id3 = id0 ^ id1 ^ id2 ^ tets[4 * tet_id + 3];
	for (int i = 0; i < DIM_L_FEAT; i++) {
		feat[i] = vol_feat[DIM_L_FEAT * id0 + i] * weights[0] + vol_feat[DIM_L_FEAT * id1 + i] * weights[1] +
			vol_feat[DIM_L_FEAT * id2 + i] * weights[2] + vol_feat[DIM_L_FEAT * id3 + i] * weights[3];
	}
	return;
}


__device__ float get_sdf_triangle32(float weights[3], float p[3], float* sites, float* sdf, int* tets, int id0, int id1, int id2) {
	// Rescale the tetrahedron
	float ps[3] = { p[0], p[1], p[2] };
	float s0[3] = { sites[3 * id0], sites[3 * id0 + 1], sites[3 * id0 + 2] };
	float s1[3] = { sites[3 * id1], sites[3 * id1 + 1], sites[3 * id1 + 2] };
	float s2[3] = { sites[3 * id2], sites[3 * id2 + 1], sites[3 * id2 + 2] };


	double tot_area = Triangle_Area_3D(s0, s1, s2);

	weights[0] = tot_area == 0.0f ? 0.33f : float(Triangle_Area_3D(ps, s1, s2) / tot_area);
	weights[1] = tot_area == 0.0f ? 0.33f : float(Triangle_Area_3D(ps, s0, s2) / tot_area);
	weights[2] = tot_area == 0.0f ? 0.33f : float(Triangle_Area_3D(ps, s0, s1) / tot_area);

	float w_tot = weights[0] + weights[1] + weights[2];
	weights[0] = weights[0] / w_tot;
	weights[1] = weights[1] / w_tot;
	weights[2] = weights[2] / w_tot;
	/*if (!((weights[0] >= 0.0f && weights[0] <= 1.0f) &&
		(weights[1] >= 0.0f && weights[1] <= 1.0f) &&
		(weights[2] >= 0.0f && weights[2] <= 1.0f) &&
		fabs((weights[0] + weights[1] + weights[2]) - 1.0f) < 1.0e-4f)) {
		return 20.0f;
	}
	else {
		weights[0] = weights[0] / w_tot;
		weights[1] = weights[1] / w_tot;
		weights[2] = weights[2] / w_tot;
	}*/

	if (sdf[id0] == -1000.0 || sdf[id1] == -1000.0 || sdf[id2] == -1000.0)
		return -1000.0;
	return sdf[id0] * weights[0] + sdf[id1] * weights[1] + sdf[id2] * weights[2];
}

__device__ float get_sdf_triangle3(float weights[3], float3 p, float3* sites, float* sdf, int4* tets, int id0, int id1, int id2) {
	// Rescale the tetrahedron
	float3 s0 = sites[id0];
	float3 s1 = sites[id1];
	float3 s2 = sites[id2];

	double tot_area = Triangle_Area_3(s0, s1, s2);

	weights[0] = tot_area == 0.0f ? 0.33f : float(Triangle_Area_3(p, s1, s2) / tot_area);
	weights[1] = tot_area == 0.0f ? 0.33f : float(Triangle_Area_3(p, s0, s2) / tot_area);
	weights[2] = tot_area == 0.0f ? 0.33f : float(Triangle_Area_3(p, s0, s1) / tot_area);

	float w_tot = weights[0] + weights[1] + weights[2];
	weights[0] = weights[0] / w_tot;
	weights[1] = weights[1] / w_tot;
	weights[2] = weights[2] / w_tot;
	
	if (sdf[id0] == -1000.0 || sdf[id1] == -1000.0 || sdf[id2] == -1000.0)
		return -1000.0;
	return sdf[id0] * weights[0] + sdf[id1] * weights[1] + sdf[id2] * weights[2];
}


__device__ void get_feat_triangle32(float *feat, float weights[3], float p[3], float* sites, float* vol_feat, int* tets, int id0, int id1, int id2) {
	for (int i = 0; i < DIM_L_FEAT; i++) {
		feat[i] = vol_feat[DIM_L_FEAT * id0 + i] * weights[0] + 
					vol_feat[DIM_L_FEAT * id1 + i] * weights[1] +
					vol_feat[DIM_L_FEAT * id2 + i] * weights[2];
	}
	return;
}

__device__ void PN_triangle_interpolation_sdf(float *sdf, float weights[3], float p[3], float* sites, float* vol_feat, int* tets, int id0, int id1, int id2) {
}


__global__ void tet32_march_cuda_kernel(
	const float inv_s,
    const size_t num_rays,                // number of rays
    const size_t num_samples,                // number of rays
    const size_t cam_id,
    const float *__restrict__ rays,       // [N_rays, 6]
    float *__restrict__ vertices,     // [N_voxels, 4] for each voxel => it's vertices
    float *__restrict__ sdf,     // [N_voxels, 4] for each voxel => it's vertices
    int *__restrict__ tets,  
    int *__restrict__ nei_tets,  
    const int *__restrict__ cam_ids,  
    const int *__restrict__ offsets_cam,  
    const int *__restrict__ cam_tets,  
    float *__restrict__ weights_samp,     // [N_voxels, 4] for each voxel => it's vertices
    float *__restrict__ z_vals,     // [N_voxels, 4] for each voxel => it's vertices
    float *__restrict__ z_sdfs,     // [N_voxels, 4] for each voxel => it's vertices
    int *__restrict__ z_ids,     // [N_voxels, 4] for each voxel => it's vertices
    int *__restrict__ counter,     // [N_voxels, 4] for each voxel => it's vertices
    int *__restrict__ activate,     // [N_voxels, 4] for each voxel => it's vertices
    int *__restrict__ offset     // [N_voxels, 4] for each voxel => it's vertices
)
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rays)
    {
        return;
    }

    int r_id = cam_ids[cam_id];

	float ray_d[3] = { rays[idx * 3], rays[idx * 3 + 1], rays[idx * 3 + 2] };
	float ray_o[3] = { vertices[r_id * 3], vertices[r_id * 3 + 1], vertices[r_id * 3 + 2] };

    float curr_site[3] {vertices[r_id * 3], vertices[r_id * 3 + 1], vertices[r_id * 3 + 2]};
	
    float *z_val_ray = &z_vals[idx*num_samples*2];
    int *z_id_ray = &z_ids[idx*num_samples*6];	
    float *z_sdf_ray = &z_sdfs[idx*num_samples*2];
    float *weights_ray = &weights_samp[idx*num_samples*6];

    // build base w.r.t ray
	float u[3]{};
	float v[3]{};

	{
		int min = 0;
		if (abs(ray_d[1]) < abs(ray_d[0])) {
			if (abs(ray_d[1]) < abs(ray_d[2]))
				min = 1;
			else
				min = 2;
		}
		else if ((abs(ray_d[2]) < abs(ray_d[0]))) {
			min = 2;
		}

		int max = 0;
		if (abs(ray_d[1]) > abs(ray_d[0])) {
			if (abs(ray_d[1]) > abs(ray_d[2]))
				max = 1;
			else
				max = 2;
		}
		else if ((abs(ray_d[2]) > abs(ray_d[0]))) {
			max = 2;
		}

		u[min] = 0.0f;
		u[(min + 1) % 3] = ray_d[(min + 2) % 3] / ray_d[max];
		u[(min + 2) % 3] = -ray_d[(min + 1) % 3] / ray_d[max];
		float t[3] = { ray_d[1] * u[2] - ray_d[2] * u[1], ray_d[2] * u[0] - ray_d[0] * u[2], ray_d[0] * u[1] - ray_d[1] * u[0] };

		min = 0;
		if (abs(t[1]) < abs(t[0])) {
			if (abs(t[1]) < abs(t[2]))
				min = 1;
			else
				min = 2;
		}
		else if ((abs(t[2]) < abs(t[0]))) {
			min = 2;
		}

		max = 0;
		if (abs(t[1]) > abs(t[0])) {
			if (abs(t[1]) > abs(t[2]))
				max = 1;
			else
				max = 2;
		}
		else if ((abs(t[2]) > abs(t[0]))) {
			max = 2;
		}

		v[0] = t[0] / t[3 - min - max];
		v[1] = t[1] / t[3 - min - max];
		v[2] = t[2] / t[3 - min - max];
	}

    // Initialisation with entry tet
	int tet_id = -1; 
	int prev_tet_id = -1;

	// id_0, id_1, id_2 are indices of vertices of exit face
	int ids[4] = { 0, 0, 0, 0 };
	float weights[3] = { 0.0f, 0.0f, 0.0f};
	float prev_weights[3] = { 0.0f, 0.0f, 0.0f};
	int id_exit_face = 3;
	float p[8]{};

	// project all vertices into the base coordinate system
	float v_new[3]{};
	float curr_dist;

	float nmle[3]{};
	float v_0[3]{};
	float v_1[3]{};
	float v_2[3]{};

    int start_cam_tet = cam_id == 0 ? 0 : offsets_cam[cam_id-1];
    int cam_adj_count = cam_id == 0 ? offsets_cam[cam_id] : offsets_cam[cam_id] - offsets_cam[cam_id-1];  

    for (int i = 1; i < cam_adj_count; i++) {
		tet_id = cam_tets[start_cam_tet + i];
		prev_tet_id = tet_id;
		ids[0] = tets[4 * tet_id];
		ids[1] = tets[4 * tet_id + 1];
		ids[2] = tets[4 * tet_id + 2];
		ids[3] = tets[4 * tet_id] ^ tets[4 * tet_id + 1] ^ tets[4 * tet_id + 2] ^ tets[4 * tet_id + 3];

		if (r_id == ids[0]) {
			ids[0] = ids[3];
			ids[3] = r_id;
			tet_id = nei_tets[4 * tet_id];
		}
		else if (r_id == ids[1]) {
			ids[1] = ids[3];
			ids[3] = r_id;
			tet_id = nei_tets[4 * tet_id + 1];
		}
		else if (r_id == ids[2]) {
			ids[2] = ids[3];
			ids[3] = r_id;
			tet_id = nei_tets[4 * tet_id + 2];
		}
		else {
			tet_id = nei_tets[4 * tet_id + 3];
		}

		for (int j = 0; j < 3; j++) {
			v_new[0] = vertices[3 * ids[j]] - ray_o[0];
			v_new[1] = vertices[3 * ids[j] + 1] - ray_o[1];
			v_new[2] = vertices[3 * ids[j] + 2] - ray_o[2];
			p[2 * j] = dot3D_gpu(u, v_new);
			p[2 * j + 1] = dot3D_gpu(v, v_new);
		}

		v_0[0] = vertices[3 * ids[0]]; v_0[1] = vertices[3 * ids[0] + 1]; v_0[2] = vertices[3 * ids[0] + 2];
		v_1[0] = vertices[3 * ids[1]]; v_1[1] = vertices[3 * ids[1] + 1]; v_1[2] = vertices[3 * ids[1] + 2];
		v_2[0] = vertices[3 * ids[2]]; v_2[1] = vertices[3 * ids[2] + 1]; v_2[2] = vertices[3 * ids[2] + 2];

		nmle[0] = (v_1[1] - v_0[1]) * (v_2[2] - v_0[2]) - (v_1[2] - v_0[2]) * (v_2[1] - v_0[1]);
		nmle[1] = (v_1[2] - v_0[2]) * (v_2[0] - v_0[0]) - (v_1[0] - v_0[0]) * (v_2[2] - v_0[2]);
		nmle[2] = (v_1[0] - v_0[0]) * (v_2[1] - v_0[1]) - (v_1[1] - v_0[1]) * (v_2[0] - v_0[0]);
		float norm_n = sqrtf(nmle[0] * nmle[0] + nmle[1] * nmle[1] + nmle[2] * nmle[2]);
		nmle[0] = norm_n == 0.0f ? 0.0f : nmle[0] / norm_n;
		nmle[1] = norm_n == 0.0f ? 0.0f : nmle[1] / norm_n;
		nmle[2] = norm_n == 0.0f ? 0.0f : nmle[2] / norm_n;

		curr_dist = dist_tri(ray_o, ray_d, v_0, v_1, v_2, nmle);

		if (curr_dist > 0.0f && OriginInTriangle_gpu(&p[0], &p[2], &p[4])) {

			v_new[0] = ray_o[0] + ray_d[0] * curr_dist / 2.0f;
			v_new[1] = ray_o[1] + ray_d[1] * curr_dist / 2.0f;
			v_new[2] = ray_o[2] + ray_d[2] * curr_dist / 2.0f;
			if (get_sdf32(v_new, weights, vertices, sdf, tets, prev_tet_id) != 30.0f) 
				break;
		}
		else {
			tet_id = -1;
		}
	}

	// Traverse tet
	prev_tet_id = -1;
	float prev_dist = 0.0f;
	int s_id = 0;
	int iter_max = 0;
	float curr_z = 0.0f;
	float curr_p[3] = { ray_o[0] + ray_d[0] * curr_z,
						ray_o[1] + ray_d[1] * curr_z,
						ray_o[2] + ray_d[2] * curr_z };
	int ids_s[6] = { 0, 0, 0, 0, 0, 0 };		
	float next_sdf;
	float prev_sdf = -1000.0f;
	float Tpartial = 1.0;
	while (tet_id >= 0 && iter_max < 10000) {
		ids_s[0] = ids[0]; ids_s[1] = ids[1]; ids_s[2] = ids[2]; 
		ids[id_exit_face] = ids[3];
		ids[3] = ids[0] ^ ids[1] ^ ids[2] ^ tets[4 * tet_id + 3]; 
		v_0[0] = vertices[3 * ids[0]]; v_0[1] = vertices[3 * ids[0] + 1]; v_0[2] = vertices[3 * ids[0] + 2];
		v_1[0] = vertices[3 * ids[1]]; v_1[1] = vertices[3 * ids[1] + 1]; v_1[2] = vertices[3 * ids[1] + 2];
		v_2[0] = vertices[3 * ids[2]]; v_2[1] = vertices[3 * ids[2] + 1]; v_2[2] = vertices[3 * ids[2] + 2];

		nmle[0] = (v_1[1] - v_0[1]) * (v_2[2] - v_0[2]) - (v_1[2] - v_0[2]) * (v_2[1] - v_0[1]);
		nmle[1] = (v_1[2] - v_0[2]) * (v_2[0] - v_0[0]) - (v_1[0] - v_0[0]) * (v_2[2] - v_0[2]);
		nmle[2] = (v_1[0] - v_0[0]) * (v_2[1] - v_0[1]) - (v_1[1] - v_0[1]) * (v_2[0] - v_0[0]);
		float norm_n = sqrtf(nmle[0] * nmle[0] + nmle[1] * nmle[1] + nmle[2] * nmle[2]);
		nmle[0] = norm_n == 0.0f ? 0.0f : nmle[0] / norm_n;
		nmle[1] = norm_n == 0.0f ? 0.0f : nmle[1] / norm_n;
		nmle[2] = norm_n == 0.0f ? 0.0f : nmle[2] / norm_n;

		curr_dist = dist_tri(ray_o, ray_d, v_0, v_1, v_2, nmle);
		curr_p[0] = ray_o[0] + ray_d[0] * curr_dist;
		curr_p[1] = ray_o[1] + ray_d[1] * curr_dist;
		curr_p[2] = ray_o[2] + ray_d[2] * curr_dist;

		next_sdf = get_sdf_triangle32(weights, curr_p, vertices, sdf, tets, ids[0], ids[1], ids[2]);
		ids_s[3] = ids[0]; ids_s[4] = ids[1]; ids_s[5] = ids[2]; 
		
		float alpha_tet = sdf2Alpha(next_sdf, prev_sdf, inv_s);
		float contrib_tet = Tpartial * (1.0f - alpha_tet);
		int fact_s = int(contrib_tet * 4.0f);
		
		if (prev_tet_id != -1) { 
			if (((prev_sdf > next_sdf && 
					(next_sdf == -1000.0f || alpha_tet < 1.0f)))) {
					//fmin(fabs(next_sdf), fabs(prev_sdf))*inv_s < 2.0*CLIP_ALPHA)))) {
				z_val_ray[2 * s_id] = prev_dist;
				z_val_ray[2 * s_id + 1] = curr_dist;

				z_id_ray[6 * s_id] = ids_s[0]; 
				z_id_ray[6 * s_id + 1] = ids_s[1];
				z_id_ray[6 * s_id + 2] = ids_s[2]; 
				z_id_ray[6 * s_id + 3] = ids_s[3]; 
				z_id_ray[6 * s_id + 4] = ids_s[4];
				z_id_ray[6 * s_id + 5] = ids_s[5]; 

				z_sdf_ray[2 * s_id] = prev_sdf;
				z_sdf_ray[2 * s_id + 1] = next_sdf;

				weights_ray[6 * s_id] = prev_weights[0];
				weights_ray[6 * s_id + 1] = prev_weights[1];
				weights_ray[6 * s_id + 2] = prev_weights[2];
				weights_ray[6 * s_id + 3] = weights[0]; 
				weights_ray[6 * s_id + 4] = weights[1]; 
				weights_ray[6 * s_id + 5] = weights[2]; 

				// activate sites here
				for (int l = 0; l < 6; l++) {
					//activate[ids_s[l]] = 1;
					atomicExch(&activate[ids_s[l]], 1);
				}

				s_id++;
				if (s_id > num_samples - 1) {
					break;
				}
			}

			if (s_id > num_samples - 1) {
				break;
			}			
		}
		
		Tpartial = Tpartial * alpha_tet;

		//if (Tpartial < STOP_TRANS) {// stop if the transmittance is low
        //    break;
        //}

		prev_dist = curr_dist;
		prev_sdf = next_sdf;
		for (int l = 0; l < 3; l++) {
			prev_weights[l] = weights[l];
		}

		prev_tet_id = tet_id;

		v_new[0] = vertices[3 * ids[3]] - ray_o[0];
		v_new[1] = vertices[3 * ids[3] + 1] - ray_o[1];
		v_new[2] = vertices[3 * ids[3] + 2] - ray_o[2];

		p[2 * id_exit_face] = p[2 * 3];
		p[2 * id_exit_face + 1] = p[2 * 3 + 1];

		p[2 * 3] = dot3D_gpu(u, v_new);
		p[2 * 3 + 1] = dot3D_gpu(v, v_new);

		id_exit_face = GetExitFaceBis(&p[0], &p[2], &p[4], &p[6]);

		tet_id = GetNextTet(tets, nei_tets, tet_id, ids[id_exit_face]);

		iter_max++;
	}

	offset[2 * idx] = atomicAdd(counter, s_id);
	offset[2 * idx + 1] = s_id;

    return;
}

__global__ void tet32_march_cuda_kernel_o(
	const float inv_s,
    const size_t num_rays,                // number of rays
    const size_t num_samples,                // number of rays
    const size_t cam_id,
    const float3 *__restrict__ rays,       // [N_rays, 6]
    float3 *__restrict__ vertices,     // [N_voxels, 4] for each voxel => it's vertices
    float *__restrict__ sdf,     // [N_voxels, 4] for each voxel => it's vertices
    int4 *__restrict__ tets,  
    int4 *__restrict__ nei_tets,  
    const int *__restrict__ cam_ids,  
    const int *__restrict__ offsets_cam,  
    const int *__restrict__ cam_tets,  
    float3 *__restrict__ weights_samp,     // [N_voxels, 4] for each voxel => it's vertices
    float2 *__restrict__ z_vals,     // [N_voxels, 4] for each voxel => it's vertices
    float2 *__restrict__ z_sdfs,     // [N_voxels, 4] for each voxel => it's vertices
    int3 *__restrict__ z_ids,     // [N_voxels, 4] for each voxel => it's vertices
    int *__restrict__ counter,     // [N_voxels, 4] for each voxel => it's vertices
    int *__restrict__ activate,     // [N_voxels, 4] for each voxel => it's vertices
    int *__restrict__ offset     // [N_voxels, 4] for each voxel => it's vertices
)
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rays)
    {
        return;
    }

    int r_id = cam_ids[cam_id];

	float3 ray_d = rays[idx];
	float3 ray_o = vertices[r_id];

    float3 curr_site = ray_o;
	
    float2 *z_val_ray = &z_vals[idx*num_samples];
    int3 *z_id_ray = &z_ids[idx*num_samples*2];	
    float2 *z_sdf_ray = &z_sdfs[idx*num_samples];
    float3 *weights_ray = &weights_samp[idx*num_samples*2];

    // build base w.r.t ray
	float ut[3]{};
	float vt[3]{};

	{
		int min = 0;
		if (abs(ray_d.y) < abs(ray_d.x)) {
			if (abs(ray_d.y) < abs(ray_d.z))
				min = 1;
			else
				min = 2;
		}
		else if ((abs(ray_d.z) < abs(ray_d.x))) {
			min = 2;
		}

		int max = 0;
		if (abs(ray_d.y) > abs(ray_d.x)) {
			if (abs(ray_d.y) > abs(ray_d.z))
				max = 1;
			else
				max = 2;
		}
		else if ((abs(ray_d.z) > abs(ray_d.x))) {
			max = 2;
		}

		ut[min] = 0.0f;
		ut[(min + 1) % 3] = elem(ray_d,(min + 2) % 3) / elem(ray_d,max);
		ut[(min + 2) % 3] = -elem(ray_d,(min + 1) % 3) / elem(ray_d,max);
		float t[3] = { ray_d.y * ut[2] - ray_d.z * ut[1], 
						ray_d.z * ut[0] - ray_d.x * ut[2], 
						ray_d.x * ut[1] - ray_d.y * ut[0] };

		min = 0;
		if (abs(t[1]) < abs(t[0])) {
			if (abs(t[1]) < abs(t[2]))
				min = 1;
			else
				min = 2;
		}
		else if ((abs(t[2]) < abs(t[0]))) {
			min = 2;
		}

		max = 0;
		if (abs(t[1]) > abs(t[0])) {
			if (abs(t[1]) > abs(t[2]))
				max = 1;
			else
				max = 2;
		}
		else if ((abs(t[2]) > abs(t[0]))) {
			max = 2;
		}

		vt[0] = t[0] / t[3 - min - max];
		vt[1] = t[1] / t[3 - min - max];
		vt[2] = t[2] / t[3 - min - max];
	}

	float3 u = make_float3(ut[0], ut[1], ut[2]);
	float3 v = make_float3(vt[0], vt[1], vt[2]);

    // Initialisation with entry tet
	int tet_id = -1; 
	int prev_tet_id = -1;

	// id_0, id_1, id_2 are indices of vertices of exit face
	int ids[4]{};
	float weights[3] =  {0.0f, 0.0f, 0.0f};
	float prev_weights[3] = {0.0f, 0.0f, 0.0f};
	int id_exit_face = 3;
	float p[8]{};

	// project all vertices into the base coordinate system
	float3 v_new = make_float3(0.0f, 0.0f, 0.0f);
	float3 nmle = make_float3(0.0f, 0.0f, 0.0f);
	float3 v_0 = make_float3(0.0f, 0.0f, 0.0f);
	float3 v_1 = make_float3(0.0f, 0.0f, 0.0f);
	float3 v_2 = make_float3(0.0f, 0.0f, 0.0f);
	float curr_dist;

    int start_cam_tet = cam_id == 0 ? 0 : offsets_cam[cam_id-1];
    int cam_adj_count = cam_id == 0 ? offsets_cam[cam_id] : offsets_cam[cam_id] - offsets_cam[cam_id-1];  

    for (int i = 1; i < cam_adj_count; i++) {
		tet_id = cam_tets[start_cam_tet + i];
		prev_tet_id = tet_id;
		ids[0] = tets[tet_id].x;
		ids[1] = tets[tet_id].y;
		ids[2] = tets[tet_id].z;
		ids[3] = ids[0] ^ ids[1] ^ ids[2] ^ tets[tet_id].w; 


		if (r_id == ids[0]) {
			ids[0] = ids[3];
			ids[3] = r_id;
			tet_id = nei_tets[tet_id].x;
		}
		else if (r_id == ids[1]) {
			ids[1] = ids[3];
			ids[3] = r_id;
			tet_id = nei_tets[tet_id].y;
		}
		else if (r_id == ids[2]) {
			ids[2] = ids[3];
			ids[3] = r_id;
			tet_id = nei_tets[tet_id].z;
		}
		else {
			tet_id = nei_tets[tet_id].w;
		}

		for (int j = 0; j < 3; j++) {
			v_new = vertices[ids[j]] - ray_o;
			p[2 * j] = dot(u, v_new);
			p[2 * j + 1] = dot(v, v_new);
		}

		v_0 = vertices[ids[0]];
		v_1 = vertices[ids[1]];
		v_2 = vertices[ids[2]];
		nmle = cross(v_1 - v_0, v_2 - v_0);
		float norm_n = norm_2(nmle);
		if (norm_n > 0.0f)
			nmle = nmle / norm_n;

		curr_dist = dist_tri3(ray_o, ray_d, v_0, v_1, v_2, nmle);

		if (curr_dist > 0.0f && OriginInTriangle_gpu(&p[0], &p[2], &p[4])) {

			v_new = ray_o + ray_d * curr_dist / 2.0f;
			if (get_sdf3(v_new, weights, vertices, sdf, tets, prev_tet_id) != 30.0f) 
				break;
		}
		else {
			tet_id = -1;
		}
	}

	// Traverse tet
	prev_tet_id = -1;
	float prev_dist = 0.0f;
	int s_id = 0;
	int iter_max = 0;
	float curr_z = 0.0f;
	float3 curr_p = ray_o + ray_d * curr_z;
	int ids_s[6] = { 0, 0, 0, 0, 0, 0 };		
	float next_sdf;
	float prev_sdf = -1000.0f;
	float Tpartial = 1.0;
	while (tet_id >= 0 && iter_max < 10000) {
		ids_s[0] = ids[0]; ids_s[1] = ids[1]; ids_s[2] = ids[2]; 
		ids[id_exit_face] = ids[3];
		ids[3] = ids[0] ^ ids[1] ^ ids[2] ^ tets[tet_id].w; 
		v_0 = vertices[ids[0]];
		v_1 = vertices[ids[1]];
		v_2 = vertices[ids[2]];

		nmle = cross(v_1 - v_0, v_2 - v_0);
		float norm_n = norm_2(nmle);
		if (norm_n > 0.0f)
			nmle = nmle / norm_n;

		curr_dist = dist_tri3(ray_o, ray_d, v_0, v_1, v_2, nmle);
		curr_p = ray_o + ray_d * curr_dist;

		next_sdf = get_sdf_triangle3(weights, curr_p, vertices, sdf, tets, ids[0], ids[1], ids[2]);
		ids_s[3] = ids[0]; ids_s[4] = ids[1]; ids_s[5] = ids[2]; 
		
		float alpha_tet = sdf2Alpha(next_sdf, prev_sdf, inv_s);
		float contrib_tet = Tpartial * (1.0f - alpha_tet);
		int fact_s = int(contrib_tet * 4.0f);
		
		if (prev_tet_id != -1) { 
			if (((prev_dist > 0.05f && prev_sdf > next_sdf && 
					(next_sdf == -1000.0f || alpha_tet < 1.0f)))) {
					//fmin(fabs(next_sdf), fabs(prev_sdf))*inv_s < 2.0*CLIP_ALPHA)))) {
				z_val_ray[s_id] = make_float2(prev_dist, curr_dist);

				z_id_ray[2 * s_id] = make_int3(ids_s[0], ids_s[1], ids_s[2]); 
				z_id_ray[2 * s_id + 1] = make_int3(ids_s[3], ids_s[4], ids_s[5]); 

				z_sdf_ray[s_id] = make_float2(prev_sdf, next_sdf);

				weights_ray[2 * s_id] = make_float3(prev_weights[0], prev_weights[1], prev_weights[2]);
				weights_ray[2 * s_id + 1] = make_float3(weights[0], weights[1], weights[2]); 

				// activate sites here
				for (int l = 0; l < 6; l++) {
					activate[ids_s[l]] = 1;
				}

				s_id++;
				if (s_id > num_samples - 1) {
					break;
				}
			}

			if (s_id > num_samples - 1) {
				break;
			}			
		}
		
		Tpartial = Tpartial * alpha_tet;

		if (Tpartial < STOP_TRANS) {// stop if the transmittance is low
            break;
        }

		prev_dist = curr_dist;
		prev_sdf = next_sdf;
		for (int l = 0; l < 3; l++) {
			prev_weights[l] = weights[l];
		}

		prev_tet_id = tet_id;

		v_new = vertices[ids[3]] - ray_o;

		p[2 * id_exit_face] = p[2 * 3];
		p[2 * id_exit_face + 1] = p[2 * 3 + 1];

		p[2 * 3] = dot(u, v_new);
		p[2 * 3 + 1] = dot(v, v_new);

		id_exit_face = GetExitFaceBis(&p[0], &p[2], &p[4], &p[6]);

		tet_id = GetNextTet3(tets, nei_tets, tet_id, ids[id_exit_face]);

		iter_max++;
	}

	offset[2 * idx] = atomicAdd(counter, s_id);
	offset[2 * idx + 1] = s_id;

    return;
}


__global__ void fill_samples_kernel(
    const size_t num_rays,                // number of rays
    const size_t num_samples,                // number of rays
    const float *__restrict__ rays_o,       // [N_rays, 6]
    const float *__restrict__ rays_d,       // [N_rays, 6]
    const float *__restrict__ sites,       // [N_rays, 6]
    float *__restrict__ in_z,       // [N_rays, 6]
    float *__restrict__ in_sdf,       // [N_rays, 6]
    float *__restrict__ in_feat,       // [N_rays, 6]
    float *__restrict__ in_weights,     // [N_voxels, 4] for each voxel => it's vertices
    float *__restrict__ in_grads,     // [N_voxels, 4] for each voxel => it's vertices
    int *__restrict__ in_ids,       // [N_rays, 6]
    float *__restrict__ out_z,     // [N_voxels, 4] for each voxel => it's vertices
    float *__restrict__ out_sdf,     // [N_voxels, 4] for each voxel => it's vertices
    float *__restrict__ out_feat,     // [N_voxels, 4] for each voxel => it's vertices
    float *__restrict__ out_weights,     // [N_voxels, 4] for each voxel => it's vertices
    float *__restrict__ out_grads,     // [N_voxels, 4] for each voxel => it's vertices
    int *__restrict__ out_ids,     // [N_voxels, 4] for each voxel => it's vertices
    int *__restrict__ offset,     // [N_voxels, 4] for each voxel => it's vertices
    float *__restrict__ samples,     // [N_voxels, 4] for each voxel => it's vertices
    float *__restrict__ sample_rays     // [N_voxels, 4] for each voxel => it's vertices
)
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rays)
    {
        return;
    }

    Ray ray = {
        {rays_o[idx * 3], rays_o[idx * 3 + 1], rays_o[idx * 3 + 2]},
        {rays_d[idx * 3], rays_d[idx * 3 + 1], rays_d[idx * 3 + 2]}};
    int num_knn = 24;

    float* in_z_rays = &in_z[2*num_samples * idx];
    int* in_ids_rays = &in_ids[6*num_samples * idx];
    float* in_sdf_rays = &in_sdf[2*num_samples * idx];
    float* in_weights_rays = &in_weights[6*num_samples * idx];

    int start = offset[2*idx];
    int end = offset[2*idx+1];
    int s_id = 0;
    for (int i = start; i < start+end; i++) {
        out_z[2*i] = in_z_rays[2*s_id];
		out_z[2*i + 1] = in_z_rays[2*s_id+1];

        out_sdf[3*i] = in_sdf_rays[2*s_id];
		out_sdf[3*i + 1] = in_sdf_rays[2*s_id+1];

		float lambda = 0.5f;
		if (out_sdf[3*i]*out_sdf[3*i+1] <= 0.0f) {
			lambda = fabs(out_sdf[3*i+1])/(fabs(out_sdf[3*i])+fabs(out_sdf[3*i+1]));
			if (lambda < 0.5f) {
				out_sdf[3*i] = 2.0f*lambda*out_sdf[3*i] + (1.0f-2.0f*lambda)*out_sdf[3*i+1];
			} else {
				out_sdf[3*i+1] = (1.0-2.0f*lambda)*out_sdf[3*i] + (1.0f-(1.0-2.0f*lambda))*out_sdf[3*i+1];
			}
		}
		out_sdf[3*i+2] = lambda;

		for (int l = 0; l < 6; l++) {
        	out_ids[6*i + l] = in_ids_rays[6 * s_id + l];
		}

		for (int l = 0; l < 12; l++) {
			out_grads[12*i + l] = lambda*(in_weights_rays[6 * s_id + 0]*in_grads[12*in_ids_rays[6 * s_id + 0] + l] +
												in_weights_rays[6 * s_id + 1]*in_grads[12*in_ids_rays[6 * s_id + 1] + l] +
												in_weights_rays[6 * s_id + 2]*in_grads[12*in_ids_rays[6 * s_id + 2] + l])
							+ (1.0f-lambda)*(in_weights_rays[6 * s_id + 3]*in_grads[12*in_ids_rays[6 * s_id + 3] + l] +
												in_weights_rays[6 * s_id + 4]*in_grads[12*in_ids_rays[6 * s_id + 4] + l] +
												in_weights_rays[6 * s_id + 5]*in_grads[12*in_ids_rays[6 * s_id + 5] + l]);
		}

		for (int l = 0; l < DIM_L_FEAT; l++) {
			out_feat[DIM_L_FEAT*i+l] = lambda*(in_weights_rays[6 * s_id + 0]*in_feat[DIM_L_FEAT*in_ids_rays[6 * s_id + 0] + l] +
												in_weights_rays[6 * s_id + 1]*in_feat[DIM_L_FEAT*in_ids_rays[6 * s_id + 1] + l] +
												in_weights_rays[6 * s_id + 2]*in_feat[DIM_L_FEAT*in_ids_rays[6 * s_id + 2] + l])
							+ (1.0f-lambda)*(in_weights_rays[6 * s_id + 3]*in_feat[DIM_L_FEAT*in_ids_rays[6 * s_id + 3] + l] +
												in_weights_rays[6 * s_id + 4]*in_feat[DIM_L_FEAT*in_ids_rays[6 * s_id + 4] + l] +
												in_weights_rays[6 * s_id + 5]*in_feat[DIM_L_FEAT*in_ids_rays[6 * s_id + 5] + l]);			
		}

        sample_rays[3 * i] = ray.direction[0];
        sample_rays[3 * i + 1] = ray.direction[1];
        sample_rays[3 * i + 2] = ray.direction[2];		

		for (int l = 0; l < 6; l++) {
        	out_weights[7*i + l] = in_weights_rays[6 * s_id + l];
		}
		out_weights[7*i + 6] = lambda;

		if (lambda < 0.5f) {
			samples[3 * i] = ray.origin[0] + (2.0f*lambda*out_z[2*i] + (1.0f-2.0f*lambda)*out_z[2*i+1])*ray.direction[0];
			samples[3 * i + 1] = ray.origin[1] + (2.0f*lambda*out_z[2*i] + (1.0f-2.0f*lambda)*out_z[2*i+1])*ray.direction[1];
			samples[3 * i + 2] = ray.origin[2] + (2.0f*lambda*out_z[2*i] + (1.0f-2.0f*lambda)*out_z[2*i+1])*ray.direction[2];
		} else {
			samples[3 * i] = ray.origin[0] + ((1.0-2.0f*lambda)*out_z[2*i] + (1.0f-(1.0-2.0f*lambda))*out_z[2*i+1])*ray.direction[0];
			samples[3 * i + 1] = ray.origin[1] + ((1.0-2.0f*lambda)*out_z[2*i] + (1.0f-(1.0-2.0f*lambda))*out_z[2*i+1])*ray.direction[1];
			samples[3 * i + 2] = ray.origin[2] + ((1.0-2.0f*lambda)*out_z[2*i] + (1.0f-(1.0-2.0f*lambda))*out_z[2*i+1])*ray.direction[2];	
		}

        s_id++;
    }

    return;
}

__global__ void fill_samples_kernel_o(
    const size_t num_rays,                // number of rays
    const size_t num_samples,                // number of rays
    const float3 *__restrict__ rays_o,       // [N_rays, 6]
    const float3 *__restrict__ rays_d,       // [N_rays, 6]
    const float3 *__restrict__ sites,       // [N_rays, 6]
    float2 *__restrict__ in_z,       // [N_rays, 6]
    float2 *__restrict__ in_sdf,       // [N_rays, 6]
    float4 *__restrict__ in_feat,       // [N_rays, 6]
    float3 *__restrict__ in_weights,     // [N_voxels, 4] for each voxel => it's vertices
    float3 *__restrict__ in_grads,     // [N_voxels, 4] for each voxel => it's vertices
    int3 *__restrict__ in_ids,       // [N_rays, 6]
    float2 *__restrict__ out_z,     // [N_voxels, 4] for each voxel => it's vertices
    float3 *__restrict__ out_sdf,     // [N_voxels, 4] for each voxel => it's vertices
    float4 *__restrict__ out_feat,     // [N_voxels, 4] for each voxel => it's vertices
    float3 *__restrict__ out_weights,     // [N_voxels, 4] for each voxel => it's vertices
    float3 *__restrict__ out_grads,     // [N_voxels, 4] for each voxel => it's vertices
    int3 *__restrict__ out_ids,     // [N_voxels, 4] for each voxel => it's vertices
    int *__restrict__ offset,     // [N_voxels, 4] for each voxel => it's vertices
    float3 *__restrict__ samples,     // [N_voxels, 4] for each voxel => it's vertices
    float3 *__restrict__ sample_rays     // [N_voxels, 4] for each voxel => it's vertices
)
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rays)
    {
        return;
    }

	float3 ray_o = rays_o[idx];
	float3 ray_d = rays_d[idx];

    float2* in_z_rays = &in_z[num_samples * idx];
    int3* in_ids_rays = &in_ids[2*num_samples * idx];
    float2* in_sdf_rays = &in_sdf[num_samples * idx];
    float3* in_weights_rays = &in_weights[2*num_samples * idx];

    int start = offset[2*idx];
    int end = offset[2*idx+1];
    int s_id = 0;
    for (int i = start; i < start+end; i++) {
		out_z[i] = in_z_rays[s_id];

		float2 c_sdf = in_sdf_rays[s_id];
		float lambda = 0.5f;
		if (c_sdf.x*c_sdf.y <= 0.0f) {
			lambda = fabs(c_sdf.y)/(fabs(c_sdf.x)+fabs(c_sdf.y));
			if (lambda < 0.5f) {
				c_sdf.x = 2.0f*lambda*c_sdf.x + (1.0f-2.0f*lambda)*c_sdf.y;
			} else {
				c_sdf.y = (1.0-2.0f*lambda)*c_sdf.x + (1.0f-(1.0-2.0f*lambda))*c_sdf.y;
			}
		}
		out_sdf[i] = make_float3(c_sdf.x, c_sdf.y, lambda);

		out_ids[2*i] = in_ids_rays[2 * s_id];
		out_ids[2*i + 1] = in_ids_rays[2 * s_id + 1];

		float3 c_weights = in_weights_rays[2 * s_id];
		float3 c_weights_n = in_weights_rays[2 * s_id + 1];
		int3 c_ids = in_ids_rays[2 * s_id];
		int3 c_ids_n = in_ids_rays[2 * s_id + 1];
		for (int l = 0; l < 4; l++) {
			out_grads[4*i + l] = lambda*(c_weights.x*in_grads[4*c_ids.x + l] +
												c_weights.y*in_grads[4*c_ids.y + l] +
												c_weights.z*in_grads[4*c_ids.z + l])
							+ (1.0f-lambda)*(c_weights_n.x*in_grads[4*c_ids_n.x + l] +
												c_weights_n.y*in_grads[4*c_ids_n.y + l] +
												c_weights_n.z*in_grads[4*c_ids_n.z + l]);
		}

		for (int l = 0; l < 8; l++) { // dim feats = 32 = 4*8
			out_feat[8*i+l] = lambda*(c_weights.x*in_feat[8*c_ids.x + l] +
												c_weights.y*in_feat[8*c_ids.y + l] +
												c_weights.z*in_feat[8*c_ids.z + l])
							+ (1.0f-lambda)*(c_weights_n.x*in_feat[8*c_ids_n.x + l] +
												c_weights_n.y*in_feat[8*c_ids_n.y + l] +
												c_weights_n.z*in_feat[8*c_ids_n.z + l]);
		}

        sample_rays[i] = ray_d;	
		out_weights[2*i] = in_weights_rays[2 * s_id];
		out_weights[2*i + 1] = in_weights_rays[2 * s_id + 1];

		if (lambda < 0.5f) {
			samples[i] = ray_o + (2.0f*lambda*out_z[i].x + (1.0f-2.0f*lambda)*out_z[i].y)*ray_d;
		} else {
			samples[i] = ray_o + ((1.0-2.0f*lambda)*out_z[i].x + (1.0f-(1.0-2.0f*lambda))*out_z[i].y)*ray_d;	
		}

        s_id++;
    }

    return;
}

/** CPU functions **/
/** CPU functions **/
/** CPU functions **/


// Ray marching in 3D
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
)   {
	
        int* counter;
        cudaMalloc((void**)&counter, sizeof(int));
        cudaMemset(counter, 0, sizeof(int));

        const int threads = 512;
        const int blocks = (num_rays + threads - 1) / threads; // ceil for example 8192 + 255 / 256 = 32
        /*AT_DISPATCH_FLOATING_TYPES( rays.type(),"tet32_march_cuda", ([&] {  
            tet32_march_cuda_kernel CUDA_KERNEL(blocks,threads) (
	 			inv_s,
                num_rays,
                num_samples,
                cam_id,
                rays.data_ptr<float>(),
                vertices.data_ptr<float>(),
                sdf.data_ptr<float>(),
                tets.data_ptr<int>(),
                nei_tets.data_ptr<int>(),
                cam_ids.data_ptr<int>(),
                offsets_cam.data_ptr<int>(),
                cam_tets.data_ptr<int>(),
                weights.data_ptr<float>(),
                z_vals.data_ptr<float>(),
                z_sdfs.data_ptr<float>(),
                z_ids.data_ptr<int>(),
                counter,
                activate.data_ptr<int>(),
                offset.data_ptr<int>()); 
    	}));*/

		AT_DISPATCH_FLOATING_TYPES( rays.type(),"tet32_march_cuda_kernel_o", ([&] {  
            tet32_march_cuda_kernel_o CUDA_KERNEL(blocks,threads) (
	 			inv_s,
                num_rays,
                num_samples,
                cam_id,
                (float3*)thrust::raw_pointer_cast(rays.data_ptr<float>()),
                (float3*)thrust::raw_pointer_cast(vertices.data_ptr<float>()),
                sdf.data_ptr<float>(),
                (int4*)thrust::raw_pointer_cast(tets.data_ptr<int>()),
                (int4*)thrust::raw_pointer_cast(nei_tets.data_ptr<int>()),
                cam_ids.data_ptr<int>(),
                offsets_cam.data_ptr<int>(),
                cam_tets.data_ptr<int>(),
                (float3*)thrust::raw_pointer_cast(weights.data_ptr<float>()),
                (float2*)thrust::raw_pointer_cast(z_vals.data_ptr<float>()),
                (float2*)thrust::raw_pointer_cast(z_sdfs.data_ptr<float>()),
                (int3*)thrust::raw_pointer_cast(z_ids.data_ptr<int>()),
                counter,
                activate.data_ptr<int>(),
                offset.data_ptr<int>()); 
    	}));
	
		cudaDeviceSynchronize();
		int res = 0;
		cudaMemcpy(&res, counter, sizeof(int), cudaMemcpyDeviceToHost);
		cudaFree(counter);

		return res;
    }

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
    torch::Tensor out_grads,     // [N_voxels, 4] for each voxel => it's vertices
    torch::Tensor out_ids,     // [N_voxels, 4] for each voxel => it's vertices
    torch::Tensor offset,     // [N_voxels, 4] for each voxel => it's vertices
    torch::Tensor samples,     // [N_voxels, 4] for each voxel => it's vertices
    torch::Tensor sample_rays     // [N_voxels, 4] for each voxel => it's vertices
)   {
        const int threads = 1024;
        const int blocks = (num_rays + threads - 1) / threads; // ceil for example 8192 + 255 / 256 = 32
        AT_DISPATCH_FLOATING_TYPES( rays_o.type(),"fill_samples_kernel_o", ([&] {  
            fill_samples_kernel_o CUDA_KERNEL(blocks,threads) (
                num_rays,
                num_samples,
                (float3*)thrust::raw_pointer_cast(rays_o.data_ptr<float>()),
                (float3*)thrust::raw_pointer_cast(rays_d.data_ptr<float>()),
                (float3*)thrust::raw_pointer_cast(sites.data_ptr<float>()),
                (float2*)thrust::raw_pointer_cast(in_z.data_ptr<float>()),
                (float2*)thrust::raw_pointer_cast(in_sdf.data_ptr<float>()),  
                (float4*)thrust::raw_pointer_cast(in_feat.data_ptr<float>()),    
                (float3*)thrust::raw_pointer_cast(in_weights.data_ptr<float>()),    
                (float3*)thrust::raw_pointer_cast(in_grads.data_ptr<float>()),     
                (int3*)thrust::raw_pointer_cast(in_ids.data_ptr<int>()),       
                (float2*)thrust::raw_pointer_cast(out_z.data_ptr<float>()),    
                (float3*)thrust::raw_pointer_cast(out_sdf.data_ptr<float>()),    
                (float4*)thrust::raw_pointer_cast(out_feat.data_ptr<float>()),    
                (float3*)thrust::raw_pointer_cast(out_weights.data_ptr<float>()),     
                (float3*)thrust::raw_pointer_cast(out_grads.data_ptr<float>()),    
                (int3*)thrust::raw_pointer_cast(out_ids.data_ptr<int>()),    
                offset.data_ptr<int>(),     
                (float3*)thrust::raw_pointer_cast(samples.data_ptr<float>()),   
                (float3*)thrust::raw_pointer_cast(sample_rays.data_ptr<float>())); 

    }));
    }