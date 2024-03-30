#include <torch/extension.h>

#include <vector>


#include <cuda.h>
#include <cuda_runtime.h>

#include <device_launch_parameters.h>

#ifdef __INTELLISENSE__
#define CUDA_KERNEL(...)
#define FAKEINIT = {0}
#else
#define CUDA_KERNEL(...) <<< __VA_ARGS__ >>>
#define FAKEINIT
#endif

#define STOP_TRANS 1.0e-8
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


__device__ float Triangle_Area_3D(float a[3], float b[3], float c[3])
{
	float ab[3] = { b[0] - a[0], b[1] - a[1], b[2] - a[2] };
	float ac[3] = { c[0] - a[0], c[1] - a[1], c[2] - a[2] };
	float cross[3] = { ab[1] * ac[2] - ab[2] * ac[1],
						ab[2] * ac[0] - ab[0] * ac[2],
						ab[0] * ac[1] - ab[1] * ac[0] };
	return sqrt(cross[0]*cross[0] + cross[1]*cross[1] + cross[2]*cross[2]) / 2.0f;
}


__device__ float get_sdf32(float* weights, float p[3], float* sites, float* sdf, int* tets, int tet_id) {
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




__global__ void tet32_march_cuda_adapt_kernel(
	const float STEP,
    const size_t num_rays,                // number of rays
    const size_t num_knn,                // number of rays
    const size_t num_samples,                // number of rays
    const size_t cam_id,
    const float *__restrict__ rays,       // [N_rays, 6]
    const int *__restrict__ neighbors,  // [N_voxels, 4] for each voxel => it's neighbors
    float *__restrict__ vertices,     // [N_voxels, 4] for each voxel => it's vertices
    float *__restrict__ sdf,     // [N_voxels, 4] for each voxel => it's vertices
    float *__restrict__ vol_feat,     // [N_voxels, 4] for each voxel => it's vertices
    int *__restrict__ tets,  
    int *__restrict__ nei_tets,  
    const int *__restrict__ cam_ids,  
    const int *__restrict__ offsets_cam,  
    const int *__restrict__ cam_tets,  
    float *__restrict__ weights_samp,     // [N_voxels, 4] for each voxel => it's vertices
    float *__restrict__ z_vals,     // [N_voxels, 4] for each voxel => it's vertices
    float *__restrict__ z_sdfs,     // [N_voxels, 4] for each voxel => it's vertices
    float *__restrict__ z_feat,     // [N_voxels, 4] for each voxel => it's vertices
    int *__restrict__ z_ids,     // [N_voxels, 4] for each voxel => it's vertices
    int *__restrict__ counter,     // [N_voxels, 4] for each voxel => it's vertices
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
	
	////////////////////////Linear interpolation//////////////////////////
	//////////////////////////////////////////////////////////////
    /*float *z_sdf_ray = &z_sdfs[idx*num_samples*2];
    float *z_feat_ray = &z_feat[idx*num_samples*12];
    float *weights_ray = &weights_samp[idx*num_samples*6];*/
	
	////////////////////////Network interpolation//////////////////////////
	//////////////////////////////////////////////////////////////
    float *z_sdf_ray = &z_sdfs[idx*num_samples*8];
    float *z_feat_ray = &z_feat[idx*num_samples*36];
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
	int prev_prev_tet_id = -1;
	int prev_tet_id = -1;

	// id_0, id_1, id_2 are indices of vertices of exit face
	int ids[4] = { 0, 0, 0, 0 };
	float weights[3] = { 0.0f, 0.0f, 0.0f };
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
	float prev_dist = 0.0f;
	int s_id = 0;
	int iter_max = 0;
	float curr_z = 0.0f;
	float curr_p[3] = { ray_o[0] + ray_d[0] * curr_z,
						ray_o[1] + ray_d[1] * curr_z,
						ray_o[2] + ray_d[2] * curr_z };
	float prev_weights[3] = { 0.0f, 0.0f, 0.0f };
	float prev_feat[6] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
	float curr_feat[6] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
	
    float prev_sdf_weights[24];// {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float curr_sdf_weights[24];// {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

	
	float length_seg, curr_seg_dist, it_seg, curr_seg_sdf, step_size;

	float prev_sdf = 20.0f;
	float curr_sdf = 20.0f;
	bool flag = false;
	float Tpartial = 1.0;
	int samples_count = 0;
	int prev_ids[3] = { 0,0,0 };
	float sigma = 0.0001f;
    float sdf_tot, weights_tot, dist;
	int knn_id;
	int prev_closest_id = 0;
	while (tet_id >= 0 && iter_max < 1000) {
		prev_ids[0] = ids[0]; prev_ids[1] = ids[1]; prev_ids[2] = ids[2]; ; prev_ids[3] = ids[3];
		prev_weights[0] = weights[0]; prev_weights[1] = weights[1]; prev_weights[2] = weights[2];
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

		/* Get sdf value */
		curr_z = curr_dist; 
		curr_p[0] = ray_o[0] + ray_d[0] * curr_z;
		curr_p[1] = ray_o[1] + ray_d[1] * curr_z;
		curr_p[2] = ray_o[2] + ray_d[2] * curr_z;
		/*float min_dist = min(curr_p[0] - bbox[0], bbox[3] - curr_p[0]);
		min_dist = min(min_dist, curr_p[1] - bbox[1]);
		min_dist = min(min_dist, bbox[4] - curr_p[1]);
		min_dist = min(min_dist, curr_p[2] - bbox[2]);
		min_dist = min(min_dist, bbox[5] - curr_p[2]);*/

		//if (min_dist > 0.1f) {
			//curr_sdf = get_sdf_triangle32(weights, curr_p, vertices, sdf, tets, ids[0], ids[1], ids[2]);
			//float alpha = sdf2Alpha32(curr_sdf, prev_sdf, inv_s);
			//float contrib = Tpartial * (1.0f - alpha);

		// compute averaged sdf

		// Search clossest id
		/*int closest_id = 0;
		float min_dist = (vertices[3*ids[0]] - curr_p[0])*(vertices[3*ids[0]] - curr_p[0]) + 
							(vertices[3*ids[0]+1] - curr_p[1])*(vertices[3*ids[0]+1] - curr_p[1])+ 
							(vertices[3*ids[0]+2] - curr_p[2])*(vertices[3*ids[0]+2] - curr_p[2]);

        for (int i = 1; i < 3; i++) {
            dist = (vertices[3*ids[i]] - curr_p[0])*(vertices[3*ids[i]] - curr_p[0]) + 
                    (vertices[3*ids[i]+1] - curr_p[1])*(vertices[3*ids[i]+1] - curr_p[1])+ 
                    (vertices[3*ids[i]+2] - curr_p[2])*(vertices[3*ids[i]+2] - curr_p[2]);
			if (dist < min_dist) {
				closest_id = i;
				min_dist = dist;
			}
		}


		weights_tot = 0.0f;
        sdf_tot = 0.0f;
        for (int i = 0; i < num_knn; i++) {
            knn_id = neighbors[num_knn*ids[closest_id] + i];
            curr_sdf_weights[i] = 0.0f;

            if (knn_id == ids[closest_id])
                continue;

            dist = (vertices[3*knn_id] - curr_p[0])*(vertices[3*knn_id] - curr_p[0]) + 
                    (vertices[3*knn_id+1] - curr_p[1])*(vertices[3*knn_id+1] - curr_p[1])+ 
                    (vertices[3*knn_id+2] - curr_p[2])*(vertices[3*knn_id+2] - curr_p[2]);

            curr_sdf_weights[i] = exp(-dist/sigma);
            sdf_tot = sdf_tot + exp(-dist/sigma) * sdf[knn_id];
            weights_tot = weights_tot + exp(-dist/sigma);
        }

        dist = (vertices[3*ids[closest_id]] - curr_p[0])*(vertices[3*ids[closest_id]] - curr_p[0]) + 
                (vertices[3*ids[closest_id]+1] - curr_p[1])*(vertices[3*ids[closest_id]+1] - curr_p[2])+ 
                (vertices[3*ids[closest_id]+2] - curr_p[2])*(vertices[3*ids[closest_id]+2] - curr_p[2]);

        curr_sdf_weights[num_knn] = exp(-dist/sigma);
        sdf_tot = sdf_tot + exp(-dist/sigma) * sdf[ids[closest_id]];
        weights_tot = weights_tot + exp(-dist/sigma);*/

		/*if (curr_dist < prev_dist) {
			offset[2 * idx] = atomicAdd(counter, s_id);
			offset[2 * idx + 1] = -1;
			return;
		}*/

        if (prev_tet_id != -1) 
            //curr_sdf = weights_tot > 0.0f ? sdf_tot/weights_tot : (sdf[ids[0]] + sdf[ids[1]] + sdf[ids[2]]) / 3.0f;
			curr_sdf = get_sdf_triangle32(weights, curr_p, vertices, sdf, tets, ids[0], ids[1], ids[2]);
			get_feat_triangle32(curr_feat, weights, curr_p, vertices, vol_feat, tets, ids[0], ids[1], ids[2]);

			if (prev_prev_tet_id != -1 && curr_dist > prev_dist && prev_sdf > curr_sdf) { //(contrib > 1.0e-10/) && prev_sdf != 20.0f) {

				/*step_size = 0.01f; // 1.0f/exp(-inv_s*min(fabs(prev_sdf), fabs(curr_sdf)))
				step_size = max(0.0001f, min(fabs(prev_sdf), fabs(curr_sdf)) / 2.0f); //min(fabs(prev_sdf), fabs(curr_sdf)) < 0.2f || prev_sdf * curr_sdf < 0.0f ? 0.01f : 0.1f;
				length_seg = (curr_dist - prev_dist)/step_size;
				curr_seg_dist = prev_dist + step_size;
				it_seg = 0.0;
				curr_seg_sdf = prev_sdf + (curr_sdf-prev_sdf) * (it_seg/length_seg);
				while(curr_seg_dist < curr_dist) {
					z_val_ray[2 * s_id] = prev_dist;
					z_val_ray[2 * s_id + 1] = curr_seg_dist;

					z_sdf_ray[2 * s_id] = prev_sdf + (curr_sdf-prev_sdf) * (it_seg/length_seg);
					z_sdf_ray[2 * s_id + 1] = prev_sdf + (curr_sdf-prev_sdf) * ((it_seg+1.0f)/length_seg);

					for (int l = 0; l < 6; l++) {
						z_feat_ray[12 * s_id + l] = prev_feat[l] + (curr_feat[l]-prev_feat[l]) * (it_seg/length_seg);
						z_feat_ray[12 * s_id + 6 + l] = prev_feat[l] + (curr_feat[l]-prev_feat[l]) * ((it_seg+1.0f)/length_seg);
					}

					z_id_ray[6 * s_id] = prev_ids[0]; 
					z_id_ray[6 * s_id + 1] = prev_ids[1];
					z_id_ray[6 * s_id + 2] = prev_ids[2]; 
					
					z_id_ray[6 * s_id + 3] = ids[0]; 
					z_id_ray[6 * s_id + 3 + 1] = ids[1];
					z_id_ray[6 * s_id + 3 + 2] = ids[2]; 
					
					weights_ray[6 * s_id] = prev_weights[0]*(1.0f - (it_seg/length_seg))*(2.0f/length_seg); //prev_weights[0] + (weights[0]-prev_weights[0]) * (it_seg/length_seg);
					weights_ray[6 * s_id + 1] = prev_weights[1]*(1.0f - (it_seg/length_seg))*(2.0f/length_seg); //prev_weights[1] + (weights[1]-prev_weights[1]) * (it_seg/length_seg);
					weights_ray[6 * s_id + 2] = prev_weights[2]*(1.0f - (it_seg/length_seg))*(2.0f/length_seg); //prev_weights[2] + (weights[2]-prev_weights[2]) * (it_seg/length_seg); 
					
					weights_ray[6 * s_id + 3] = weights[0]*((it_seg+1.0f)/length_seg)*(2.0f/length_seg); //prev_weights[0] + (weights[0]-prev_weights[0]) * ((it_seg+1.0f)/length_seg);
					weights_ray[6 * s_id + 3 + 1] = weights[1]*((it_seg+1.0f)/length_seg)*(2.0f/length_seg); //prev_weights[1] + (weights[1]-prev_weights[1]) * ((it_seg+1.0f)/length_seg);
					weights_ray[6 * s_id + 3 + 2] = weights[2]*((it_seg+1.0f)/length_seg)*(2.0f/length_seg); //prev_weights[2] + (weights[2]-prev_weights[2]) * ((it_seg+1.0f)/length_seg);

					prev_dist = curr_seg_dist;
					curr_seg_dist = prev_dist + step_size;
					it_seg += 1.0f;

					s_id++;
					if (s_id > num_samples - 1) {
						break;
					}
				}
				if (s_id > num_samples - 1) {
					break;
				}
				prev_sdf = prev_sdf + (curr_sdf-prev_sdf) * (it_seg/length_seg);

				for (int l = 0; l < 6; l++) {
					prev_feat[l] = prev_feat[l] + (curr_feat[l]-prev_feat[l]) * (it_seg/length_seg);
				}*/

				z_val_ray[2 * s_id] = prev_dist;
				z_val_ray[2 * s_id + 1] = curr_dist;
				
				
				
				////////////////////////Linear interpolation//////////////////////////
				//////////////////////////////////////////////////////////////
				/*z_sdf_ray[2 * s_id] = prev_sdf;
				z_sdf_ray[2 * s_id + 1] = curr_sdf;
				
				z_id_ray[6 * s_id] = prev_ids[0]; 
				z_id_ray[6 * s_id + 1] = prev_ids[1];
				z_id_ray[6 * s_id + 2] = prev_ids[2]; 
				
				z_id_ray[6 * s_id + 3] = ids[0]; 
				z_id_ray[6 * s_id + 3 + 1] = ids[1];
				z_id_ray[6 * s_id + 3 + 2] = ids[2]; 

				for (int l = 0; l < 6; l++) {
					z_feat_ray[12 * s_id + l] = prev_feat[l];
					z_feat_ray[12 * s_id + 6 + l] = curr_feat[l];
				}

				weights_ray[6 * s_id] = prev_weights[0];// * (1.0f - it_seg/length_seg)*(2.0f/length_seg); 
				weights_ray[6 * s_id + 1] = prev_weights[1];// * (1.0f - it_seg/length_seg)*(2.0f/length_seg);
				weights_ray[6 * s_id + 2] = prev_weights[2];// * (1.0f - it_seg/length_seg)*(2.0f/length_seg); 
				
				weights_ray[6 * s_id + 3] = weights[0];// * (2.0f/length_seg); 
				weights_ray[6 * s_id + 3 + 1] = weights[1];// * (2.0f/length_seg);
				weights_ray[6 * s_id + 3 + 2] = weights[2];// * (2.0f/length_seg); */

				
				////////////////////////Network interpolation//////////////////////////
				//////////////////////////////////////////////////////////////
				z_id_ray[6 * s_id] = prev_ids[0]; 
				z_id_ray[6 * s_id + 1] = prev_ids[1];
				z_id_ray[6 * s_id + 2] = prev_ids[2]; 				
				//z_id_ray[6 * s_id + 3] = prev_ids[3]; 

				z_id_ray[6 * s_id + 3] = ids[0]; 
				z_id_ray[6 * s_id + 3 + 1] = ids[1];
				z_id_ray[6 * s_id + 3 + 2] = ids[2]; 

				z_sdf_ray[8 * s_id] = prev_sdf;
				z_sdf_ray[8 * s_id + 1] = curr_sdf;
				z_sdf_ray[8 * s_id + 2] = sdf[prev_ids[0]];
				z_sdf_ray[8 * s_id + 3] = sdf[prev_ids[1]];
				z_sdf_ray[8 * s_id + 4] = sdf[prev_ids[2]];
				z_sdf_ray[8 * s_id + 5] = sdf[ids[0]];
				z_sdf_ray[8 * s_id + 6] = sdf[ids[1]];
				z_sdf_ray[8 * s_id + 7] = sdf[ids[2]];
				
				/*for (int k = 0; k < 4; k++) {
					for (int l = 0; l < 6; l++) {
						z_feat_ray[24 * s_id + 6*k + l] = vol_feat[6*prev_ids[k]+l];
					}
				}*/
				for (int k = 0; k < 3; k++) {
					for (int l = 0; l < 6; l++) {
						z_feat_ray[36 * s_id + 6*k + l] = vol_feat[6*prev_ids[k]+l];
						z_feat_ray[36 * s_id + 18 + 6*k + l] = vol_feat[6*ids[k]+l];
					}
				}

				weights_ray[6 * s_id] = prev_weights[0];
				weights_ray[6 * s_id + 1] = prev_weights[1];
				weights_ray[6 * s_id + 2] = prev_weights[2];
				//weights_ray[8 * s_id + 3] = 0.0f; 

				weights_ray[6 * s_id + 3] = weights[0];
				weights_ray[6 * s_id + 4] = weights[1];
				weights_ray[6 * s_id + 5] = weights[2];
				//weights_ray[8 * s_id + 7] = 0.0f; 
				
				/*weights_ray[8 * s_id + 4] = 0.0f; weights_ray[8 * s_id + 5] = 0.0f; weights_ray[8 * s_id + 6] = 0.0f; weights_ray[8 * s_id + 7] = 0.0f; 
				for(int l = 0; l < 3; l++) {
					if (prev_ids[0] == ids[l])
						weights_ray[8 * s_id + 4] = weights[l];
					if (prev_ids[1] == ids[l])
						weights_ray[8 * s_id + 5] = weights[l];
					if (prev_ids[2] == ids[l])
						weights_ray[8 * s_id + 6] = weights[l];
					if (prev_ids[3] == ids[l])
						weights_ray[8 * s_id + 7] = weights[l];
				}*/

				s_id++;
				if (s_id > num_samples - 1) {
					break;
				}
			}

			prev_sdf = curr_sdf;
			for (int l = 0; l < 6; l++)
				prev_feat[l] = curr_feat[l];

			//Tpartial = Tpartial * alpha;
			//if (Tpartial < 1.0e-10f)
			//	break;
		//}

		prev_dist = curr_dist;
        prev_prev_tet_id = prev_tet_id;
		prev_tet_id = tet_id;
		//prev_closest_id = ids[closest_id];

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

__global__ void tet32_march_cuda_kernel(
	const float STEP_IN,
	const float inv_s,
	const float sigma,
    const size_t num_rays,                // number of rays
    const size_t num_knn,                // number of rays
    const size_t num_samples,                // number of rays
    const size_t cam_id,
    const float *__restrict__ rays,       // [N_rays, 6]
    const int *__restrict__ neighbors,  // [N_voxels, 4] for each voxel => it's neighbors
    float *__restrict__ vertices,     // [N_voxels, 4] for each voxel => it's vertices
    float *__restrict__ sdf,     // [N_voxels, 4] for each voxel => it's vertices
    float *__restrict__ vol_feat,     // [N_voxels, 4] for each voxel => it's vertices
    int *__restrict__ tets,  
    int *__restrict__ nei_tets,  
    const int *__restrict__ cam_ids,  
    const int *__restrict__ offsets_cam,  
    const int *__restrict__ cam_tets,  
    float *__restrict__ weights_samp,     // [N_voxels, 4] for each voxel => it's vertices
    float *__restrict__ z_vals,     // [N_voxels, 4] for each voxel => it's vertices
    float *__restrict__ z_sdfs,     // [N_voxels, 4] for each voxel => it's vertices
    float *__restrict__ z_feat,     // [N_voxels, 4] for each voxel => it's vertices
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
    int *z_id_ray = &z_ids[idx*num_samples*12];	
    float *z_sdf_ray = &z_sdfs[idx*num_samples*2];
    float *z_feat_ray = &z_feat[idx*num_samples*2*DIM_L_FEAT];
    float *weights_ray = &weights_samp[idx*num_samples*12];

	//float STEP = 0.01f;
    
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
	int prev_prev_tet_id = -1;
	int prev_tet_id = -1;

	// id_0, id_1, id_2 are indices of vertices of exit face
	int ids[4] = { 0, 0, 0, 0 };
	float weights[6] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
	float prev_weights[6] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
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
	float prev_dist = 0.0f;
	int s_id = 0;
	int iter_max = 0;
	float curr_z = 0.0f;
	float curr_p[3] = { ray_o[0] + ray_d[0] * curr_z,
						ray_o[1] + ray_d[1] * curr_z,
						ray_o[2] + ray_d[2] * curr_z };
	float prev_weights_tet[6] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f  };
	float next_weights[6] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f  };

	float next_feat[DIM_L_FEAT] = { };
	float prev_feat_tet[DIM_L_FEAT] = { };
	float prev_feat[DIM_L_FEAT] = { };
	float curr_feat[DIM_L_FEAT] = { };
	
	int ids_s[6] = { 0, 0, 0, 0, 0, 0 };
	int prev_ids_s[6] = { 0, 0, 0, 0, 0, 0 };
		
	float next_sdf, prev_sdf_tet, prev_dist_tet = 0.0f, lambda;
	float prev_sdf = 20.0f;
	float curr_sdf = 20.0f;
	bool flag = false;
	float Tpartial = 1.0;
	int samples_count = 0;
	int prev_ids[4] = { 0,0,0,0 };
	int prev_prev_ids[4] = { 0,0,0,0 };
	//float sigma = 0.0001f;
    float sdf_tot, weights_tot, dist, dist0, dist1, dist2;
	int knn_id;
	int prev_closest_id = 0;
	float STEP = STEP_IN;
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

		//next_weights[3] = next_weights[0]; next_weights[4] = next_weights[1]; next_weights[5] = next_weights[2]; 
		next_sdf = get_sdf_triangle32(next_weights, curr_p, vertices, sdf, tets, ids[0], ids[1], ids[2]);
		get_feat_triangle32(next_feat, next_weights, curr_p, vertices, vol_feat, tets, ids[0], ids[1], ids[2]);
		ids_s[3] = ids[0]; ids_s[4] = ids[1]; ids_s[5] = ids[2]; 
		
		dist0 = (curr_p[0] - v_0[0])*(curr_p[0] - v_0[0]) + 
						(curr_p[1] - v_0[1])*(curr_p[1] - v_0[1]) + 
						(curr_p[2] - v_0[2])*(curr_p[2] - v_0[2]);

		dist1 = (curr_p[0] - v_1[0])*(curr_p[0] - v_1[0]) + 
						(curr_p[1] - v_1[1])*(curr_p[1] - v_1[1]) + 
						(curr_p[2] - v_1[2])*(curr_p[2] - v_1[2]);
						
		dist2 = (curr_p[0] - v_2[0])*(curr_p[0] - v_2[0]) + 
						(curr_p[1] - v_2[1])*(curr_p[1] - v_2[1]) + 
						(curr_p[2] - v_2[2])*(curr_p[2] - v_2[2]);

		float STEP_CURR = STEP_IN;
		if (prev_tet_id == -1) {
			curr_z = curr_dist;
			prev_dist_tet = curr_dist;
			prev_sdf_tet = next_sdf;
			for (int l = 0; l < DIM_L_FEAT; l++)
				prev_feat_tet[l] = next_feat[l];
			for (int l = 0; l < 4; l++)
				prev_weights_tet[l] = next_weights[l];
		}

		float alpha_tet = sdf2Alpha(next_sdf, prev_sdf_tet, inv_s);
		float contrib_tet = Tpartial * (1.0f - alpha_tet);
		int fact_s = int(contrib_tet * 4.0f);
		/*for (int l = 0; l < fact_s; l++)
			STEP_CURR = STEP_CURR/ 2.0f;*/

		/*if (prev_prev_tet_id != -1 && prev_dist_tet < curr_dist) {
			if (STEP > (curr_dist - prev_dist_tet) / 2.0f)
				STEP = STEP/ 2.0f;
			if (STEP > (curr_dist - prev_dist_tet) / 2.0f)
				STEP = STEP/ 2.0f;
			if (STEP > (curr_dist - prev_dist_tet) / 2.0f)
				STEP = STEP/ 2.0f;
			if (STEP > (curr_dist - prev_dist_tet) / 2.0f && ((curr_dist - prev_dist_tet) / 2.0f) > 1.0e-6f)
				STEP = (curr_dist - prev_dist_tet) / 2.0f;
			//while (STEP > (curr_dist - prev_dist_tet) / 2.0f)
			//	STEP = STEP/ 2.0f;
			//min(STEP_IN, (curr_dist - prev_dist_tet) / 2.0f);
		}*/

		//float min_sdf = min(fabs(next_sdf), fabs(prev_sdf_tet)); && prev_sdf_tet > next_sdf
		//lambda = (curr_dist - prev_dist_tet) < 1.0e-6f ? 0.0f : STEP / (curr_dist - prev_dist_tet);
		//float contrib_in = (1.0f - sdf2Alpha((1.0f-lambda)*prev_sdf_tet + lambda * next_sdf, prev_sdf_tet, inv_s));
		//float contrib_out = (1.0f - sdf2Alpha(next_sdf, (1.0f-lambda)*next_sdf + lambda * prev_sdf_tet, inv_s));
        if (prev_tet_id != -1) { //})  && lambda > 0.0f && (contrib_in > 0.0f || contrib_out > 0.0f || next_sdf*prev_sdf_tet < 0.0f)) {
			if (true) { //(curr_z + STEP > curr_dist) {
				//if (prev_sdf_tet > next_sdf) {
				if (prev_sdf_tet*next_sdf <= 0.0f || 
					(prev_sdf_tet > next_sdf && fmin(fabs(next_sdf), fabs(prev_sdf_tet))*inv_s < 2.0*CLIP_ALPHA)) {
					z_val_ray[2 * s_id] = prev_dist_tet;
					z_val_ray[2 * s_id + 1] = curr_dist;

					z_id_ray[12 * s_id] = prev_tet_id; //prev_ids_s[0]; 
					z_id_ray[12 * s_id + 1] = prev_ids_s[1];
					z_id_ray[12 * s_id + 2] = prev_ids_s[2]; 
					z_id_ray[12 * s_id + 3] = prev_ids_s[3]; 
					z_id_ray[12 * s_id + 4] = prev_ids_s[4];
					z_id_ray[12 * s_id + 5] = prev_ids_s[5]; 
					
					z_id_ray[12 * s_id + 6] = ids_s[0]; 
					z_id_ray[12 * s_id + 7] = ids_s[1];
					z_id_ray[12 * s_id + 8] = ids_s[2]; 
					z_id_ray[12 * s_id + 9] = ids_s[3]; 
					z_id_ray[12 * s_id + 10] = ids_s[4];
					z_id_ray[12 * s_id + 11] = ids_s[5]; 

					z_sdf_ray[2 * s_id] = prev_sdf_tet;
					z_sdf_ray[2 * s_id + 1] = next_sdf;
					
					for (int l = 0; l < DIM_L_FEAT; l++) {
						z_feat_ray[2*DIM_L_FEAT * s_id + l] = prev_feat_tet[l];
						z_feat_ray[2*DIM_L_FEAT * s_id + DIM_L_FEAT + l] = next_feat[l];
					}

					weights_ray[12 * s_id] = prev_weights_tet[0];
					weights_ray[12 * s_id + 1] = prev_weights_tet[1];
					weights_ray[12 * s_id + 2] = prev_weights_tet[2];
					weights_ray[12 * s_id + 3] = prev_weights_tet[3]; 
					weights_ray[12 * s_id + 4] = prev_weights_tet[4]; 
					weights_ray[12 * s_id + 5] = prev_weights_tet[5]; 

					weights_ray[12 * s_id + 6] = next_weights[0];//*exp(-dist0/(sigma*sigma));
					weights_ray[12 * s_id + 7] = next_weights[1];//*exp(-dist1/(sigma*sigma));
					weights_ray[12 * s_id + 8] = next_weights[2];//*exp(-dist2/(sigma*sigma));
					weights_ray[12 * s_id + 9] = next_weights[3];
					weights_ray[12 * s_id + 10] = next_weights[4];
					weights_ray[12 * s_id + 11] = next_weights[5];

					// activate sites here
					for (int l = 0; l < 6; l++) {
						activate[ids_s[l]] = 1;
						//atomicExch(&activate[ids_s[l]], 1);
					}

					s_id++;
					if (s_id > num_samples - 1) {
						break;
					}
				}

				prev_dist = curr_dist;
				prev_sdf = next_sdf;
				for (int l = 0; l < DIM_L_FEAT; l++)
					prev_feat[l] = next_feat[l];
				for (int l = 0; l < 6; l++) {
					prev_weights[l] = next_weights[l];
					prev_ids_s[l] = ids_s[l];
				}
				
			}

			if (s_id > num_samples - 1) {
				break;
			}
			
			/*while (curr_z + STEP <= curr_dist) {
				/// Get sdf value
				curr_z += STEP; 
				//STEP = STEP_CURR;
				curr_p[0] = ray_o[0] + ray_d[0] * curr_z;
				curr_p[1] = ray_o[1] + ray_d[1] * curr_z;
				curr_p[2] = ray_o[2] + ray_d[2] * curr_z;

				lambda = (curr_dist - prev_dist_tet) < 1.0e-10f ? 0.0f : (curr_z-prev_dist_tet) / (curr_dist - prev_dist_tet);
				curr_sdf = (1.0f-lambda)*prev_sdf_tet + lambda * next_sdf;
				for (int l = 0; l < 6; l++) {
					curr_feat[l] = (1.0f-lambda)*prev_feat_tet[l] + lambda * next_feat[l];
				}

				for (int l = 0; l < 3; l++) {
					weights[l] = (1.0f-lambda)*prev_weights_tet[l];
					weights[3+l] = lambda * next_weights[l];
				}
				
				//double sdf_prev_clamp = fmin(CLIP_ALPHA, fmax(double(prev_sdf * inv_s), -CLIP_ALPHA));
				//double sdf_clamp = fmin(CLIP_ALPHA, fmax(double(curr_sdf * inv_s), -CLIP_ALPHA));
				//double inv_clipped_p = (fabs(prev_sdf) < CLIP_ALPHA / inv_s) ? double(inv_s) : sdf_prev_clamp / double(prev_sdf);
				//double inv_clipped = (fabs(curr_sdf) < CLIP_ALPHA / inv_s) ? double(inv_s) : sdf_clamp / double(curr_sdf);
				//alpha = min(1.0f, __double2float_rn((1.0 + exp(-sdf_prev_clamp)) / (1.0 + exp(-sdf_clamp))));
				//float dalpha_dsdf_p = __double2float_rn(-inv_clipped_p * exp(-sdf_prev_clamp) / (1.0 + exp(-sdf_clamp)));
				//float dalpha_dsdf_n = __double2float_rn((1.0 + exp(-sdf_prev_clamp)) * ((inv_clipped * exp(-sdf_clamp)) / ((1.0 + exp(-sdf_clamp)) * (1.0 + exp(-sdf_clamp)))));
		
				//float alpha = sdf2Alpha(curr_sdf, prev_sdf, inv_s);
				//if ((alpha < 1.0f) && (prev_sdf > curr_sdf)) {
				if (prev_sdf*curr_sdf <= 0.0f || 
					(prev_sdf > curr_sdf && (fabs(curr_sdf)*inv_s < 3.0*CLIP_ALPHA || fabs(prev_sdf)*inv_s < 3.0*CLIP_ALPHA))) {

					z_val_ray[2 * s_id] = prev_dist;
					z_val_ray[2 * s_id + 1] = curr_z;

					z_id_ray[12 * s_id] = prev_ids_s[0]; 
					z_id_ray[12 * s_id + 1] = prev_ids_s[1];
					z_id_ray[12 * s_id + 2] = prev_ids_s[2]; 
					z_id_ray[12 * s_id + 3] = prev_ids_s[3]; 
					z_id_ray[12 * s_id + 4] = prev_ids_s[4];
					z_id_ray[12 * s_id + 5] = prev_ids_s[5]; 
					
					z_id_ray[12 * s_id + 6] = ids_s[0]; 
					z_id_ray[12 * s_id + 7] = ids_s[1];
					z_id_ray[12 * s_id + 8] = ids_s[2]; 
					z_id_ray[12 * s_id + 9] = ids_s[3]; 
					z_id_ray[12 * s_id + 10] = ids_s[4];
					z_id_ray[12 * s_id + 11] = ids_s[5]; 

					z_sdf_ray[2 * s_id] = prev_sdf;
					z_sdf_ray[2 * s_id + 1] = curr_sdf;
					
					for (int l = 0; l < 6; l++) {
						z_feat_ray[12 * s_id + l] = prev_feat[l];
						z_feat_ray[12 * s_id + 6 + l] = curr_feat[l];
					}

					weights_ray[12 * s_id] = prev_weights[0];
					weights_ray[12 * s_id + 1] = prev_weights[1];
					weights_ray[12 * s_id + 2] = prev_weights[2];
					weights_ray[12 * s_id + 3] = prev_weights[3]; 
					weights_ray[12 * s_id + 4] = prev_weights[4]; 
					weights_ray[12 * s_id + 5] = prev_weights[5]; 

					weights_ray[12 * s_id + 6] = weights[0];
					weights_ray[12 * s_id + 7] = weights[1];
					weights_ray[12 * s_id + 8] = weights[2];
					weights_ray[12 * s_id + 9] = weights[3];
					weights_ray[12 * s_id + 10] = weights[4];
					weights_ray[12 * s_id + 11] = weights[5];

					s_id++;
					if (s_id > num_samples - 1) {
						break;
					}
				}

				prev_dist = curr_z;
				prev_sdf = curr_sdf;
				for (int l = 0; l < 6; l++)
					prev_feat[l] = curr_feat[l];
				for (int l = 0; l < 6; l++) {
					prev_weights[l] = weights[l];
					prev_ids_s[l] = ids_s[l];
				}

				//Tpartial = Tpartial * alpha;
				//if (Tpartial < 1.0e-10f)
				//	break;
			}
			
			if (s_id > num_samples - 1) {
				break;
			}*/
		}
		
		Tpartial = Tpartial * alpha_tet;

		if (Tpartial < STOP_TRANS) {// stop if the transmittance is low
            break;
        }

		prev_dist_tet = curr_dist;
		prev_sdf_tet = next_sdf;
		for (int l = 0; l < DIM_L_FEAT; l++)
			prev_feat_tet[l] = next_feat[l];
		for (int l = 0; l < 4; l++)
			prev_weights_tet[l] = next_weights[l];
		prev_weights_tet[0] = prev_weights_tet[0];//*exp(-dist0/(sigma*sigma));
		prev_weights_tet[1] = prev_weights_tet[1];//*exp(-dist1/(sigma*sigma));
		prev_weights_tet[2] = prev_weights_tet[2];//*exp(-dist2/(sigma*sigma));
        prev_prev_tet_id = prev_tet_id;
		prev_tet_id = tet_id;
		//prev_closest_id = ids[closest_id];

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


__global__ void fill_samples_adapt_kernel(
    const size_t num_rays,                // number of rays
    const size_t num_samples,                // number of rays
    const float *__restrict__ rays_o,       // [N_rays, 6]
    const float *__restrict__ rays_d,       // [N_rays, 6]
    const float *__restrict__ sites,       // [N_rays, 6]
    float *__restrict__ in_z,       // [N_rays, 6]
    float *__restrict__ in_sdf,       // [N_rays, 6]
    float *__restrict__ in_feat,       // [N_rays, 6]
    float *__restrict__ in_weights,     // [N_voxels, 4] for each voxel => it's vertices
    int *__restrict__ in_ids,       // [N_rays, 6]
    float *__restrict__ out_z,     // [N_voxels, 4] for each voxel => it's vertices
    float *__restrict__ out_sdf,     // [N_voxels, 4] for each voxel => it's vertices
    float *__restrict__ out_feat,     // [N_voxels, 4] for each voxel => it's vertices
    float *__restrict__ out_weights,     // [N_voxels, 4] for each voxel => it's vertices
    int *__restrict__ out_ids,     // [N_voxels, 4] for each voxel => it's vertices
    int *__restrict__ offset,     // [N_voxels, 4] for each voxel => it's vertices
    float *__restrict__ samples,     // [N_voxels, 4] for each voxel => it's vertices
    float *__restrict__ samples_loc,     // [N_voxels, 4] for each voxel => it's vertices
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

	////////////////////////Linear interpolation//////////////////////////
	//////////////////////////////////////////////////////////////
    /*float* in_sdf_rays = &in_sdf[2*num_samples * idx];
    float* in_feat_rays = &in_feat[12*num_samples * idx];
    float* in_weights_rays = &in_weights[6*num_samples * idx];*/

	////////////////////////Network interpolation//////////////////////////
	//////////////////////////////////////////////////////////////
    float* in_sdf_rays = &in_sdf[8*num_samples * idx];
    float* in_feat_rays = &in_feat[36*num_samples * idx];
    float* in_weights_rays = &in_weights[6*num_samples * idx];

    int start = offset[2*idx];
    int end = offset[2*idx+1];
    int s_id = 0;
    for (int i = start; i < start+end; i++) {
        out_z[2*i] = in_z_rays[2*s_id];
		out_z[2*i + 1] = in_z_rays[2*s_id+1];
        
        out_ids[6*i] = in_ids_rays[6 * s_id];
        out_ids[6*i+1] = in_ids_rays[6 * s_id+1];
        out_ids[6*i+2] = in_ids_rays[6 * s_id+2];
		
        out_ids[6*i + 3] = in_ids_rays[6 * s_id + 3];
        out_ids[6*i + 3 +1] = in_ids_rays[6 * s_id + 3 +1];
        out_ids[6*i + 3 +2] = in_ids_rays[6 * s_id + 3 +2];

        /*samples[3 * i] = ray.origin[0] + out_z[2*i]*ray.direction[0];
        samples[3 * i + 1] = ray.origin[1] + out_z[2*i]*ray.direction[1];
        samples[3 * i + 2] = ray.origin[2] + out_z[2*i]*ray.direction[2];

        samples_loc[3 * i] = ray.origin[0] + out_z[2*i+1]*ray.direction[0];
        samples_loc[3 * i + 1] = ray.origin[1] + out_z[2*i+1]*ray.direction[1];
        samples_loc[3 * i + 2] = ray.origin[2] + out_z[2*i+1]*ray.direction[2];
		
		
		for (int l = 0; l < 6; l++) {
			out_feat[12*i+l] = in_feat_rays[12 * s_id+l];
			out_feat[12*i+6+l] = in_feat_rays[12 * s_id+6+l];
		}*/
        
        samples_loc[3 * i] = ray.origin[0]; //samples[3 * i] - sites[3*out_ids[6*i+1]];
        samples_loc[3 * i + 1] = ray.origin[1]; //samples[3 * i + 1] - sites[3*out_ids[6*i+1] + 1];
        samples_loc[3 * i + 2] = ray.origin[2]; //samples[3 * i + 2] - sites[3*out_ids[6*i+1] + 2];

        sample_rays[3 * i] = ray.direction[0];
        sample_rays[3 * i + 1] = ray.direction[1];
        sample_rays[3 * i + 2] = ray.direction[2];		

		//////////////////////Linear interpolation///////////////////////////
		//////////////////////////////////////////////////////////////
		
        /*
		float lambda = 0.5f;
		if (out_sdf[2*i]*out_sdf[2*i+1] <= 0.0f) {
			lambda = fabs(out_sdf[2*i+1])/(fabs(out_sdf[2*i])+fabs(out_sdf[2*i+1]));
		}

        samples[3 * i] = ray.origin[0] + (lambda*out_z[2*i] + (1.0f-lambda)*out_z[2*i+1])*ray.direction[0];
        samples[3 * i + 1] = ray.origin[1] + (lambda*out_z[2*i] + (1.0f-lambda)*out_z[2*i+1])*ray.direction[1];
        samples[3 * i + 2] = ray.origin[2] + (lambda*out_z[2*i] + (1.0f-lambda)*out_z[2*i+1])*ray.direction[2];
		
		out_sdf[2*i] = in_sdf_rays[2 * s_id];
        out_sdf[2*i+1] = in_sdf_rays[2 * s_id+1];

		for (int l = 0; l < 6; l++) {
			out_feat[12*i+l] = lambda*in_feat_rays[12 * s_id+l] + (1.0f-lambda)*in_feat_rays[12 * s_id+6+l];
			out_feat[12*i+6+l] = in_feat_rays[12 * s_id+6+l];
		}
        
        out_weights[7*i] = in_weights_rays[6 * s_id];
        out_weights[7*i+1] = in_weights_rays[6 * s_id+1];
        out_weights[7*i+2] = in_weights_rays[6 * s_id+2];
		
        out_weights[7*i + 3] = in_weights_rays[6 * s_id + 3];
        out_weights[7*i + 3 +1] = in_weights_rays[6 * s_id + 3 +1];
        out_weights[7*i + 3 +2] = in_weights_rays[6 * s_id + 3 +2];

        out_weights[7*i + 6] = lambda;*/

		////////////////////////Network interpolation//////////////////////////
		//////////////////////////////////////////////////////////////

		for (int l = 0; l < 8; l++) {
        	out_sdf[8*i + l] = in_sdf_rays[8 * s_id + l];
		}


		float lambda = 0.5f;
		if (out_sdf[8*i]*out_sdf[8*i+1] <= 0.0f) {
			lambda = fabs(out_sdf[8*i+1])/(fabs(out_sdf[8*i])+fabs(out_sdf[8*i+1]));
			if (lambda < 0.5f) {
				out_sdf[8*i] = 2.0f*lambda*out_sdf[8*i] + (1.0f-2.0f*lambda)*out_sdf[8*i+1];
			} else {
				out_sdf[8*i+1] = (1.0-2.0f*lambda)*out_sdf[8*i] + (1.0f-(1.0-2.0f*lambda))*out_sdf[8*i+1];
			}
		}

        samples[3 * i] = ray.origin[0] + (lambda*out_z[2*i] + (1.0f-lambda)*out_z[2*i+1])*ray.direction[0];
        samples[3 * i + 1] = ray.origin[1] + (lambda*out_z[2*i] + (1.0f-lambda)*out_z[2*i+1])*ray.direction[1];
        samples[3 * i + 2] = ray.origin[2] + (lambda*out_z[2*i] + (1.0f-lambda)*out_z[2*i+1])*ray.direction[2];

		for (int l = 0; l < 36; l++) {
			out_feat[39*i+l] = in_feat_rays[36*s_id+l];
		}
		
		for (int l = 0; l < 3; l++) {
			float in_feat_val = in_feat_rays[36*s_id + l]*in_weights_rays[6 * s_id + 0] + 
								in_feat_rays[36*s_id + 6 + l]*in_weights_rays[6 * s_id + 1] + 
								in_feat_rays[36*s_id + 12 + l]*in_weights_rays[6 * s_id + 2];
			float out_feat_val = in_feat_rays[36*s_id + 18+l]*in_weights_rays[6 * s_id + 3] + 
									in_feat_rays[36*s_id + 24 + l]*in_weights_rays[6 * s_id + 4] + 
									in_feat_rays[36*s_id + 30 + l]*in_weights_rays[6 * s_id + 5];
			out_feat[39*i+36+l] = (lambda*in_feat_val + (1.0f-lambda)*out_feat_val);
		}

		for (int l = 0; l < 6; l++) {
        	out_weights[7*i + l] = in_weights_rays[6 * s_id + l];
		}
		out_weights[7*i + 6] = lambda;

        s_id++;
    }

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
    float *__restrict__ samples_loc,     // [N_voxels, 4] for each voxel => it's vertices
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
    int* in_ids_rays = &in_ids[12*num_samples * idx];
    float* in_sdf_rays = &in_sdf[2*num_samples * idx];
    float* in_feat_rays = &in_feat[2*DIM_L_FEAT*num_samples * idx];
    float* in_weights_rays = &in_weights[12*num_samples * idx];

    int start = offset[2*idx];
    int end = offset[2*idx+1];
    int s_id = 0;
    for (int i = start; i < start+end; i++) {
        out_z[2*i] = in_z_rays[2*s_id];
		out_z[2*i + 1] = in_z_rays[2*s_id+1];

        out_sdf[2*i] = in_sdf_rays[2*s_id];
		out_sdf[2*i + 1] = in_sdf_rays[2*s_id+1];

		for (int l = 0; l < 3; l++) {
			out_grads[6*i + l] = in_weights_rays[12 * s_id + 0]*in_grads[3*in_ids_rays[12 * s_id + 6 + 0] + l] +
							in_weights_rays[12 * s_id + 1]*in_grads[3*in_ids_rays[12 * s_id + 6 + 1] + l] +
							in_weights_rays[12 * s_id + 2]*in_grads[3*in_ids_rays[12 * s_id + 6 + 2] + l];
							
			out_grads[6*i + 3 + l] = in_weights_rays[12 * s_id + 6 + 0]*in_grads[3*in_ids_rays[12 * s_id + 9 + 0] + l] +
							in_weights_rays[12 * s_id + 6 + 1]*in_grads[3*in_ids_rays[12 * s_id + 9 + 1] + l] +
							in_weights_rays[12 * s_id + 6 + 2]*in_grads[3*in_ids_rays[12 * s_id + 9 + 2] + l];
		}
		
		float lambda = 0.5f;
		if (out_sdf[2*i]*out_sdf[2*i+1] <= 0.0f) {
			lambda = fabs(out_sdf[2*i+1])/(fabs(out_sdf[2*i])+fabs(out_sdf[2*i+1]));
			if (lambda < 0.5f) {
				out_sdf[2*i] = 2.0f*lambda*out_sdf[2*i] + (1.0f-2.0f*lambda)*out_sdf[2*i+1];
			} else {
				out_sdf[2*i+1] = (1.0-2.0f*lambda)*out_sdf[2*i] + (1.0f-(1.0-2.0f*lambda))*out_sdf[2*i+1];
			}
		}

		for (int l = 0; l < 12; l++) {
        	out_ids[12*i + l] = in_ids_rays[12 * s_id + l];
		}

		/*for (int l = 0; l < DIM_L_FEAT; l++) {
			out_feat[DIM_L_FEAT*i+l] = (lambda*in_feat_rays[2*DIM_L_FEAT*s_id + l] + (1.0f-lambda)*in_feat_rays[2*DIM_L_FEAT*s_id + DIM_L_FEAT + l]);
		}*/

		for (int l = 0; l < DIM_L_FEAT; l++) {
			out_feat[2*DIM_L_FEAT*i+l] = in_feat_rays[2*DIM_L_FEAT*s_id + l];
			out_feat[2*DIM_L_FEAT*i+ DIM_L_FEAT + l] = in_feat_rays[2*DIM_L_FEAT*s_id + DIM_L_FEAT + l];
		}

        samples_loc[6 * i] = ray.origin[0] + out_z[2*i]*ray.direction[0]; //samples[3 * i] - sites[3*out_ids[6*i+1]];
        samples_loc[6 * i + 1] = ray.origin[1] + out_z[2*i]*ray.direction[1]; //samples[3 * i + 1] - sites[3*out_ids[6*i+1] + 1];
        samples_loc[6 * i + 2] = ray.origin[2] + out_z[2*i]*ray.direction[2]; //samples[3 * i + 2] - sites[3*out_ids[6*i+1] + 2];

        samples_loc[6 * i + 3] = ray.origin[0] + out_z[2*i + 1]*ray.direction[0]; //samples[3 * i] - sites[3*out_ids[6*i+1]];
        samples_loc[6 * i + 4] = ray.origin[1] + out_z[2*i + 1]*ray.direction[1]; //samples[3 * i + 1] - sites[3*out_ids[6*i+1] + 1];
        samples_loc[6 * i + 5] = ray.origin[2] + out_z[2*i + 1]*ray.direction[2]; //samples[3 * i + 2] - sites[3*out_ids[6*i+1] + 2];

        sample_rays[3 * i] = ray.direction[0];
        sample_rays[3 * i + 1] = ray.direction[1];
        sample_rays[3 * i + 2] = ray.direction[2];		

		for (int l = 0; l < 12; l++) {
        	out_weights[13*i + l] = in_weights_rays[12 * s_id + l];
		}
		out_weights[13*i + 12] = lambda;

		//if (out_sdf[2*i]*out_sdf[2*i+1] > 0.0f) {
		//	lambda = 0.5f;//0.1f + 0.8f*float((i+idx)%1000)/1000.0f;
		//}

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

/** CPU functions **/
/** CPU functions **/
/** CPU functions **/


// Ray marching in 3D
int tet32_march_cuda(
	float STEP,
	float inv_s,
	float sigma,
    size_t num_rays,
    size_t num_knn,
    size_t num_samples,
    size_t cam_id,
    torch::Tensor rays,      // [N_rays, 6]
    torch::Tensor neighbors, // [N_voxels, 26] for each voxel => it's neighbors
    torch::Tensor vertices, // [N_voxels, 26] for each voxel => it's neighbors
    torch::Tensor sdf, // [N_voxels, 26] for each voxel => it's neighbors
    torch::Tensor vol_feat, // [N_voxels, 26] for each voxel => it's neighbors
    torch::Tensor tets, // [N_voxels, 26] for each voxel => it's neighbors
    torch::Tensor nei_tets, // [N_voxels, 26] for each voxel => it's neighbors
    torch::Tensor cam_ids,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor offsets_cam,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor cam_tets,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor weights,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor z_vals,     // [N_voxels, 4] for each voxel => it's vertices
    torch::Tensor z_sdfs,     // [N_voxels, 4] for each voxel => it's vertices
    torch::Tensor z_feat,     // [N_voxels, 4] for each voxel => it's vertices
    torch::Tensor z_ids,     // [N_voxels, 4] for each voxel => it's vertices
    torch::Tensor activate,     // [N_voxels, 4] for each voxel => it's vertices
    torch::Tensor offset     // [N_voxels, 4] for each voxel => it's vertices
)   {
	
        int* counter;
        cudaMalloc((void**)&counter, sizeof(int));
        cudaMemset(counter, 0, sizeof(int));

        const int threads = 256;
        const int blocks = (num_rays + threads - 1) / threads; // ceil for example 8192 + 255 / 256 = 32
        AT_DISPATCH_FLOATING_TYPES( rays.type(),"tet32_march_cuda", ([&] {  
            tet32_march_cuda_kernel CUDA_KERNEL(blocks,threads) (
				STEP,
	 			inv_s,
	 			sigma,
                num_rays,
    			num_knn,
                num_samples,
                cam_id,
                rays.data_ptr<float>(),
                neighbors.data_ptr<int>(),
                vertices.data_ptr<float>(),
                sdf.data_ptr<float>(),
                vol_feat.data_ptr<float>(),
                tets.data_ptr<int>(),
                nei_tets.data_ptr<int>(),
                cam_ids.data_ptr<int>(),
                offsets_cam.data_ptr<int>(),
                cam_tets.data_ptr<int>(),
                weights.data_ptr<float>(),
                z_vals.data_ptr<float>(),
                z_sdfs.data_ptr<float>(),
                z_feat.data_ptr<float>(),
                z_ids.data_ptr<int>(),
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
    torch::Tensor samples_loc,     // [N_voxels, 4] for each voxel => it's vertices
    torch::Tensor sample_rays     // [N_voxels, 4] for each voxel => it's vertices
)   {
        const int threads = 1024;
        const int blocks = (num_rays + threads - 1) / threads; // ceil for example 8192 + 255 / 256 = 32
        AT_DISPATCH_FLOATING_TYPES( rays_o.type(),"fill_samples_kernel", ([&] {  
            fill_samples_kernel CUDA_KERNEL(blocks,threads) (
                num_rays,
                num_samples,
                rays_o.data_ptr<float>(),
                rays_d.data_ptr<float>(),
                sites.data_ptr<float>(),
                in_z.data_ptr<float>(),
                in_sdf.data_ptr<float>(),  
                in_feat.data_ptr<float>(),    
                in_weights.data_ptr<float>(),    
                in_grads.data_ptr<float>(),     
                in_ids.data_ptr<int>(),       
                out_z.data_ptr<float>(),    
                out_sdf.data_ptr<float>(),    
                out_feat.data_ptr<float>(),    
                out_weights.data_ptr<float>(),     
                out_grads.data_ptr<float>(),    
                out_ids.data_ptr<int>(),    
                offset.data_ptr<int>(),     
                samples.data_ptr<float>(),   
                samples_loc.data_ptr<float>(),    
                sample_rays.data_ptr<float>()); 

    }));
    }