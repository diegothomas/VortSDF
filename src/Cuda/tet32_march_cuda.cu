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



/** Device functions **/
/** Device functions **/
/** Device functions **/

struct Ray
{
    float origin[3];
    float direction[3];
};

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


__device__ float get_sdf32(float p[3], float* sites, float* sdf, int* tets, int tet_id) {
	int id0 = tets[4 * tet_id];
	int id1 = tets[4 * tet_id + 1];
	int id2 = tets[4 * tet_id + 2];
	int id3 = id0 ^ id1 ^ id2 ^ tets[4 * tet_id + 3];
	float tot_vol = volume_tetrahedron_32(&sites[3 * id0], &sites[3 * id1],
		&sites[3 * id2], &sites[3 * id3]);

	float w_1 = tot_vol == 0.0f ? 0.25f : volume_tetrahedron_32(p, &sites[3 * id1],
		&sites[3 * id2], &sites[3 * id3]) / tot_vol;
	float w_2 = tot_vol == 0.0f ? 0.25f : volume_tetrahedron_32(p, &sites[3 * id0],
		&sites[3 * id2], &sites[3 * id3]) / tot_vol;
	float w_3 = tot_vol == 0.0f ? 0.25f : volume_tetrahedron_32(p, &sites[3 * id0],
		&sites[3 * id1], &sites[3 * id3]) / tot_vol;
	float w_4 = tot_vol == 0.0f ? 0.25f : volume_tetrahedron_32(p, &sites[3 * id0],
		&sites[3 * id1], &sites[3 * id2]) / tot_vol;

	float sum_weights = w_1 + w_2 + w_3 + w_4;
	if (sum_weights > 0.0f) {
		w_1 = w_1 / sum_weights;
		w_2 = w_2 / sum_weights;
		w_3 = w_3 / sum_weights;
		w_4 = w_4 / sum_weights;
	}
	else {
		w_1 = 0.25f;
		w_2 = 0.25f;
		w_3 = 0.25f;
		w_4 = 0.25f;
	}

	return sdf[id0] * w_1 + sdf[id1] * w_2 +
		sdf[id2] * w_3 + sdf[id3] * w_4;
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
	if (!((weights[0] >= 0.0f && weights[0] <= 1.0f) &&
		(weights[1] >= 0.0f && weights[1] <= 1.0f) &&
		(weights[2] >= 0.0f && weights[2] <= 1.0f) &&
		fabs((weights[0] + weights[1] + weights[2]) - 1.0f) < 1.0e-4f)) {
		return 20.0f;
	}
	else {
		weights[0] = weights[0] / w_tot;
		weights[1] = weights[1] / w_tot;
		weights[2] = weights[2] / w_tot;
	}

	return sdf[id0] * weights[0] + sdf[id1] * weights[1] + sdf[id2] * weights[2];
}


__global__ void tet32_march_cuda_kernel(
    const size_t num_rays,                // number of rays
    const size_t num_knn,                // number of rays
    const size_t num_samples,                // number of rays
    const size_t cam_id,
    const float *__restrict__ rays,       // [N_rays, 6]
    const int *__restrict__ neighbors,  // [N_voxels, 4] for each voxel => it's neighbors
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
    float *z_sdf_ray = &z_sdfs[idx*num_samples*2];
    float *weights_ray = &weights_samp[idx*num_samples*2*(num_knn+1)];
    int *z_id_ray = &z_ids[idx*num_samples*3];
    
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

    for (int i = 0; i < cam_adj_count; i++) {
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
			if (get_sdf32(v_new, vertices, sdf, tets, prev_tet_id) != 30.0f) 
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
	float weights[3] = { 0.0f, 0.0f, 0.0f };
	float prev_feat[6] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
	float curr_feat[6] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
	
    float prev_sdf_weights[24];// {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float curr_sdf_weights[24];// {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

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
	while (tet_id >= 0 && iter_max < 10000) {
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
		int closest_id = 0;
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
        weights_tot = weights_tot + exp(-dist/sigma);

        if (prev_tet_id != -1) 
            curr_sdf = weights_tot > 0.0f ? sdf_tot/weights_tot : (sdf[ids[0]] + sdf[ids[1]] + sdf[ids[2]]) / 3.0f;
			//curr_sdf = get_sdf_triangle32(weights, curr_p, vertices, sdf, tets, ids[0], ids[1], ids[2]);

			if (prev_prev_tet_id != -1) { //(contrib > 1.0e-10/) && prev_sdf != 20.0f) {
				//get_feat_triangle32(curr_feat, weights, curr_p, vertices, vol_feat, tets, ids[0], ids[1], ids[2]);

				z_val_ray[2 * s_id] = prev_dist;
				z_val_ray[2 * s_id + 1] = curr_dist;

				z_sdf_ray[2 * s_id] = prev_sdf;
				z_sdf_ray[2 * s_id + 1] = curr_sdf;

				z_id_ray[3 * s_id] = prev_closest_id; //prev_prev_tet_id;
				z_id_ray[3 * s_id + 1] = prev_tet_id;
				z_id_ray[3 * s_id + 2] = ids[closest_id]; //tet_id;

				for (int i = 0; i < num_knn+1; i++) {
					weights_ray[2*(num_knn+1)*s_id + i] = prev_sdf_weights[i];
					weights_ray[2*(num_knn+1)*s_id + i + num_knn+1] = weights_tot > 0.0f ? curr_sdf_weights[i]/weights_tot : 0.0f;
					prev_sdf_weights[i] = weights_tot > 0.0f ? curr_sdf_weights[i]/weights_tot : 0.0f;
				}

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
		prev_closest_id = ids[closest_id];

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

__global__ void fill_samples_kernel(
    const size_t num_rays,                // number of rays
    const size_t num_samples,                // number of rays
    const float *__restrict__ rays_o,       // [N_rays, 6]
    const float *__restrict__ rays_d,       // [N_rays, 6]
    const float *__restrict__ sites,       // [N_rays, 6]
    float *__restrict__ in_z,       // [N_rays, 6]
    float *__restrict__ in_sdf,       // [N_rays, 6]
    float *__restrict__ in_weights,     // [N_voxels, 4] for each voxel => it's vertices
    int *__restrict__ in_ids,       // [N_rays, 6]
    float *__restrict__ out_z,     // [N_voxels, 4] for each voxel => it's vertices
    float *__restrict__ out_sdf,     // [N_voxels, 4] for each voxel => it's vertices
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
    float* in_sdf_rays = &in_sdf[2*num_samples * idx];
    float* in_weights_rays = &in_weights[2*(num_knn+1)*num_samples * idx];
    int* in_ids_rays = &in_ids[3*num_samples * idx];

    int start = offset[2*idx];
    int end = offset[2*idx+1];
    int s_id = 0;
    for (int i = start; i < start+end; i++) {
        out_z[i] = in_z_rays[2*s_id]; //(in_z_rays[2*s_id] + in_z_rays[2*s_id+1])/2.0f;

        out_sdf[2*i] = in_sdf_rays[2 * s_id];
        out_sdf[2*i+1] = in_sdf_rays[2 * s_id+1];
        
        out_ids[3*i] = in_ids_rays[3 * s_id];
        out_ids[3*i+1] = in_ids_rays[3 * s_id+1];
        out_ids[3*i+2] = in_ids_rays[3 * s_id+2];

        samples[3 * i] = ray.origin[0] + out_z[i]*ray.direction[0];
        samples[3 * i + 1] = ray.origin[1] + out_z[i]*ray.direction[1];
        samples[3 * i + 2] = ray.origin[2] + out_z[i]*ray.direction[2];
        
        samples_loc[3 * i] = samples[3 * i] - sites[3*out_ids[3*i+1]];
        samples_loc[3 * i + 1] = samples[3 * i + 1] - sites[3*out_ids[3*i+1] + 1];
        samples_loc[3 * i + 2] = samples[3 * i + 2] - sites[3*out_ids[3*i+1] + 2];

        sample_rays[3 * i] = ray.direction[0];
        sample_rays[3 * i + 1] = ray.direction[1];
        sample_rays[3 * i + 2] = ray.direction[2];

        for (int j = 0; j < num_knn+1; j++) {
            out_weights[2*(num_knn+1)*i + j] = in_weights_rays[2*(num_knn+1)*s_id + j];
            out_weights[2*(num_knn+1)*i + j + num_knn+1] = in_weights_rays[2*(num_knn+1)*s_id + j + num_knn+1];
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
    size_t num_rays,
    size_t num_knn,
    size_t num_samples,
    size_t cam_id,
    torch::Tensor rays,      // [N_rays, 6]
    torch::Tensor neighbors, // [N_voxels, 26] for each voxel => it's neighbors
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
    torch::Tensor offset     // [N_voxels, 4] for each voxel => it's vertices
)   {
	
        int* counter;
        cudaMalloc((void**)&counter, sizeof(int));
        cudaMemset(counter, 0, sizeof(int));

        const int threads = 256;
        const int blocks = (num_rays + threads - 1) / threads; // ceil for example 8192 + 255 / 256 = 32
        AT_DISPATCH_FLOATING_TYPES( rays.type(),"tet32_march_cuda", ([&] {  
            tet32_march_cuda_kernel CUDA_KERNEL(blocks,threads) (
                num_rays,
    			num_knn,
                num_samples,
                cam_id,
                rays.data_ptr<float>(),
                neighbors.data_ptr<int>(),
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
    torch::Tensor in_weights,       // [N_rays, 6]
    torch::Tensor in_ids,       // [N_rays, 6]
    torch::Tensor out_z,     // [N_voxels, 4] for each voxel => it's vertices
    torch::Tensor out_sdf,     // [N_voxels, 4] for each voxel => it's vertices
    torch::Tensor out_weights,     // [N_voxels, 4] for each voxel => it's vertices
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
                in_weights.data_ptr<float>(),     
                in_ids.data_ptr<int>(),       
                out_z.data_ptr<float>(),    
                out_sdf.data_ptr<float>(),    
                out_weights.data_ptr<float>(),     
                out_ids.data_ptr<int>(),    
                offset.data_ptr<int>(),     
                samples.data_ptr<float>(),   
                samples_loc.data_ptr<float>(),    
                sample_rays.data_ptr<float>()); 

    }));
    }