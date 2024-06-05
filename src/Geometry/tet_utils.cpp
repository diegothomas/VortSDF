#include <torch/extension.h>
#include <iostream>
#include <vector>
#include <map>
#include <set>
#include <tuple>

using namespace std;

int upsample_counter_cuda(
    size_t nb_edges,
    float sigma,
    torch::Tensor edges,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor sites,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor sdf,
    torch::Tensor active_sites
);

int upsample_counter_tet_cuda(
    size_t nb_tets,
    float sigma,
    torch::Tensor tets,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor sites,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor sdf
);

void upsample_cuda(
    size_t nb_edges,
    float sigma,
    torch::Tensor edges,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor sites,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor sdf,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor true_sdf,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor feats,
    torch::Tensor new_sites,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor new_sdf,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor new_feats
);

void upsample_tet_cuda(
    size_t nb_tets,
    float sigma,
    torch::Tensor tets,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor sites,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor sdf,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor feats,
    torch::Tensor new_sites,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor new_sdf,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor new_feats
);

void vertex_adjacencies_cuda(
    size_t nb_tets,
    size_t nb_sites,
    torch::Tensor tetras,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor summits,
    int ** adjacencies,
    int ** offset
);
    

void make_adjacencies_cuda(
    size_t nb_tets,
    torch::Tensor tetras,    // [N_sites, 3] for each voxel => it's vertices
    int * offset,
    int * adjacencies,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor neighbors
);

void cameras_adjacencies_cuda(
    size_t nb_tets,
    size_t nb_cams,
    torch::Tensor tetras,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor cam_ids,
    torch::Tensor adjacencies,
    torch::Tensor offset
);

int count_cam_neighbors_cuda(
    size_t nb_tets,
    size_t nb_cams,
    torch::Tensor tetras,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor cam_ids,
    torch::Tensor offset
);

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// ***************************
int upsample_counter(
    size_t nb_edges,
    float sigma,
    torch::Tensor edges,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor sites,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor sdf,
    torch::Tensor active_sites
) {
    //std::cout << "Backprop feature gradients" << std::endl; 
    return upsample_counter_cuda(
        nb_edges,
        sigma,
        edges,    // [N_sites, 3] for each voxel => it's vertices
        sites,    // [N_sites, 3] for each voxel => it's vertices
        sdf,
        active_sites);
}

int upsample_counter_tet(
    size_t nb_tets,
    float sigma,
    torch::Tensor tets,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor sites,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor sdf
)  {
    //std::cout << "Backprop feature gradients" << std::endl; 
    return upsample_counter_tet_cuda(
        nb_tets,
        sigma,
        tets,    // [N_sites, 3] for each voxel => it's vertices
        sites,    // [N_sites, 3] for each voxel => it's vertices
        sdf);
}


void upsample(
    size_t nb_edges,
    float sigma,
    torch::Tensor edges,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor sites,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor sdf,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor true_sdf,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor feats,
    torch::Tensor new_sites,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor new_sdf,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor new_feats
) {
    //std::cout << "Backprop feature gradients" << std::endl; 
    upsample_cuda(
        nb_edges,
        sigma,
        edges,    // [N_sites, 3] for each voxel => it's vertices
        sites,    // [N_sites, 3] for each voxel => it's vertices
        sdf,
        true_sdf,    // [N_sites, 3] for each voxel => it's vertices
        feats,    // [N_sites, 3] for each voxel => it's vertices
        new_sites,    // [N_sites, 3] for each voxel => it's vertices
        new_sdf,
        new_feats);
}

void upsample_tet(
    size_t nb_tets,
    float sigma,
    torch::Tensor tets,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor sites,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor sdf,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor feats,
    torch::Tensor new_sites,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor new_sdf,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor new_feats
)  {
    //std::cout << "Backprop feature gradients" << std::endl; 
    upsample_tet_cuda(
        nb_tets,
        sigma,
        tets,    // [N_sites, 3] for each voxel => it's vertices
        sites,    // [N_sites, 3] for each voxel => it's vertices
        sdf,
        feats,    // [N_sites, 3] for each voxel => it's vertices
        new_sites,    // [N_sites, 3] for each voxel => it's vertices
        new_sdf,
        new_feats);
}

/*void vertex_adjacencies(
    size_t nb_tets,
    size_t nb_sites,
    torch::Tensor tetras,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor summits,
    torch::Tensor adjacencies,
    int * offset,
)
{
    vertex_adjacencies_cuda(
    nb_tets,
    nb_sites,
    tetras,    // [N_sites, 3] for each voxel => it's vertices
    summits,
    adjacencies,
    offset);
}
    

void make_adjacencies(
    size_t nb_tets,
    torch::Tensor tetras,    // [N_sites, 3] for each voxel => it's vertices
    int * offset,
    int * adjacencies,    // [N_sites, 3] for each voxel => it's vertices
    torch::Tensor neighbors
)
{
    make_adjacencies_cuda(
    nb_tets,
    tetras,    // [N_sites, 3] for each voxel => it's vertices
    offset,
    adjacencies,    // [N_sites, 3] for each voxel => it's vertices
    neighbors
    );
}*/

vector<tuple<int, int, int>> get_faces_from_tetrahedron(const int* tetrahedron) {
    return {
        make_tuple(tetrahedron[1], tetrahedron[2], tetrahedron[3]),
        make_tuple(tetrahedron[0], tetrahedron[2], tetrahedron[3]),
        make_tuple(tetrahedron[0], tetrahedron[1], tetrahedron[3]),
        make_tuple(tetrahedron[0], tetrahedron[1], tetrahedron[2])
    };
}

void compute_neighbors(size_t nb_tets, size_t nb_sites, torch::Tensor tetras_t, torch::Tensor summits_t, torch::Tensor neighbors_t) {

    int *adj_t;
    int *offset;
    vertex_adjacencies_cuda(nb_tets, nb_sites, tetras_t, summits_t, &adj_t, &offset);
    cout << "Faces adjacencies computed" << endl;
    
    make_adjacencies_cuda(nb_tets, tetras_t, offset, adj_t, neighbors_t);
    cout << "Neighboors computed" << endl;


    /*int* tetras = tetras_t.data_ptr<int>();
    int* summits = summits_t.data_ptr<int>();
    int* neighbors = neighbors_t.data_ptr<int>();

    map<tuple<int, int, int>, vector<int>> faces_to_tetrahedron; 

//#pragma omp parallel for
    for (int i = 0; i < nb_tets; ++i) {
        vector<tuple<int, int, int>> faces = get_faces_from_tetrahedron(&tetras[4*i]);
        for (const auto& face : faces) {
            faces_to_tetrahedron[face].push_back(i);
        }
    }

    cout << "Faces adjacencies computed" << endl;

//#pragma omp parallel for
    for (int i = 0; i < nb_tets; ++i) {
        vector<tuple<int, int, int>> faces = get_faces_from_tetrahedron(&tetras[4*i]);
        //if (faces.size() != 4)
        //    cout << "ERROR in nb of faces in a tet !!" << endl;

        for (int j = 0; j < faces.size(); ++j) {
            vector<int> adjacencies = faces_to_tetrahedron[faces[j]];
            if (adjacencies.size() == 1) {
                neighbors[4*i+j] = -1;
            } else if (adjacencies[0] == i) {
                neighbors[4*i+j] = adjacencies[1];
            } else {
                neighbors[4*i+j] = adjacencies[0];
            }
        }

        // make last summit index as xor
        summits[4*i] = tetras[4*i]; summits[4*i+1] = tetras[4*i+1]; summits[4*i+2] = tetras[4*i+2]; 
        summits[4*i+3] = tetras[4*i] ^ tetras[4*i+1] ^ tetras[4*i+2] ^ tetras[4*i+3];
    }*/
    
    //cout << "Neighboors computed" << endl;
}


int count_cam_neighbors(size_t nb_tets, size_t nb_cams, torch::Tensor tetras_t, torch::Tensor cam_ids_t, torch::Tensor offset) {
    return count_cam_neighbors_cuda(nb_tets, nb_cams, tetras_t, cam_ids_t, offset);
}

void compute_cam_neighbors(size_t nb_tets, size_t nb_cams, torch::Tensor tetras_t, torch::Tensor cam_ids_t, torch::Tensor adj_t, torch::Tensor offset) {
    cameras_adjacencies_cuda(nb_tets, nb_cams, tetras_t, cam_ids_t, adj_t, offset);
    cout << "Cam adjacencies computed" << endl;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compute_neighbors", &compute_neighbors, "compute_neighbors (CPP)");
    m.def("upsample_counter", &upsample_counter, "upsample_counter (CPP)");
    m.def("upsample", &upsample, "upsample (CPP)");
    m.def("upsample_counter_tet", &upsample_counter_tet, "upsample_counter_tet (CPP)");
    m.def("upsample_tet", &upsample_tet, "upsample_tet (CPP)");
    m.def("count_cam_neighbors", &count_cam_neighbors, "count_cam_neighbors (CPP)");
    m.def("compute_cam_neighbors", &compute_cam_neighbors, "compute_cam_neighbors (CPP)");
}