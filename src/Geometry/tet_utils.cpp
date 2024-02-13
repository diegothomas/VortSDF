#include <torch/extension.h>
#include <iostream>
#include <vector>
#include <map>
#include <set>
#include <tuple>

using namespace std;

vector<tuple<int, int, int>> get_faces_from_tetrahedron(const int* tetrahedron) {
    return {
        make_tuple(tetrahedron[1], tetrahedron[2], tetrahedron[3]),
        make_tuple(tetrahedron[0], tetrahedron[2], tetrahedron[3]),
        make_tuple(tetrahedron[0], tetrahedron[1], tetrahedron[3]),
        make_tuple(tetrahedron[0], tetrahedron[1], tetrahedron[2])
    };
}

void compute_neighbors(size_t nb_tets, torch::Tensor tetras_t, torch::Tensor summits_t, torch::Tensor neighbors_t) {

    int* tetras = tetras_t.data_ptr<int>();
    int* summits = summits_t.data_ptr<int>();
    int* neighbors = neighbors_t.data_ptr<int>();

    map<tuple<int, int, int>, vector<int>> faces_to_tetrahedron; 

    for (int i = 0; i < nb_tets; ++i) {
        vector<tuple<int, int, int>> faces = get_faces_from_tetrahedron(&tetras[4*i]);
        for (const auto& face : faces) {
            faces_to_tetrahedron[face].push_back(i);
        }
    }

    cout << "Faces adjacencies computed" << endl;

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
    }
    
    cout << "Neighboors computed" << endl;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compute_neighbors", &compute_neighbors, "compute_neighbors (CPP)");
}