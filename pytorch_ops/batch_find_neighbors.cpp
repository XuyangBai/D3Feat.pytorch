#include "torch/extension.h"
#include "neighbors/neighbors.h"
#include  "vector"

torch::Tensor batch_find_neighbors(
    torch::Tensor query_points,
    torch::Tensor support_points,
    torch::Tensor query_batches,
    torch::Tensor support_batches,
    float radius
    ){

    // Points Dimensions
    int Nq = (int)query_points.size(0);
    int Ns = (int)support_points.size(0);

    // Number of batches
    int Nb = (int)query_batches.size(0);

    // get the data as std vector of points
    vector<PointXYZ> queries = vector<PointXYZ>((PointXYZ*)query_points.data<float>(),
                                                (PointXYZ*)query_points.data<float>() + Nq);
    vector<PointXYZ> supports = vector<PointXYZ>((PointXYZ*)support_points.data<float>(),
                                                 (PointXYZ*)support_points.data<float>()+ Ns);

        // Batches lengths
    vector<int> q_batches = vector<int>((int*)query_batches.data<int>(),
                                        (int*)query_batches.data<int>() + Nb);
    vector<int> s_batches = vector<int>((int*)support_batches.data<int>(),
                                        (int*)support_batches.data<int>() + Nb);

    // Create result containers
    vector<int> neighbors_indices;

    // Compute results
    //batch_ordered_neighbors(queries, supports, q_batches, s_batches, neighbors_indices, radius);
    batch_nanoflann_neighbors(queries, supports, q_batches, s_batches, neighbors_indices, radius);

    // Maximal number of neighbors
    int max_neighbors = neighbors_indices.size() / Nq;

    // create output shape
    auto output_tensor = torch::zeros({Nq, max_neighbors});

    // Fill output tensor
    // output_tensor = torch::from_blob(&neighbors_indices);
    // https://discuss.pytorch.org/t/can-i-initialize-tensor-from-std-vector-in-libtorch/33236
    output_tensor = torch::tensor(neighbors_indices);
    // for (int i = 0; i < Nq; i++)
    // {
    //     for (int j = 0; j < max_neighbors; j++)
    //     { 
    //         // std::cout << neighbors_indices[max_neighbors * i + j] << std::endl;
    //         output_tensor[i][j] = neighbors_indices[max_neighbors * i + j];
    //     }
    // }

    return output_tensor;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("compute", &batch_find_neighbors, "batch find neighbors");
}