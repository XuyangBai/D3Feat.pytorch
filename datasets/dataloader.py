import open3d
import numpy as np
from utils.pointcloud import make_point_cloud
from functools import partial
import torch
import cpp_wrappers.cpp_subsampling.grid_subsampling as cpp_subsampling
from utils.timer import Timer
import batch_find_neighbors


def find_neighbors(query_points, support_points, radius, max_neighbor):
    pcd = make_point_cloud(support_points)
    kdtree = open3d.KDTreeFlann(pcd)
    neighbor_indices_list = []
    for i, point in enumerate(query_points):
        [k, idx, dis] = kdtree.search_radius_vector_3d(point, radius)
        if k > max_neighbor:
            idx = np.random.choice(idx, max_neighbor, replace=False)
        else:
            # if not enough neighbor points, then add the dustbin point.
            idx = list(idx) + [len(support_points)] * (max_neighbor - k)
        neighbor_indices_list.append([idx])
    neighbors = np.concatenate(neighbor_indices_list, axis=0)
    return torch.from_numpy(neighbors)

def batch_find_neighbors_wrapper(query_points, support_points, query_batches, support_batches, radius, max_neighbors):
    if True:
        cpp = batch_find_neighbors_cpp(query_points, support_points, query_batches, support_batches, radius, max_neighbors)
        cpp = cpp.reshape([query_points.shape[0], -1])
        cpp = cpp[:, :max_neighbors]
        return cpp
    else:
        py = batch_find_neighbors_py(query_points, support_points, query_batches, support_batches, radius, max_neighbors)
        py = py[:, :max_neighbors]
        return py

def batch_find_neighbors_cpp(query_points, support_points, query_batches, support_batches, radius, max_neighbors):
    outputs = batch_find_neighbors.compute(query_points, support_points, query_batches, support_batches, radius)
    outputs = outputs.long()
    return outputs

def batch_find_neighbors_py(query_points, support_points, query_batches, support_batches, radius, max_neighbors):
    num_batches = len(support_batches)
    # Create kdtree for each pcd in support_points
    kdtrees = []
    start_ind = 0
    for length in support_batches:
        pcd = make_point_cloud(support_points[start_ind:start_ind + length])
        kdtrees.append(open3d.KDTreeFlann(pcd))
        start_ind += length
    assert len(kdtrees) == num_batches
    # Search neigbors indices
    neighbors_indices_list = []
    start_ind = 0
    support_start_ind = 0
    dustbin_ind = len(support_points)
    for i_batch, length in enumerate(query_batches):
        for i_pts, pts in enumerate(query_points[start_ind:start_ind + length]):
            [k, idx, dis] = kdtrees[i_batch].search_radius_vector_3d(pts, radius)
            if k > max_neighbors:
                # idx = np.random.choice(idx, max_neighbors, replace=False)
                # If i use random, the closest_pool will not work as expected.
                idx = list(idx[0:max_neighbors])
            else:
                # if not enough neighbor points, then add dustbin index. Careful !!!
                idx = list(idx) + [dustbin_ind - support_start_ind] * (max_neighbors - k)
            idx = np.array(idx) + support_start_ind
            neighbors_indices_list.append(idx)
        # finish one query_points, update the start_ind
        start_ind += int(query_batches[i_batch])
        support_start_ind += int(support_batches[i_batch])
    return torch.from_numpy(np.array(neighbors_indices_list)).long()

def grid_subsampling(points, features=None, labels=None, sampleDl=0.1, verbose=0):
    """
    CPP wrapper for a grid subsampling (method = barycenter for points and features
    :param points: (N, 3) matrix of input points
    :param features: optional (N, d) matrix of features (floating number)
    :param labels: optional (N,) matrix of integer labels
    :param sampleDl: parameter defining the size of grid voxels
    :param verbose: 1 to display
    :return: subsampled points, with features and/or labels depending of the input
    """

    if (features is None) and (labels is None):
        return cpp_subsampling.compute(points, sampleDl=sampleDl, verbose=verbose)
    elif (labels is None):
        return cpp_subsampling.compute(points, features=features, sampleDl=sampleDl, verbose=verbose)
    elif (features is None):
        return cpp_subsampling.compute(points, classes=labels, sampleDl=sampleDl, verbose=verbose)
    else:
        return cpp_subsampling.compute(points, features=features, classes=labels, sampleDl=sampleDl, verbose=verbose)

def batch_grid_subsampling(points, batches_len, sampleDl=0.1):
    """
    CPP wrapper for a batch grid subsampling (method = barycenter for points and features
    :param points: (N, 3) matrix of input points
    :param batches_len: lengths of batched input points
    :param sampleDl: parameter defining the size of grid voxels
    :return:
    """
    subsampled_points_list = []
    subsampled_batches_len_list = []
    start_ind = 0
    for length in batches_len:
        b_origin_points = points[start_ind:start_ind + length]
        b_subsampled_points = grid_subsampling(b_origin_points, sampleDl=sampleDl)
        start_ind += length
        subsampled_points_list.append(b_subsampled_points)
        subsampled_batches_len_list.append(len(b_subsampled_points))
    subsampled_points = torch.from_numpy(np.concatenate(subsampled_points_list, axis=0))
    subsampled_batches_len = torch.from_numpy(np.array(subsampled_batches_len_list)).int()
    return subsampled_points, subsampled_batches_len

def collate_fn_descriptor(list_data, config, neighborhood_limits):
    batched_points_list = []
    batched_features_list = []
    batched_lengths_list = []
    assert len(list_data) == 1
    
    for ind, (pts0, pts1, feat0, feat1, sel_corr, dist_keypts) in enumerate(list_data):
        batched_points_list.append(pts0)
        batched_points_list.append(pts1)
        batched_features_list.append(feat0)
        batched_features_list.append(feat1)
        batched_lengths_list.append(len(pts0))
        batched_lengths_list.append(len(pts1))
    
    batched_features = torch.from_numpy(np.concatenate(batched_features_list, axis=0))
    batched_points = torch.from_numpy(np.concatenate(batched_points_list, axis=0))
    batched_lengths = torch.from_numpy(np.array(batched_lengths_list)).int()

    # Starting radius of convolutions
    r_normal = config.first_subsampling_dl * config.KP_extent * 2.5

    # Starting layer
    layer_blocks = []
    layer = 0

    # Lists of inputs
    input_points = []
    input_neighbors = []
    input_pools = []
    input_upsamples = []
    input_batches_len = []

    for block_i, block in enumerate(config.architecture):

        # Stop when meeting a global pooling or upsampling
        if 'global' in block or 'upsample' in block:
            break

        # Get all blocks of the layer
        if not ('pool' in block or 'strided' in block):
            layer_blocks += [block]
            if block_i < len(config.architecture) - 1 and not ('upsample' in config.architecture[block_i + 1]):
                continue

        # Convolution neighbors indices
        # *****************************

        if layer_blocks:
            # Convolutions are done in this layer, compute the neighbors with the good radius
            if np.any(['deformable' in blck for blck in layer_blocks[:-1]]):
                r = r_normal * config.density_parameter / (config.KP_extent * 2.5)
            else:
                r = r_normal
            conv_i = batch_find_neighbors_wrapper(batched_points, batched_points, batched_lengths, batched_lengths, r, neighborhood_limits[layer])

        else:
            # This layer only perform pooling, no neighbors required
            conv_i = torch.zeros((0, 1), dtype=torch.int64)

        # Pooling neighbors indices
        # *************************

        # If end of layer is a pooling operation
        if 'pool' in block or 'strided' in block:

            # New subsampling length
            dl = 2 * r_normal / (config.KP_extent * 2.5)

            # Subsampled points
            pool_p, pool_b = batch_grid_subsampling(batched_points, batched_lengths, sampleDl=dl)

            # Radius of pooled neighbors
            if 'deformable' in block:
                r = r_normal * config.density_parameter / (config.KP_extent * 2.5)
            else:
                r = r_normal

            # Subsample indices
            pool_i = batch_find_neighbors_wrapper(pool_p, batched_points, pool_b, batched_lengths, r, neighborhood_limits[layer])

            # Upsample indices (with the radius of the next layer to keep wanted density)
            up_i = batch_find_neighbors_wrapper(batched_points, pool_p, batched_lengths, pool_b, 2 * r, neighborhood_limits[layer])

        else:
            # No pooling in the end of this layer, no pooling indices required
            pool_i = torch.zeros((0, 1), dtype=torch.int64)
            pool_p = torch.zeros((0, 3), dtype=torch.float32)
            pool_b = torch.zeros((0,), dtype=torch.int64)
            up_i = torch.zeros((0, 1), dtype=torch.int64)

        # Updating input lists
        input_points += [batched_points.float()]
        input_neighbors += [conv_i.long()]
        input_pools += [pool_i.long()]
        input_upsamples += [up_i.long()]
        input_batches_len += [batched_lengths]

        # New points for next layer
        batched_points = pool_p
        batched_lengths = pool_b

        # Update radius and reset blocks
        r_normal *= 2
        layer += 1
        layer_blocks = []

    ###############
    # Return inputs
    ###############
    dict_inputs = {
        'points': input_points,
        'neighbors': input_neighbors,
        'pools': input_pools,
        'upsamples': input_upsamples,
        'features': batched_features.float(),
        'stack_lengths': input_batches_len,
        'corr': torch.from_numpy(sel_corr),
        'dist_keypts': torch.from_numpy(dist_keypts),
    }

    return dict_inputs

def calibrate_neighbors(dataset, config, collate_fn, keep_ratio=0.8, samples_threshold=5000):
    timer = Timer()
    last_display = timer.total_time

    # From config parameter, compute higher bound of neighbors number in a neighborhood
    hist_n = int(np.ceil(4 / 3 * np.pi * (config.density_parameter + 1) ** 3))
    neighb_hists = np.zeros((config.num_layers, hist_n), dtype=np.int32)

    # Get histogram of neighborhood sizes i in 1 epoch max.
    for i in range(len(dataset)):
        timer.tic()
        batched_input = collate_fn([dataset[i]], config, neighborhood_limits=[hist_n] * 5)

        # update histogram
        counts = [torch.sum(neighb_mat < neighb_mat.shape[0], dim=1).numpy() for neighb_mat in batched_input['neighbors']]
        hists = [np.bincount(c, minlength=hist_n)[:hist_n] for c in counts]
        neighb_hists += np.vstack(hists)
        timer.toc()

        if timer.total_time - last_display > 0.1:
            last_display = timer.total_time
            print(f"Calib Neighbors {i:08d}: timings {timer.total_time:4.2f}s")

        if np.min(np.sum(neighb_hists, axis=1)) > samples_threshold:
            break

    cumsum = np.cumsum(neighb_hists.T, axis=0)
    percentiles = np.sum(cumsum < (keep_ratio * cumsum[hist_n - 1, :]), axis=0)

    neighborhood_limits = percentiles
    print('\n')

    return neighborhood_limits

def get_dataloader(dataset, batch_size=2, num_workers=4, shuffle=True):
    neighborhood_limits = calibrate_neighbors(dataset, dataset.config, collate_fn=collate_fn_descriptor)
    print("neighborhood:", neighborhood_limits)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        # https://discuss.pytorch.org/t/supplying-arguments-to-collate-fn/25754/4
        collate_fn=partial(collate_fn_descriptor, config=dataset.config, neighborhood_limits=neighborhood_limits),
        drop_last=False
    )
    return dataloader


if __name__ == '__main__':
    from training_3DMatch import ThreeDMatchConfig
    from datasets.ThreeDMatch import  ThreeDMatchDataset
    config = ThreeDMatchConfig()
    dset = ThreeDMatchDataset(root='/home/xybai/KPConv/data/3DMatch', split='val', config=config)
    dataloader = get_dataloader(dset, batch_size=1, num_workers=1)
    for iter, inputs in enumerate(dataloader):
        print(iter)
        import pdb
        pdb.set_trace()