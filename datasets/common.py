import open3d
import numpy as np
from datasets.dataloader import find_neighbors, grid_subsampling
import torch


def segmentation_inputs(points, features, labels, config):
    # TODO: originally I use this function to prepare the segmentation inputs for one single point cloud inputs and the next function collate_fn
    # to aggregate the inputs to a batch.

    # Starting radius of convolutions
    r_normal = config.first_subsampling_dl * config.KP_extent * 2.5

    # Starting layer
    layer_blocks = []

    # Lists of inputs
    input_points = []
    input_neighbors = []
    input_pools = []
    input_upsamples = []
    input_batches_len = []
    for block_i, block in enumerate(config.architecture):

        # TODO: stop early ?
        # Stop when meeting a global pooling or upsampling
        if 'upsample' in block:
            # if 'global' in block or 'upsample' in block:
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
            conv_i = find_neighbors(points, points, r, max_neighbor=40)
        else:
            # This layer only perform pooling, no neighbors required
            conv_i = torch.zeros((0, 1), dtype=torch.int32)

        # Pooling neighbors indices
        # *************************
        # If end of layer is a pooling operation
        if 'pool' in block or 'strided' in block:

            # New subsampling length
            dl = 2 * r_normal / (config.KP_extent * 2.5)

            # Subsampled points
            pool_p = grid_subsampling(points, sampleDl=dl)
            pool_b = torch.from_numpy(np.array([len(pool_p)]))

            # Radius of pooled neighbors
            if 'deformable' in block:
                r = r_normal * config.density_parameter / (config.KP_extent * 2.5)
            else:
                r = r_normal

            # Subsample indices
            pool_i = find_neighbors(pool_p, points, r, max_neighbor=40)

            # Upsample indices (with the radius of the next layer to keep wanted density)
            up_i = find_neighbors(points, pool_p, 2 * r, max_neighbor=40)

        else:
            # No pooling in the end of this layer, no pooling indices required
            pool_i = torch.zeros((0, 1), dtype=torch.int32)
            pool_p = torch.zeros((0, 3), dtype=torch.float32)
            pool_b = torch.zeros((0,), dtype=torch.int32)
            up_i = torch.zeros((0, 1), dtype=torch.int32)

        # TODO: Instead of eliminating the furthest point here, I select as most "max_neighbor" neighbors in find_neighbors function
        # Reduce size of neighbors matrices by eliminating furthest point

        # Updating input lists
        input_points += [torch.from_numpy(points)]
        input_neighbors += [conv_i]
        input_pools += [pool_i]
        input_upsamples += [up_i]
        # input_batches_len += [stacked_lengths]

        # New points for next layer
        points = pool_p
        # stacked_lengths = pool_b

        # Update radius and reset blocks
        r_normal *= 2
        layer_blocks = []

    ###############
    # Return inputs
    ###############

    dict_inputs = {
        'points': input_points,
        'neighbors': input_neighbors,
        'pools': input_pools,
        'upsamples': input_upsamples,
        'features': features,
        'labels': labels
    }

    return dict_inputs


def collate_fn(list_data):
    # TODO: the problem is, for 'neighbors', 'pools' and 'upsamples', there are dustbin indices for points with less
    # neighboring points than max_neighbors, but when we simply do "list_data[i_pcd]['points'][i_layer] + start_ind"
    # the dustbin index will be the first point in the next point cloud.
    batched_input = {
        'points': [],
        'neighbors': [],
        'pools': [],
        'upsamples': [],
        'features': [],
        'labels': [],
        'batches_len': [],
    }
    num_layers = len(list_data[0]['points'])
    num_pcd = len(list_data)
    # process inputs['points']
    for i_layer in range(num_layers):
        layer_points_list = []
        start_ind = 0
        for i_pcd in range(num_pcd):
            layer_points_list.append(list_data[i_pcd]['points'][i_layer] + start_ind)
            start_ind += len(list_data[i_pcd]['points'][i_layer])
        layer_points = torch.cat(layer_points_list, dim=0)
        batched_input['points'].append(layer_points)

    # process inputs['neighbors']
    for i_layer in range(num_layers):
        layer_neighbors_list = []
        start_ind = 0
        for i_pcd in range(num_pcd):
            layer_neighbors_list.append(list_data[i_pcd]['neighbors'][i_layer] + start_ind)
            start_ind += len(list_data[i_pcd]['neighbors'][i_layer])
        layer_neighbors = torch.cat(layer_neighbors_list, dim=0)
        batched_input['neighbors'].append(layer_neighbors)

    return batched_input
