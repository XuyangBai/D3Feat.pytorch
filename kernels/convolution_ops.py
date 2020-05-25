import numpy as np
from kernels.kernel_points import load_kernels as create_kernel_points
import torch
from torch import nn


def unary_convolution(features,
                      K_values):
    """
    Simple unary convolution in tensorflow. Equivalent to matrix multiplication (space projection) for each features
    :param features: float32[n_points, in_fdim] - input features
    :param K_values: float32[in_fdim, out_fdim] - weights of the kernel
    :return: output_features float32[n_points, out_fdim]
    """

    return torch.matmul(features, K_values)


def radius_gaussian(sq_r, sig, eps=1e-9):
    """
    Compute a radius gaussian (gaussian of distance)
    :param sq_r: input radiuses [dn, ..., d1, d0]
    :param sig: extents of gaussians [d1, d0] or [d0] or float
    :return: gaussian of sq_r [dn, ..., d1, d0]
    """
    return torch.exp(-sq_r / (2 * torch.pow(sig, 2) + eps))


def KPConv(query_points,
           support_points,
           neighbors_indices,
           features,
           K_values,
           fixed='center',
           KP_extent=1.0,
           KP_influence='linear',
           aggregation_mode='sum'):
    """
    This function initiates the kernel point disposition before building KPConv graph ops
    :param query_points: float32[n_points, dim] - input query points (center of neighborhoods)
    :param support_points: float32[n0_points, dim] - input support points (from which neighbors are taken)
    :param neighbors_indices: int32[n_points, n_neighbors] - indices of neighbors of each point
    :param features: float32[n_points, in_fdim] - input features
    :param K_values: float32[n_kpoints, in_fdim, out_fdim] - weights of the kernel
    :param fixed: string in ('none', 'center' or 'verticals') - fix position of certain kernel points
    :param KP_extent: float32 - influence radius of each kernel point
    :param KP_influence: string in ('constant', 'linear', 'gaussian') - influence function of the kernel points
    :param aggregation_mode: string in ('closest', 'sum') - whether to sum influences, or only keep the closest
    :return: output_features float32[n_points, out_fdim]
    """

    # Initial kernel extent for this layer
    K_radius = 1.5 * KP_extent

    # Number of kernel points
    num_kpoints = int(K_values.shape[0])

    # Check point dimension (currently only 3D is supported)
    points_dim = int(query_points.shape[1])

    # Create one kernel disposition (as numpy array). Choose the KP distance to center thanks to the KP extent
    K_points_numpy = create_kernel_points(K_radius,
                                          num_kpoints,
                                          num_kernels=1,
                                          dimension=points_dim,
                                          fixed=fixed)
    K_points_numpy = K_points_numpy.reshape((num_kpoints, points_dim))

    # Create the tensorflow variable
    K_points = torch.from_numpy(K_points_numpy.astype(np.float32))
    if K_values.is_cuda:
        K_points = K_points.to(K_values.device)

    return KPConv_ops(query_points,
                      support_points,
                      neighbors_indices,
                      features,
                      K_points,
                      K_values,
                      KP_extent,
                      KP_influence,
                      aggregation_mode)


def KPConv_ops(query_points,
               support_points,
               neighbors_indices,
               features,
               K_points,
               K_values,
               KP_extent,
               KP_influence,
               aggregation_mode):
    """
    This function creates a graph of operations to define Kernel Point Convolution in tensorflow. See KPConv function
    above for a description of each parameter

    :param query_points:        [n_points, dim]
    :param support_points:      [n0_points, dim]
    :param neighbors_indices:   [n_points, n_neighbors]
    :param features:            [n_points, in_fdim]
    :param K_points:            [n_kpoints, dim]
    :param K_values:            [n_kpoints, in_fdim, out_fdim]
    :param KP_extent:           float32
    :param KP_influence:        string
    :param aggregation_mode:    string
    :return:                    [n_points, out_fdim]
    """

    # Get variables
    n_kp = int(K_points.shape[0])

    # Add a fake point in the last row for shadow neighbors
    shadow_point = torch.ones_like(support_points[:1, :]) * 1e6
    support_points = torch.cat([support_points, shadow_point], dim=0)

    # Get neighbor points [n_points, n_neighbors, dim]
    neighbors = support_points[neighbors_indices, :]

    # Center every neighborhood
    neighbors = neighbors - query_points.unsqueeze(1)

    # Get all difference matrices [n_points, n_neighbors, n_kpoints, dim]
    differences = neighbors.unsqueeze(2) - K_points

    # Get the square distances [n_points, n_neighbors, n_kpoints]
    sq_distances = torch.sum(torch.mul(differences, differences), dim=3)

    # Get Kernel point influences [n_points, n_kpoints, n_neighbors]
    if KP_influence == 'constant':
        # Every point get an influence of 1.
        all_weights = torch.ones_like(sq_distances)
        all_weights = all_weights.transpose(1, 2)

    elif KP_influence == 'linear':
        # Influence decrease linearly with the distance, and get to zero when d = KP_extent.
        corr = 1 - torch.sqrt(sq_distances + 1e-10) / KP_extent
        all_weights = torch.max(corr, torch.zeros_like(sq_distances))
        all_weights = all_weights.transpose(1, 2)

    elif KP_influence == 'gaussian':
        # Influence in gaussian of the distance.
        sigma = KP_extent * 0.3
        all_weights = radius_gaussian(sq_distances, sigma)
        all_weights = all_weights.transpose(1, 2)
    else:
        raise ValueError('Unknown influence function type (config.KP_influence)')

    # In case of closest mode, only the closest KP can influence each point
    if aggregation_mode == 'closest':
        neighbors_1nn = torch.argmin(sq_distances, dim=2)
        all_weights *= torch.transpose(nn.functional.one_hot(neighbors_1nn, n_kp), 1, 2)

    elif aggregation_mode != 'sum':
        raise ValueError("Unknown convolution mode. Should be 'closest' or 'sum'")

    features = torch.cat([features, torch.zeros_like(features[:1, :])], dim=0)

    # Get the features of each neighborhood [n_points, n_neighbors, in_fdim]
    # neighborhood_features = features[neighbors_indices, :]
    from models.network_blocks import gather
    neighborhood_features = gather(features, neighbors_indices)

    # Apply distance weights [n_points, n_kpoints, in_fdim]
    weighted_features = torch.matmul(all_weights, neighborhood_features)

    # Apply network weights [n_kpoints, n_points, out_fdim]
    weighted_features = weighted_features.transpose(0, 1)
    kernel_outputs = torch.matmul(weighted_features, K_values)

    # Convolution sum to get [n_points, out_fdim]
    output_features = torch.sum(kernel_outputs, dim=0, keepdim=False)

    # normalization term.
    # neighbor_features_sum = torch.sum(neighborhood_features, dim=-1)
    # neighbor_num = torch.sum(torch.gt(neighbor_features_sum, 0.0), dim=-1)
    # neighbor_num = torch.max(neighbor_num, torch.ones_like(neighbor_num))
    # output_features = output_features / neighbor_num.unsqueeze(1)

    return output_features


def KPConv_deformable(query_points,
                      support_points,
                      neighbors_indices,
                      features,
                      K_values,
                      w0, b0,
                      fixed='center',
                      KP_extent=1.0,
                      KP_influence='linear',
                      aggregation_mode='sum',
                      modulated=False):
    """
    This function initiates the kernel point disposition before building deformable KPConv graph ops

    :param query_points: float32[n_points, dim] - input query points (center of neighborhoods)
    :param support_points: float32[n0_points, dim] - input support points (from which neighbors are taken)
    :param neighbors_indices: int32[n_points, n_neighbors] - indices of neighbors of each point
    :param features: float32[n_points, in_fdim] - input features
    :param K_values: float32[n_kpoints, in_fdim, out_fdim] - weights of the kernel
    :param w0: float32[n_points, dim * n_kpoints] - weights of the rigid KPConv for offsets
    :param b0: float32[dim * n_kpoints] - bias of the rigid KPConv for offsets.
    :param fixed: string in ('none', 'center' or 'verticals') - fix position of certain kernel points
    :param KP_extent: float32 - influence radius of each kernel point
    :param KP_influence: string in ('constant', 'linear', 'gaussian') - influence function of the kernel points
    :param aggregation_mode: string in ('closest', 'sum') - behavior of the convolution
    :param modulated: bool - If deformable conv should be modulated

    :return: output_features float32[n_points, out_fdim]
    """

    # Initial kernel extent for this layer
    K_radius = 1.5 * KP_extent

    # Number of kernel points
    num_kpoints = int(K_values.shape[0])

    # Check point dimension (currently only 3D is supported)
    points_dim = int(query_points.shape[1])

    # Create one kernel disposition (as numpy array). Choose the KP distance to center thanks to the KP extent
    K_points_numpy = create_kernel_points(K_radius,
                                          num_kpoints,
                                          num_kernels=1,
                                          dimension=points_dim,
                                          fixed=fixed)
    K_points_numpy = K_points_numpy.reshape((num_kpoints, points_dim))

    # Create the pytorch variable
    K_points = torch.from_numpy(K_points_numpy.astype(np.float32))
    if K_values.is_cuda:
        K_points = K_points.to(K_values.device)

    #############################
    # Standard KPConv for offsets
    #############################

    # Create independant weight for the first convolution and a bias term as no batch normalization happen
    # if modulated:
    #     offset_dim = (points_dim + 1) * num_kpoints
    # else:
    #     offset_dim = points_dim * num_kpoints
    # shape0 = list(K_values.shape)
    # shape0[-1] = offset_dim  # [n_kpoints, in_fdim, offset_dim]
    # w0 = torch.zeros(shape0, dtype=torch.float32)  # offset_conv_weights
    # b0 = torch.zeros(offset_dim, dtype=torch.float32)  # offset_conv_bias

    # Get features from standard convolution
    features0 = KPConv_ops(query_points,
                           support_points,
                           neighbors_indices,
                           features,
                           K_points,
                           w0,
                           KP_extent,
                           KP_influence,
                           aggregation_mode) + b0
    if modulated:

        # Get offset (in normalized scale) from features
        offsets = features0[:, :points_dim * num_kpoints]
        offsets = offsets.reshape([-1, num_kpoints, points_dim])

        # Get modulations
        modulations = 2 * torch.sigmoid(features0[:, points_dim * num_kpoints:])

    else:

        # Get offset (in normalized scale) from features
        offsets = features0.reshape([-1, num_kpoints, points_dim])

        # No modulations
        modulations = None

    # Rescale offset for this layer
    offsets *= KP_extent
    ###############################
    # Build deformable KPConv graph
    ###############################

    # Apply deformed convolution
    return KPConv_deform_ops(query_points,
                             support_points,
                             neighbors_indices,
                             features,
                             K_points,
                             offsets,
                             modulations,
                             K_values,
                             KP_extent,
                             KP_influence,
                             aggregation_mode)


def KPConv_deformable_v2(query_points,
                         support_points,
                         neighbors_indices,
                         features,
                         K_values,
                         w0, b0,
                         fixed='center',
                         KP_extent=1.0,
                         KP_influence='linear',
                         aggregation_mode='sum',
                         modulated=False):
    """
    This alternate version uses a pointwise MLP instead of KPConv to get the offset. It has thus less parameters.
    It also fixes the center point to remain in the center in any case. This definition offers similar performances

    :param query_points: float32[n_points, dim] - input query points (center of neighborhoods)
    :param support_points: float32[n0_points, dim] - input support points (from which neighbors are taken)
    :param neighbors_indices: int32[n_points, n_neighbors] - indices of neighbors of each point
    :param features: float32[n_points, in_fdim] - input features
    :param K_values: float32[n_kpoints, in_fdim, out_fdim] - weights of the kernel
    :param w0: float32[n_points, dim * n_kpoints] - weights of the unary for offsets
    :param b0: float32[dim * n_kpoints] - bias of the unary for offsets.
    :param fixed: string in ('none', 'center' or 'verticals') - fix position of certain kernel points
    :param KP_extent: float32 - influence radius of each kernel point
    :param KP_influence: string in ('constant', 'linear', 'gaussian') - influence function of the kernel points
    :param aggregation_mode: string in ('closest', 'sum') - behavior of the convolution
    :param modulated: bool - If deformable conv should be modulated

    :return: output_features float32[n_points, out_fdim]
    """
    # Initial kernel extent for this layer
    K_radius = 1.5 * KP_extent

    # Number of kernel points
    num_kpoints = int(K_values.shape[0])

    # Check point dimension (currently only 3D is supported)
    points_dim = int(query_points.shape[1])

    # Create one kernel disposition (as numpy array). Choose the KP distance to center thanks to the KP extent
    K_points_numpy = create_kernel_points(K_radius,
                                          num_kpoints,
                                          num_kernels=1,
                                          dimension=points_dim,
                                          fixed=fixed)
    K_points_numpy = K_points_numpy.reshape((num_kpoints, points_dim))

    # Create the pytorch variable
    K_points = torch.from_numpy(K_points_numpy.astype(np.float32))
    if K_values.is_cuda:
        K_points = K_points.to(K_values.device)

    #############################
    # Pointwise MLP for offsets
    #############################
    # Create independant weight for the first convolution and a bias term as no batch normalization happen
    # if modulated:
    #     offset_dim = (points_dim + 1) * num_kpoints
    # else:
    #     offset_dim = points_dim * num_kpoints
    # shape0 = K_values.shape.as_list()
    # w0 = torch.zeros([shape0[1], offset_dim], dtype=torch.float32)  # offset_mlp_weights
    # b0 = torch.zeros(offset_dim, dtype=torch.float32)  # offset_mlp_bias

    # Get features from mlp
    features0 = unary_convolution(features, w0) + b0
    # TODO: need to do something to reduce the point size from len(support_points) to len(query_points).

    if modulated:

        # Get offset (in normalized scale) from features
        offsets = features0[:, :points_dim * (num_kpoints - 1)]
        offsets = offsets.reshape([-1, (num_kpoints - 1), points_dim])

        # Get modulations
        modulations = 2 * torch.sigmoid(features0[:, points_dim * (num_kpoints-1):])

        # No offset for the first Kernel points
        offsets = torch.cat([torch.zeros_like(offsets[:, :1, :]), offsets], dim=1)
        modulations = torch.cat([torch.zeros_like(modulations[:, :1]), modulations], dim=1)

    else:

        # Get offset (in normalized scale) from features
        offsets = features0.reshape([-1, (num_kpoints-1), points_dim])

        # No offset for the first kernle points
        offsets = torch.cat([torch.zeros_like(offsets[:, :1, :]), offsets], dim=1)

        # No modulations
        modulations = None

    # Rescale offset for this layer
    offsets *= KP_extent

    ###############################
    # Build deformable KPConv graph
    ###############################

    # Apply deformed convolution
    return KPConv_deform_ops(query_points,
                             support_points,
                             neighbors_indices,
                             features,
                             K_points,
                             offsets,
                             modulations,
                             K_values,
                             KP_extent,
                             KP_influence,
                             aggregation_mode)


def KPConv_deform_ops(query_points,
                      support_points,
                      neighbors_indices,
                      features,
                      K_points,
                      offsets,
                      modulations,
                      K_values,
                      KP_extent,
                      KP_influence,
                      mode):
    """
    This function creates a graph of operations to define Deformable Kernel Point Convolution in tensorflow. See
    KPConv_deformable function above for a description of each parameter

    :param query_points:        [n_points, dim]
    :param support_points:      [n0_points, dim]
    :param neighbors_indices:   [n_points, n_neighbors]
    :param features:            [n_points, in_fdim]
    :param K_points:            [n_kpoints, dim]
    :param offsets:             [n_points, n_kpoints, dim]
    :param modulations:         [n_points, n_kpoints] or None
    :param K_values:            [n_kpoints, in_fdim, out_fdim]
    :param KP_extent:           float32
    :param KP_influence:        string
    :param mode:                string

    :return:                    [n_points, out_fdim]
    """

    # Get variables
    n_kp = int(K_points.shape[0])
    shadow_ind = support_points.shape[0]

    # Add a fake point in the last row for shadow neighbors
    shadow_point = torch.ones_like(support_points[:1, :]) * 1e6
    support_points = torch.cat([support_points, shadow_point], dim=0)

    # Get neighbor points [n_points, n_neighbors, dim]
    neighbors = support_points[neighbors_indices, :]

    # Center every neighborhood
    neighbors = neighbors - query_points.unsqueeze(1)

    # Apply offsets to kernel points [n_points, n_kpoints, dim]
    deformed_K_points = torch.add(offsets, K_points)

    # Get all difference matrices [n_points, n_neighbors, n_kpoints, dim]
    differences = neighbors.unsqueeze(2) - deformed_K_points.unsqueeze(1)

    # Get the square distances [n_points, n_neighbors, n_kpoints]
    sq_distances = torch.sum(torch.mul(differences, differences), dim=3)

    # Boolean of the neighbors in range of a kernel point [n_points, n_neighbors]
    in_range = torch.any((sq_distances < KP_extent ** 2), dim=2).int()

    # New value of max neighbors
    new_max_neighb = torch.max(torch.sum(in_range, dim=1))

    # For each row of neighbors, indices of the ones that are in range [n_points, new_max_neighb]
    new_neighb_bool, new_neighb_inds = in_range.topk(k=int(new_max_neighb))
    
    # Gather new neighbor indices [n_points, new_max_neighb]
    # new_neighbors_indices = tf.batch_gather(neighbors_indices, new_neigh_inds)
    new_neighbors_indices = torch.gather(neighbors_indices, dim=1, index=new_neighb_inds)

    # Gather new distances to KP [n_points, new_max_neighb, n_kpoints]
    # new_sq_distances = tf.batch_gather(sq_distances, new_neighb_inds)
    # https://pytorch.org/docs/stable/torch.html#torch.gather
    # https://discuss.pytorch.org/t/question-about-torch-gather-with-3-dimensions/19891/2
    new_sq_distances = sq_distances.gather(dim=1, index=new_neighb_inds.unsqueeze(-1).repeat(1,1,sq_distances.shape[-1])) 

    # New shadow neighbors have to point to the last shadow point
    new_neighbors_indices *= new_neighb_bool.long()
    new_neighbors_indices += (1 - new_neighb_bool.long()) * shadow_ind

    # Get Kernel point influences [n_points, n_kpoints, n_neighbors]
    if KP_influence == 'constant':
        # Every point get an influence of 1.
        all_weights = (new_sq_distances < KP_extent ** 2).float32()
        all_weights = all_weights.transpose(1, 2)

    elif KP_influence == 'linear':
        # Influence decrease linearly with the distance, and get to zero when d = KP_extent.
        corr = 1 - torch.sqrt(new_sq_distances + 1e-10) / KP_extent
        all_weights = torch.max(corr, torch.zeros_like(new_sq_distances))
        all_weights = all_weights.transpose(1, 2)

    elif KP_influence == 'gaussian':
        # Influence in gaussian of the distance.
        sigma = KP_extent * 0.3
        all_weights = radius_gaussian(sq_distances, sigma)
        all_weights = all_weights.transpose(1, 2)
    else:
        raise ValueError('Unknown influence function type (config.KP_influence)')

    # In case of closest mode, only the closest KP can influence each point
    if mode == 'closest':
        pass
    elif mode != 'sum':
        raise ValueError("Unknown convolution mode. Should be 'closest' or 'sum'")

    features = torch.cat([features, torch.zeros_like(features[:1, :])], dim=0)

    # Get the features of each neighborhood [n_points, n_neighbors, in_fdim]
    neighborhood_features = features[new_neighbors_indices, :]

    # Apply distance weights [n_points, n_kpoints, in_fdim]
    weighted_features = torch.matmul(all_weights, neighborhood_features)

    # Apply modulations
    if modulations is not None:
        weighted_features *= modulations.unsqueeze(2)

    # Apply network weights [n_kpoints, n_points, out_fdim]
    weighted_features = weighted_features.transpose(0, 1)
    kernel_outputs = torch.matmul(weighted_features, K_values)

    # Convolution sum to get [n_points, out_fdim]
    output_features = torch.sum(kernel_outputs, dim=0, keepdim=False)

    # normalization term.
    # neighbor_features_sum = torch.sum(neighborhood_features, dim=-1)
    # neighbor_num = torch.sum(torch.gt(neighbor_features_sum, 0.0), dim=-1)
    # neighbor_num = torch.max(neighbor_num, torch.ones_like(neighbor_num))
    # output_features = output_features / neighbor_num.unsqueeze(1)

    return output_features
