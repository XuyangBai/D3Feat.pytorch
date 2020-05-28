import numpy as np
import kernels.convolution_ops as conv_ops
import torch
from torch import nn
from torch.nn import functional as F


# ----------------------------------------------------------------------------------------------------------------------
#
#           Utilities
#       \***************/
#

def weight_variable(size):
    # https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/21
    initial = np.random.normal(scale=np.sqrt(2 / size[-1]), size=size)
    initial[initial > 2 * np.sqrt(2 / size[-1])] = 0  # truncated
    initial[initial < -2 * np.sqrt(2 / size[-1])] = 0  # truncated
    weight = nn.Parameter(torch.from_numpy(initial).float(), requires_grad=True)
    return weight


def bias_variable(size):
    initial = torch.zeros(size=size, dtype=torch.float32)
    bias = nn.Parameter(initial, requires_grad=True)
    return bias


def leaky_relu_layer(negative_slope=0.1):
    return nn.LeakyReLU(negative_slope=negative_slope)


def ind_max_pool(features, inds):
    """
    This pytorch operation compute a maxpooling according to the list of indices 'inds'.
    > x = [n1, d] features matrix
    > inds = [n2, max_num] each row of this tensor is a list of indices of features to be pooled together
    >> output = [n2, d] pooled features matrix
    """

    # Add a last row with minimum features for shadow pools
    features = torch.cat([features, torch.min(features, dim=0, keepdim=True)[0]], dim=0)

    # Get features for each pooling cell [n2, max_num, d]
    pool_features = features[inds, :]

    # Pool the maximum
    return torch.max(pool_features, dim=1)[0]

def gather(x, idx, method=2):
    """
    implementation of a custom gather operation for faster backwards.
    :param x: input with shape [N, D_1, ... D_d]
    :param idx: indexing with shape [n_1, ..., n_m]
    :param method: Choice of the method
    :return: x[idx] with shape [n_1, ..., n_m, D_1, ... D_d]
    """

    if method == 0:
        return x[idx]
    elif method == 1:
        x = x.unsqueeze(1)
        x = x.expand((-1, idx.shape[-1], -1))
        idx = idx.unsqueeze(2)
        idx = idx.expand((-1, -1, x.shape[-1]))
        return x.gather(0, idx)
    elif method == 2:
        for i, ni in enumerate(idx.size()[1:]):
            x = x.unsqueeze(i+1)
            new_s = list(x.size())
            new_s[i+1] = ni
            x = x.expand(new_s)
        n = len(idx.size())
        for i, di in enumerate(x.size()[n:]):
            idx = idx.unsqueeze(i+n)
            new_s = list(idx.size())
            new_s[i+n] = di
            idx = idx.expand(new_s)
        return x.gather(0, idx)
    else:
        raise ValueError('Unkown method')


def closest_pool(features, upsample_indices):
    """
    This tensorflow operation compute a pooling according to the list of indices 'inds'.
    > features = [n1, d] features matrix
    > upsample_indices = [n2, max_num] We only use the first column of this which should be the closest points too pooled positions
    >> output = [n2, d] pooled features matrix
    """

    # Add a last row with minimum features for shadow pools
    features = torch.cat([features, torch.zeros_like(features[:1, :])], dim=0)

    # Get features for each pooling cell [n2, d]
    # pool_features = features[upsample_indices[:, 0], :]
    pool_features = gather(features, upsample_indices[:, 0], method=2)

    return pool_features


# def KPConv(query_points, support_points, neighbors_indices, features, K_values, radius, config):
#     """
#     Returns the output features of a KPConv
#     """
#
#     # Get KP extent from current radius and config density
#     extent = config.KP_extent * radius / config.density_parameter
#
#     # Convolution
#     return conv_ops.KPConv(query_points,
#                            support_points,
#                            neighbors_indices,
#                            features,
#                            K_values,
#                            fixed=config.fixed_kernel_points,
#                            KP_extent=extent,
#                            KP_influence=config.KP_influence,
#                            aggregation_mode=config.convolution_mode, )


class unary_block(nn.Module):
    def __init__(self, config, in_fdim, out_fdim):
        super(unary_block, self).__init__()
        self.config = config
        self.in_fdim, self.out_fdim = in_fdim, out_fdim
        self.weight = weight_variable([in_fdim, out_fdim])
        if config.use_batch_norm:
            self.bn = nn.BatchNorm1d(out_fdim, momentum=config.batch_norm_momentum, eps=1e-6)
        self.relu = leaky_relu_layer()

    def forward(self, query_points, support_points, neighbors_indices, features):
        """
        This module performs a unary 1x1 convolution (same with MLP)
        :param features: float32[n_points, in_fdim] - input features
        :return: output_features float32[n_points, out_fdim]
        """
        x = conv_ops.unary_convolution(features, self.weight)
        if self.config.use_batch_norm:
            x = self.relu(self.bn(x))
        else:
            x = self.relu(x)
        return x

    def __repr__(self):
        return f'unary(in_fdim={self.in_fdim}, out_fdim={self.out_fdim})'


class last_unary_block(nn.Module):
    def __init__(self, config, in_fdim, out_fdim):
        super(last_unary_block, self).__init__()
        self.config = config
        self.in_fdim, self.out_fdim = in_fdim, out_fdim
        self.weight = weight_variable([in_fdim, out_fdim])

    def forward(self, query_points, support_points, neighbors_indices, features):
        x = conv_ops.unary_convolution(features, self.weight)
        return x

    def __repr__(self):
        return f'last_unary(in_fdim={self.in_fdim}, out_fdim={self.out_fdim})'


class simple_block(nn.Module):
    def __init__(self, config, in_fdim, out_fdim, radius, strided=False):
        super(simple_block, self).__init__()
        self.config = config
        self.radius = radius
        self.strided = strided
        self.in_fdim, self.out_fdim = in_fdim, out_fdim // 2

        # kernel points weight
        self.weight = weight_variable([config.num_kernel_points, in_fdim, out_fdim])
        if config.use_batch_norm:
            self.bn = nn.BatchNorm1d(out_fdim, momentum=config.batch_norm_momentum, eps=1e-6)
        self.relu = leaky_relu_layer()

    def forward(self, query_points, support_points, neighbors_indices, features):
        """
        This module performs a Kernel Point Convolution. (both normal and strided version)
        :param query_points: float32[n_points, dim] - input query points (center of neighborhoods)
        :param support_points: float32[n0_points, dim] - input support points (from which neighbors are taken)
        :param neighbors_indices: int32[n_points, n_neighbors] - indices of neighbors of each point
        :param features: float32[n_points, in_fdim] - input features
        :return: output_features float32[n_points, out_fdim]
        """
        x = conv_ops.KPConv(query_points,
                            support_points,
                            neighbors_indices,
                            features,
                            K_values=self.weight,
                            fixed=self.config.fixed_kernel_points,
                            KP_extent=self.config.KP_extent * self.radius / self.config.density_parameter,
                            KP_influence=self.config.KP_influence,
                            aggregation_mode=self.config.convolution_mode, )
        if self.config.use_batch_norm:
            x = self.relu(self.bn(x))
        else:
            x = self.relu(x)
        return x

    def __repr__(self):
        return f'simple(in_fdim={self.in_fdim}, out_fdim={self.out_fdim})'


class simple_deformable_block(nn.Module):
    def __init__(self, config, in_fdim, out_fdim, radius, strided=False):
        super(simple_deformable_block, self).__init__()
        self.config = config
        self.radius = radius
        self.strided = strided
        self.in_fdim, self.out_fdim = in_fdim, out_fdim
        self.deformable_v2 = False

        # kernel points weight
        self.weight = weight_variable([config.num_kernel_points, in_fdim, out_fdim])
        if config.use_batch_norm:
            self.bn = nn.BatchNorm1d(out_fdim, momentum=config.batch_norm_momentum, eps=1e-6)
        self.relu = leaky_relu_layer()

        point_dim = 4 if self.config.modulated else 3
        if self.deformable_v2:
            # for deformable_v2, there is no offset for the first kernel point.
            offset_dim = point_dim * (config.num_kernel_points - 1)
            self.offset_weight = weight_variable([in_fdim, offset_dim])
            self.offset_bias = bias_variable([offset_dim])
        else:
            offset_dim = point_dim * config.num_kernel_points
            self.offset_weight = weight_variable([config.num_kernel_points, in_fdim, offset_dim])
            self.offset_bias = bias_variable([offset_dim])

    def forward(self, query_points, support_points, neighbors_indices, features):
        """
        This module performs a Kernel Point Convolution. (both normal and strided version)
        :param query_points: float32[n_points, dim] - input query points (center of neighborhoods)
        :param support_points: float32[n0_points, dim] - input support points (from which neighbors are taken)
        :param neighbors_indices: int32[n_points, n_neighbors] - indices of neighbors of each point
        :param features: float32[n_points, in_fdim] - input features
        :return: output_features float32[n_points, out_fdim]
        """
        if self.deformable_v2:
            x = conv_ops.KPConv_deformable_v2(query_points,
                                              support_points,
                                              neighbors_indices,
                                              features,
                                              K_values=self.weight,
                                              w0=self.offset_weight,
                                              b0=self.offset_bias,
                                              fixed=self.config.fixed_kernel_points,
                                              KP_extent=self.config.KP_extent * self.radius / self.config.density_parameter,
                                              KP_influence=self.config.KP_influence,
                                              aggregation_mode=self.config.convolution_mode,
                                              modulated=self.config.modulated)
        else:
            x = conv_ops.KPConv_deformable(query_points,
                                           support_points,
                                           neighbors_indices,
                                           features,
                                           K_values=self.weight,
                                           w0=self.offset_weight,
                                           b0=self.offset_bias,
                                           fixed=self.config.fixed_kernel_points,
                                           KP_extent=self.config.KP_extent * self.radius / self.config.density_parameter,
                                           KP_influence=self.config.KP_influence,
                                           aggregation_mode=self.config.convolution_mode,
                                           modulated=self.config.modulated)
        if self.config.use_batch_norm:
            x = self.relu(self.bn(x))
        else:
            x = self.relu(x)
        return x

    def __repr__(self):
        return f'simple_deformable(in_fdim={self.in_fdim}, out_fdim={self.out_fdim})'


class resnet_block(nn.Module):
    def __init__(self, config, in_fdim, out_fdim, radius):
        super(resnet_block, self).__init__()
        self.config = config
        self.radius = radius
        self.in_fdim, self.out_fdim = in_fdim, out_fdim
        self.conv1 = simple_block(config, in_fdim, out_fdim, radius)
        self.conv2 = simple_block(config, out_fdim, out_fdim, radius)
        self.shortcut = unary_block(config, in_fdim, out_fdim)
        self.relu = leaky_relu_layer()

    def forward(self, query_points, support_points, neighbors_indices, features):
        """
        This module performs a resnet double convolution (two convolution vgglike and a shortcut)
        :param query_points: float32[n_points, dim] - input query points (center of neighborhoods)
        :param support_points: float32[n0_points, dim] - input support points (from which neighbors are taken)
        :param neighbors_indices: int32[n_points, n_neighbors] - indices of neighbors of each point
        :param features: float32[n_points, in_fdim] - input features
        :return: output_features float32[n_points, out_fdim]
        """
        shortcut = self.shortcut(query_points, support_points, neighbors_indices, features)
        features = self.conv1(query_points, support_points, neighbors_indices, features)
        features = self.conv2(query_points, support_points, neighbors_indices, features)
        return self.relu(shortcut + features)


class resnetb_block(nn.Module):
    def __init__(self, config, in_fdim, out_fdim, radius, strided):
        super(resnetb_block, self).__init__()
        self.config = config
        self.radius = radius
        self.strided = strided
        self.in_fdim, self.out_fdim = in_fdim, out_fdim
        self.conv1 = unary_block(config, in_fdim, out_fdim // 4)
        self.conv2 = simple_block(config, out_fdim // 4, out_fdim // 4, radius, strided=strided)
        # TODO: origin implementation this last conv change feature dim to out_fdim * 2
        self.conv3 = unary_block(config, out_fdim // 4, out_fdim)
        self.shortcut = unary_block(config, in_fdim, out_fdim)
        self.relu = leaky_relu_layer()

    def forward(self, query_points, support_points, neighbors_indices, features):
        """
        This module performs a resnet bottleneck convolution (1conv > KPconv > 1conv + shortcut)
        Both resnetb and resnetb_strided use the module.
        :param query_points: float32[n_points, dim] - input query points (center of neighborhoods)
        :param support_points: float32[n0_points, dim] - input support points (from which neighbors are taken)
        :param neighbors_indices: int32[n_points, n_neighbors] - indices of neighbors of each point
        :param features: float32[n_points, in_fdim] - input features
        :return: output_features float32[n_points, out_fdim]
        """
        origin_features = features  # save for shortcut
        features = self.conv1(query_points, support_points, neighbors_indices, features)
        features = self.conv2(query_points, support_points, neighbors_indices, features)
        features = self.conv3(query_points, support_points, neighbors_indices, features)
        # TODO: origin implementation has two kinds of shortcut.
        if self.strided is False:  # for resnetb
            shortcut = self.shortcut(query_points, support_points, neighbors_indices, origin_features)
        else:  # for resnetb_strided
            pool_features = ind_max_pool(origin_features, neighbors_indices)
            shortcut = self.shortcut(query_points, support_points, neighbors_indices, pool_features)
        return self.relu(shortcut + features)


class resnetb_deformable_block(nn.Module):
    def __init__(self, config, in_fdim, out_fdim, radius, strided):
        super(resnetb_deformable_block, self).__init__()
        self.config = config
        self.radius = radius
        self.strided = strided
        self.in_fdim, self.out_fdim = in_fdim, out_fdim
        self.conv1 = unary_block(config, in_fdim, out_fdim // 4)
        self.conv2 = simple_deformable_block(config, out_fdim // 4, out_fdim // 4, radius, strided=strided)
        # TODO: origin implementation this last conv change feature dim to out_fdim * 2
        self.conv3 = unary_block(config, out_fdim // 4, out_fdim)
        self.shortcut = unary_block(config, in_fdim, out_fdim)
        self.relu = leaky_relu_layer()

    def forward(self, query_points, support_points, neighbors_indices, features):
        """
        This module performs a resnet deformable bottleneck convolution (1conv > KPconv > 1conv + shortcut)
        Both resnetb_deformable and resnetb_deformable_strided use the module.
        :param query_points: float32[n_points, dim] - input query points (center of neighborhoods)
        :param support_points: float32[n0_points, dim] - input support points (from which neighbors are taken)
        :param neighbors_indices: int32[n_points, n_neighbors] - indices of neighbors of each point
        :param features: float32[n_points, in_fdim] - input features
        :return: output_features float32[n_points, out_fdim]
        """
        origin_features = features  # save for shortcut
        features = self.conv1(query_points, support_points, neighbors_indices, features)
        features = self.conv2(query_points, support_points, neighbors_indices, features)
        features = self.conv3(query_points, support_points, neighbors_indices, features)
        # TODO: origin implementation has two kinds of shortcut.
        if self.strided is False:  # for resnetb
            shortcut = self.shortcut(query_points, support_points, neighbors_indices, origin_features)
        else:  # for resnetb_strided
            pool_features = ind_max_pool(origin_features, neighbors_indices)
            shortcut = self.shortcut(query_points, support_points, neighbors_indices, pool_features)
        return self.relu(shortcut + features)


class nearest_upsample_block(nn.Module):
    def __init__(self, config):
        super(nearest_upsample_block, self).__init__()
        self.config = config

    def forward(self, upsample_indices, features):
        """
        This module performs an upsampling by nearest interpolation
        :param TODO: upsample_indices
        :param features: float32[n_points, in_fdim] - input features
        :return:
        """
        pool_features = closest_pool(features, upsample_indices)
        return pool_features


class global_average_block(nn.Module):
    def __init__(self, config):
        super(global_average_block, self).__init__()
        self.config = config

    def forward(self, stack_lengths, features):
        """
        This module performs a global average over batch pooling
        :param features: float32[n_points, in_fdim] - input features
        :return: output_features: float32[batch_size, in_fdim]
        """
        start_ind = 0
        average_feature_list = []
        for length in stack_lengths[-1]:
            tmp = torch.mean(features[start_ind:start_ind + length], dim=0, keepdim=True)
            average_feature_list.append(tmp)
            start_ind += length
        return torch.cat(average_feature_list, dim=0)


def get_block(block_name, config, in_fdim, out_fdim, radius, strided):
    if block_name == 'unary':
        return unary_block(config, in_fdim, out_fdim)
    if block_name == 'last_unary':
        return last_unary_block(config, in_fdim, out_fdim)
    if block_name == 'simple' or block_name == 'simple_strided':
        return simple_block(config, in_fdim, out_fdim, radius=radius, strided=strided)
    if block_name == 'nearest_upsample':
        return nearest_upsample_block(config)
    if block_name == 'global_average':
        return global_average_block(config)
    if block_name == 'resnet':  # or block_name == 'resnet_strided':
        return resnet_block(config, in_fdim, out_fdim, radius=radius)
    if block_name == 'resnetb' or block_name == 'resnetb_strided':
        return resnetb_block(config, in_fdim, out_fdim, radius=radius, strided=strided)
    if block_name == 'resnetb_deformable' or block_name == 'resnetb_deformable_strided':
        return resnetb_deformable_block(config, in_fdim, out_fdim, radius=radius, strided=strided)
