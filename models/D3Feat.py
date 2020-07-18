from models.network_blocks import get_block
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class KPCNN(nn.Module):
    def __init__(self, config):
        super(KPCNN, self).__init__()
        self.config = config
        self.blocks = nn.ModuleDict()

        # Feature Extraction Module
        r = config.first_subsampling_dl * config.deform_radius
        in_fdim = config.in_features_dim
        out_fdim = config.first_features_dim
        layer = 0
        for block_i, block in enumerate(config.architecture):
            # Detect upsampling block to stop
            if 'upsample' in block:
                break

            is_strided = 'strided' in block
            self.blocks[f'layer{layer}/{block}'] = get_block(block, config, in_fdim, out_fdim, radius=r, strided=is_strided)

            # Update dimension of input from output
            in_fdim = out_fdim

            if 'pool' in block or 'strided' in block:
                # Update radius and feature dimension for next layer
                out_fdim *= 2
                r *= 2
                layer += 1

    def forward(self, inputs):
        F = self.feature_extraction(inputs)
        return F

    def feature_extraction(self, inputs):
        # Current radius of convolution and feature dimension
        r = self.config.first_subsampling_dl * self.config.deform_radius
        layer = 0
        fdim = self.config.first_features_dim

        # Input features
        features = inputs['features']
        F = []

        # Loop over consecutive blocks
        block_in_layer = 0
        for block_i, block in enumerate(self.config.architecture):

            # Detect change to next layer
            if np.any([tmp in block for tmp in ['pool', 'strided', 'upsample', 'global']]):
                # Save this layer features
                F += [features]

            # Detect upsampling block to stop
            if 'upsample' in block:
                break

            # Get the function for this layer
            block_ops = self.blocks[f'layer{layer}/{block}']

            # Apply the layer function defining tf ops
            if block == 'global_average':
                stack_lengths = inputs['stack_lengths']
                features = block_ops(stack_lengths, features)
            else:
                if block in ['unary', 'simple', 'resnet', 'resnetb', 'resnetb_deformable']:
                    query_points = inputs['points'][layer]
                    support_points = inputs['points'][layer]
                    neighbors_indices = inputs['neighbors'][layer]
                elif block in ['simple_strided', 'resnetb_strided', 'resnetb_deformable_strided']:
                    query_points = inputs['points'][layer + 1]
                    support_points = inputs['points'][layer]
                    neighbors_indices = inputs['pools'][layer]
                else:
                    raise ValueError("Unknown block type.")
                features = block_ops(query_points, support_points, neighbors_indices, features)

            # Index of block in this layer
            block_in_layer += 1

            # Detect change to a subsampled layer
            if 'pool' in block or 'strided' in block:
                # Update radius and feature dimension for next layer
                layer += 1
                r *= 2
                fdim *= 2
                block_in_layer = 0

            # Save feature vector after global pooling
            if 'global' in block:
                # Save this layer features
                F += [features]
        return F


class KPFCNN(nn.Module):
    def __init__(self, config):
        super(KPFCNN, self).__init__()
        self.encoder = KPCNN(config)
        self.config = config
        self.blocks = nn.ModuleDict()

        # Feature Extraction Module
        # Find first upsampling block
        start_i = 0
        for block_i, block in enumerate(config.architecture):
            if 'upsample' in block:
                start_i = block_i
                break

        layer = config.num_layers - 1
        r = config.first_subsampling_dl * config.deform_radius * 2 ** layer
        in_fdim = config.first_features_dim * 2 ** layer
        out_fdim = in_fdim
        for block_i, block in enumerate(config.architecture[start_i:]):

            is_strided = 'strided' in block
            if block != 'last_unary':
                self.blocks[f'layer{layer}/{block}'] = get_block(block, config, int(1.5 * in_fdim), out_fdim, radius=r, strided=is_strided)
            else:
                self.blocks[f'layer{layer}/{block}'] = get_block(block, config, in_fdim, out_fdim, radius=r, strided=is_strided)

            # update feature dimension
            in_fdim = out_fdim

            # Detect change to a subsampled layer
            if 'upsample' in block:
                # Update radius and feature dimension for next layer
                out_fdim = out_fdim // 2
                r *= 0.5
                layer -= 1

        # print(list(self.named_parameters()))

    def forward(self, inputs):
        features = self.feature_extraction(inputs)
        scores = self.detection_scores(inputs, features)
        features = F.normalize(features, p=2, dim=-1)
        return features, scores

    def feature_extraction(self, inputs):
        F = self.encoder.feature_extraction(inputs)
        features = F[-1]

        # Current radius of convolution and feature dimension
        layer = self.config.num_layers - 1
        r = self.config.first_subsampling_dl * self.config.deform_radius * 2 ** layer
        fdim = self.config.first_features_dim * 2 ** layer

        # Find first upsampling block
        start_i = 0
        for block_i, block in enumerate(self.config.architecture):
            if 'upsample' in block:
                start_i = block_i
                break

        # Loop over upsampling blocks
        for block_i, block in enumerate(self.config.architecture[start_i:]):

            # Get the function for this layer
            block_ops = self.blocks[f'layer{layer}/{block}']

            # Apply the layer function defining tf ops
            if 'upsample' in block:
                if block == 'nearest_upsample':
                    upsample_indices = inputs['upsamples'][layer - 1]
                else:
                    raise ValueError(f"Unknown block type. {block}")
                features = block_ops(upsample_indices, features)
            else:
                if block in ['unary', 'simple', 'resnet', 'resnetb', 'last_unary']:
                    query_points = inputs['points'][layer]
                    support_points = inputs['points'][layer]
                    neighbors_indices = inputs['neighbors'][layer]
                elif block in ['simple_strided', 'resnetb_strided', 'resnetb_deformable_strided']:
                    query_points = inputs['points'][layer + 1]
                    support_points = inputs['points'][layer]
                    neighbors_indices = inputs['pools'][layer]
                else:
                    raise ValueError(f"Unknown block type. {block}")
                features = block_ops(query_points, support_points, neighbors_indices, features)

            # Detect change to a subsampled layer
            if 'upsample' in block:
                # Update radius and feature dimension for next layer
                layer -= 1
                r *= 0.5
                fdim = fdim // 2

                # Concatenate with CNN feature map
                features = torch.cat((features, F[layer]), dim=1)

        return features

    def detection_scores(self, inputs, features):
        neighbor = inputs['neighbors'][0]  # [n_points, n_neighbors]
        first_pcd_length, second_pcd_length = inputs['stack_lengths'][0]

        first_pcd_indices = torch.arange(first_pcd_length)
        second_pcd_indices = torch.arange(first_pcd_length, first_pcd_length+second_pcd_length)

        # add a fake point in the last row for shadow neighbors
        shadow_features = torch.zeros_like(features[:1, :])
        features = torch.cat([features, shadow_features], dim=0)
        shadow_neighbor = torch.ones_like(neighbor[:1, :]) * (first_pcd_length + second_pcd_length)
        neighbor = torch.cat([neighbor, shadow_neighbor], dim=0)

        # #  normalize the feature to avoid overflow
        # point_cloud_feature0 = torch.max(features[first_pcd_indices])
        # point_cloud_feature1 = torch.max(features[second_pcd_indices])
        # max_per_sample =  torch.cat([
        #     torch.stack([point_cloud_feature0] * first_pcd_length, dim=0),
        #     torch.stack([point_cloud_feature1] * (second_pcd_length+1), dim=0)
        # ], dim=0)
        features = features / (torch.max(features) + 1e-6)

        # local max score (saliency score)
        neighbor_features = features[neighbor, :] # [n_points, n_neighbors, 64]
        neighbor_features_sum = torch.sum(neighbor_features, dim=-1)  # [n_points, n_neighbors]
        neighbor_num = (neighbor_features_sum != 0).sum(dim=-1, keepdims=True)  # [n_points, 1]
        neighbor_num = torch.max(neighbor_num, torch.ones_like(neighbor_num))
        mean_features = torch.sum(neighbor_features, dim=1) / neighbor_num  # [n_points, 64]
        local_max_score = F.softplus(features - mean_features)  # [n_points, 64]

        # calculate the depth-wise max score
        depth_wise_max = torch.max(features, dim=1, keepdims=True)[0]  # [n_points, 1]
        depth_wise_max_score = features / (1e-6 + depth_wise_max)  # [n_points, 64]

        all_scores = local_max_score * depth_wise_max_score
        # use the max score among channel to be the score of a single point. 
        scores = torch.max(all_scores, dim=1, keepdims=True)[0]  # [n_points, 1]

        # hard selection (used during test)
        if self.training is False:
            local_max = torch.max(neighbor_features, dim=1)[0]
            is_local_max = (features == local_max)
            # print(f"Local Max Num: {float(is_local_max.sum().detach().cpu())}")
            detected = torch.max(is_local_max.float(), dim=1, keepdims=True)[0]
            scores = scores * detected


        return scores[:-1, :]
