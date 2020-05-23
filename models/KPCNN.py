import numpy as np
from models.network_blocks import get_block, weight_variable
import torch
import torch.nn as nn


class KPCNN(nn.Module):
    def __init__(self, config):
        # TODO: One big difference is in tensorflow version, each block receives same parameters (layer_ind, inputs, features, radius, fdim, config)
        # which in my opinion is because it will be easy to write a unified code for creating models of different architecture, but the readability
        # of each block is not satisfactory since the block functions are coupling with the data prepartion pipelines.
        # Instead, in pytorch version, I save the fixed parameter(like radius, in_fdim, out_fdim) for each layer (or module) when creating the layer
        # and only provide the necessary inputs (query_points, supporting_points, neighbors_indices, features) during forward, so that the function
        # of each layer is more clear. And when we want to modify the data preparation pipeline, there is no need to change to code for KPConv layers.
        # We only need to modify the code for parsing the architecture list and feed each layer with the correct inputs.

        super(KPCNN, self).__init__()
        self.config = config
        self.blocks = nn.ModuleDict()

        # Feature Extraction Module
        r = config.first_subsampling_dl * config.density_parameter
        in_fdim = config.in_features_dim
        out_fdim = config.first_features_dim
        layer = 0
        block_in_layer = 0
        for block_i, block in enumerate(config.architecture):
            # Detect upsampling block to stop
            if 'upsample' in block:
                break

            is_strided = 'strided' in block
            self.blocks[f'layer{layer}/{block}'] = get_block(block, config, in_fdim, out_fdim, radius=r, strided=is_strided)

            # update feature dimension
            in_fdim = out_fdim
            block_in_layer += 1

            if 'pool' in block or 'strided' in block:
                # Update radius and feature dimension for next layer
                out_fdim *= 2
                r *= 2
                layer += 1
                block_in_layer = 0

        # Classification Head
        self.blocks['classification_head'] = nn.Sequential(
            nn.Linear(out_fdim, 1024),
            # nn.BatchNorm1d(1024, momentum=config.batch_norm_momentum, eps=1e-6),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(p=config.dropout),
            nn.Linear(1024, config.num_classes)
        )

        # print(list(self.parameters()))

    def forward(self, inputs):
        F = self.feature_extraction(inputs)
        logits = self.classification_head(F[-1])
        return logits

    def feature_extraction(self, inputs):
        # Current radius of convolution and feature dimension
        r = self.config.first_subsampling_dl * self.config.density_parameter
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

    def classification_head(self, features):
        logits = self.blocks['classification_head'](features)
        return logits


if __name__ == '__main__':
    from training_ShapeNetCls import ShapeNetPartConfig
    from datasets.ShapeNet import ShapeNetDataset
    from datasets.dataloader import get_dataloader

    config = ShapeNetPartConfig()
    datapath = "./data/shapenetcore_partanno_segmentation_benchmark_v0"
    dset = ShapeNetDataset(root=datapath, config=config, first_subsampling_dl=0.01, classification=True)
    dataloader = get_dataloader(dset, batch_size=4)
    model = KPCNN(config)

    for iter, input in enumerate(dataloader):
        output = model(input)
        print("Predict:", output)
        print("GT:", input['labels'])
        break
