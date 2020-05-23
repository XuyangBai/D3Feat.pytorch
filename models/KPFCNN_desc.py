from models.network_blocks import get_block
from models.KPCNN import KPCNN
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        r = config.first_subsampling_dl * config.density_parameter * 2 ** layer
        in_fdim = config.first_features_dim * 2 ** layer
        out_fdim = in_fdim
        block_in_layer = 0
        for block_i, block in enumerate(config.architecture[start_i:]):

            is_strided = 'strided' in block
            self.blocks[f'layer{layer}/{block}'] = get_block(block, config, int(1.5 * in_fdim), out_fdim, radius=r, strided=is_strided)

            # update feature dimension
            in_fdim = out_fdim
            block_in_layer += 1

            # Detect change to a subsampled layer
            if 'upsample' in block:
                # Update radius and feature dimension for next layer
                out_fdim = out_fdim // 2
                r *= 0.5
                layer -= 1
                block_in_layer = 0

        # print(list(self.named_parameters()))

    def forward(self, inputs):
        features = self.feature_extraction(inputs)
        features = F.normalize(features, p=2, dim=-1)
        return features

    def feature_extraction(self, inputs):
        F = self.encoder.feature_extraction(inputs)
        features = F[-1]

        # Current radius of convolution and feature dimension
        layer = self.config.num_layers - 1
        r = self.config.first_subsampling_dl * self.config.density_parameter * 2 ** layer
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
                if block in ['unary', 'simple', 'resnet', 'resnetb']:
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


if __name__ == '__main__':
    from training_ShapeNetPart import ShapeNetPartConfig
    from datasets.ShapeNet import ShapeNetDataset
    from datasets.dataloader import get_dataloader

    config = ShapeNetPartConfig()
    datapath = "./data/shapenetcore_partanno_segmentation_benchmark_v0"
    dset = ShapeNetDataset(root=datapath, config=config, first_subsampling_dl=0.01, classification=False)
    dataloader = get_dataloader(dset, batch_size=1)
    model = KPFCNN(config)

    for iter, input in enumerate(dataloader):
        output = model(input)
        print(output.shape)
        break
