## KPConv.pytorch

This repo is implementation for KPConv(https://arxiv.org/abs/1904.08889) in pytorch.

## TODO
There are still some works to be done:
- [x] Deformable KPConv. Currently I have only implemented the rigid KPConv.
  - [ ] Regularization loss for the deformable convolution needs to be implemented. I have tried using the deformable convolution layer in part segmention on shapenet without the regularization term, the performance is similar with the rigid convolution counterparts.
- [x] Speed up. For current implementation, the `collate_fn` where the neighbor indices and pooling indices are calculated, is too slow. In the tf version, the author implement 2 tensroflow C++ wrapper which is quite efficient. I am planing to write C++ extention using pytorch. 
  - [ ] But after I implemented the C++ extention, the evaluation time reduces significantly while the model forward and backward pass still cost about 0.8s per iteration.
- [ ] Maybe other datasets.


## Installation

1. Create an environment from the environment.yml file,
```
conda env create -f environment.yml
```
2. Compile the customized Tensorflow operators and C++ extension module following the [installation instructions](https://github.com/HuguesTHOMAS/KPConv/blob/master/INSTALL.md) provided by the authors.
3. Go to `pytorch_ops` dictionary and run `python setup.py install` to build and install the C++ extension for `batch_find_neighbors` function.


## Experiments

Due to the time limitation, I have just implemented the experiments on ShapeNet(classification and part segmentation) and ModelNet40. 

- Shape Classification on ModelNet40 or ShapeNet.
```
python training_ModelNet.py[training_ShapeNetCls.py]
```

- Part Segmentation on ShapeNet. (I have only implemented the single class part segmentation.)
```
python training_ShapeNetPart.py
```

## Acknowledgment

Thank @HuguesTHOMAS for sharing the tensorflow version and valuable explainations. 

