### PointNet++: *Deep Hierarchical Feature Learning on Point Sets in a Metric Space*
Created by <a href="http://charlesrqi.com" target="_blank">Charles R. Qi</a>, <a href="http://stanford.edu/~ericyi">Li (Eric) Yi</a>, <a href="http://ai.stanford.edu/~haosu/" target="_blank">Hao Su</a>, <a href="http://geometry.stanford.edu/member/guibas/" target="_blank">Leonidas J. Guibas</a> from Stanford University.

![prediction example](https://github.com/charlesq34/pointnet2/blob/master/doc/teaser.jpg)

### Citation
If you find our work useful in your research, please consider citing:

        @article{qi2017pointnetplusplus,
          title={PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space},
          author={Qi, Charles R and Yi, Li and Su, Hao and Guibas, Leonidas J},
          journal={arXiv preprint arXiv:1706.02413},
          year={2017}
        }

### Introduction
This work is based on our NIPS'17 paper. You can find arXiv version of the paper <a href="https://arxiv.org/pdf/1706.02413.pdf">here</a> or check <a href="http://stanford.edu/~rqi/pointnet2">project webpage</a> for a quick overview. PointNet++ is a follow-up project that builds on and extends <a href="https://github.com/charlesq34/pointnet">PointNet</a>. It is version 2.0 of the PointNet architecture.

PointNet (the v1 model) either transforms features of *individual points* independently or process global features of the *entire point set*. However, in many cases there are well defined distance metrics such as Euclidean distance for 3D point clouds collected by 3D sensors or geodesic distance for manifolds like isometric shape surfaces. In PointNet++ we want to respect *spatial localities* of those point sets. PointNet++ learns hierarchical features with increasing scales of contexts, just like that in convolutional neural networks. Besides, we also observe one challenge that is not present in convnets (with images) -- non-uniform densities in natural point clouds. To deal with those non-uniform densities, we further propose special layers that are able to intelligently aggregate information from different scales.

In this repository we release code and data for our PointNet++ classification and segmentation networks as well as a few utility scripts for training, testing and data processing and visualization.

### Installation

Install <a href="https://www.tensorflow.org/install/">TensorFlow</a>. The code is tested under TF1.2 GPU version and Python 2.7 (version 3 should also work) on Ubuntu 14.04. There are also some dependencies for a few Python libraries for data processing and visualizations like `cv2`, `h5py` etc. It's highly recommended that you have access to GPUs.

#### Compile Customized TF Operators
The TF operators are included under `tf_ops`, you need to compile them (check `tf_xxx_compile.sh` under each ops subfolder) first. Update `nvcc` and `python` path if necessary. The code is tested under TF1.2.0. If you are using earlier version it's possible that you need to remove the `-D_GLIBCXX_USE_CXX11_ABI=0` flag in g++ command in order to compile correctly.

To compile the operators in TF version >=1.4, you need to modify the compile scripts slightly.

First, find Tensorflow include and library paths.

        TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
        TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
        
Then, add flags of `-I$TF_INC/external/nsync/public -L$TF_LIB -ltensorflow_framework` to the `g++` commands.

### Usage

#### Shape Classification

To train a PointNet++ model to classify ModelNet40 shapes (using point clouds with XYZ coordinates):

        python train.py

To see all optional arguments for training:

        python train.py -h

If you have multiple GPUs on your machine, you can also run the multi-GPU version training (our implementation is similar to the tensorflow <a href="https://github.com/tensorflow/models/tree/master/tutorials/image/cifar10">cifar10 tutorial</a>):

        CUDA_VISIBLE_DEVICES=0,1 python train_multi_gpu.py --num_gpus 2

After training, to evaluate the classification accuracies (with optional multi-angle voting):

        python evaluate.py --num_votes 12 

<i>Side Note:</i> For the XYZ+normal experiment reported in our paper: (1) 5000 points are used and (2) a further random data dropout augmentation is used during training (see commented line after `augment_batch_data` in `train.py` and (3) the model architecture is updated such that the `nsample=128` in the first two set abstraction levels, which is suited for the larger point density in 5000-point samplings.

To use normal features for classification: You can get our sampled point clouds of ModelNet40 (XYZ and normal from mesh, 10k points per shape) <a href="https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip">here (1.6GB)</a>. Move the uncompressed data folder to `data/modelnet40_normal_resampled`

#### Object Part Segmentation

To train a model to segment object parts for ShapeNet models:

        cd part_seg
        python train.py

Preprocessed ShapeNetPart dataset (XYZ, normal and part labels) can be found <a href="https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip">here (674MB)</a>. Move the uncompressed data folder to `data/shapenetcore_partanno_segmentation_benchmark_v0_normal`

#### Semantic Scene Parsing

See `scannet/README` and `scannet/train.py` for details.

#### Visualization Tools
We have provided a handy point cloud visualization tool under `utils`. Run `sh compile_render_balls_so.sh` to compile it and then you can try the demo with `python show3d_balls.py` The original code is from <a href="http://github.com/fanhqme/PointSetGeneration">here</a>.

#### Prepare Your Own Data
You can refer to <a href="https://github.com/charlesq34/3dmodel_feature/blob/master/io/write_hdf5.py">here</a> on how to prepare your own HDF5 files for either classification or segmentation. Or you can refer to `modelnet_dataset.py` on how to read raw data files and prepare mini-batches from them. A more advanced way is to use TensorFlow's dataset APIs, for which you can find more documentations <a href="https://www.tensorflow.org/programmers_guide/datasets">here</a>.

### License
Our code is released under MIT License (see LICENSE file for details).

### Updates
* 02/23/2018: Added support for multi-gpu training for the classification task.
* 02/23/2018: Adopted a new way for data loading. No longer require manual data downloading to train a classification network.
* 02/06/2018: Added sample training code for ScanNet semantic segmentation.

### Related Projects

* <a href="http://stanford.edu/~rqi/pointnet" target="_blank">PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation</a> by Qi et al. (CVPR 2017 Oral Presentation). Code and data released in <a href="https://github.com/charlesq34/pointnet">GitHub</a>.
* <a href="https://arxiv.org/abs/1711.08488" target="_blank">Frustum PointNets for 3D Object Detection from RGB-D Data</a> by Qi et al. (CVPR 2018) A novel framework for 3D object detection with RGB-D data. Based on 2D boxes from a 2D object detector on RGB images, we extrude the depth maps in 2D boxes to point clouds in 3D space and then realize instance segmentation and 3D bounding box estimation using PointNet/PointNet++. The method proposed has achieved first place on KITTI 3D object detection benchmark on all categories (last checked on 11/30/2017). Code and data release TBD.
