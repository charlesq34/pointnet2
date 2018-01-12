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
This is work is based on our NIPS'17 paper. You can find arXiv version of the paper <a href="https://arxiv.org/pdf/1706.02413.pdf">here</a> or check <a href="http://stanford.edu/~rqi/pointnet2">project webpage</a> for a quick overview. PointNet++ is a follow-up project that builds on and extends <a href="https://github.com/charlesq34/pointnet">PointNet</a>. It's version 2.0 of the PointNet architecture.

While PointNet (the v1 model) either transforms features of *individual points* independently or process global features of the *entire point set*, in many cases, there are well defined distance metrics such as Euclidean distance for 3D point clouds collected by 3D sensors or geodesic distance for manifolds like isometric shape surfaces -- it'll be desired to respect the localities of those point sets data. Therefore we invent PointNet++ that learns hierarchical features with increasing scales of context, just like that in convolutional neural networks. Besides, we also observe one problem that's very different from conv nets on images -- non-uniform densities in natural point clouds. Thus we further propose special layers that's able to learn how to make use of points in regions with different densities.

In this repository we release code and data for our PointNet++ classification and segmentation networks as well as a few utility scripts for training, testing and data processing and visualization.

### Installation

Install <a href="https://www.tensorflow.org/install/">TensorFlow</a>. The code is tested under TF1.2 GPU version and Python 2.7 (version 3 should also work) on Ubuntu 14.04. There are also some dependencies for a few Python libraries for data processing and visualizations like `cv2`, `h5py` etc. It's highly recommended that you have access to GPUs.

The TF operators are included under `tf_ops`, you need to compile them (check `tf_xxx_compile.sh` under each ops subfolder) first. Update `nvcc` and `python` path if necessary. The code is tested under TF1.2.0. If you are using earlier version it's possible that you need to remove the `-D_GLIBCXX_USE_CXX11_ABI=0` flag in g++ command in order to compile correctly.

There is also a handy point cloud visualization tool under `utils`, run `sh compile_render_balls_so.sh` to compile it and you can try the demo with `python show3d_balls.py` The original code is from <a href="://github.com/fanhqme/PointSetGeneration">here</a>.

### Point Cloud Data
You can get our sampled point clouds of ModelNet40 (XYZ and normal from mesh, 10k points per shape) at this <a href="https://1drv.ms/u/s!ApbTjxa06z9CgQfKl99yUDHL_wHs">OneDrive link</a>. The ShapeNetPart dataset (XYZ, normal and part labels) can be found <a href="https://1drv.ms/u/s!ApbTjxa06z9CgQnl-Qm6KI3Ywbe1">here</a>. Uncompress them to the data folder such that it becomes:

        data/modelnet40_normal_resampled
        data/shapenetcore_partanno_segmentation_benchmark_v0_normal

so that training and testing scripts can successfully locate them.

### Usage

#### Shape Classification

To train a model to classify point clouds sampled from ModelNet40 shapes:

        cd classification
        python train.py

#### Object Part Segmentation

To train a model to segment object parts for ShapeNet models:

        cd part_seg
        python train.py

#### Scene Parsing
TBA

### License
Our code is released under MIT License (see LICENSE file for details).

### Related Projects

* <a href="http://stanford.edu/~rqi/pointnet" target="_blank">PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space</a> by Qi et al. (CVPR 2017 Oral Presentation). Code and data released in <a href="https://github.com/charlesq34/pointnet">GitHub</a>.
* <a href="https://arxiv.org/abs/1711.08488" target="_blank">Frustum PointNets for 3D Object Detection from RGB-D Data</a> by Qi et al. (arXiv) A novel framework for 3D object detection with RGB-D data. The method proposed has achieved first place on KITTI 3D object detection benchmark on all categories (last checked on 11/30/2017). Code and data release TBD.
