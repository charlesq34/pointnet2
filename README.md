### PointNet++: *Deep Hierarchical Feature Learning on Point Sets in a Metric Space*
Created by <a href="http://charlesrqi.com" target="_blank">Charles R. Qi</a>, <a href="http://stanford.edu/~ericyi">Li (Eric) Yi</a>, <a href="http://ai.stanford.edu/~haosu/" target="_blank">Hao Su</a>, <a href="http://geometry.stanford.edu/member/guibas/" target="_blank">Leonidas J. Guibas</a> from Stanford University.

This is work is based on our paper linked <a href="https://arxiv.org/pdf/1706.02413.pdf">here</a>. The code release is still in an ongoing process... Stay tuned!

Current release includes TF operators (CPU and GPU), some core pointnet++ layers and a few example network models.

The TF operators are included under `tf_ops`, you need to compile them (check `tf_xxx_compile.sh` under each ops subfolder) first. Update `nvcc` and `python` path if necessary. The code is tested under TF1.2.0. If you are using earlier version it's possible that you need to remove the `-D_GLIBCXX_USE_CXX11_ABI=0` flag in g++ command in order to compile correctly.

TF and pointnet++ utility layers are defined under `utils/tf_util.py` and `utils/pointnet_util.py`

Under `models`, two classification models (SSG and MSG) and SSG models for part and semantic segmentation have been included.

#### Point Cloud Data
You can get our sampled point clouds of ModelNet40 (XYZ and normal from mesh, 10k points per shape) at this <a href="https://1drv.ms/u/s!ApbTjxa06z9CgQfKl99yUDHL_wHs">OneDrive link</a>.
