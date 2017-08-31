''' Furthest point sampling
Original author: Haoqiang Fan
Modified by Charles R. Qi
All Rights Reserved. 2017. 
'''
import tensorflow as tf
from tensorflow.python.framework import ops
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sampling_module=tf.load_op_library(os.path.join(BASE_DIR, 'tf_sampling_so.so'))
def prob_sample(inp,inpr):
    '''
input:
    batch_size * ncategory float32
    batch_size * npoints   float32
returns:
    batch_size * npoints   int32
    '''
    return sampling_module.prob_sample(inp,inpr)
ops.NoGradient('ProbSample')
# TF1.0 API requires set shape in C++
#@tf.RegisterShape('ProbSample')
#def _prob_sample_shape(op):
#    shape1=op.inputs[0].get_shape().with_rank(2)
#    shape2=op.inputs[1].get_shape().with_rank(2)
#    return [tf.TensorShape([shape2.dims[0],shape2.dims[1]])]
def gather_point(inp,idx):
    '''
input:
    batch_size * ndataset * 3   float32
    batch_size * npoints        int32
returns:
    batch_size * npoints * 3    float32
    '''
    return sampling_module.gather_point(inp,idx)
#@tf.RegisterShape('GatherPoint')
#def _gather_point_shape(op):
#    shape1=op.inputs[0].get_shape().with_rank(3)
#    shape2=op.inputs[1].get_shape().with_rank(2)
#    return [tf.TensorShape([shape1.dims[0],shape2.dims[1],shape1.dims[2]])]
@tf.RegisterGradient('GatherPoint')
def _gather_point_grad(op,out_g):
    inp=op.inputs[0]
    idx=op.inputs[1]
    return [sampling_module.gather_point_grad(inp,idx,out_g),None]
def farthest_point_sample(npoint,inp):
    '''
input:
    int32
    batch_size * ndataset * 3   float32
returns:
    batch_size * npoint         int32
    '''
    return sampling_module.farthest_point_sample(inp, npoint)
ops.NoGradient('FarthestPointSample')
    

if __name__=='__main__':
    import numpy as np
    np.random.seed(100)
    triangles=np.random.rand(1,5,3,3).astype('float32')
    with tf.device('/gpu:1'):
        inp=tf.constant(triangles)
        tria=inp[:,:,0,:]
        trib=inp[:,:,1,:]
        tric=inp[:,:,2,:]
        areas=tf.sqrt(tf.reduce_sum(tf.cross(trib-tria,tric-tria)**2,2)+1e-9)
        randomnumbers=tf.random_uniform((1,8192))
        triids=prob_sample(areas,randomnumbers)
        tria_sample=gather_point(tria,triids)
        trib_sample=gather_point(trib,triids)
        tric_sample=gather_point(tric,triids)
        us=tf.random_uniform((1,8192))
        vs=tf.random_uniform((1,8192))
        uplusv=1-tf.abs(us+vs-1)
        uminusv=us-vs
        us=(uplusv+uminusv)*0.5
        vs=(uplusv-uminusv)*0.5
        pt_sample=tria_sample+(trib_sample-tria_sample)*tf.expand_dims(us,-1)+(tric_sample-tria_sample)*tf.expand_dims(vs,-1)
        print 'pt_sample: ', pt_sample
        reduced_sample=gather_point(pt_sample,farthest_point_sample(1024,pt_sample))
        print reduced_sample
    with tf.Session('') as sess:
        ret=sess.run(reduced_sample)
    print ret.shape,ret.dtype
    import cPickle as pickle
    pickle.dump(ret,open('1.pkl','wb'),-1)
