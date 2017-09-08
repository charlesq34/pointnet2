import tensorflow as tf
from tensorflow.python.framework import ops
import sys
import os
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
interpolate_module=tf.load_op_library(os.path.join(BASE_DIR, 'tf_interpolate_so.so'))
def three_nn(xyz1, xyz2):
    '''
    Input:
        xyz1: (b,n,3) float32 array, unknown points
        xyz2: (b,m,3) float32 array, known points
    Output:
        dist: (b,n,3) float32 array, distances to known points
        idx: (b,n,3) int32 array, indices to known points
    '''
    return interpolate_module.three_nn(xyz1, xyz2)
ops.NoGradient('ThreeNN')
def three_interpolate(points, idx, weight):
    '''
    Input:
        points: (b,m,c) float32 array, known points
        idx: (b,n,3) int32 array, indices to known points
        weight: (b,n,3) float32 array, weights on known points
    Output:
        out: (b,n,c) float32 array, interpolated point values
    '''
    return interpolate_module.three_interpolate(points, idx, weight)
@tf.RegisterGradient('ThreeInterpolate')
def _three_interpolate_grad(op, grad_out):
    points = op.inputs[0]
    idx = op.inputs[1]
    weight = op.inputs[2]
    return [interpolate_module.three_interpolate_grad(points, idx, weight, grad_out), None, None]

if __name__=='__main__':
    import numpy as np
    import time
    np.random.seed(100)
    pts = np.random.random((32,128,64)).astype('float32')
    tmp1 = np.random.random((32,512,3)).astype('float32')
    tmp2 = np.random.random((32,128,3)).astype('float32')
    with tf.device('/cpu:0'):
        points = tf.constant(pts)
        xyz1 = tf.constant(tmp1)
        xyz2 = tf.constant(tmp2)
        dist, idx = three_nn(xyz1, xyz2)
        weight = tf.ones_like(dist)/3.0
        interpolated_points = three_interpolate(points, idx, weight)
    with tf.Session('') as sess:
        now = time.time() 
        for _ in range(100):
            ret = sess.run(interpolated_points)
        print(time.time() - now)
        print(ret.shape, ret.dtype)
        #print ret
    
    
    
