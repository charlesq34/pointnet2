''' Visualize part segmentation '''
import os
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('/home/rqi/Projects/toolkits/visualization')
from show3d_balls import showpoints
import numpy as np
from tf_interpolate import three_nn, three_interpolate
import tensorflow as tf


pts2 = np.array([[0,0,1],[1,0,0],[0,1,0],[1,1,0]]).astype('float32')
xyz1 = np.random.random((100,3)).astype('float32')
xyz2 = np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,1]]).astype('float32')

def fun(xyz1,xyz2,pts2):
    with tf.device('/cpu:0'):
        points = tf.constant(np.expand_dims(pts2,0))
        xyz1 = tf.constant(np.expand_dims(xyz1,0))
        xyz2 = tf.constant(np.expand_dims(xyz2,0))
        dist, idx = three_nn(xyz1, xyz2)
        #weight = tf.ones_like(dist)/3.0
        dist = tf.maximum(dist, 1e-10)
        norm = tf.reduce_sum((1.0/dist),axis=2,keep_dims=True)
        norm = tf.tile(norm, [1,1,3])
        print norm
        weight = (1.0/dist) / norm
        interpolated_points = three_interpolate(points, idx, weight)
    with tf.Session('') as sess:
        tmp,pts1,d,w = sess.run([xyz1, interpolated_points, dist, weight])
        #print w
        pts1 = pts1.squeeze()
    return pts1

pts1 = fun(xyz1,xyz2,pts2) 
all_pts = np.zeros((104,3))
all_pts[0:100,:] = pts1
all_pts[100:,:] = pts2
all_xyz = np.zeros((104,3))
all_xyz[0:100,:]=xyz1
all_xyz[100:,:]=xyz2
showpoints(xyz2, pts2, ballradius=8)
showpoints(xyz1, pts1, ballradius=8)
showpoints(all_xyz, all_pts, ballradius=8)
