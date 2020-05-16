'''
    Predict class of single pointcloud data.
'''
import tensorflow as tf
import numpy as np
import argparse
import importlib
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

import provider

from pprint import pprint

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet2_cls_ssg', help='Model name. [default: pointnet2_cls_ssg]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--model_path', default='log/model.ckpt', help='model checkpoint file path [default: log/model.ckpt]')
parser.add_argument('--dump_dir', default='dump', help='dump folder path [dump]')
parser.add_argument('--normal', action='store_true', help='Whether to use normal information')
parser.add_argument('--num_votes', type=int, default=1, help='Aggregate classification scores from multiple rotations [default: 1]')
parser.add_argument('-v', '--visualize', action='store_true', help='Visualize input pointcloud data')
parser.add_argument('--path', help='Path of pointcloud txt')

FLAGS = parser.parse_args()

NUM_POINT = FLAGS.num_point
MODEL_PATH = FLAGS.model_path
GPU_INDEX = FLAGS.gpu
MODEL = importlib.import_module(FLAGS.model) # import network module
DUMP_DIR = FLAGS.dump_dir
VISUALIZE = FLAGS.visualize
if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)
LOG_FOUT = open(os.path.join(DUMP_DIR, 'log_evaluate.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

NUM_CLASSES = 40
SHAPE_NAMES = [line.rstrip() for line in \
    open(os.path.join(ROOT_DIR, 'data/modelnet40_ply_hdf5_2048/shape_names.txt'))] 


PC_PATH = FLAGS.path
# Get first n dimensions, must change with normal flag
npoints = 3 


def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def evaluate():
     
    with tf.device('/gpu:'+str(GPU_INDEX)):
        pointclouds_pl, labels_pl = MODEL.placeholder_inputs(1, NUM_POINT)
        is_training_pl = tf.placeholder(tf.bool, shape=())

        # simple model
        pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl)
        
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        
    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    sess = tf.Session(config=config)

    # Restore variables from disk.
    saver.restore(sess, MODEL_PATH)
    log_string("Model restored.")

    ops = {'pointclouds_pl': pointclouds_pl,
           'labels_pl': labels_pl,
           'is_training_pl': is_training_pl,
           'pred': pred
    }

    # Load pointcloud data from txt file
    point_set = np.loadtxt(PC_PATH, delimiter=',').astype(np.float32)
    # Get indexes for random points from pointcloud
    random_idx = np.random.randint(point_set.shape[0], size=1024) 

    #point_set = point_set[random_idx,0:npoints]
    point_set = point_set[:NUM_POINT, 0:npoints]
    point_set = np.array([point_set])

    pred_one(sess, ops, point_set)

def pred_one(sess, ops, pointcloud_data):
    is_training = False
    num_votes = FLAGS.num_votes

    pred_val_sum = np.zeros((1, NUM_CLASSES))

    for vote_idx in range(num_votes):

        rotation = vote_idx/float(num_votes) * np.pi * 2
        rotated_data = provider.rotate_point_cloud_by_angle(pointcloud_data, rotation)

        feed_dict = {ops['pointclouds_pl']: rotated_data,
                        ops['is_training_pl']: is_training}

        pred_val = sess.run([ops['pred']], feed_dict=feed_dict)[0]
        pred_val_sum += pred_val
        idx = np.argmax(pred_val)

        print("Predicted shape as: '{}' with rotation: {}".format(SHAPE_NAMES[idx], np.degrees(rotation)) )
    
    final_idx = np.argmax(pred_val_sum)
    print("Final prediction:", SHAPE_NAMES[final_idx])

    if VISUALIZE:
        from show3d_balls import showpoints

        showpoints(pointcloud_data[0])


if __name__=='__main__':
    with tf.Graph().as_default():
        evaluate()
    LOG_FOUT.close()

