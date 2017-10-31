import sys
import os

BASE_DIR = os.path.dirname(__file__)

sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../'))

import numpy as np
import pc_util

data = np.load('scannet_scenes/scene0001_01.npy')
scene_points = data[:,0:3]
colors = data[:,3:6]
instance_labels = data[:,6]
semantic_labels = data[:,7]


output_folder = 'demo_output'
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

# Write scene as OBJ file for visualization
pc_util.write_ply_rgb(scene_points, colors, os.path.join(output_folder, 'scene.obj'))
pc_util.write_ply_color(scene_points, instance_labels, os.path.join(output_folder, 'scene_instance.obj'))
pc_util.write_ply_color(scene_points, semantic_labels, os.path.join(output_folder, 'scene_semantic.obj'))
