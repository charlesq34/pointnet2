import scannet_util

CLASS_NAMES = scannet_util.g_label_names
RAW2SCANNET = scannet_util.g_raw2scannet

import os
import json
import sys
import numpy as np
BASE_DIR = os.path.dirname(__file__)

sys.path.append(BASE_DIR)
sys.path.append('../')
import pc_util

SCANNET_DIR = 'scannet_clean_2'
SCENE_NAMES = [line.rstrip() for line in open('scannet_all.txt')]

def collect_one_scene_data_label(scene_name, out_filename):
    # Over-segmented segments: maps from segment to vertex/point IDs
    data_folder = os.path.join(SCANNET_DIR, scene_name)
    mesh_seg_filename = os.path.join(data_folder, '%s_vh_clean_2.0.010000.segs.json'%(scene_name))
    #print mesh_seg_filename
    with open(mesh_seg_filename) as jsondata:
        d = json.load(jsondata)
        seg = d['segIndices']
        #print len(seg)
    segid_to_pointid = {}
    for i in range(len(seg)):
        if seg[i] not in segid_to_pointid:
            segid_to_pointid[seg[i]] = []
        segid_to_pointid[seg[i]].append(i)
    
    # Raw points in XYZRGBA
    ply_filename = os.path.join(data_folder, '%s_vh_clean_2.ply' % (scene_name))
    points = pc_util.read_ply_xyzrgb(ply_filename)
    log_string(str(points.shape))
    
    # Instances over-segmented segment IDs: annotation on segments
    instance_segids = []
    labels = []
    annotation_filename = os.path.join(data_folder, '%s.aggregation.json'%(scene_name))
    #print annotation_filename
    with open(annotation_filename) as jsondata:
        d = json.load(jsondata)
        for x in d['segGroups']:
            instance_segids.append(x['segments'])
            labels.append(x['label'])
    
    #print len(instance_segids)
    #print labels
    
    # Each instance's points
    instance_points_list = []
    instance_labels_list = []
    semantic_labels_list = []
    for i in range(len(instance_segids)):
       segids = instance_segids[i]
       pointids = []
       for segid in segids:
           pointids += segid_to_pointid[segid]
       instance_points = points[np.array(pointids),:]
       instance_points_list.append(instance_points)
       instance_labels_list.append(np.ones((instance_points.shape[0], 1))*i)   
       if labels[i] not in RAW2SCANNET:
           label = 'unannotated'
       else:
           label = RAW2SCANNET[labels[i]]
       label = CLASS_NAMES.index(label)
       semantic_labels_list.append(np.ones((instance_points.shape[0], 1))*label)
       
    # Refactor data format
    scene_points = np.concatenate(instance_points_list, 0)
    scene_points = scene_points[:,0:6] # XYZRGB, disregarding the A
    instance_labels = np.concatenate(instance_labels_list, 0) 
    semantic_labels = np.concatenate(semantic_labels_list, 0)
    data = np.concatenate((scene_points, instance_labels, semantic_labels), 1)
    np.save(out_filename, data)


LOG_FOUT = open('log.txt','w')
def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


if __name__=='__main__':
    output_folder = 'scannet_scenes'
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    
    for scene_name in SCENE_NAMES:
        log_string(scene_name)
        try:
            out_filename = scene_name+'.npy' # scene0000_00.npy
            collect_one_scene_data_label(scene_name, os.path.join(output_folder, out_filename))
        except Exception, e:
            log_string(scene_name+'ERROR!!')
            log_string(str(e))
    
    LOG_FOUT.close()
