''' scanning through annotation files for all the scenes to get a complete list of categories '''

import os
import json
scannet_dir = './scannet/'
scene_names = [line.rstrip() for line in open('scannet_all.txt')]

labels = set()
for scene_name in scene_names:
    path = os.path.join(scannet_dir, scene_name)
    agg_filename = os.path.join(path, scene_name+'.aggregation.json')
    with open(agg_filename) as jsondata:
        d = json.load(jsondata)
        for x in d['segGroups']:
            labels.add(x['label']) 

fout = open('class_names.txt', 'w')
for label in list(labels):
    print label
    try:
        fout.write(label+'\n')
    except:
        pass
fout.close()
