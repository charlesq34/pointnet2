'''
    ModelNet dataset. Support ModelNet40, ModelNet10, XYZ and normal channels. Up to 10000 points.
'''

import os
import os.path
import json
import numpy as np
import sys

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

class ModelNetDataset():
    def __init__(self, root, npoints = 1024, split='train', normalize=True, normal_channel=True, modelnet10=False, modelnet30=False):
        self.npoints = npoints
        self.root = root
        self.normalize = normalize
        if modelnet10:
            self.catfile = os.path.join(self.root, 'modelnet10_shape_names.txt')
        else:
            if modelnet30:
                self.catfile = os.path.join(self.root, 'modelnet30_shape_names.txt')
            else:
                self.catfile = os.path.join(self.root, 'shape_names.txt')
        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))  
        self.normal_channel = normal_channel
        
        shape_ids = {}
        if modelnet10:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_train.txt'))] 
            shape_ids['test']= [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_test.txt'))]
        else:
            if modelnet30:
                shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet30_train.txt'))] 
                shape_ids['test']= [line.rstrip() for line in open(os.path.join(self.root, 'modelnet30_test.txt'))]
            else:
                shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))] 
                shape_ids['test']= [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]
        assert(split=='train' or split=='test')
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        # list of (shape_name, shape_txt_file_path) tuple
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i])+'.txt') for i in range(len(shape_ids[split]))]

        self.cache = {} # from index to (point_set, cls) tuple
        self.cache_size = 15000
               
    def __getitem__(self, index):
        if index in self.cache:
            point_set, cls = self.cache[index]
        else:
            fn = self.datapath[index]
            cls = self.classes[self.datapath[index][0]]
            cls = np.array([cls]).astype(np.int32)
            point_set = np.loadtxt(fn[1],delimiter=',').astype(np.float32)
            # Take the first npoints
            point_set = point_set[0:self.npoints,:]
            if self.normalize:
                point_set[:,0:3] = pc_normalize(point_set[:,0:3])
            if not self.normal_channel:
                point_set = point_set[:,0:3]
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls)
                
        #choice = np.random.choice(len(seg), self.npoints, replace=True)
        #resample
        #point_set = point_set[choice, :]
        return point_set, cls
        
    def __len__(self):
        return len(self.datapath)

    def num_channel(self):
        if self.normal_channel:
            return 6
        else:
            return 3

if __name__ == '__main__':
    d = ModelNetDataset(root = '../data/modelnet40_normal_resampled', split='test', modelnet30=True)
    print(len(d))
    import time
    tic = time.time()
    for i in range(100):
        ps, cls = d[i]
    print(time.time() - tic)
    print(ps.shape, type(ps), cls)
