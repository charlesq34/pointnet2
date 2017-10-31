import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import numpy as np
from sklearn.neighbors import NearestNeighbors
from numpy import linalg as la
import scipy.io as sio

def cart2sph(xyz):
  xy = xyz[:,0]**2+xyz[:,1]**2
  aer = np.zeros(xyz.shape)
  aer[:,2] = np.sqrt(xy+xyz[:,2]**2)
  aer[:,1] = np.arctan2(xyz[:,2],np.sqrt(xy))
  aer[:,0] = np.arctan2(xyz[:,1],xyz[:,0])
  return aer

# generate virtual scan of a scene by subsampling the point cloud
def virtual_scan(xyz, mode=-1):
  camloc = np.mean(xyz,axis=0)
  camloc[2] = 1.5 # human height
  if mode==-1:
    view_dr = np.array([2*np.pi*np.random.random(), np.pi/10*(np.random.random()-0.75)])
    camloc[:2] -= (0.8+0.7*np.random.random())*np.array([np.cos(view_dr[0]),np.sin(view_dr[0])])
  else:
    view_dr = np.array([np.pi/4*mode, 0])
    camloc[:2] -= np.array([np.cos(view_dr[0]),np.sin(view_dr[0])])
  ct_ray_dr = np.array([np.cos(view_dr[1])*np.cos(view_dr[0]), np.cos(view_dr[1])*np.sin(view_dr[0]), np.sin(view_dr[1])])
  hr_dr = np.cross(ct_ray_dr, np.array([0,0,1]))
  hr_dr /= la.norm(hr_dr)
  vt_dr = np.cross(hr_dr, ct_ray_dr)
  vt_dr /= la.norm(vt_dr)
  xx = np.linspace(-0.6,0.6,200) #200
  yy = np.linspace(-0.45,0.45,150) #150
  xx, yy = np.meshgrid(xx,yy)
  xx = xx.reshape(-1,1)
  yy = yy.reshape(-1,1)
  rays = xx*hr_dr.reshape(1,-1)+yy*vt_dr.reshape(1,-1)+ct_ray_dr.reshape(1,-1)
  rays_aer = cart2sph(rays)
  local_xyz = xyz-camloc.reshape(1,-1)
  local_aer = cart2sph(local_xyz)
  nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(rays_aer[:,:2])
  mindd, minidx = nbrs.kneighbors(local_aer[:,:2])
  mindd = mindd.reshape(-1)
  minidx = minidx.reshape(-1)

  sub_idx = mindd<0.01
  if sum(sub_idx)<100:
    return np.ones(0)
  sub_r = local_aer[sub_idx,2]
  sub_minidx = minidx[sub_idx]
  min_r = float('inf')*np.ones(np.max(sub_minidx)+1)
  for i in xrange(len(sub_r)):
    if sub_r[i]<min_r[sub_minidx[i]]:
      min_r[sub_minidx[i]] = sub_r[i]
  sub_smpidx = np.ones(len(sub_r))
  for i in xrange(len(sub_r)):
    if sub_r[i]>min_r[sub_minidx[i]]:
      sub_smpidx[i] = 0
  smpidx = np.where(sub_idx)[0]
  smpidx = smpidx[sub_smpidx==1]
  return smpidx

if __name__=='__main__':
  pc = np.load('scannet_dataset/scannet_scenes/scene0015_00.npy')
  print pc.shape
  xyz = pc[:,:3]
  seg = pc[:,7]
  smpidx = virtual_scan(xyz,mode=2)
  xyz = xyz[smpidx,:]
  seg = seg[smpidx]
  sio.savemat('tmp.mat',{'pc':xyz,'seg':seg})
