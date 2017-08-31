import tensorflow as tf
import numpy as np
from tf_interpolate import three_nn, three_interpolate

class GroupPointTest(tf.test.TestCase):
  def test(self):
    pass

  def test_grad(self):
    with self.test_session():
      points = tf.constant(np.random.random((1,8,16)).astype('float32'))
      print points
      xyz1 = tf.constant(np.random.random((1,128,3)).astype('float32'))
      xyz2 = tf.constant(np.random.random((1,8,3)).astype('float32'))
      dist, idx = three_nn(xyz1, xyz2)
      weight = tf.ones_like(dist)/3.0
      interpolated_points = three_interpolate(points, idx, weight)
      print interpolated_points
      err = tf.test.compute_gradient_error(points, (1,8,16), interpolated_points, (1,128,16))
      print err
      self.assertLess(err, 1e-4) 

if __name__=='__main__':
  tf.test.main() 
