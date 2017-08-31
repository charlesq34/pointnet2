import tensorflow as tf
import numpy as np

np.random.seed(0)


a_val = np.random.random((2,5,3))
b_val = np.random.random((2,2,3))
for b in range(2):
    print '--- ', b
    t1 = a_val[b,:,:]
    t2 = b_val[b,:,:]
    for i in range(2): #npoint in b
        print '-- point b: ', i
        for j in range(5): # npoint in a
            d = np.sum((t2[i,:]-t1[j,:])**2)
            print d
            


a = tf.constant(a_val)
b = tf.constant(b_val)
print a.get_shape()
k = 3

a = tf.tile(tf.reshape(a, (2,1,5,3)), [1,2,1,1])
b = tf.tile(tf.reshape(b, (2,2,1,3)), [1,1,5,1])

dist = -tf.reduce_sum((a-b)**2, -1)
print dist

val, idx = tf.nn.top_k(dist, k=k)
print val, idx
sess = tf.Session()
print sess.run(a)
print sess.run(b)
print sess.run(dist)
print sess.run(val)
print sess.run(idx)
print sess.run(idx).shape
