import tensorflow as tf
from numpy.core.tests.test_mem_overlap import xrange

b = tf.Variable(tf.zeros([100]))
W = tf.Variable(tf.random_uniform([784,100],-1,1))
x = tf.placeholder(name="x")
relu = tf.nn.relu(tf.matmul(W,x) + b)
C = [...]

s = tf.Session()

tf.Session.run()