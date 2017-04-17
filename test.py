import tensorflow as tf
import numpy as np


# 定义变量
state = tf.Variable(0, name = 'counter')
print(state.name)

# 定义常量
one = tf.constant(1)
# 定义新的变量，值等于state + one
new_value = tf.add(state, one)
# 定义update操作，将new_value赋值给state
update = tf.assign(state, new_value)

# 定义的变量激活，如果定义了Variable，必须有
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        # 执行update操作，update操作包括add,assign两部分
        sess.run(update)
        # 输出state结果，必须放到sess.run()中
        print(sess.run(state))

# # 使用 NumPy 生成假数据(phony data), 总共 100 个点.
# from numpy.core.tests.test_mem_overlap import xrange
#
# x_data = np.float32(np.random.rand(2, 100)) # 随机输入
# y_data = np.dot([0.100, 0.200], x_data) + 0.300
#
# # 构造一个线性模型
# #
# b = tf.Variable(tf.zeros([1]))
# W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
# y = tf.matmul(W, x_data) + b
#
# # 最小化方差
# loss = tf.reduce_mean(tf.square(y - y_data))
# optimizer = tf.train.GradientDescentOptimizer(0.5)
# train = optimizer.minimize(loss)
#
# # 初始化变量
# init = tf.initialize_all_variables()
#
# # 启动图 (graph)
# sess = tf.Session()
# sess.run(init)
#
# # 拟合平面
# for step in xrange(0, 201):
#     sess.run(train)
#     if step % 20 == 0:
#         print(step, sess.run(W), sess.run(b))

# node1 = tf.constant(3.0,tf.float32)
# node2 = tf.constant(4.0)
# print(node1,node2)
#
# sess = tf.Session()
# print(sess.run([node1,node2]))
#
# node3 = tf.add(node1,node2)
# print("node3:",node3)
# print("sess.run(ndoe3)",sess.run(node3))
#
# a = tf.placeholder(tf.float32)
# b = tf.placeholder(tf.float32)
#
# adder_node = a + b
# print(sess.run(adder_node,{a:3,b:4}))
# print(sess.run(adder_node,{a:[1,3],b:[1,4]}))
#
# add_and_triple = adder_node * 3
# print(sess.run(add_and_triple,{a:3,b:4.5}))


# #creat two variables
# weigths = tf.Variable(tf.random_normal([784,200],stddev=0.35),name="weights")  #Variable是一个构造函数，初始化tensor，可以是常量也可以是变量
# biases = tf.Variable(tf.zeros([200]),name="biases")
#
# #添加一个op初始化这个variables
# init_op = tf.initialize_all_variables()
#
# saver = tf.train.Saver()
#
# with tf.Session() as sess:
#     sess.run(init_op)
#     ...
#     # Use the model
#     ...
# save_path = saver.save(sess, "/tmp/model.ckpt")
# print("Model saved in file: ", save_path)
#
# #Create another variable with the same value as 'weights'.
# w2 = tf.Variable(weigths.initialized_value(),name="w2")
# # Create another variable with twice the value of 'weights'
# w3 = tf.Variable(weigths.initialized_value()*2, name="w3")



