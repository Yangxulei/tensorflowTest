import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def add_layer(layoutname, inputs, in_size, out_size, act = None):
    with tf.name_scope(layoutname):
        with tf.name_scope('weights'):
            weights = tf.Variable(tf.random_normal([in_size, out_size]), name = 'weights')
            w_hist = tf.summary.histogram('weights', weights)
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name = 'biases')
            b_hist = tf.summary.histogram('biases', biases)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.add(tf.matmul(inputs, weights), biases)

        if act is None:
            outputs = Wx_plus_b
        else :
            outputs = act(Wx_plus_b)
        return outputs

x_data = np.linspace(-1, 1, 300)[:,np.newaxis]
noise = np.random.normal(0,0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

with tf.name_scope('Input'):
    xs = tf.placeholder(tf.float32, [None, 1], name = "input_x")
    ys = tf.placeholder(tf.float32, [None, 1], name = "target_y")


l1 = add_layer("first_layer", xs, 1, 10, act = tf.nn.relu)
l1_hist = tf.summary.histogram('l1', l1)

y = add_layer("second_layout", l1, 10, 1, act = None)
y_hist = tf.summary.histogram('y', y)

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - y),
                            reduction_indices = [1]))
    tf.summary.histogram('loss ', loss)
    tf.summary.scalar('loss', loss)

with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()
merged = tf.summary.merge_all()

with tf.Session() as sess:
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x_data, y_data)
    plt.ion()
    plt.show()

    writer = tf.summary.FileWriter('logs/', sess.graph)
    sess.run(init)

    for train in range(1000):
        sess.run(train_step, feed_dict = {xs: x_data, ys: y_data})
        if train % 50 == 0:
            try:
                ax.lines.remove(lines[0])
            except Exception:
                pass
            summary_str = sess.run(merged, feed_dict = {xs: x_data, ys: y_data})
            writer.add_summary(summary_str, train)

            print(train, sess.run(loss, feed_dict = {xs: x_data, ys: y_data}))

            prediction_value = sess.run(y, feed_dict = {xs: x_data})
            lines = ax.plot(x_data, prediction_value, 'r-', lw = 5)
            plt.pause(1)


# import numpy as np
# import tensorflow as tf
#
# # Model parameters
# W = tf.Variable([.3], tf.float32)
# b = tf.Variable([-.3], tf.float32)
# # Model input and output
# x = tf.placeholder(tf.float32)
# linear_model = W * x + b
# y = tf.placeholder(tf.float32)
# # loss
# loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of the squares
# # optimizer
# optimizer = tf.train.GradientDescentOptimizer(0.01)
# train = optimizer.minimize(loss)
# # training data
# x_train = [1,2,3,4]
# y_train = [0,-1,-2,-3]
# # training loop
# init = tf.global_variables_initializer()
# sess = tf.Session()
# sess.run(init) # reset values to wrong
# for i in range(1000):
#   sess.run(train, {x:x_train, y:y_train})
#
# # evaluate training accuracy
# curr_W, curr_b, curr_loss  = sess.run([W, b, loss], {x:x_train, y:y_train})
# print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))
