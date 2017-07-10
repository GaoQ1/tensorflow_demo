import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result

def weight_variable(shape):
    pass

def bias_variable(shape):
    pass

def conv2d(x, W):
    pass

def max_pool2x2(x):
    pass

# define placeholder for inputs to network
 xs = tf.placeholder(tf.float32, [None, 784])
 ys = tf.placeholder(tf.float32, [None, 10])
 keep_prob = tf.placeholder(tf.float32)

 ## conv1 layer
 ## conv2 layer
 ## func1 layer
 ## func2 layer

# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys* tf.log(prediction), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(cross_entropy)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    
