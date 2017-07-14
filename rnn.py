import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# this is data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# hyperparamters
lr = 0.001
training_iters = 100000
batch_size = 128
display_step = 10

n_inputs = 28
n_steps = 28
n_hidden_unit = 128
n_classes = 10

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])


# 32
