# This file is the layers of deep learning network that DHN and AlexNet uses

# imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np

# 用于输出网络参数
def print_actications(t):
    print(t.op.name, '', t.get_shape().as_list())
    return

# 对input数据创建一层卷积层, group用来模拟alexnet提出时的分组做法，但训练过程不会真的分组
def createConv(input, filter_height, filter_width, num_filters,
               stride_y, stride_x, name, padding='SAME', groups = 1):
    input_channels = int(input.get_shape()[-1])
    # lambda表达式，定义卷积核
    convolve = lambda i,ker: tf.nn.conv2d(i, ker, strides=[1,stride_y,stride_x,1], padding=padding)

    with tf.variable_scope(name) as scope:
        kernel = tf.get_variable('weights', shape=[filter_height, filter_width, input_channels/groups, num_filters])
        biases = tf.get_variable('biases', shape=[num_filters])

        if groups == 1:
            conv = convolve(input, kernel)
        else:
            input_group = tf.split(axis=3, num_or_size_splits=groups, value=input)
            weight_group = tf.split(axis=3, num_or_size_splits=groups, value=kernel)
            output_groups = [convolve(i,k) for i,k in zip(input_group,weight_group)]

            conv = tf.concat(axis=3, values=output_groups)

        # 设置偏置
        bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())

        # 使用relu函数作为激活函数
        conv_ans = tf.nn.relu(bias, name = scope.name)

    return conv_ans

# 对输入进行归一化
def createLrn(input, depth_radius, alpha, beta, name, bias=1.0):
    with tf.name_scope(name) as scope:
        lrn = tf.nn.local_response_normalization(input, alpha=alpha, beta=beta,
                                                 depth_radius=depth_radius, bias=bias, name=name)
    return lrn

# 创建降采样层(pooling)
def createMaxPool(input, filter_height, filter_width, stride_y, stride_x, name, padding='VALID'):
    pool = tf.nn.max_pool(input, ksize=[1,filter_height,filter_width,1], strides=[1,stride_y,stride_x,1],
                          padding=padding, name=name)
    return pool

# 创建dropout层
def createDropout(input, keep_prob):
    return tf.nn.dropout(input, keep_prob)

# 全连接层，此处有两个激活函数，relu和tanh，relu默认开启
def createFullConnect(input, num_in, num_out, name, relu = True, tanh = False):
    with tf.variable_scope(name) as scope:
        weights = tf.get_variable('weights', shape=[num_in, num_out], trainable=True)
        biases = tf.get_variable('biases', shape=[num_out], trainable=True)

        full = tf.nn.xw_plus_b(input,weights,biases,name=scope.name)

        if relu:
            rec_relu = tf.nn.relu(full)
            return rec_relu
        elif tanh:
            rec_tanh = tf.nn.tanh(full)
            return rec_tanh
        else:
            return full
