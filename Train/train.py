# This file is for train the DHN, the checkpoint file is stored in checkpoints folder

import sys
sys.path.append('../DataGenerator')
sys.path.append('../Architecture')
import DataGenerator
import DeepHashNet as DHN
import LossFunction as Loss
from datetime import datetime
import tensorflow as tf
import numpy as np
import os

# 网络训练参数
EPOCH = 20
LEARN_RATE = 0.001
BATCH_SIZE = 128
LAMBDA = 0.2
DROP_RATE = 0.5
HASHING_BITS = 48
NUM_CLASS = 10
DATA_PATH_TRAIN ='../data/train'

# 要训练的层
TRAIN_LAYERS = ['fch','fc7']
display_step = 1

# Path for tf.summary.FileWriter and to store model checkpoints
filewriter_path = os.path.join(os.getcwd(), "../checkpoints/tensorboard_result")
checkpoint_path = os.path.join(os.getcwd(), "../checkpoints")

# 初始化训练变量和测试变量
train_generator = DataGenerator.ImageDataGenerator(DATA_PATH_TRAIN,
                                     horizontal_flip=True, shuffle=True)

# place holder
x = tf.placeholder(tf.float32, [BATCH_SIZE, 227, 227, 3])
y = tf.placeholder(tf.float32, [None, NUM_CLASS])
keep_prob = tf.placeholder(tf.float32)

# 构建模型
model = DHN.DeepHashingNet(x, keep_prob, HASHING_BITS, skip_layer=TRAIN_LAYERS)
hash = model.fch

# 需要训练的参数列表
var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in TRAIN_LAYERS]

# 损失函数
with tf.name_scope('loss_func'):
    loss = Loss.pairwise_cross_entropy_loss(hash, y) + tf.multiply(tf.Variable(LAMBDA), tf.reduce_mean(
        tf.square(tf.subtract(tf.abs(hash), tf.constant(1.0)))))

# 训练
with tf.name_scope('train'):
    # 计算梯度
    gradients = tf.gradients(loss, var_list)
    gradients = list(zip(gradients, var_list))

    # 优化器和梯度下降
    optimizer = tf.train.GradientDescentOptimizer(LEARN_RATE)
    train_op = optimizer.apply_gradients(grads_and_vars=gradients)

# 添加一堆summary
# Add gradients to summary
for gradient, var in gradients:
    tf.summary.histogram(var.name + '/gradient', gradient)
# Add the variables we train to the summary
for var in var_list:
    tf.summary.histogram(var.name, var)
# Add the loss to summary
tf.summary.scalar('loss_function', loss)
merged_summary = tf.summary.merge_all()

# saver用于保存检查点，writer用于写文件
saver = tf.train.Saver()
writer = tf.summary.FileWriter(filewriter_path)

# 对于每个epoch有多少个batch
train_batches_per_epoch = np.floor(train_generator.dataSize / BATCH_SIZE).astype(np.int16)

# 开启tensorflow会话
with tf.Session() as sess:
    # 初始化所有变量
    sess.run(tf.global_variables_initializer())

    # 添加网络图到tensorboard
    writer.add_graph(sess.graph)

    # 加载预训练参数
    model.loadInitialWeights(sess)

    print("{} Start training...".format(datetime.now()))
    print("{} Open Tensorboard at --logdir {}".format(datetime.now(),filewriter_path))

    # 循环epoch次
    for epoch in range(EPOCH):

        print("{} Epoch number: {}".format(datetime.now(), epoch + 1))

        step = 1

        while step < train_batches_per_epoch:

            print("{} Batch number: {}".format(datetime.now(), step))
            # 获取每个batch的图像和标签
            batch_xs, batch_ys = train_generator.next_batch(BATCH_SIZE)

            # And run the training op
            sess.run(train_op, feed_dict={x: batch_xs,
                                          y: batch_ys,
                                          keep_prob: DROP_RATE})

            if step % display_step == 0:
                print("{} Adding Summary".format(datetime.now()))
                s = sess.run(merged_summary, feed_dict={x: batch_xs,
                                                        y: batch_ys,
                                                        keep_prob: 1.})
                writer.add_summary(s, epoch * train_batches_per_epoch + step)

            step += 1

        # 重置指针
        train_generator.reset()

        # 保存检查点文件
        print("{} Saving checkpoint of model...".format(datetime.now()))
        checkpoint_name = os.path.join(checkpoint_path, 'model_epoch' + str(epoch + 1) + '.ckpt')
        save_path = saver.save(sess, checkpoint_name)
        print("{} Model checkpoint saved at {}".format(datetime.now(), checkpoint_name))