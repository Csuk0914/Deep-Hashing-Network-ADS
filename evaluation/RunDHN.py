# This file is for run the net on a certain set of data

import os
import tensorflow as tf
import sys
sys.path.append('../DataGenerator')
sys.path.append('../Architecture')
import DataGenerator
import DeepHashNet as DHN

# 生成二进制串
def toBinaryString(binary_like_values):
    numOfImage, bit_length = binary_like_values.shape
    list_string_binary = []
    for i in range(numOfImage):
        str = ''
        for j in range(bit_length):
            str += '0' if binary_like_values[i][j] <= 0 else '1'
        list_string_binary.append(str)
    return list_string_binary

# 生成字符串
def toString(binary_like_values):
    numOfImage, bit_length = binary_like_values.shape
    list_string_binary = []
    for i in range(numOfImage):
        st = ''
        for j in range(bit_length):
            st += str(binary_like_values[i][j]) + ' '
        list_string_binary.append(st)
    return list_string_binary

BATCH_SIZE = 128
HASHING_BITS = 48

DATA_PATH_TEST = '../data/test'
DATA_PATH_TRAIN ='../data/train'

checkpoint_path = os.path.join(os.getcwd(), "../checkpoints")

# 构建
image = tf.placeholder(tf.float32, [BATCH_SIZE, 227, 227, 3], name='image')
Dhn = DHN.DeepHashingNet(image, 1, HASHING_BITS, skip_layer=[])
fch = Dhn.fch

# 读入检查点
print("Reading checkpoints...")
ckpt = tf.train.get_checkpoint_state(checkpoint_path)
saver = tf.train.Saver(tf.all_variables())

# 建立会话
sess = tf.Session()

# 数据加载
train_generator = DataGenerator.ImageDataGenerator(DATA_PATH_TRAIN,
                                     horizontal_flip=False, shuffle=False)
val_generator = DataGenerator.ImageDataGenerator(DATA_PATH_TEST, shuffle=False)
file_res = open('result_dhn.txt', 'w')

if ckpt and ckpt.model_checkpoint_path:
    ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    saver.restore(sess, os.path.join(checkpoint_path, ckpt_name))

    print('Loading success, global_step is %s' % global_step)

    k = 0
    for i in range(len(val_generator.images) // BATCH_SIZE):
        batch_xs, batch_ys = val_generator.next_batch(BATCH_SIZE)

        eval_sess = sess.run(Dhn.fch, feed_dict={image: batch_xs})
        print(eval_sess)
        w_bin = toBinaryString(eval_sess)
        w_res = toString(eval_sess)

        for j in range(BATCH_SIZE):
            file_res.write(w_res[j] + '\t' + w_bin[j] + '\t' + str(val_generator.images[k]) + '\n')
            # file_res.write(w_bin[j] + '\t' + str(val_generator.images[k]) + '\n')
            print('write number %d' % k)
            k+=1
    k = 0
    for i in range(len(train_generator.images) // BATCH_SIZE):
        batch_xs, batch_ys = train_generator.next_batch(BATCH_SIZE)

        eval_sess = sess.run(Dhn.fch, feed_dict={image: batch_xs})

        w_bin = toBinaryString(eval_sess)
        w_res = toString(eval_sess)

        for j in range(BATCH_SIZE):
            file_res.write(w_res[j] + '\t' + w_bin[j] + '\t' + str(train_generator.images[k]) + '\n')
            # file_res.write(w_bin[j] + '\t' + str(train_generator.images[k]) + '\n')
            print('write number %d' % k)
            k+=1
file_res.close()
sess.close()
