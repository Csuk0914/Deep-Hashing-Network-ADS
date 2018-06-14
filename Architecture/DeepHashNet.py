# This file is used to build a deep hashing net
from model_layers import *
import tensorflow as tf

class DeepHashingNet():
    # 构造函数，images为输入图像，是一个num*227*227*3的张量，skiplayer是要训练的层
    # weights_path是DEFAULT时在项目目录寻找npy（预训练参数）文件
    def __init__(self, images, keep_prob, hash_K, skip_layer, weights_path = 'DEFAULT'):
        self.images = images
        self.hash_K = hash_K
        self.KEEP_PROB = keep_prob
        self.SKIP_LAYER = skip_layer

        if weights_path == 'DEFAULT':
            self.WEIGHTS_PATH = '../Architecture/bvlc_alexnet.npy'
        else:
            self.WEIGHTS_PATH = weights_path

        # 开始创建
        self.create()

    def create(self):
        # 1st Layer: Conv (w ReLu) -> Pool -> Lrn
        conv1 = createConv(self.images, 11, 11, 96, 4, 4, padding='VALID', name='conv1')
        pool1 = createMaxPool(conv1, 3, 3, 2, 2, padding='VALID', name='pool1')
        norm1 = createLrn(pool1, 2, 2e-05, 0.75, name='norm1')

        # 2nd Layer: Conv (w ReLu) -> Pool -> Lrn with 2 groups
        conv2 = createConv(norm1, 5, 5, 256, 1, 1, groups=2, name='conv2')
        pool2 = createMaxPool(conv2, 3, 3, 2, 2, padding='VALID', name='pool2')
        norm2 = createLrn(pool2, 2, 2e-05, 0.75, name='norm2')

        # 3rd Layer: Conv (w ReLu)
        conv3 = createConv(norm2, 3, 3, 384, 1, 1, name='conv3')

        # 4th Layer: Conv (w ReLu) splitted into two groups
        conv4 = createConv(conv3, 3, 3, 384, 1, 1, groups=2, name='conv4')

        # 5th Layer: Conv (w ReLu) -> Pool splitted into two groups
        conv5 = createConv(conv4, 3, 3, 256, 1, 1, groups=2, name='conv5')
        pool5 = createMaxPool(conv5, 3, 3, 2, 2, padding='VALID', name='pool5')

        # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
        flattened = tf.reshape(pool5, [-1, 6 * 6 * 256])
        fc6 = createFullConnect(flattened, 6 * 6 * 256, 4096, name='fc6')
        dropout6 = createDropout(fc6, self.KEEP_PROB)

        # 7th Layer: FC (w ReLu) -> Dropout
        fc7 = createFullConnect(dropout6, 4096, 4096, name='fc7')
        #dropout7 = createDropout(fc7, self.KEEP_PROB)

        # 8th Layer: FC and return tanh activation
        self.fch = createFullConnect(fc7, 4096, self.hash_K, relu=False, name='fch', tanh=True)
        return

    def loadInitialWeights(self, session):
        # load the weight
        weights_dict = np.load(self.WEIGHTS_PATH, encoding='bytes').item()

        for op_name in weights_dict:
            # 在跳过的层中，说明不使用外部参数，则需要进行学习，DHN没有fc8层，不需要学习
            if op_name in self.SKIP_LAYER or op_name == 'fc8':
                continue
            with tf.variable_scope(op_name, reuse=True):
                # loop the list
                for data in weights_dict[op_name]:
                    # Biases
                    if len(data.shape) == 1:
                        bia = tf.get_variable('biases',trainable=False)
                        session.run(bia.assign(data))
                    else:
                        var = tf.get_variable('weights', trainable=False)
                        session.run(var.assign(data))