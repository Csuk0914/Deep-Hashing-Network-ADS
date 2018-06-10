from alexnet import *

class DeepHashingNet():
    def __init__(self, images, keep_prob, hash_K, skip_layer, weights_path = 'DEFAULT'):
        '''
        INPUTS:
        :param images: tf.placeholder, input the images
        :param keep_prob: tf.placeholder, for the dropout rate
        :param num_classes: int, number of classes of the dataset
        :param skip_layer: list of strings, names of the layers you want to reinitialize
        :param weights_path: path string, to the pretrained weights(if blvc npy is not in the folder)
        '''
        self.images = images
        self.hash_K = hash_K
        self.KEEP_PROB = keep_prob
        self.SKIP_LAYER = skip_layer
        # self.IS_TRAINING = is_training
        if weights_path == 'DEFAULT':
            self.WEIGHTS_PATH = 'bvlc_alexnet.npy'
        else:
            self.WEIGHTS_PATH = weights_path
        # Call the create function to build the computational graph of AlexNet
        self.create()

    def create(self):
        """Build the Alexnet model
        参数：
        训练图像集
        返回：
        pool5：卷积层的最后一个输出
        paras：得到的每一卷积层的weights和biases
        """
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
        dropout7 = createDropout(fc7, self.KEEP_PROB)

        # 8th Layer: FC and return unscaled activations (for tf.nn.softmax_cross_entropy_with_logits)
        self.fch = createFullConnect(dropout7, 4096, self.hash_K, relu=False, name='fch')
        return

    def loadInitialWeights(self, session):
        # load the weight
        weights_dict = np.load(self.WEIGHTS_PATH, encoding='bytes').item()

        for op_name in weights_dict:
            # 在跳过的层中，说明不使用外部参数，则需要进行学习
            if op_name in self.SKIP_LAYER:
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
