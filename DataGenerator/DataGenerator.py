# This file is for data generate

import numpy as np
import cv2
import os

# 输入trainpath为文件保存路径， horizontal_flip决定是否有0.5的概率将图片反转。shuffle决定是否要随即打乱
class ImageDataGenerator:
    def __init__(self, train_path, horizontal_flip=False, shuffle=False,
                 mean = np.array([104., 117., 124.]), scale_size=(227, 227)):
        self.horizontal_flip = horizontal_flip
        self.shuffle = shuffle
        self.mean = mean
        self.scaleSize = scale_size
        self.index = 0

        self.trainPath = train_path
        self.read_data()

        if (self.shuffle):
            self.shuffle_data()

    def read_data(self):
        # scan the images and labels
        train_path = os.path.join(os.getcwd(), self.trainPath)
        print(train_path)
        self.images = []
        self.labels = []

        classTable = dict()
        tempClassNum = 0

        ls = os.listdir(train_path)
        for filename in ls:
            path = os.path.join(train_path, filename)
            if not os.path.isfile(path):
                continue
            prefix, suffix = os.path.splitext(filename)

            label = prefix.split('_')[0]
            self.images.append(filename)

            if not classTable.__contains__(label):
                classTable[label] = tempClassNum
                tempClassNum += 1

            self.labels.append(classTable[label])

        self.dataSize = len(self.images)
        self.n_classes = tempClassNum
        return

    def shuffle_data(self):
        # random the input data
        images = self.images.copy()
        labels = self.labels.copy()
        self.images = []
        self.labels = []

        indexs = np.random.permutation(len(labels))
        for i in indexs:
            self.images.append(images[i])
            self.labels.append(labels[i])

    def reset(self):
        self.index = 0
        if self.shuffle:
            self.shuffle_data()

    def next_batch(self, batch_size):
        # 将images中的名称读入下一个batch
        images = np.ndarray([batch_size, self.scaleSize[0], self.scaleSize[1], 3])

        for i in range(0,batch_size):
            img = cv2.imread('./' + self.trainPath + '/' + self.images[self.index + i])

            # 水平翻转
            if self.horizontal_flip and np.random.random() < 0.5:
                img = cv2.flip(img, 1)

            # 缩放尺寸
            img = cv2.resize(img, (self.scaleSize[0], self.scaleSize[1]))
            img = img.astype(np.float32)

            # 减去平均值
            img -= self.mean

            images[i] = img

        #构建类别
        oneHotLabels = np.zeros((batch_size, self.n_classes))
        for i in range(0,batch_size):
            oneHotLabels[i][self.labels[i + self.index]] = 1.0

        self.index += batch_size

        return images, oneHotLabels
