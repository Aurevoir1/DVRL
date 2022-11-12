# codeing=utf-8
"""
DVRL 主程序的实现
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
from sklearn import linear_model
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import keras

from keras import applications
from keras import layers
from keras import models

import data_loading
import dvrl
import dvrl_metrics

def main(args):
    """
    Main function of DVRL with transfer learning for image data 针对图像数据使用迁移学习的DVRL主函数

    使用transfer learning 进行损坏样本发现以及鲁棒学习
    """

    # 定义数据集名称  cifar10 or cifar100
    data_name = args.data_name

    # 训练与验证样本的数量
    dict_no = dict()
    dict_no['train'] = args.train_no
    dict_no['valid'] = args.valid_no
    dict_no['test'] = args.test_no

    noise_rate = args.noise_rate

    checkpoint_file_name = args.checkpoint_file_name

    noise_idx = data_loading.load_image_data(data_name, dict_no, noise_rate)

    print('Finish data loading!')

    # 提取特征和标签
    x_train, y_train, x_valid, y_valid, x_test, y_test = data_loading.load_image_data_from_file('train.npz', 'valid.npz', 'test.npz')
    print('finish data preprocess')

    # Encodes samples
    preprocess_function = applications.inception_v3.preprocess_input
    input_shape = (299,299)

    def encoder_model(architecture='inception_v3',pre_trained_dataset="imagenet",downsample_factor=8):
        tf_input = layers.Input(shape=(input_shape[0],input_shape[1],3))
        if architecture == 'inception_v3':
            model = applications.inception_v3.InceptionV3(
                input_tensor=tf_input,weights=pre_trained_dataset,include_top=False)

            output_pooled = layers.AveragePooling2D((downsample_factor,downsample_factor),
                                                     strides=(downsample_factor,
                                                              downsample_factor))(model.output)
        else:
            raise NameError('Invalid architecture')
        return models.Model(model.input,output_pooled)


    # 训练样本编码
    enc_x_train = data_loading.encode_image(x_train,encoder_model,input_shape,preprocess_function)