# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Data loading and preprocessing functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import os

import zipfile

import numpy as np
import pandas as pd

from six.moves import urllib
from sklearn import preprocessing
import tensorflow.compat.v1 as tf
# from tensorflow.compat.v1.keras import backend
# from tensorflow.compat.v1.keras import datasets
from keras import backend
from keras import datasets

# from dvrl import dvrl_utils
import dvrl_utils


def load_tabular_data(data_name, dict_no, noise_rate):
    """Loads Adult Income and Blog Feedback datasets.
    This module loads the two tabular datasets and saves train.csv, valid.csv and
    test.csv files under data_files directory.
    UCI Adult data link: https://archive.ics.uci.edu/ml/datasets/Adult
    UCI Blog data link: https://archive.ics.uci.edu/ml/datasets/BlogFeedback
    If noise_rate > 0.0, adds noise on the datasets.
    Then, saves train.csv, valid.csv, test.csv on './data_files/' directory
    Args:
      data_name: 'adult' or 'blog'   dataset name  'adult'  'blog'
      dict_no: training and validation set numbers  训练集 与 验证集 数量
      noise_rate: label corruption ratio
    Returns:
      noise_idx: indices of noisy samples
    """

    # Loads datasets from links
    uci_base_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/'

    # Adult Income dataset
    if data_name == 'adult':
        # 加载adult数据集
        train_url = uci_base_url + 'adult/adult.data'
        test_url = uci_base_url + 'adult/adult.test'
        # 加载图片数据集
        data_train = pd.read_csv(train_url, header=None)
        data_test = pd.read_csv(test_url, skiprows=1, header=None)

        df = pd.concat((data_train, data_test), axis=0)  # 将训练数据和测试数据合并

        # Column names
        df.columns = ['Age', 'WorkClass', 'fnlwgt', 'Education', 'EducationNum',
                      'MaritalStatus', 'Occupation', 'Relationship', 'Race',
                      'Gender', 'CapitalGain', 'CapitalLoss', 'HoursPerWeek',
                      'NativeCountry', 'Income']

        # Creates binary labels  创建二进制标签
        df['Income'] = df['Income'].map({' <=50K': 0, ' >50K': 1,
                                         ' <=50K.': 0, ' >50K.': 1})

        # Changes string to float  将string格式转换成float格式
        df.Age = df.Age.astype(float)
        df.fnlwgt = df.fnlwgt.astype(float)
        df.EducationNum = df.EducationNum.astype(float)
        df.EducationNum = df.EducationNum.astype(float)
        df.CapitalGain = df.CapitalGain.astype(float)
        df.CapitalLoss = df.CapitalLoss.astype(float)

        # One-hot encoding
        # get_dummies 是pandas中转换成one-hot编码表示的方式
        df = pd.get_dummies(df, columns=['WorkClass', 'Education', 'MaritalStatus',
                                         'Occupation', 'Relationship',
                                         'Race', 'Gender', 'NativeCountry'])

        # Sets label name as Y
        df = df.rename(columns={'Income': 'Y'})
        df['Y'] = df['Y'].astype(int)

        # Resets index
        df = df.reset_index()  # 重置索引  原来的index列自动变成第一列数据列被保留下来
        df = df.drop(columns=['index'])  # 将保存下来的原有的index列 drop掉

    # Blog Feedback dataset
    elif data_name == 'blog':

        resp = urllib.request.urlopen(uci_base_url + '00304/BlogFeedback.zip')
        zip_file = zipfile.ZipFile(io.BytesIO(resp.read()))

        # Loads train dataset
        train_file_name = 'blogData_train.csv'
        data_train = pd.read_csv(zip_file.open(train_file_name), header=None)

        # Loads test dataset
        data_test = []
        for i in range(29):
            if i < 9:
                file_name = 'blogData_test-2012.02.0' + str(i + 1) + '.00_00.csv'
            else:
                file_name = 'blogData_test-2012.02.' + str(i + 1) + '.00_00.csv'

            temp_data = pd.read_csv(zip_file.open(file_name), header=None)

            if i == 0:
                data_test = temp_data
            else:
                data_test = pd.concat((data_test, temp_data), axis=0)

        for i in range(31):
            if i < 9:
                file_name = 'blogData_test-2012.03.0' + str(i + 1) + '.00_00.csv'
            elif i < 25:
                file_name = 'blogData_test-2012.03.' + str(i + 1) + '.00_00.csv'
            else:
                file_name = 'blogData_test-2012.03.' + str(i + 1) + '.01_00.csv'

            temp_data = pd.read_csv(zip_file.open(file_name), header=None)

            data_test = pd.concat((data_test, temp_data), axis=0)

        df = pd.concat((data_train, data_test), axis=0)

        # Removes rows with missing data
        df = df.dropna()

        # Sets label and named as Y
        df.columns = df.columns.astype(str)

        df['280'] = 1 * (df['280'] > 0)
        df = df.rename(columns={'280': 'Y'})
        df['Y'] = df['Y'].astype(int)

        # Resets index
        df = df.reset_index()
        df = df.drop(columns=['index'])

    # Splits train, valid and test sets  划分数据集
    train_idx = range(len(data_train))
    train = df.loc[train_idx]

    test_idx = range(len(data_train), len(df))
    test = df.loc[test_idx]

    train_idx_final = np.random.permutation(len(train))[:dict_no['train']]

    temp_idx = np.random.permutation(len(test))
    valid_idx_final = temp_idx[:dict_no['valid']] + len(data_train)
    test_idx_final = temp_idx[dict_no['valid']:] + len(data_train)

    train = train.loc[train_idx_final]
    valid = test.loc[valid_idx_final]
    test = test.loc[test_idx_final]

    # Adds noise on labels
    y_train = np.asarray(train['Y'])
    y_train, noise_idx = dvrl_utils.corrupt_label(y_train, noise_rate)
    train['Y'] = y_train

    # Saves data
    if not os.path.exists('data_files'):
        os.makedirs('data_files')

    train.to_csv('./data_files/train.csv', index=False)
    valid.to_csv('./data_files/valid.csv', index=False)
    test.to_csv('./data_files/test.csv', index=False)

    # Returns indices of noisy samples
    return noise_idx


def load_rossmann_data(dict_no, setting, test_store_type):
    """Loads Rossmann data.
    This module loads Rossmann data for a domain adaptation application.
    Rossmann data link: https://www.kaggle.com/c/rossmann-store-sales
    The users should download 'rossmann-store-sales.zip' from the above link and
    save it in './data_files/' directory
    Args:
      dict_no: the number of source and valid samples
      setting: 'train-on-all', 'train-on-rest', or 'train-on-specific'
      test_store_type: 'A', 'B', 'C', or 'D'
    """

    # Loads datasets
    zip_file = zipfile.ZipFile('./data_files/rossmann-store-sales.zip')
    train_data = pd.read_csv(zip_file.open('train.csv'))
    store_data = pd.read_csv(zip_file.open('store.csv'))

    # Extracts features
    train_data = train_data[['Store', 'Sales', 'DayOfWeek', 'Customers', 'Open',
                             'Promo', 'StateHoliday', 'SchoolHoliday']]
    store_data = store_data[['Store', 'StoreType', 'Assortment',
                             'CompetitionDistance', 'CompetitionOpenSinceMonth',
                             'Promo2', 'Promo2SinceWeek']]

    # Data preprocessing
    # Fill na to 0
    store_data = store_data.fillna(0)
    # Converts string to int
    train_data['StateHoliday'] = train_data['StateHoliday'].replace(['a', 'b',
                                                                     'c'], 1)

    # One-hot encoding
    store_data = pd.get_dummies(store_data)

    # Combines store data and train data
    data_x = pd.merge(train_data, store_data, on='Store')

    # Removes the samples when close
    remove_idx = data_x.index[data_x['Sales'] == 0].tolist()
    data_x = data_x.drop(remove_idx, axis=0)

    # Renames target variable to 'Y'
    data_x = data_x.rename(columns={'Sales': 'Y'})

    # Defines store types
    data_c = data_x[['StoreType_a', 'StoreType_b', 'StoreType_c', 'StoreType_d']]
    data_c = data_c.rename(columns={'StoreType_a': 'A', 'StoreType_b': 'B',
                                    'StoreType_c': 'C', 'StoreType_d': 'D'})

    # Defines features
    data_x = data_x.drop(['StoreType_a', 'StoreType_b',
                          'StoreType_c', 'StoreType_d'], axis=1)

    # Resets index
    data_x = data_x.reset_index()
    data_c = data_c.reset_index()

    data_x = data_x.drop(['index'], axis=1)
    data_c = data_c.drop(['index'], axis=1)

    # Splits source, valid, and target sets
    # Random partitioning
    idx = np.random.permutation(len(data_x))

    source_idx = idx[:dict_no['source']]
    valid_idx = idx[dict_no['source']:(dict_no['source'] + dict_no['valid'])]
    target_idx = idx[(dict_no['source'] + dict_no['valid']):]

    x_source = data_x.loc[source_idx]
    c_source = data_c.loc[source_idx]

    x_valid = data_x.loc[valid_idx]
    c_valid = data_c.loc[valid_idx]

    x_target = data_x.loc[target_idx]
    c_target = data_c.loc[target_idx]

    # Selects source dataset based on the setting and test_store_type
    if setting == 'train-on-all':
        source_sub_idx = c_source.index[c_source[test_store_type] >= 0].tolist()
    elif setting == 'train-on-rest':
        source_sub_idx = c_source.index[c_source[test_store_type] == 0].tolist()
    elif setting == 'train-on-specific':
        source_sub_idx = c_source.index[c_source[test_store_type] == 1].tolist()

    # Selects valid and target datasets based on test_store_type
    valid_sub_idx = c_valid.index[c_valid[test_store_type] == 1].tolist()
    target_sub_idx = c_target.index[c_target[test_store_type] == 1].tolist()

    # Divides source, valid, and target datasets
    source = x_source.loc[source_sub_idx]
    valid = x_valid.loc[valid_sub_idx]
    target = x_target.loc[target_sub_idx]

    source.to_csv('./data_files/source.csv', index=False)
    valid.to_csv('./data_files/valid.csv', index=False)
    target.to_csv('./data_files/target.csv', index=False)

    return


def preprocess_data(normalization,
                    train_file_name, valid_file_name, test_file_name):
    """Loads datasets, divides features and labels, and normalizes features.
    Args:
      normalization: 'minmax' or 'standard'
      train_file_name: file name of training set
      valid_file_name: file name of validation set
      test_file_name: file name of testing set
    Returns:
      x_train: training features
      y_train: training labels
      x_valid: validation features
      y_valid: validation labels
      x_test: testing features
      y_test: testing labels
      col_names: column names
    """

    # Loads datasets
    train = pd.read_csv('./data_files/' + train_file_name)
    valid = pd.read_csv('./data_files/' + valid_file_name)
    test = pd.read_csv('./data_files/' + test_file_name)

    # Extracts label
    y_train = np.asarray(train['Y'])
    y_valid = np.asarray(valid['Y'])
    y_test = np.asarray(test['Y'])

    # Drops label
    train = train.drop(columns=['Y'])
    valid = valid.drop(columns=['Y'])
    test = test.drop(columns=['Y'])

    # Column names
    col_names = train.columns.values.astype(str)

    # Concatenates train, valid, test for normalization
    df = pd.concat((train, valid, test), axis=0)

    # Normalization
    if normalization == 'minmax':
        scaler = preprocessing.MinMaxScaler()  # 归一化
    elif normalization == 'standard':
        scaler = preprocessing.StandardScaler()  # 标准化

    scaler.fit(df)
    df = scaler.transform(df)

    # Divides df into train / valid / test sets
    # 划分 train / valid / test 数据集
    train_no = len(train)
    valid_no = len(valid)
    test_no = len(test)

    x_train = df[range(train_no), :]
    x_valid = df[range(train_no, train_no + valid_no), :]
    x_test = df[range(train_no + valid_no, train_no + valid_no + test_no), :]

    return x_train, y_train, x_valid, y_valid, x_test, y_test, col_names


def load_image_data(data_name, dict_no, noise_rate):
    """Loads image datasets.  加载图片数据集的方法
    This module loads CIFAR10 and CIFAR100 datasets and
    saves train.npz, valid.npz and test.npz files under data_files directory.
    If noise_rate > 0.0, adds noise on the datasets.
    Args:
      data_name: 'cifar10' or 'cifar100'
      dict_no: Training and validation set numbers
      noise_rate: Label corruption ratio
    Returns:
      noise_idx: Indices of noisy samples  损坏样本指数
    """

    # Loads datasets
    if data_name == 'cifar10':
        (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
    elif data_name == 'cifar100':
        (x_train, y_train), (x_test, y_test) = datasets.cifar100.load_data()

    # Splits train, valid and test sets
    train_idx = np.random.permutation(len(x_train))  # permutation函数将len长度的list随机排序

    valid_idx = train_idx[:dict_no['valid']]
    train_idx = train_idx[dict_no['valid']:(dict_no['train'] + dict_no['valid'])]

    test_idx = np.random.permutation(len(x_test))[:dict_no['test']]

    x_valid = x_train[valid_idx]
    x_train = x_train[train_idx]
    x_test = x_test[test_idx]

    y_valid = y_train[valid_idx].flatten()  # flatten函数将array降为1维
    y_train = y_train[train_idx].flatten()
    y_test = y_test[test_idx].flatten()

    # Adds noise on labels  添加噪声
    y_train, noise_idx = dvrl_utils.corrupt_label(y_train, noise_rate)

    # Saves data
    if not os.path.exists('data_files'):
        os.makedirs('data_files')

    np.savez_compressed('./data_files/train.npz',
                        x_train=x_train, y_train=y_train)
    np.savez_compressed('./data_files/valid.npz',
                        x_valid=x_valid, y_valid=y_valid)
    np.savez_compressed('./data_files/test.npz',
                        x_test=x_test, y_test=y_test)

    return noise_idx


def load_image_data_from_file(train_file_name, valid_file_name, test_file_name):
    """Loads image datasets from npz files and divides features and labels.
    Args:
      train_file_name: file name of training set
      valid_file_name: file name of validation set
      test_file_name: file name of testing set
    Returns:
      x_train: training features
      y_train: training labels
      x_valid: validation features
      y_valid: validation labels
      x_test: testing features
      y_test: testing labels
    """

    # Loads images datasets
    train = np.load('./data_files/' + train_file_name)
    valid = np.load('./data_files/' + valid_file_name)
    test = np.load('./data_files/' + test_file_name)

    # Divides features and labels
    x_train = train['x_train']
    y_train = train['y_train']

    x_valid = valid['x_valid']
    y_valid = valid['y_valid']

    x_test = test['x_test']
    y_test = test['y_test']

    return x_train, y_train, x_valid, y_valid, x_test, y_test


def encode_image(features, encoder_model, input_shape, preprocess_function,
                 batch_size=16):
    """Encodes images using pre-trained encoder model.
    需要使用预训练的model来对图像数据进行编码
    在main函数中的  数据编码部分  对x_train x_valid x_test 进行了编码 得到enc_x_train/valid/test
    We apply encoding step with a pre-trained model and save the encoded
    representation to input them
    directly to DVRL, for computational efficiency.  将encoded表示直接输入给DVRL  来获得高效的计算
    Args:
      features: Input image
      encoder_model: Model for encoding images (e.g., InceptionV3)  图像预处理函数-InceptionV3
      input_shape: Input shape of the encoder model
      preprocess_function: Preprocessing function from features to input_shape of
                           encoder_model   预处理函数：encoder model从特征到输入形状的预处理函数
      batch_size: Number of mini-batches for encoding images
    Returns:
      encoded_features: Encoded images
    """

    # Number of samples
    n_features = len(features)  # 输入image的len

    # Placeholder for batch of images
    batch_of_images_placeholder = tf.placeholder('uint8',
                                                 (None, features.shape[1],
                                                  features.shape[2],
                                                  features.shape[3]))
    # 使用tf.image.resize_images  调整输入图片的尺寸大小
    tf_resize_op = tf.image.resize_images(batch_of_images_placeholder,
                                          input_shape, method=0)

    # Data extraction function  数据提取函数
    def data_generator(sess, data):
        """Generates preprocessed data.  生成预处理的data   一个batch一个batch的加工
        Args:
          sess: Session
          data: Image data
        Returns:
          generator: Generator function
        """

        def generator():
            """Subfunction of data generator in the session.  session中数据生成器的子程序
            Yields:
              batch_of_image_preprocessed: preprocessed data  预处理数据
            """
            start = 0
            end = start + batch_size  # 一个batch的长度
            n = data.shape[0]
            while True:
                batch_of_images_resized = sess.run(tf_resize_op,
                                                   {batch_of_images_placeholder:
                                                        data[start:end]})  # 一个batch中resized过的image
                batch_of_images__preprocessed = \
                    preprocess_function(batch_of_images_resized)  # 使用预处理函数将图片进行加工
                start = start + batch_size
                end = end + batch_size # 向后推移一个batch
                if start >= n:
                    start = 0
                    end = batch_size
                yield batch_of_images__preprocessed

        return generator

    with tf.Session() as sess:

        backend.set_session(sess)
        model = encoder_model()
        data_gen = data_generator(sess, features)
        ftrs_training = model.predict_generator(data_gen(),
                                                n_features / batch_size,
                                                verbose=1)

    encoded_features = \
        np.array([ftrs_training[i].flatten() for i in range(n_features)])

    return encoded_features
