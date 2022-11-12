# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
from operator import le 
import os
from tabnanny import verbose 

import zipfile
from lightgbm import train

import numpy as np
import pandas as pd 
from six.moves import urllib
from sklearn import preprocessing
import tensorflow as tf

from keras import backend
from keras import datasets

import dvrl_utils

def load_tabular_data(data_name, dict_no, noise_rate):
    # Loads datasets from links
    uci_base_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/'

    # Adult Income dataset
    if data_name == 'adult':

        train_url = uci_base_url + 'adult/adult.data'
        test_url = uci_base_url + 'adult/adult.test'

        data_train = pd.read_csv(train_url, header=None)
        data_test = pd.read_csv(test_url, skiprows=1, header=None)

        df = pd.concat((data_train, data_test), axis=0)

        # Column names
        df.columns = ['Age', 'WorkClass', 'fnlwgt', 'Education', 'EducationNum',
                      'MaritalStatus', 'Occupation', 'Relationship', 'Race',
                      'Gender', 'CapitalGain', 'CapitalLoss', 'HoursPerWeek',
                      'NativeCountry', 'Income']

        # Creates binary labels
        df['Income'] = df['Income'].map({' <=50K': 0, ' >50K': 1,
                                         ' <=50K.': 0, ' >50K.': 1})

        # Changes string to float
        df.Age = df.Age.astype(float)
        df.fnlwgt = df.fnlwgt.astype(float)
        df.EducationNum = df.EducationNum.astype(float)
        df.EducationNum = df.EducationNum.astype(float)
        df.CapitalGain = df.CapitalGain.astype(float)
        df.CapitalLoss = df.CapitalLoss.astype(float)

        # One-hot encoding
        df = pd.get_dummies(df, columns=['WorkClass', 'Education', 'MaritalStatus',
                                         'Occupation', 'Relationship',
                                         'Race', 'Gender', 'NativeCountry'])

        # Sets label name as Y
        df = df.rename(columns={'Income': 'Y'})
        df['Y'] = df['Y'].astype(int)

        # Resets index
        df = df.reset_index()
        df = df.drop(columns=['index'])

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

    # Splits train, valid and test sets
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

def load_rossman_data(dict_no, setting, test_store_type):

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

def preprocess_data(normalization, train_file_name, valid_file_name, test_file_name):
    # Loads datasets
    train = pd.read_csv('./data_files/' + train_file_name)
    valid = pd.read_csv('./data_files/' + valid_file_name)
    test = pd.read_csv('./data_files/' + test_file_name)

    # 提取标签
    y_train = np.asarray(train['Y'])
    y_valid = np.asarray(valid['Y'])
    y_test = np.asarray(test['Y'])

    # Drops label
    train = train.drop(columns=['Y'])
    valid = valid.drop(columns=['Y'])
    test = test.drop(columns=['Y'])

    col_names = train.columns.values.astype(str)

    # 将train valid test合并之后进行正则化normalization
    df = pd.concat((train,valid,test),axis=0)

    if normalization == 'minmax':
        scaler = preprocessing.MinMaxScaler()
    elif normalization == 'standard':
        scaler = preprocessing.StandardScaler()

    scaler.fit(df)
    df = scaler.transform(df)

    # 将df中的训练、验证、测试集分开
    train_no = len(train)
    valid_no = len(train)
    test_no = len(test)

    x_train = df[range(train_no),:]
    x_valid = df[range(train_no,train_no+valid_no),:]
    x_test = df[range(train_no+valid_no,train_no+valid_no+test_no),:]

    return x_train, y_train, x_valid, y_valid, x_test, y_test, col_names

def load_image_data(data_name,dict_no,noise_rate):
    """
    加载图片数据集的方法
    加载的图片数据集可以是cifar10 or cifar100
    """

    # Load datasets
    if data_name == 'cifar10':
        (x_train,y_train),(x_test,y_test) = datasets.cifar10.load_data()
    elif data_name == 'cifar100':
        (x_train,y_train),(x_test,y_test) = datasets.cifar100.load_data()

    # 划分训练、验证、测试集
    # 其中permeation函数的作用是对与len长度的列表进行随机排序  打乱顺序
    train_idx = np.random.permutation(len(x_train))

    valid_idx = train_idx[:dict_no['valid']]
    train_idx = train_idx[dict_no['valid']:(dict_no['train'] + dict_no['valid'])]

    test_idx = np.random.permutation(len(x_test))[:dict_no['test']]

    x_valid = x_train[valid_idx] 
    x_train = x_train[train_idx]
    x_test = x_test[test_idx]

    y_valid = y_train[valid_idx].flatten()
    y_train = y_train[train_idx].flatten()
    y_test = y_test[test_idx].flatten()

    # 添加噪声
    y_train, noise_idx = dvrl_utils.corrupt_label(y_train, noise_rate)

    # 保存数据
    if not os.path.exists('data_files'):
        os.makedirs("data_files")
    
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

def encode_image(feature,encoder_model,input_shape,preprocess_function,batch_size=16):
    """
    使用预训练的encoder model来给图像编码
    参数列表：
        feature - 输入的图片
        encoder_model - 使用Inception V3来编码图片  图像预处理函数
        input_shape - encoder model的输入大小
        preprocess_function - 预处理函数  对于encoder model 特征到输入形状的预处理
    """

    # Numbers of samples
    n_feature = len(feature)

    # Placeholder for batch of images
    batch_of_image_placeholder = tf.placeholder('uniot8',(None,feature.shape[1],feature[2],feature[3]))

    # 调整输入图片尺寸大小
    tf_resize_op = tf.image.resize_image(batch_of_image_placeholder,input_shape,method=0)

    def data_generator(sess,data):
        """
        生成预处理的数据   一个batch一个batch的加工数据
        """
        def generator():
            start = 0
            end = start + batch_size
            n = data.shape[0]
            while True:
                batch_of_images_resized = sess.run(tf_resize_op,{batch_of_image_placeholder:data[start:end]})
                batch_of_imagees__preprocessed = preprocess_function(batch_of_images_resized)
                start = start + batch_size
                end = end + batch_size

                if start >= n:
                    start = 0
                    end = batch_size
                yield batch_of_imagees__preprocessed
            
        return generator

    with tf.Session() as sess:
        backend.set_session(sess)
        model = encoder_model()
        data_gen = data_generator(sess,feature)
        ftrs_training = model.predict_generator(data_gen(),
                                                n_feature/batch_size,
                                                verbose=1)

     encoded_features = \
        np.array([ftrs_training[i].flatten() for i in range(n_features)])

    return encoded_features