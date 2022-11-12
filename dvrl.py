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

"""Data Valuation using Reinforcement Learning (DVRL)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import os
import numpy as np
from sklearn import metrics
import tensorflow.compat.v1 as tf
import tqdm
# from dvrl import dvrl_metrics
import dvrl_metrics
from tensorflow.contrib import layers as contrib_layers


class Dvrl(object):
    """Data Valuation using Reinforcement Learning (DVRL) class.
    使用强化学习进行数据评估
      Attributes:
        x_train: training feature  训练特征
        y_train: training labels   训练label标签
        x_valid: validation features  验证特征
        y_valid: validation labels    验证label
        problem: 'regression' or 'classification'  回归/分类问题
        pred_model: predictive model (object)      预测模型
        parameters: network parameters such as hidden_dim, iterations,  网络参数：隐藏层维度、迭代次数、
                    activation function, layer_number, learning rate  激活函数、网络层数、学习率
        checkpoint_file_name: File name for saving and loading the trained model
        flags: flag for training with stochastic gradient descent (flag_sgd)
               and flag for using pre-trained model (flag_pretrain)  使用SGD或预训练model的flag
    """

    # 对于Dvrl模型的初始化
    def __init__(self, x_train, y_train, x_valid, y_valid,
                 problem, pred_model, parameters, checkpoint_file_name, flags):
        """Initializes DVRL."""

        # Inputs  确定DVRL模型的输入
        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid

        self.problem = problem  # 确定是 回归 or 分类 问题

        # One-hot encoded labels  使用np.eye()[]  ()为长度  [] 为待转换数组数组
        # 将标签数据转化为one-hot编码
        if self.problem == 'classification':  # 分类问题
            self.y_train_onehot = \
                np.eye(len(np.unique(y_train)))[y_train.astype(int)]  # np.eye  将数组转换成one-hot形式
            self.y_valid_onehot = \
                np.eye(len(np.unique(y_train)))[y_valid.astype(int)]
        elif self.problem == 'regression':  # 回归问题
            self.y_train_onehot = np.reshape(y_train, [len(y_train), 1])
            self.y_valid_onehot = np.reshape(y_valid, [len(y_valid), 1])

        # Network parameters   网络参数
        self.hidden_dim = parameters['hidden_dim']
        self.comb_dim = parameters['comb_dim']
        self.outer_iterations = parameters['iterations']
        self.act_fn = parameters['activation']
        self.layer_number = parameters['layer_number']
        self.batch_size = np.min([parameters['batch_size'], len(x_train[:, 0])])
        self.learning_rate = parameters['learning_rate']

        # Basic parameters  基准参数
        self.epsilon = 1e-8  # Adds to the log to avoid overflow
        self.threshold = 0.9  # Encourages exploration   鼓励探索

        # Flags
        self.flag_sgd = flags['sgd']
        self.flag_pretrain = flags['pretrain']

        # If the pred_model uses stochastic gradient descent (SGD) for training
        if self.flag_sgd:
            self.inner_iterations = parameters['inner_iterations']
            self.batch_size_predictor = np.min([parameters['batch_size_predictor'], 
                                                len(x_valid[:, 0])])  # 预测器的batch- size

        # Checkpoint file name
        self.checkpoint_file_name = checkpoint_file_name

        # Basic parameters
        self.data_dim = len(x_train[0, :])  # 数据维度
        self.label_dim = len(self.y_train_onehot[0, :])  # 标签维度

        # Training Inputs  model训练的inputs
        # x_input can be raw input or its encoded representation, e.g. using a
        # pre-trained neural network. Using encoded representation can be beneficial
        # to reduce computational cost for high dimensional inputs, like images.
        # x_input 可以是原始输入或者是其编码表示。 比如：用一个预训练的神经网络。
        # 使用编码表示可以有效减少高维度向量的计算消耗。

        self.x_input = tf.placeholder(tf.float32, [None, self.data_dim])
        self.y_input = tf.placeholder(tf.float32, [None, self.label_dim])

        # Prediction difference
        # y_hat_input is the prediction difference between predictive models
        # trained on the training set and validation set.   
        # (adding y_hat_input into data value estimator as the additional input
        # is observed to improve data value estimation quality in some cases)
        # y_hat_input  预测器在训练集上训练与预测器在验证集上训练的预测差异
        # y_hat_input就是论文中指的 Marginal information  边际信息
        # （将y_hat_input添加到DVE中作为额外的输入在某些情况下可以提高DVE性能）
        self.y_hat_input = tf.placeholder(tf.float32, [None, self.label_dim])

        # Selection vector  选择向量
        self.s_input = tf.placeholder(tf.float32, [None, 1])

        # Rewards (Reinforcement signal)
        self.reward_input = tf.placeholder(tf.float32)

        # Pred model (Note that any model architecture can be used as the predictor
        # model, either randomly initialized or pre-trained with the training data.
        # The condition for predictor model to have fit (e.g. using certain number
        # of back-propagation iterations) and predict functions as its subfunctions.
        # 任何模型结构都可以被用做为预测模型，如 随机初始化的或是使用训练集预训练的模型。
        self.pred_model = pred_model  # 预测模型

        # Final model
        self.final_model = pred_model

        # With randomly initialized predictor  随机初始化预测器
        if (not self.flag_pretrain) & self.flag_sgd:
            if not os.path.exists('tmp'):
                os.makedirs('tmp')
            pred_model.fit(self.x_train, self.y_train_onehot,
                           batch_size=len(self.x_train), epochs=0)  # 预测模型训练epochs=0
            # Saves initial randomization
            pred_model.save_weights('tmp/pred_model.h5')
            # With pre-trained model, pre-trained model should be saved as
            # 'tmp/pred_model.h5'
            # pred_model

        # Baseline model
        if self.flag_sgd:
            self.ori_model = copy.copy(self.pred_model)
            self.ori_model.load_weights('tmp/pred_model.h5')

            # Trains the model
            self.ori_model.fit(x_train, self.y_train_onehot,
                               batch_size=self.batch_size_predictor,
                               epochs=self.inner_iterations, verbose=False)
        else:
            self.ori_model = copy.copy(self.pred_model)
            self.ori_model.fit(x_train, y_train)

        # Valid baseline model
        if 'summary' in dir(self.pred_model):
            self.val_model = copy.copy(self.pred_model)
            self.val_model.load_weights('tmp/pred_model.h5')

            # Trains the model
            self.val_model.fit(x_valid, self.y_valid_onehot,
                               batch_size=self.batch_size_predictor,
                               epochs=self.inner_iterations, verbose=False)
        else:
            self.val_model = copy.copy(self.pred_model)
            self.val_model.fit(x_valid, y_valid)

    def data_value_evaluator(self):
        """Returns data value evaluator model.
        Here, we assume a simple multi-layer perceptron architecture for the data
        value evaluator model. For data types like tabular, multi-layer perceptron
        is already efficient at extracting the relevant information.
        For high-dimensional data types like images or text,
        it is important to introduce inductive biases to the architecture to
        extract information efficiently. In such cases, there are two options:
        (i) Input the encoded representations (e.g. the last layer activations of
        ResNet for images, or the last layer activations of BERT for  text) and use
        the multi-layer perceptron on top of it. The encoded representations can
        simply come from a pre-trained predictor model using the entire dataset.
        (ii) Modify the data value evaluator model definition below to have the
        appropriate inductive bias (e.g. using convolutional layers for images,
        or attention layers text).
        Returns:
          dve: data value estimations
        我们假设DVE（数据估值器）是由多层感知机结构组成，对于表格这种数据类型，多层感知机能够有效的提取相关信息
        对于高维的数据类型比如图片、文本，在结构中引入感应biases来高效的获取信息。
        在某些情况下有两种选择：
        （1）输入一个编码表示（例如，最后的激活层 对图像使用ResNet，对文本使用BERT） 使用其上面的多层感知机
            编码表示就可以使用整个数据集从预训练的预测器模型中得到
        （2）修改DVE模型定义，以获得适当的感应偏差（例如对图像使用卷积层，对文本使用注意层）
        """
        with tf.variable_scope('data_value_estimator', reuse=tf.AUTO_REUSE):
            inputs = tf.concat((self.x_input, self.y_input), axis=1)

            # Stacks multi-layered perceptron  堆叠多层感知机
            inter_layer = contrib_layers.fully_connected(  # 创建一个全连接层
                inputs, self.hidden_dim, activation_fn=self.act_fn)
            for _ in range(int(self.layer_number - 3)):
                inter_layer = contrib_layers.fully_connected(
                    inter_layer, self.hidden_dim, activation_fn=self.act_fn)  # 迭代创建中间的全连接层
            inter_layer = contrib_layers.fully_connected(
                inter_layer, self.comb_dim, activation_fn=self.act_fn)

            # Combines with y_hat  和y_hat结合
            comb_layer = tf.concat((inter_layer, self.y_hat_input), axis=1)
            comb_layer = contrib_layers.fully_connected(
                comb_layer, self.comb_dim, activation_fn=self.act_fn)
            dve = contrib_layers.fully_connected(
                comb_layer, 1, activation_fn=tf.nn.sigmoid)  # 最终输出的是dve（数据价值评估）的结果

        return dve

    def train_dvrl(self, perf_metric):
        """Trains DVRL based on the specified objective function.
        Args:
          性能度量
          perf_metric: 'auc', 'accuracy', 'log-loss' for classification
                       'mae', 'mse', 'rmspe' for regression
        """

        # Generates selected probability
        est_data_value = self.data_value_evaluator()  # 生成数据估值器DVE
        # 强化学习算法  生成器的loss
        # Generator loss (REINFORCE algorithm)
        prob = tf.reduce_sum(self.s_input * tf.log(est_data_value + self.epsilon) + \
                             (1 - self.s_input) * \
                             tf.log(1 - est_data_value + self.epsilon))
        dve_loss = (-self.reward_input * prob) + \
                   1e3 * (tf.maximum(tf.reduce_mean(est_data_value) \
                                     - self.threshold, 0) + \
                          tf.maximum((1 - self.threshold) - \
                                     tf.reduce_mean(est_data_value), 0))

        # Variable  DVE中可训练的variables
        dve_vars = [v for v in tf.trainable_variables() \
                    if v.name.startswith('data_value_estimator')]

        # Solver   使用Adam优化器来对DVE的loss进行优化 min dve_loss
        dve_solver = tf.train.AdamOptimizer(self.learning_rate).minimize(
            dve_loss, var_list=dve_vars)

        # Baseline performance
        if self.flag_sgd:
            y_valid_hat = self.ori_model.predict(self.x_valid)
        else:
            if self.problem == 'classification':  # 分类问题
                y_valid_hat = self.ori_model.predict_proba(self.x_valid)
            elif self.problem == 'regression':  # 回归问题
                y_valid_hat = self.ori_model.predict(self.x_valid)
        # 性能度量部分
        if perf_metric == 'auc':
            valid_perf = metrics.roc_auc_score(self.y_valid, y_valid_hat[:, 1])
        elif perf_metric == 'accuracy':
            valid_perf = metrics.accuracy_score(self.y_valid, np.argmax(y_valid_hat,
                                                                        axis=1))
        elif perf_metric == 'log_loss':
            valid_perf = -metrics.log_loss(self.y_valid, y_valid_hat)
        elif perf_metric == 'rmspe':
            valid_perf = dvrl_metrics.rmspe(self.y_valid, y_valid_hat)
        elif perf_metric == 'mae':
            valid_perf = metrics.mean_absolute_error(self.y_valid, y_valid_hat)
        elif perf_metric == 'mse':
            valid_perf = metrics.mean_squared_error(self.y_valid, y_valid_hat)

        # Prediction differences  预测差异
        if self.flag_sgd:
            y_train_valid_pred = self.val_model.predict(self.x_train)
        else:
            if self.problem == 'classification':
                y_train_valid_pred = self.val_model.predict_proba(self.x_train)
            elif self.problem == 'regression':
                y_train_valid_pred = self.val_model.predict(self.x_train)
                y_train_valid_pred = np.reshape(y_train_valid_pred, [-1, 1])

        if self.problem == 'classification':
            # np.abs 表示返回()中的绝对值
            y_pred_diff = np.abs(self.y_train_onehot - y_train_valid_pred)  # diff为onehot表示-pred结果
        elif self.problem == 'regression':
            y_pred_diff = \
                np.abs(self.y_train_onehot - y_train_valid_pred) / self.y_train_onehot  # 再取个平均值

        # Main session
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())  # 参数全部初始化

        # Model save at the end  保存和加载模型
        saver = tf.train.Saver(dve_vars)

        # tqdm为进度条信息显示包
        for _ in tqdm.tqdm(range(self.outer_iterations)):  # 外循环 共2000个epoch 

            # Batch selection
            # np.random.permutation() 随机对序列进行排序
            batch_idx = \
                np.random.permutation(len(self.x_train[:, 0]))[:self.batch_size]

            x_batch = self.x_train[batch_idx, :]
            y_batch_onehot = self.y_train_onehot[batch_idx]
            y_batch = self.y_train[batch_idx]
            y_hat_batch = y_pred_diff[batch_idx]

            # Generates selection probability  生成选择概率
            est_dv_curr = sess.run(  # 生成当前数据价值
                est_data_value,
                feed_dict={
                    self.x_input: x_batch,
                    self.y_input: y_batch_onehot,
                    self.y_hat_input: y_hat_batch  # 每次进一个batch
                })

            # Samples the selection probability  采样当前选择的概率
            # 一次抛5枚硬币，每枚硬币正面朝上概率为0.5，做10次试验，求每次试验发生正面朝上的硬币个数
            # np.random.binomial(5, 0.5, 10)  其中est_dv_curr就是选择的概率
            sel_prob_curr = np.random.binomial(1, est_dv_curr, est_dv_curr.shape)

            # Exception (When selection probability is 0) 例外（当选择概率为0的时候）
            if np.sum(sel_prob_curr) == 0:
                est_dv_curr = 0.5 * np.ones(np.shape(est_dv_curr))
                sel_prob_curr = np.random.binomial(1, est_dv_curr, est_dv_curr.shape)  # 使用0.5

            # Trains predictor  训练预测器
            # If the predictor is neural network  神经网络
            if 'summary' in dir(self.pred_model):

                new_model = self.pred_model
                new_model.load_weights('tmp/pred_model.h5')

                # Train the model
                new_model.fit(x_batch, y_batch_onehot,
                              sample_weight=sel_prob_curr[:, 0],
                              batch_size=self.batch_size_predictor,
                              epochs=self.inner_iterations, verbose=False)

                y_valid_hat = new_model.predict(self.x_valid)

            else:
                new_model = self.pred_model
                new_model.fit(x_batch, y_batch, sel_prob_curr[:, 0])

            # Prediction
            if 'summary' in dir(new_model):
                y_valid_hat = new_model.predict(self.x_valid)
            else:
                if self.problem == 'classification':
                    y_valid_hat = new_model.predict_proba(self.x_valid)
                elif self.problem == 'regression':
                    y_valid_hat = new_model.predict(self.x_valid)

            # Reward computation
            if perf_metric == 'auc':
                dvrl_perf = metrics.roc_auc_score(self.y_valid, y_valid_hat[:, 1])
            elif perf_metric == 'accuracy':
                dvrl_perf = metrics.accuracy_score(self.y_valid, np.argmax(y_valid_hat,
                                                                           axis=1))
            elif perf_metric == 'log_loss':
                dvrl_perf = -metrics.log_loss(self.y_valid, y_valid_hat)
            elif perf_metric == 'rmspe':
                dvrl_perf = dvrl_metrics.rmspe(self.y_valid, y_valid_hat)
            elif perf_metric == 'mae':
                dvrl_perf = metrics.mean_absolute_error(self.y_valid, y_valid_hat)
            elif perf_metric == 'mse':
                dvrl_perf = metrics.mean_squared_error(self.y_valid, y_valid_hat)

            if self.problem == 'classification':
                reward_curr = dvrl_perf - valid_perf
            elif self.problem == 'regression':
                reward_curr = valid_perf - dvrl_perf

            # Trains the generator
            _, _ = sess.run(
                [dve_solver, dve_loss],
                feed_dict={
                    self.x_input: x_batch,
                    self.y_input: y_batch_onehot,
                    self.y_hat_input: y_hat_batch,
                    self.s_input: sel_prob_curr,
                    self.reward_input: reward_curr
                })

        # Saves trained model
        saver.save(sess, self.checkpoint_file_name)

        # Trains DVRL predictor
        # Generate data values
        final_data_value = sess.run(
            est_data_value, feed_dict={
                self.x_input: self.x_train,
                self.y_input: self.y_train_onehot,
                self.y_hat_input: y_pred_diff})[:, 0]

        # Trains final model
        # If the final model is neural network
        if 'summary' in dir(self.pred_model):
            self.final_model.load_weights('tmp/pred_model.h5')
            # Train the model
            self.final_model.fit(self.x_train, self.y_train_onehot,
                                 sample_weight=final_data_value,
                                 batch_size=self.batch_size_predictor,
                                 epochs=self.inner_iterations, verbose=False)  # 训练predict预测器model
        else:
            self.final_model.fit(self.x_train, self.y_train, final_data_value)

    def data_valuator(self, x_train, y_train):
        """Returns data values using the data valuator model.
        Args:
          x_train: training features
          y_train: training labels
        Returns:
          final_dat_value: final data values of the training samples  训练样本的最终价值
        """

        # One-hot encoded labels
        if self.problem == 'classification':
            y_train_onehot = np.eye(len(np.unique(y_train)))[y_train.astype(int)]
            y_train_valid_pred = self.val_model.predict_proba(x_train)
        elif self.problem == 'regression':
            y_train_onehot = np.reshape(y_train, [len(y_train), 1])
            y_train_valid_pred = np.reshape(self.val_model.predict(x_train),
                                            [-1, 1])

        # Generates y_train_hat
        if self.problem == 'classification':
            y_train_hat = np.abs(y_train_onehot - y_train_valid_pred)
        elif self.problem == 'regression':
            y_train_hat = np.abs(y_train_onehot - y_train_valid_pred) / y_train_onehot

        # Restores the saved model
        imported_graph = \
            tf.train.import_meta_graph(self.checkpoint_file_name + '.meta')

        sess = tf.Session()
        imported_graph.restore(sess, self.checkpoint_file_name)

        # Estimates data value  估计数据价值
        est_data_value = self.data_value_evaluator()

        final_data_value = sess.run(
            est_data_value, feed_dict={
                self.x_input: x_train,
                self.y_input: y_train_onehot,
                self.y_hat_input: y_train_hat})[:, 0]

        return final_data_value

    def dvrl_predictor(self, x_test):
        """Returns predictions using the predictor model.
        Args:
          x_test: testing features
        Returns:
          y_test_hat: predictions of the predictive model with DVRL
        """

        if self.flag_sgd:
            y_test_hat = self.final_model.predict(x_test)
        else:
            if self.problem == 'classification':
                y_test_hat = self.final_model.predict_proba(x_test)
            elif self.problem == 'regression':
                y_test_hat = self.final_model.predict(x_test)

        return y_test_hat
