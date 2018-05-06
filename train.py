# encoding: utf-8

import tensorflow as tf
import inference
import math
import eval
import os
import numpy as np

# 配置神经网络参数
BATCH_SIZE = 32  # batch大小
# LEARNING_RATE_BASE = 0.8 #最开始的学习率
# LEARNING_RATE_DECAY = 0.99 #学习率削减率
LEARNING_RATE = 1e-8
# REGULARIZATION_RATE = 0.0001 #正则化的lambda
EPOCH = 30000  # 总的训练轮数
DEV_RATE = 0.01
# 模型保存的路径和文件名
MODEL_SAVE_PATH = "nlp_model/"
MODEL_NAME = "mlp_model"


# x_train = np.random.randn(500,20,100)
# y_train = np.random.randint(0,2,(500,20))
# x_dev = np.random.randn(50,20,100)
# y_dev = np.random.randint(0,2,(50,20))


def data_helper():
    y_inputdata = np.load('../data/label_preprocess.npy')
    y_length = np.load('data/doc_len.npy')
    print(y_inputdata.shape)
    x_input = np.load('../data/vector_preprocess.npz')
    x = x_input["arr_0"]
    print(x.shape)
    np.random.seed(3)
    sample_total = x.shape[0]
    devsize = int(sample_total * DEV_RATE)
    x_train = x[0: sample_total - devsize, :, :]
    y_train = y_inputdata[0:sample_total - devsize, :]
    x_dev = x[-devsize:, :, :]
    y_dev = y_inputdata[-devsize:, :]
    length_train = y_length[0: sample_total - devsize]
    length_test = y_length[-devsize:]

    print(x_train.shape)
    print(y_train.shape)
    print(x_dev.shape)
    print(y_dev.shape)
    print('length shape :',length_train.shape)
    return x_train, y_train, x_dev, y_dev, length_train, length_test

def computeloss(y, y_, doc_length):
    totalloss = 0
    for i in range(BATCH_SIZE):
        cross_entropy = (-y_[i] * tf.log(y[i]) - (1 - y_[i]) * tf.log(1 - y[i])) * doc_length[i]
        temploss = tf.reduce_sum(cross_entropy) / tf.cast(tf.count_nonzero(doc_length[i]), tf.float32)
        totalloss += temploss
    loss = totalloss / BATCH_SIZE
    # y = tf.clip_by_value(y, 1e-15, 1)
    # cross_entropy = (-y_ * tf.log(y) - (1 - y_) * tf.log(1 - y)) * doc_length
    # loss = tf.reduce_mean(cross_entropy)
    return loss


def train(x_train, y_train, x_dev, y_dev, length_train, length_dev):  # 传入训练集和验证集 维度eg.X:[M, 20, 100] Y[M, 20]
    # 定义输入输出
    x = tf.placeholder(tf.float32, [
        None,  # batch_size
        inference.MAX_LEN_DOC,  # 一个文档最多几句话
        inference.SENTENCE_LEN],  # sentence2vec维度对应channels
                       name='x-input')
    y_ = tf.placeholder(tf.float32, [None, inference.MAX_LEN_DOC], name='y-input')
    doc_length = tf.placeholder(tf.float32, [None, inference.MAX_LEN_DOC], name='doc_length')

    # 正则化
    # regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    # 前向传播
    y = inference.inference(x)

    # 交叉熵
    # cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.clip_by_value(y, 1e-15, 1), labels=y_)
    # cross_entropy_mean = tf.reduce_mean
    # # loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses')) #交叉熵加上正则化 loss function
    # loss = cross_entropy_mean()

    loss = computeloss(y, y_, doc_length)
    # 用指数衰减设置learing rate衰减
    # learning_rate = tf.train.exponential_decay(
    #     LEARNING_RATE_BASE,
    #     global_step,
    #     mnist.train.num_examples / BATCH_SIZE, LEARNING_RATE_DECAY,
    #     staircase=True)

    # 定义优化算法
    train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)

    # 模型持久化
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # 初始化
        tf.global_variables_initializer().run()
        data_size = x_train.shape[0]
        for i in range(EPOCH):
            for j in range(math.ceil(data_size / BATCH_SIZE)):
                start = j * BATCH_SIZE
                end = min(start + BATCH_SIZE, data_size)
                batch_x = x_train[start:end, :, :]
                batch_y = y_train[start:end, :]
                length = length_train[start:end, :]
                _, loss_value = sess.run([train_step, loss], feed_dict={x: batch_x, y_: batch_y, doc_length:length})
                print("After %d training step(s), loss on training batch is %g." % (
                    i * math.ceil(data_size / BATCH_SIZE) + j, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME))
            #eval.evaluate(x_dev, y_dev, i)


def main(argv=None):
    x_train, y_train, x_dev, y_dev, length_train, length_dev = data_helper()
    train(x_train, y_train, x_dev, y_dev, length_train, length_dev)


if __name__ == '__main__':
    # data_helper()
    tf.app.run()
