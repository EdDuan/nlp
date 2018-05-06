# encoding: utf-8
import tensorflow as tf

# 配置参数
MAX_LEN_DOC = 125  # 一个文档最多有的句子数
SENTENCE_LEN = 100  # sentence2vec维度

LAYERS = [2048, 1024, 512, 256, 128, 64, 32]  # channels 数量
FILITER_SIZE = [1, 2, 3]  # 卷积核大小
FILITER_LEN = len(FILITER_SIZE)  # 用的卷积核个数


def inference(input_tensor):
    # 用FILITER_SIZE里每一个size的卷积核做一维卷积，然后concat
    # 第1层
    conv_outputs1 = []
    for size in FILITER_SIZE:
        with tf.variable_scope('conv1_%d' % size):
            conv1_weights = tf.get_variable("weight", [size, SENTENCE_LEN, LAYERS[0]],
                                            initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv1_biases = tf.get_variable("bias", [LAYERS[0]], initializer=tf.constant_initializer(0.0))
            conv1 = tf.nn.conv1d(input_tensor, conv1_weights, stride=1, padding='SAME')
            relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
            conv_outputs1.append(relu1)
    conv_output1 = tf.concat(conv_outputs1, 2)
    # 第2层
    conv_outputs2 = []
    for size in FILITER_SIZE:
        with tf.variable_scope('conv2_%d' % size):
            conv2_weights = tf.get_variable("weight", [size, LAYERS[0] * FILITER_LEN, LAYERS[1]],
                                            initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv2_biases = tf.get_variable("bias", [LAYERS[1]], initializer=tf.constant_initializer(0.0))
            conv2 = tf.nn.conv1d(conv_output1, conv2_weights, stride=1, padding='SAME')
            relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
            conv_outputs2.append(relu2)
    conv_output2 = tf.concat(conv_outputs2, 2)
    # 第3层
    conv_outputs3 = []
    for size in FILITER_SIZE:
        with tf.variable_scope('conv3_%d' % size):
            conv3_weights = tf.get_variable("weight", [size, LAYERS[1] * FILITER_LEN, LAYERS[2]],
                                            initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv3_biases = tf.get_variable("bias", [LAYERS[2]], initializer=tf.constant_initializer(0.0))
            conv3 = tf.nn.conv1d(conv_output2, conv3_weights, stride=1, padding='SAME')
            relu3 = tf.nn.relu(tf.nn.bias_add(conv3, conv3_biases))
            conv_outputs3.append(relu3)
    conv_output3 = tf.concat(conv_outputs3, 2)
    # 第4层
    conv_outputs4 = []
    for size in FILITER_SIZE:
        with tf.variable_scope('conv4_%d' % size):
            conv4_weights = tf.get_variable("weight", [size, LAYERS[2] * FILITER_LEN, LAYERS[3]],
                                            initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv4_biases = tf.get_variable("bias", [LAYERS[3]], initializer=tf.constant_initializer(0.0))
            conv4 = tf.nn.conv1d(conv_output3, conv4_weights, stride=1, padding='SAME')
            relu4 = tf.nn.relu(tf.nn.bias_add(conv4, conv4_biases))
            conv_outputs4.append(relu4)
    conv_output4 = tf.concat(conv_outputs4, 2)
    # 第5层
    conv_outputs5 = []
    for size in FILITER_SIZE:
        with tf.variable_scope('conv5_%d' % size):
            conv5_weights = tf.get_variable("weight", [size, LAYERS[3] * FILITER_LEN, LAYERS[4]],
                                            initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv5_biases = tf.get_variable("bias", [LAYERS[4]], initializer=tf.constant_initializer(0.0))
            conv5 = tf.nn.conv1d(conv_output4, conv5_weights, stride=1, padding='SAME')
            relu5 = tf.nn.relu(tf.nn.bias_add(conv5, conv5_biases))
            conv_outputs5.append(relu5)
    conv_output5 = tf.concat(conv_outputs5, 2)
    # 第6层
    conv_outputs6 = []
    for size in FILITER_SIZE:
        with tf.variable_scope('conv6_%d' % size):
            conv6_weights = tf.get_variable("weight", [size, LAYERS[4] * FILITER_LEN, LAYERS[5]],
                                            initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv6_biases = tf.get_variable("bias", [LAYERS[5]], initializer=tf.constant_initializer(0.0))
            conv6 = tf.nn.conv1d(conv_output5, conv6_weights, stride=1, padding='SAME')
            relu6 = tf.nn.relu(tf.nn.bias_add(conv6, conv6_biases))
            conv_outputs6.append(relu6)
    conv_output6 = tf.concat(conv_outputs6, 2)
    # 第7层
    conv_outputs7 = []
    for size in FILITER_SIZE:
        with tf.variable_scope('conv7_%d' % size):
            conv7_weights = tf.get_variable("weight", [size, LAYERS[5] * FILITER_LEN, LAYERS[6]],
                                            initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv7_biases = tf.get_variable("bias", [LAYERS[6]], initializer=tf.constant_initializer(0.0))
            conv7 = tf.nn.conv1d(conv_output6, conv7_weights, stride=1, padding='SAME')
            relu7 = tf.nn.relu(tf.nn.bias_add(conv7, conv7_biases))
            conv_outputs7.append(relu7)
    conv_output7 = tf.concat(conv_outputs7, 2)
    # 上一层输出为[batch_size, MAX_LEN_DOC, 3]的tensor，最后用一个大小为1的卷积核得到最后结果
    with tf.variable_scope('output'):
        output_weights = tf.get_variable("weight", [1, LAYERS[3] * FILITER_LEN, 1],
                                         initializer=tf.truncated_normal_initializer(stddev=0.1))
        output_biases = tf.get_variable("bias", [1], initializer=tf.constant_initializer(0.0))
        conv_output = tf.nn.conv1d(conv_output4, output_weights, stride=1, padding='SAME')
        logit = tf.nn.bias_add(conv_output, output_biases)  # 激活函数用sigmoid
        logit = tf.squeeze(logit)

    return logit
