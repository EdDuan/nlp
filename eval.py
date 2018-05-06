import tensorflow as tf
import inference
import train
import os


def evaluate(x_dev, y_dev, iteration):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [
            None,  # batch_size
            inference.MAX_LEN_DOC,  # 一个文档最多几句话
            inference.SENTENCE_LEN],  # sentence2vec维度对应channels
                           name='x-input')
        y_ = tf.placeholder(tf.float32, [None, inference.MAX_LEN_DOC], name='y-input')
        validate_feed = {x: x_dev, y_: y_dev}

        # 因为是验证集，不加正则化
        y = inference.inference(x)
        prediction = tf.cast(y >= 0.5, tf.float32)
        correct_prediction = tf.equal(prediction, y_)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        saver = tf.train.Saver()
        with tf.Session() as sess:
            # tf.train.get_checkpoint_state通过checkpoint文件自动找到目录中最新模型文件名
            ckpt = tf.train.get_checkpoint_state(train.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                # 加载模型
                saver.restore(sess, ckpt.model_checkpoint_path)
                accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
                print("After %s training step(s), validation accuracy = %g" % (iteration, accuracy_score))
            else:
                print('No checkpoint file found')
                return
