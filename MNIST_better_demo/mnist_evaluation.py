import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_inference
import mnist_train
import time


def evaluate(mnist):
    with tf.Graph().as_default() as graph:
        x = tf.placeholder(dtype=tf.float32, shape=[None, mnist_inference.INPUT_NODE], name='x-input')
        y_ = tf.placeholder(dtype=tf.float32, shape=[None, mnist_inference.OUTPUT_NODE], name='y-input')
        validata_feed = {x: mnist.validation.images, y_: mnist.validation.labels}

        test_feed = {x: mnist.test.images, y_: mnist.test.labels}

        y = mnist_inference.inference(input_tensor=x, regularizer=None)

        # 计算正确率
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # 通过变量重命名的方式加载模型
        variable_averages = tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVERAGR_DECAY)
        variable_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variable_to_restore)

        with tf.Session() as sess:
            # 找到最新模型
            ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                # 加载模型
                saver.restore(sess=sess, save_path=ckpt.model_checkpoint_path)
                # 通过文件名找到模型保存时迭代的轮数
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                validate_accuracy_score = sess.run(fetches=accuracy, feed_dict=validata_feed)
                test_accuracy_score = sess.run(fetches=accuracy,feed_dict=test_feed)
                print('After %s training steps, validation accuracy is: %g' % (global_step, validate_accuracy_score))
                print('After %s training steps, test accuracy is: %g' % (global_step, test_accuracy_score))
            else:
                print('No checkpoint file found!')
                return


def main():
    mnist = input_data.read_data_sets('/home/gbzhu/datasets/MNIST_data', one_hot=True)  # for linux
    evaluate(mnist)


if __name__ == '__main__':
    main()
