import tensorflow as tf
import mnist_inference
import os

# 配置神经网络参数
from tensorflow.examples.tutorials.mnist import input_data

BATCH_SIZE = 100
REGULARAZTION_RATE = 0.0001
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
TRAINING_STEPS = 15000
MOVING_AVERAGR_DECAY = 0.99
# 模型保存的路径和文件名
MODEL_SAVE_PATH = os.getcwd() + '/mnist_model/'
MODEL_NAME = 'model.ckpt'


def train(mnist):
    x = tf.placeholder(name='x-input', shape=[None, mnist_inference.INPUT_NODE], dtype=tf.float32)
    y_ = tf.placeholder(name='y-input', shape=[None, mnist_inference.OUTPUT_NODE], dtype=tf.float32)
    # 定义L2正则化类
    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    # 使用前向传播计算结果
    y = mnist_inference.inference(input_tensor=x, regularizer=regularizer)

    # 定义当前训练的轮数
    global_step = tf.Variable(0, trainable=False)

    # 定义滑动平均模型
    variable_average = tf.train.ExponentialMovingAverage(decay=MOVING_AVERAGR_DECAY, num_updates=global_step)
    variable_average_op = variable_average.apply(tf.trainable_variables())

    # 定义交叉熵与最终损失函数
    # 这个函数的参数要显示地传参
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('loss'))

    # 定义学习率
    learning_rate = tf.train.exponential_decay(learning_rate=LEARNING_RATE_BASE, decay_rate=LEARNING_RATE_DECAY,
                                               decay_steps=TRAINING_STEPS / BATCH_SIZE, global_step=global_step,
                                               staircase=True)
    train_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss=loss,
                                                                                         global_step=global_step)
    with tf.control_dependencies(control_inputs=[train_step, variable_average_op]):
        train_op = tf.no_op(name='train')

    # 初始化tensorflow 持久化类
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value, step = sess.run(fetches=[train_op, loss, global_step], feed_dict={x: xs, y_: ys})
            if i % 1000 == 0:
                print('After %d training steps,loss on traning batch is %g' % (step, loss_value))
                # 每训练1000步保存当前的模型
                saver.save(sess=sess, save_path=os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)


def main():
    # mnist = input_data.read_data_sets('/Users/gbzhu/dataset/MNIST_data', one_hot=True)  # for mac
    mnist = input_data.read_data_sets('/home/gbzhu/datasets/MNIST_data', one_hot=True)  # for linux
    train(mnist)


if __name__ == '__main__':
    main()
