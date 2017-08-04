"""
demo for MNIST data
author: gbzhu
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 定义 MNIST 数据集相关的常数
INPUT_NODE = 784
OUTPUT_NODE = 10

# 配置神经网络的参数
LAYER_1_NODE = 500  # 隐藏层节点数
BATCH_SIZE = 100  # 一个batch的大小
LEARNING_RATE_BASE = 0.8  # 学习率
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 10000
MOVING_AVERAGE_DECAY = 0.99


# 计算前向传播的结果，分为两种
def inference(input_tensor, avg_class, reuse=True):
    with tf.variable_scope('layer_1', reuse=reuse):
        weights = tf.get_variable('weights', [INPUT_NODE, LAYER_1_NODE],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable('biases', [LAYER_1_NODE], initializer=tf.constant_initializer(0.0))

        if avg_class is None:
            # 计算隐藏层的前向传播结果
            layer_1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)
        else:  # 使用了滑动平均类
            layer_1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights)) + avg_class.average(biases))

    with tf.variable_scope('layer_2', reuse=reuse):
        weights = tf.get_variable('weights', [LAYER_1_NODE, OUTPUT_NODE],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable('biases', [OUTPUT_NODE], initializer=tf.constant_initializer(0.0))
        # 没有使用滑动平均类
        if avg_class is None:
            return tf.matmul(layer_1, weights) + biases
        else:  # 使用了滑动平均类
            return tf.matmul(layer_1, avg_class.average(weights)) + avg_class.average(biases)


# 神经网络的训练过程
def train(mnist):
    x = tf.placeholder(tf.float32, shape=[None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, shape=[None, OUTPUT_NODE], name='y-input')

    # 计算当前参数下的前向传播结果，不使用滑动平均类
    y = inference(input_tensor=x, avg_class=None, reuse=False)

    # 定义存储训练轮数的变量，不需要计算滑动平均值，指定为不可训练的变量
    global_step = tf.Variable(0, trainable=False)

    # 初始化滑动平均类，根据给定的平均衰减率和训练轮数
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    # 指定哪些变量要使用滑动平均,设置了trainable = False的变量不会使用
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    # 计算使用滑动平均之后的前向传播的结果
    average_y = inference(input_tensor=x, avg_class=variable_averages, reuse=True)

    # 计算损失函数（交叉熵）
    # tf.argmax(): [0,0,0,0,0,0,0,0,1]  =>  9
    # 这个函数的参数要显示地传参
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    # 计算当前batch中所用样例的交叉熵的平均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # 使用L2正则化正则损失函数
    # L2正则化函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    # 计算模型的正则化损失
    print(tf.get_variable_scope().reuse)

    # 这一步需要要，很关键，不然拿不到weights的值
    with tf.variable_scope('', reuse=True):
        regularization = regularizer(tf.get_variable('layer_1/weights')) + regularizer(
            tf.get_variable('layer_2/weights'))
    # 总损失 = 交叉熵 + 正则化损失
    loss = cross_entropy_mean + regularization

    # 设置学习率（指数衰减）
    learning_rate = tf.train.exponential_decay(learning_rate=LEARNING_RATE_BASE, global_step=global_step,
                                               decay_steps=mnist.train.num_examples / BATCH_SIZE,
                                               decay_rate=LEARNING_RATE_DECAY)
    # 优化损失
    train_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss=loss,
                                                                                         global_step=global_step)
    # 更新参数和参数的滑动平均值
    train_op = tf.group(train_step, variable_averages_op)

    correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 开始训练
    with tf.Session() as sess:
        tf.initialize_all_variables().run()

        # 准备验证数据集
        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
        # 准备测试数据集
        test_feed = {x: mnist.test.images, y_: mnist.test.labels}

        # 迭代训练神经网络
        for i in range(TRAINING_STEPS):
            if i % 1000 == 0:
                # 计算滑动平均模型在验证数据集上的效果
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print('经过 %d 步训练，验证数据集在滑动模型上的评估是：%g' % (i, validate_acc))
            # 产生这一轮batch的训练数据
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x: xs, y_: ys})
        # 训练之后，用测试集最终正确率
        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print('测试数据集在滑动模型上的评估是：%g' % test_acc)


def main():
    # mnist = input_data.read_data_sets('/Users/gbzhu/dataset/MNIST_data', one_hot=True)  # for mac
    mnist = input_data.read_data_sets('/home/gbzhu/datasets/MNIST_data', one_hot=True)   # for linux
    train(mnist)

if __name__ == '__main__':
    main()
