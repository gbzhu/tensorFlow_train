import tensorflow as tf
import mnist_inference
import os

# 配置神经网络参数
BATCH_SIZE = 100
REGULARAZTION_RATE = 0.0001
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
TRAINING_STEPS = 15000
MOVING_AVERAGR_DECAY = 0.99
# 模型保存的路径和文件名
MODEL_SAVE_PATH = os.getcwd() + 'mnist_model'
MODEL_NAME = 'model.ckpt'


def train(mnist):
    x = tf.placeholder(name='x-input', shape=[None, mnist_inference.INPUT_NODE], dtype=tf.float32)
    y_ = tf.placeholder(name='y-input', shape=[None, mnist_inference.OUTPUT_NODE], dtype=tf.float32)
    regularizer = tf.contrib.layer.l2_regularizer(REGULARAZTION_RATE)
    # 使用前向传播计算结果
    y = mnist_inference.inference(input_tensor=x, regularizer=regularizer)

    # 定义当前训练的轮数
    global_step = tf.Variable(0, trainable=False)

    # 定义滑动平均模型
    variable_average = tf.train.ExponentialMovingAverage(decay=MOVING_AVERAGR_DECAY, num_updates=global_step)
    variable_average_op = variable_average.apply(tf.trainable_variables())

    # 定义交叉熵与最终损失函数
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(y, tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('loss'))

    # 定义学习率
    learning_rate = tf.train.exponential_decay(learning_rate=LEARNING_RATE_BASE, decay_rate=LEARNING_RATE_DECAY,
                                               decay_steps=TRAINING_STEPS / BATCH_SIZE, global_step=global_step)
    train_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss=loss)
    with tf.control_dependencies(control_inputs=[train_step, variable_average_op]):
        train_op = tf.no_op(name='train')