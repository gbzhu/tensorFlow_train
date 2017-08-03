import tensorflow as tf

# 定义神经网络的参数
INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER_1_NODE = 500


def get_weights(shape, regularizer):
    weights = tf.get_variable(name='weights', shape=shape,
                              initializer=tf.truncated_normal_initializer(stddev=0.1))
    # 将变量的正则化损失加入到损失集合
    if regularizer is not None:
        tf.add_to_collection('loss', regularizer(weights))
    return weights


def inference(input_tensor, regularizer):
    with tf.variable_scope('layer_1'):
        weights = get_weights(shape=[INPUT_NODE, LAYER_1_NODE], regularizer=regularizer)
        biases = tf.get_variable(name='biases', shape=[LAYER_1_NODE],
                                 initializer=tf.constant_initializer(0.0))
        layer_1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)

    with tf.variable_scope('layer_2'):
        weights = get_weights(shape=[LAYER_1_NODE, OUTPUT_NODE], regularizer=regularizer)
        biases = tf.get_variable(name='biases', shape=[OUTPUT_NODE],
                                 initializer=tf.constant_initializer(0.0))
        return tf.matmul(layer_1, weights) + biases
