import tensorflow as tf

def conv2d_1(x, kernel, bias, strides, use_relu=True, use_BN = True, Reuse = None):
    x_padded = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
    out = tf.nn.conv2d(x_padded, kernel, strides, padding='VALID')
    out = tf.nn.bias_add(out, bias)

    if use_BN:
        out = tf.layers.batch_normalization(out, trainable=True, reuse=Reuse)
    if use_relu:
        out = tf.nn.relu(out)
    return out

def discriminator1(input, reuse, scope_name):
    with tf.variable_scope(scope_name, reuse=reuse):
        with tf.variable_scope('conv1'):
            weight = tf.get_variable(name='weight', shape=[3, 3, 1, 16], dtype=tf.float32,
                                     initializer=tf.random_normal_initializer(mean=0, stddev=0.1), trainable=True)
            bias = tf.get_variable(name='bias', shape=[16], dtype=tf.float32, initializer=tf.constant_initializer(0), trainable=True)
            out = conv2d_1(input, weight, bias, [1, 2, 2, 1], use_relu=True, use_BN=False, Reuse=reuse)
        with tf.variable_scope('conv2'):
            weight = tf.get_variable(name='weight', shape=[3, 3, 16, 32], dtype=tf.float32,
                                     initializer=tf.random_normal_initializer(mean=0, stddev=0.1), trainable=True)
            bias = tf.get_variable(name='bias', shape=[32], dtype=tf.float32, initializer=tf.constant_initializer(0), trainable=True)
            out = conv2d_1(out, weight, bias, [1, 2, 2, 1], use_relu=True, use_BN=True, Reuse=reuse)
        with tf.variable_scope('conv3'):
            weight = tf.get_variable(name='weight', shape=[3, 3, 32, 64], dtype=tf.float32,
                                     initializer=tf.random_normal_initializer(mean=0, stddev=0.1), trainable=True)
            bias = tf.get_variable(name='bias', shape=[64], dtype=tf.float32, initializer=tf.constant_initializer(0), trainable=True)
            out = conv2d_1(out, weight, bias, [1, 2, 2, 1], use_relu=True, use_BN=True, Reuse=reuse)
        out = tf.reshape(out, [-1, int(out.shape[1] * int(out.shape[2] * int(out.shape[3])))])
        print("out.shape  ",out.shape)
        with tf.variable_scope('disc_flatten1'):
            out = tf.layers.dense(inputs=out, units=1, activation=tf.nn.tanh, use_bias=True, trainable=True, reuse=reuse)
        # out = out / 2 + 0.5
        return out