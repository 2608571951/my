# # # # # # # # # # # # # # # # # # # # # # #
# 4层UNet
# 跨层连接特征相加(add)
# 模型及相关文件的后缀是 XX2_X
# # # # # # # # # # # # # # # # # # # # # # #

import tensorflow as tf
import numpy as np
from scipy.signal import convolve2d
import math
from sklearn import metrics
# import matplotlib.pylab as plt

SEED = 0
PATCH_SIZE = 256

def convolution(input_data, filters_shape, trainable, name, downsample=False, activate=True, bn=True):
    with tf.variable_scope(name):
        if downsample:
            pad_h, pad_w = (filters_shape[0] - 2) // 2 + 1, (filters_shape[1] - 2) // 2 + 1
            paddings = tf.constant([[0, 0], [pad_h, pad_h], [pad_w, pad_w], [0, 0]])
            input_data = tf.pad(input_data, paddings, 'CONSTANT')
            strides = (1, 2, 2, 1)
            padding = 'VALID'
        # output.shape = input.shape
        else:
            strides = (1, 1, 1, 1)
            padding = "SAME"
        weight = tf.get_variable(name='weight', dtype=tf.float32, trainable=True,
                                 shape=filters_shape, initializer=tf.random_normal_initializer(stddev=0.01, seed=SEED))
        # print(weight.name)
        conv = tf.nn.conv2d(input=input_data, filter=weight, strides=strides, padding=padding)

        if bn:
            conv = tf.layers.batch_normalization(conv, beta_initializer=tf.zeros_initializer(),
                                                 gamma_initializer=tf.ones_initializer(),
                                                 moving_mean_initializer=tf.zeros_initializer(),
                                                 moving_variance_initializer=tf.ones_initializer(), training=trainable)
        else:
            bias = tf.get_variable(name='bias', shape=filters_shape[-1], trainable=True,
                                   dtype=tf.float32, initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, bias)

        if activate is True:
            conv = tf.nn.relu(conv)

    return conv

def upsample(input, name, width=8, height=8, method='deconv'):
    assert method in ['resize', 'deconv']
    if method=='resize':
        with tf.variable_scope(name):
            ss = [height, width]
            ss = np.array(ss)
            output = tf.image.resize_nearest_neighbor(images=input, size=ss)
    if method=='deconv':
        num_filter = input.shape.as_list()[-1]
        output = tf.layers.conv2d_transpose(inputs=input, filters=num_filter//2, kernel_size=4, padding='same', strides=2,
                                            kernel_initializer=tf.random_normal_initializer(seed=SEED))
    return output


def UNet(input, batch_size=1, patch_size=1,input_channel=2, trainable=True):
    endpoints={}
    with tf.name_scope('conv1'):
        conv1_1 = convolution(input_data=input, filters_shape=[3, 3, input_channel, 64], trainable=trainable, name='conv1_1')
        print('conv1_1:',conv1_1.shape)
        # return conv1_1
        conv1_2 = convolution(input_data=conv1_1, filters_shape=[3, 3, 64, 64], trainable=trainable, name='conv1_2')
        print('conv1_2:', conv1_2.shape)
        # conv1_2 = tf.concat([input, conv1_2], axis=-1)
        # print('concat1_2:', conv1_2.shape)
        endpoints['C1']=conv1_2

    with tf.name_scope('pool1'):
        pool1 = tf.nn.max_pool(value=conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        # pool1 = tf.nn.avg_pool(value=conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        print('pool1:', pool1.shape)
    with tf.name_scope('conv2'):
        conv2_1 = convolution(input_data=pool1, filters_shape=[3, 3, pool1.shape[3], 128], trainable=trainable, name='conv2_1',)
        print('conv2_1:', conv2_1.shape)
        conv2_2 = convolution(input_data=conv2_1, filters_shape=[3, 3, 128, 128], trainable=trainable, name='conv2_2')
        print('conv2_2:', conv2_2.shape)
        # conv2_2 = tf.concat([pool1, conv2_2], axis=-1)
        # print('concat2_2:', conv2_2.shape)
        endpoints['C2'] = conv2_2

    with tf.name_scope('pool2'):
        pool2 = tf.nn.max_pool(value=conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        # pool2 = tf.nn.avg_pool(value=conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        print('pool2:', pool2.shape)
    with tf.name_scope('conv3'):
        conv3_1 = convolution(input_data=pool2, filters_shape=[3, 3, pool2.shape[3], 256], trainable=trainable, name='conv3_1')
        print('conv3_1:', conv3_1.shape)
        conv3_2 = convolution(input_data=conv3_1, filters_shape=[3, 3, 256, 256], trainable=trainable, name='conv3_2')
        print('conv3_2:', conv3_2.shape)
        # conv3_2 = tf.concat([pool2, conv3_2], axis=-1)
        # print('concat3_2:', conv3_2.shape)
        endpoints['C3'] = conv3_2

    with tf.name_scope('pool3'):
        pool3 = tf.nn.max_pool(value=conv3_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        # pool3 = tf.nn.avg_pool(value=conv3_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        print('pool3:', pool3.shape)
    with tf.name_scope('conv4'):
        conv4_1 = convolution(input_data=pool3, filters_shape=[3, 3, pool3.shape[3], 512], trainable=trainable, name='conv4_1')
        print('conv4_1:', conv4_1.shape)
        conv4_2 = convolution(input_data=conv4_1, filters_shape=[3, 3, 512, 512], trainable=trainable, name='conv4_2')
        print('conv4_2:', conv4_2.shape)
        # conv4_2 = tf.concat([pool3, conv4_2], axis=-1)
        # print('concat4_2:', conv4_2.shape)
        endpoints['C4'] = conv4_2
        conv = conv4_2


    for i in range(4, 1, -1):
        with tf.variable_scope('Ronghe%d' % i):
            shape1 = endpoints['C%d' % (i - 1)].shape
            # print('same_layer.shape: %s' % (endpoints['C%d' % (i - 1)].shape))
            w = shape1[1]
            h = shape1[2]
            # print('w: %d, h: %d' %(w, h))
            uplayer = upsample(conv, 'deconv%d' % (5 - i), w, h, method="deconv")
            print('uplayer%d.shape: %s' %(8-i, uplayer.shape))
            # concat = tf.concat([endpoints['C%d' % (i - 1)], uplayer], axis=-1)
            concat = tf.add(endpoints['C%d' % (i - 1)], uplayer)
            print('concat%d.shape: %s' %(8-i, concat.shape))
            dim = concat.get_shape()[-1].value
            conv = convolution(input_data=concat, filters_shape = [3, 3, dim, dim], trainable=trainable, name='conv1')
            print('C%d1.shape: %s' % (8 - i, conv.shape))
            conv = convolution(input_data=conv, filters_shape=[3, 3, dim, dim], trainable=trainable, name='conv2')
            print('C%d2.shape: %s' % (8 - i, conv.shape))
    out = convolution(input_data=conv, filters_shape=[3, 3, dim, 1], trainable=trainable, name='out', activate=False, bn=False)
    print("out.shape: ", out.shape)
    return out




# 拉普拉斯算子，计算图像梯度
def grad(img):
    kernel = tf.constant([[1/8, 1/8, 1/8], [1/8, -1, 1/8], [1/8, 1/8, 1/8]])
    kernel = tf.expand_dims(kernel, axis=-1)
    kernel = tf.expand_dims(kernel, axis=-1)
    g = tf.nn.conv2d(img, tf.cast(kernel, tf.float32), strides=[1, 1, 1, 1], padding='SAME')
    return g
def grad_Scharr(img):
    kernel_x = tf.constant([[-3, 0, +3], [-10, 0, +10], [-3, 0, +3]])
    kernel_y = tf.constant([[-3, -10, -3], [0, 0, 0], [+3, +10, +3]])
    kernel_x = tf.expand_dims(kernel_x, axis=-1)
    kernel_x = tf.expand_dims(kernel_x, axis=-1)
    kernel_y = tf.expand_dims(kernel_y, axis=-1)
    kernel_y = tf.expand_dims(kernel_y, axis=-1)
    g_x = tf.nn.conv2d(img, tf.cast(kernel_x, tf.float32), strides=[1, 1, 1, 1], padding='SAME')
    g_y = tf.nn.conv2d(img, tf.cast(kernel_y, tf.float32), strides=[1, 1, 1, 1], padding='SAME')
    g = g_x + g_y
    return g_x


# 计算MR-T2的二值图
def bi_image(img):
   b = tf.less_equal(img, 127)
   c = tf.where(condition=b, x=img, y=img-img)
   d = tf.where(condition=b, x=c-c+1, y=c)
   return d

# ssim_loss
# def matlab_style_gauss2D(size=11, sigma=1.5):
#     x_data, y_data = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
#     x_data = np.expand_dims(x_data, axis=-1)
#     x_data = np.expand_dims(x_data, axis=-1)
#
#     y_data = np.expand_dims(y_data, axis=-1)
#     y_data = np.expand_dims(y_data, axis=-1)
#
#     x = tf.constant(x_data, dtype=tf.float32)
#     y = tf.constant(y_data, dtype=tf.float32)
#
#     g = tf.exp(-((x ** 2 + y ** 2) / (2.0 ** sigma ** 2)))
#     return g / tf.reduce_sum(g)
#
# def filter2(x, kernel, padding):
#     return tf.nn.conv2d(x, kernel, strides=[1, 1, 1, 1], padding=padding)
#
# def compute_ssim(img1, img2, k1=0.01, k2=0.03, win_size=11, L=255):
#     if not img1.shape == img2.shape:
#         print("Input images must have the same dimension...")
#         raise ValueError("Input images must have the same dimension...")
#     C1 = (k1 * L) ** 2
#     C2 = (k2 * L) ** 2
#     window = matlab_style_gauss2D()
#
#     if img1.dtype == np.uint8:
#         img1 = np.double(img1)
#     if img2.dtype == np.uint8:
#         img2 = np.double(img2)
#
#     mu1 = filter2(img1, window, 'VALID')
#     mu2 = filter2(img2, window, 'VALID')
#
#     mu1_sq = mu1 * mu1
#     mu2_sq = mu2 * mu2
#     mu1_mu2 = mu1 * mu2
#
#     sigma1_sq = filter2(img1 * img1, window, 'VALID') - mu1_sq
#     sigma2_sq = filter2(img2 * img2, window, 'VALID') - mu2_sq
#     sigma12 = filter2(img1 * img2, window, 'VALID') - mu1_mu2
#
#     ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
#     # return np.mean(ssim_map)
#     return tf.reduce_mean(ssim_map)
#
# # psnr_loss
# def compute_psnr(img1, img2):
#     mse = tf.reduce_mean((img1/1.0 - img2/1.0) ** 2)
#     if mse < 1.0e-10:
#         return  100
#     return 10 * math.log10(255.0**2/mse)

# model1
# def loss_func(input1, input2, input3):
#     input1_1 = abs(input1 - input2)
#     input1_2 = abs(input1 - input3)
#     input2_1 = grad(input1) - grad(input2)
#     fro_norm_1 = tf.square(tf.norm(input1_1, axis=[1, 2], ord='fro')) # 像素
#     fro_norm_2 = tf.square(tf.norm(input1_2, axis=[1, 2], ord='fro')) # 像素
#     L1_norm = tf.reduce_sum(tf.abs(input2_1), axis=[1, 2]) # 梯度
#     loss = 2.0 * tf.reduce_mean(L1_norm) + 0.5 * tf.reduce_mean(fro_norm_1) + 2.0 * tf.reduce_mean(fro_norm_2)
#     return loss


# model2
# def loss_func(input1, input2, input3):
#     input1_1 = input1 - input2
#     input1_2 = input1 - input3
#     fro_norm_1 = tf.square(tf.norm(input1_1, axis=[1, 2], ord='fro')) # 像素
#     fro_norm_2 = tf.square(tf.norm(input1_2, axis=[1, 2], ord='fro')) # 像素
#     loss = tf.reduce_mean(fro_norm_1) + tf.reduce_mean(fro_norm_2)
#     return loss

# model3
# def loss_func(input1, input2, input3):
#     input1_1 = input1 - input2
#     input1_2 = input1 - input3
#     mse1 = tf.reduce_mean((tf.square(input1_1)))
#     mse2 = tf.reduce_mean((tf.square(input1_2)))
#     loss = mse1 + mse2
#     return loss

# model4
# def loss_func(input1, input2, input3):
#     input1_2 = input1 - input3
#     input2_1 = grad(input1) - grad(input2)
#     mse2 = tf.reduce_mean((tf.square(input1_2)))
#     mse3 = tf.reduce_mean((tf.square(input2_1)))
#     loss = mse2 + mse3
#     return loss

# model5
# def loss_func(input1, input2, input3):
#     input1_1 = input1 - input2
#     input1_2 = input1 - input3
#     input2_1 = grad(input1) - grad(input2)
#     mse1 = tf.reduce_mean(tf.square(input1_1))
#     mse2 = tf.reduce_mean(tf.square(input1_2))
#     L1_norm = tf.reduce_mean(tf.square(input2_1)) # 梯度
#     loss = 5.0 * L1_norm +  1.0 * mse1 + 2.0 * mse2
#     return loss

# # model6
def loss_func(input1, input2, input3):
    input1_1 = input1 - input2
    input1_2 = input1 - input3
    input2_1 = grad(input1) - grad(input2)
    # input2_2 = grad(input1) - grad(input3)
    mse1 = tf.reduce_mean(tf.square(input1_1))
    mse2 = tf.reduce_mean(tf.square(input1_2))
    L1_norm = tf.reduce_mean(tf.square(input2_1)) # 梯度
    # L2_norm = tf.reduce_mean(tf.square(input2_2))  # 梯度
    loss = 15.0 * L1_norm + 5.0 * mse1 + 7.0 * mse2

    return loss

# model7
# def loss_func(input1, input2, input3):
#     input1_1 = input1 - input2
#     input1_2 = input1 - input3
#     input2_1 = grad(input1) - grad(input2)
#     mse1 = tf.reduce_mean(tf.square(input1_1))
#     mse2 = tf.reduce_mean(tf.square(input1_2))
#     L1_norm = tf.reduce_mean(tf.square(input2_1)) # 梯度
#     loss = 15.0 * L1_norm +  1.0 * mse1 + 4.0 * mse2
#     return loss




# 反向传播，优化
def training(learning_rate, train_loss):
    g_solver = tf.train.AdamOptimizer(learning_rate).minimize(train_loss)
    return g_solver
