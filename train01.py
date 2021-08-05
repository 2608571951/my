import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
import numpy as np
import h5py
import net as generator
import discriminator01 as discriminator

# 数据集存储路径
datapath1 = './dataset_After/MR-T1/train.h5'
datapath2 = './dataset_After/MR-T2/train.h5'

# 常量
BATCH_SIZE = 1
PATCH_SIZE = 256
LEARNING_RATE = 0.0001
EPOCHES = 30
N_CLASSES = 1
eps = 1e-8
SAVE_PATH = './model01_1/'  # 模型保存路径


# 读取训练数据
def read_data(path):
    with h5py.File(path, 'r') as hf:
        data = np.array(hf.get('data'))
        label = np.array(hf.get('label'))
        return data, label


def train():
    if not os.path.exists(SAVE_PATH):
        os.mkdir(SAVE_PATH)

    # 读取数据
    data1, label1 = read_data(datapath1)
    data2, label2 = read_data(datapath2)
    num_images = data1.shape[0]
    mod = num_images % BATCH_SIZE
    num_batches = int(num_images // BATCH_SIZE)
    print('Train images number: %d, Batches number: %d.\n' % (num_images, num_batches))
    if mod > 0:
        print('Train set has been trimmed %d samples\n' % mod)
        data1 = data1[:-mod]
        data2 = data2[:-mod]

    # 定义输入占位符
    source_data1 = tf.placeholder(tf.float32, shape=(BATCH_SIZE, PATCH_SIZE, PATCH_SIZE, 1), name='source_data1')
    source_data2 = tf.placeholder(tf.float32, shape=(BATCH_SIZE, PATCH_SIZE, PATCH_SIZE, 1), name='source_data2')

    # 处理输入数据
    input = tf.concat([source_data1, source_data2], axis=-1)

    # 定义操作
    train_logits_G = generator.UNet(input, BATCH_SIZE, PATCH_SIZE)

    with tf.variable_scope('dis1') as scope:
        train_logits_D1_real = discriminator.discriminator1(input=source_data1, reuse=False, scope_name='Disc1')
        train_logits_D1_fake = discriminator.discriminator1(input=train_logits_G, reuse=True, scope_name='Disc1')

    with tf.variable_scope('dis2') as scope:
        train_logits_D2_real = discriminator.discriminator1(input=source_data2, reuse=False, scope_name='Disc2')
        train_logits_D2_fake = discriminator.discriminator1(input=train_logits_G, reuse=True, scope_name='Disc2')


    # 生成器的loss
    G_loss_GAN_D1 = tf.reduce_mean(tf.square(
        train_logits_D1_fake - tf.random_uniform(shape=[BATCH_SIZE, 1], minval=1., maxval=1., dtype=tf.float32)))
    G_loss_GAN_D2 = tf.reduce_mean(tf.square(
        train_logits_D2_fake - tf.random_uniform(shape=[BATCH_SIZE, 1], minval=1., maxval=1., dtype=tf.float32)))
    train_loss_G_norm = generator.loss_func(train_logits_G, source_data1, source_data2)
    G_loss = G_loss_GAN_D1 + G_loss_GAN_D2 + 0.5 * train_loss_G_norm




    # 鉴别器的loss
    D1_loss_real = tf.reduce_mean(tf.square(
        train_logits_D1_real - tf.random_uniform(shape=[BATCH_SIZE, 1], minval=1., maxval=1., dtype=tf.float32)))
    D1_loss_fake = tf.reduce_mean(tf.square(
        train_logits_D1_fake - tf.random_uniform(shape=[BATCH_SIZE, 1], minval=0., maxval=0., dtype=tf.float32)))
    D1_loss = D1_loss_fake + D1_loss_real

    D2_loss_real = tf.reduce_mean(tf.square(
        train_logits_D2_real - tf.random_uniform(shape=[BATCH_SIZE, 1], minval=1., maxval=1., dtype=tf.float32)))
    D2_loss_fake = tf.reduce_mean(tf.square(
        train_logits_D2_fake - tf.random_uniform(shape=[BATCH_SIZE, 1], minval=0., maxval=0., dtype=tf.float32)))
    D2_loss = D2_loss_fake + D2_loss_real

    # 反向传播
    G_loss_op = generator.training(LEARNING_RATE, G_loss)
    D1_loss_op = generator.training(LEARNING_RATE, D1_loss)
    D2_loss_op = generator.training(LEARNING_RATE, D2_loss)

    # tf.summary.scalar('train_loss_G_norm', train_loss_G_norm)
    tf.summary.scalar('G_loss_GAN_D1', G_loss_GAN_D1)
    tf.summary.scalar('G_loss_GAN_D2', G_loss_GAN_D2)
    tf.summary.scalar('G_loss', G_loss)
    tf.summary.scalar('D1_loss', D1_loss)
    tf.summary.scalar('D2_loss', D2_loss)

    # log汇总记录
    summery_op = tf.summary.merge_all()

    step = 0
    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(SAVE_PATH, sess.graph)
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        for epoche in range(EPOCHES):
            for batch in range(num_batches):
                step += 1
                data1_batch = data1[int(batch * BATCH_SIZE):int(batch * BATCH_SIZE + BATCH_SIZE), :, :, :]
                data2_batch = data2[int(batch * BATCH_SIZE):int(batch * BATCH_SIZE + BATCH_SIZE), :, :, :]
                FEED_DICT = {source_data1: data1_batch, source_data2: data2_batch}

                it_g = 0
                it_d1 = 0
                it_d2 = 0

                if batch % 2 == 0:  # discriminator
                    sess.run(D1_loss_op, feed_dict=FEED_DICT)
                    sess.run(D2_loss_op, feed_dict=FEED_DICT)
                    it_d1 += 1
                    it_d2 += 1
                else:  # generator
                    sess.run(G_loss_op, feed_dict=FEED_DICT)
                    it_g += 1
                g_loss, d1_loss, d2_loss = sess.run([G_loss, D1_loss, D2_loss], feed_dict=FEED_DICT)
                # print('1_g_loss:', g_loss)

                if batch % 2 == 0:  # discriminator
                    while d1_loss >= 1.0 and it_d1 < 20:
                        sess.run(D1_loss_op, feed_dict=FEED_DICT)
                        d1_loss = sess.run(D1_loss, feed_dict=FEED_DICT)
                        it_d1 += 1
                    while d2_loss >= 1.0 and it_d2 < 20:
                        sess.run(D2_loss_op, feed_dict=FEED_DICT)
                        d2_loss = sess.run(D2_loss, feed_dict=FEED_DICT)
                        it_d2 += 1
                else:  # generator
                    while (d1_loss < 1.0) and it_g < 20:
                        sess.run([G_loss_op, D1_loss_op], feed_dict=FEED_DICT)
                        g_loss, d1_loss = sess.run([G_loss, D1_loss], feed_dict=FEED_DICT)
                        # print('2_g_loss:',g_loss)
                        it_g += 1
                    while (d2_loss < 1.0) and it_g < 20:
                        sess.run([G_loss_op, D2_loss_op], feed_dict=FEED_DICT)
                        g_loss, d2_loss = sess.run([G_loss, D2_loss], feed_dict=FEED_DICT)
                        # print('2_g_loss:',g_loss)
                        it_g += 1
                    while (g_loss > 10) and it_g < 20:
                        sess.run(G_loss_op, feed_dict=FEED_DICT)
                        g_loss = sess.run(G_loss, feed_dict=FEED_DICT)
                        # print('3_g_loss:',g_loss)
                        it_g += 1
                print('epoch: {0}; step: {1}; g_loss: {2}; d1_loss: {3}; d2_loss: {4};'.format(epoche, step, g_loss, d1_loss, d2_loss))
                summery_str = sess.run(summery_op, feed_dict=FEED_DICT)
                train_writer.add_summary(summery_str, step)

                if step % 100 == 0:
                # if g_loss < 0.1 and d1_loss == 1.0 and d2_loss == 1.0:
                    saver.save(sess, SAVE_PATH + str(step) + '.ckpt')
    print('train finish')


if __name__ == '__main__':
    train()
