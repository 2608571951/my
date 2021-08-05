import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
from scipy.misc import imread, imsave
import numpy as np
import math
import cv2
from scipy.signal import convolve2d
from sklearn import metrics
from PIL import ImageEnhance
import net as generator

BATCH_SIZE = 1
PATCH_SIZE = 256
SAVE_PATH = './model01_1/'
TEST_PATH1 = '/home/root3203/lishanshan/第一篇论文相关/TheWholeBrainAtlas_After/test/MR-T1/'
TEST_PATH2 = '/home/root3203/lishanshan/第一篇论文相关/TheWholeBrainAtlas_After/test/MR-T2/'

# TEST_PATH1 = '/home/root3203/lishanshan/CT_MR_dataset/train/CT'
# TEST_PATH2 = '/home/root3203/lishanshan/CT_MR_dataset/train/MR'

OUTPUT_PATH = './medical_fused_image01_1/'



# # # # # # # # # # # # # # # # # # # #
# 计算psnr，数值越大表示失真越小
# # # # # # # # # # )# # # # # # # # # #
def compute_psnr(img1, img2):
    mse = np.mean((img1/1.0 - img2/1.0) ** 2)
    if mse < 1.0e-10:
        return  100
    return 10 * math.log10(255.0**2/mse)




# # # # # # # # # # # # # # # # # # # #
# 计算ssim, 取值范围[-1, 1]
# 越接近1,代表相似度越高，融合质量越好
# # # # # # # # # # # # # # # # # # # #
def matlab_style_gauss2D(shape=(3, 3), sigma=0.5):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m : m + 1, -n : n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def filter2(x, kernel, mode='same'):
    return convolve2d(x, np.rot90(kernel, 2), mode=mode)

def compute_ssim(img1, img2, k1=0.01, k2=0.03, win_size=11, L=255):
    if not img1.shape == img2.shape:
        print("Input images must have the same dimension...")
        raise ValueError("Input images must have the same dimension...")
    # M, N = img1.shape
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2
    window = matlab_style_gauss2D(shape=(win_size, win_size), sigma=1.5)
    window = window / np.sum(np.sum(window))

    if img1.dtype == np.uint8:
        img1 = np.double(img1)
    if img2.dtype == np.uint8:
        img2 = np.double(img2)

    mu1 = filter2(img1, window, 'valid')
    mu2 = filter2(img2, window, 'valid')

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = filter2(img1 * img1, window, 'valid') - mu1_sq
    sigma2_sq = filter2(img2 * img2, window, 'valid') - mu2_sq
    sigma12 = filter2(img1 * img2, window, 'valid') - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return np.mean(np.mean(ssim_map))



# # # # # # # # # # # # # # # # # # # #
# 计算熵，熵越高表示融合图像的信息量越丰富，
# 质量越好
# # # # # # # # # # # # # # # # # # # #
# def compute_entropy(img):
#     tmp = []
#     for i in range(256):
#         tmp.append(0)
#     k = 0
#     res = 0
#     img = np.array(img)
#     for i in range(len(img)):
#         for j in range(len(img[i])):
#             val = img[i][j]
#             tmp[int(val)] = float(tmp[int(val)] + 1)
#             k = float(k + 1)
#     for i in range(len(tmp)):
#         tmp[i] = float(tmp[i] / k)
#     for i in range(len(tmp)):
#         if(tmp[i] == 0):
#             res = res
#         else:
#             res = float(res - tmp[i] * (math.log(tmp[i]) / math.log(2.0)))
#     return res





# # # # # # # # # # # # # # # # # # # #
# 计算标准差，值越大表示灰度级分布越分散，
# 图像携带信息就越多，融合图像质量越好。
# 均值反应亮度信息，均值适中，质量越好。
# # # # # # # # # # # # # # # # # # # #
def compute_MEAN_STD(img):
    (mean, stddv) = cv2.meanStdDev(img)
    return mean, stddv




# # # # # # # # # # # # # # # # # # # #
# 计算标准化互信息NMI, 取值范围[0, 1]
# 越接近1,代表相似度越高，融合质量越好
# # # # # # # # # # # # # # # # # # # #
def compute_NMI(img1, img2):
    nmi = metrics.normalized_mutual_info_score(img1, img2)
    return nmi




# # # # # # # # # # # # # # # # # # # #
# 测试
# # # # # # # # # # # # # # # # # # # #
def make():
    if not os.path.exists(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)

    # psnr_list = {}
    # ssim_list = {}

    # doc = open(FILE_NAME, 'w')
    for i in range(18):
        # 准备输入数据
        num = 3 * (i + 14)
        path1 = TEST_PATH1 + str(num) + '.png'
        path2 = TEST_PATH2 + str(num) + '.png'
        data1 = imread(name=path1, flatten=True) / 255.0
        data2 = imread(name=path2, flatten=True) / 255.0
        # data1 = imread(name=path1, flatten=True)
        # data2 = imread(name=path2, flatten=True)
        data1 = data1.reshape([BATCH_SIZE, PATCH_SIZE, PATCH_SIZE, 1])
        data2 = data2.reshape([BATCH_SIZE, PATCH_SIZE, PATCH_SIZE, 1])
        # print(data1.shape)
        # print(data2.shape)
        with tf.Graph().as_default():
            # 定义输入占位符
            source_data1 = tf.placeholder(tf.float32, shape=(BATCH_SIZE, PATCH_SIZE, PATCH_SIZE, 1), name='source_data1')
            source_data2 = tf.placeholder(tf.float32, shape=(BATCH_SIZE, PATCH_SIZE, PATCH_SIZE, 1), name='source_data2')

            # 处理输入数据
            input = tf.concat([source_data1, source_data2], axis=-1)

            # 定义操作
            # tf.reset_default_graph()
            logit = generator.UNet(input, BATCH_SIZE, PATCH_SIZE)

            saver = tf.train.Saver()

            with tf.Session() as sess:
                print('Reading checkpoint...')
                ckpt = tf.train.get_checkpoint_state(SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('.')[0]
                    # print(global_step)
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    print('Loading success, global_step is %s' % (global_step))
                else:
                    print('No checkpoint file found')

                fusion_img = sess.run(logit, feed_dict={source_data1 : data1, source_data2 : data2})
                print(fusion_img.shape)
                fusion_img = fusion_img.reshape([PATCH_SIZE, PATCH_SIZE])

                # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
                #增强图像对比度

                # enh_con = ImageEnhance.Contrast(fusion_img)
                # contrast = 1.5
                # fusion_img = enh_con.enhance(contrast)

                # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

                imsave(OUTPUT_PATH + str(num) + '.png', fusion_img)
                # img1 = data1.reshape([PATCH_SIZE, PATCH_SIZE])
                # img2 = data2.reshape([PATCH_SIZE, PATCH_SIZE])
                #
                # psnr_fusion_img1 = compute_psnr(fusion_img, img1)
                # psnr_fusion_img2 = compute_psnr(fusion_img, img2)
                # ssim_fusion_img1 = compute_ssim(fusion_img, img1)
                # ssim_fusion_img2 = compute_ssim(fusion_img, img2)
                # fusion_mean, fusion_stddv = compute_MEAN_STD(fusion_img)
                #
                # fusion_img_1D = fusion_img.reshape([PATCH_SIZE * PATCH_SIZE])
                # img1_1D = img1.reshape([PATCH_SIZE * PATCH_SIZE])
                # img2_1D = img2.reshape([PATCH_SIZE * PATCH_SIZE])
                #
                # nmi_fusion_img1 = compute_NMI(fusion_img_1D, img1_1D)
                # nmi_fusion_img2 = compute_NMI(fusion_img_1D, img2_1D)
                # print('\t\t\t\tPSNR\t\t\t\tSSIM\t\t\t\tNMI\t\t\t\t\tMEAN\t\t\t\tSTDDV\nfusion_img1\t\t{0}\t{1}\t{2}\t{3}\t{4}\n'
                #       'fusion_img2\t\t{5}\t{6}\t{7}'.format(psnr_fusion_img1, ssim_fusion_img1,nmi_fusion_img1,
                #       fusion_mean, fusion_stddv, psnr_fusion_img2, ssim_fusion_img2, nmi_fusion_img2), file=doc)
                # print(
                #     '\t\t\t\tPSNR\t\t\t\tSSIM\t\t\t\tNMI\t\t\t\tMEAN\t\t\t\tSTDDV\nfusion_img1\t\t{0}\t{1}\t{2}\t{3}\t{4}\n'
                #     'fusion_img2\t\t{5}\t{6}\t{7}'.format(psnr_fusion_img1, ssim_fusion_img1, nmi_fusion_img1,
                #                                           fusion_mean, fusion_stddv, psnr_fusion_img2, ssim_fusion_img2,
                #                                           nmi_fusion_img2))
    print('test finish')

if __name__ == '__main__':
    make()