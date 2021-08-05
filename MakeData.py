import os
import h5py
import numpy as np
from scipy.misc import imread
PATCH_SIZE = 256

# 数据生成
def start1():
    dir_path1 = '/home/root3203/lishanshan/第一篇论文相关/TheWholeBrainAtlas_After/train/MR-T1'
    dir_path2 = '/home/root3203/lishanshan/第一篇论文相关/TheWholeBrainAtlas_After/train/MR-T2'

    # dir_path1 = '/home/root3203/lishanshan/CT_MR_dataset/train/CT'
    # dir_path2 = '/home/root3203/lishanshan/CT_MR_dataset/train/MR'

    dir_name1 = os.path.basename(dir_path1)
    print(dir_name1)
    dir_name2 = os.path.basename(dir_path2)
    print(dir_name2)


    # 获取MR-T1和MR-T2图像的路径
    data1, data2 = prepare_data(dir_path1, dir_path2)

    # 将MR-T1和MR-T2的图片数据变成h5文件数据
    savepath1 = input_setup(data1, dir_name1)
    savepath2 = input_setup(data2, dir_name2)
    print(savepath1)
    print(savepath2)


# 获取图像路径
def prepare_data(dataset1, dataset2):
    """
    Args:
      dataset: choose train dataset or test dataset
    """
    data1 = []
    data2 = []
    filenames1 = os.listdir(dataset1)
    filenames1.sort(key=lambda x: int(x[:-4]))
    print(filenames1)

    filenames2 = os.listdir(dataset2)
    filenames2.sort(key=lambda x: int(x[:-4]))
    print(filenames2)

    for item in filenames1:
        data1.append(os.path.join(dataset1, item))
    for item in filenames2:
        data2.append(os.path.join(dataset2, item))

    return data1, data2



def make_data(data, label, data_dir):
    savepath = os.path.join('.', os.path.join('dataset_After', data_dir, 'train.h5'))
    if not os.path.exists(os.path.join('.', os.path.join('dataset_After', data_dir))):
        os.makedirs(os.path.join('.', os.path.join('dataset_After', data_dir)))
    with h5py.File(savepath, 'w') as hf:
        hf.create_dataset('data', data=data)
        hf.create_dataset('label', data=label)
    return savepath


# 读取图片文件，制作子图片并把它们保存成h5文件的形式
def input_setup(data, data_dir):
    """
    Read image files and make their sub-images and saved them as a h5 file format.
    """
    image_size = PATCH_SIZE
    label_size = PATCH_SIZE
    stride = 1

    sub_input_sequence = []
    sub_label_sequence = []

    for i in range(len(data)):

        input_ = imread(name=data[i],flatten=True) / 255.0   # flatten=True：以灰度形式读取图片
        print('input_.shape: ', input_.shape)
        label_ = input_

        if len(input_.shape) == 3:
            h, w, _ = input_.shape
        else:
            h, w = input_.shape
        # 按stride步长采样小patch
        for x in range(0, h - image_size + 1, stride):
            for y in range(0, w - image_size + 1, stride):
                sub_input = input_[x:x + image_size, y:y + image_size]
                sub_label = label_[x:x + label_size, y:y + label_size]

                # Make channel value
                sub_input = sub_input.reshape([image_size, image_size, 1])   # h*w*c
                sub_label = sub_label.reshape([label_size, label_size, 1])

                sub_input_sequence.append(sub_input)
                sub_label_sequence.append(sub_label)
    # Make list to numpy array. With this transform
    arrdata = np.asarray(sub_input_sequence)
    arrlabel = np.asarray(sub_label_sequence)
    # print(arrdata.shape)
    savepath = make_data(arrdata, arrlabel, data_dir)
    return savepath


# 读取h5文件
def read_data(path):
    with h5py.File(path, 'r') as hf:
        data = np.array(hf.get('data'))
        label = np.array(hf.get('label'))
        return data, label

if __name__ == '__main__':
    start1()