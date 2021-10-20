"""
@Date: 2021/9/7 上午10:36
@Author: Chen Zhang
@Brief:  TFRecorder，提供tfrecord的写入和读出功能
"""
import os
import math
import numpy as np
import tensorflow as tf

from typing import List
from Code.utils.datamaker.one_hot_encode import mask_to_onehot
from Code.config import cfg


class TFRecorder:

    """Data ==> '.tfrecord' ==> trainable Dataset"""

    def __init__(self, img_path=None, mask_path=None, save_path=None):
        """
        :param img_path: 图像所在的文件夹
        :param mask_path: Mask所在文件夹
        :param save_path: tfrecord文件保存路径
        """
        self.img_path = img_path
        self.mask_path = mask_path
        self.save_path = save_path

    @staticmethod
    def __byte_feature(value):
        """将string / byte类输入编码为bytes_list"""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def __float_feature(value):
        """将float / double类输入编码为float_list"""
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    @staticmethod
    def __int64_feature(value):
        """将bool / enum / int / uint类数据编码为int64_list"""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def __img_example(self, img_string, mask_string):
        """Define protocol"""
        img_shape = tf.io.decode_jpeg(img_string).shape

        # one hot encode
        mask_tensor = tf.io.decode_png(mask_string)
        mask_array = mask_tensor.numpy()  # Change tensor into numpy array
        # mask_array = np.expand_dims(mask_array, axis=2)  # (H, W) to (H, W, 1)
        palette = [[0], [1], [2], [3], [4]]  # Set Color
        mask = mask_to_onehot(mask_array, palette)  # one hot encoding

        # Protocol of encoding
        feature = {
            # 'img_height': self.__int64_feature(img_shape[0]),
            # 'img_width': self.__int64_feature(img_shape[1]),
            # 'img_depth': self.__int64_feature(img_shape[2]),
            'img': self.__byte_feature(img_string),

            # 'mask_height': _int64_feature(mask_shape[0]),
            # 'mask_width': _int64_feature(mask_shape[1]),
            # 'mask_raw': _byte_feature(mask_string)

            # one_hot
            'mask_height': self.__int64_feature(mask.shape[0]),
            'mask_width': self.__int64_feature(mask.shape[1]),
            'mask_depth': self.__int64_feature(mask.shape[2]),
            'mask': self.__byte_feature(mask.astype(np.float32).tostring())
        }

        return tf.train.Example(features=tf.train.Features(feature=feature))

    def write_to_tfrecord(self, tfrecord_name: str, file_nums: int = 1):
        """将图片数据写入tfrecord，可选择是否分多个文件写入"""
        img_names = os.listdir(self.img_path)

        # 不需要切片
        if file_nums == 1:
            with tf.io.TFRecordWriter(os.path.join(self.save_path, tfrecord_name+'.tfrecord')) as writer:
                for name in img_names:
                    # 以二进制方式读取图片数据
                    img_string = open(os.path.join(self.img_path, name), 'rb').read()
                    mask_string = open(os.path.join(self.mask_path, name.split('.')[0]+'.png'), 'rb').read()
                    tf_example = self.__img_example(img_string, mask_string)
                    writer.write(tf_example.SerializeToString())
        # 需要切片
        elif file_nums > 1:
            total = len(img_names)
            file_capacity = math.ceil(total / file_nums)
            for i in range(file_nums):
                end = (i + 1) * file_capacity
                if end < total:
                    names = img_names[i*file_capacity: end]
                else:
                    names = img_names[i*file_capacity:]
                with tf.io.TFRecordWriter(
                        os.path.join(self.save_path, tfrecord_name+'_{}.tfrecord'.format(i))
                ) as writer:
                    for name in names:
                        img_string = open(os.path.join(self.img_path, name), 'rb').read()
                        mask_string = open(os.path.join(self.mask_path, name.split('.')[0] + '.png'), 'rb').read()
                        tf_example = self.__img_example(img_string, mask_string)
                        writer.write(tf_example.SerializeToString())

    @staticmethod
    def read_from_tfrecord(tfrecord_files: List[str], train_set_nums: int = None):
        """从.tfrecord文件中读数据，可选择是否划分验证集"""
        raw_dataset = tf.data.TFRecordDataset(tfrecord_files)

        raw_dataset.shuffle(cfg.DATA.SHUFFLE_BUFFER_SIZE)

        # Protocol of decoding
        image_feature_description = {
            # 'img_height': tf.io.FixedLenFeature([], tf.int64),
            # 'img_width': tf.io.FixedLenFeature([], tf.int64),
            # 'img_depth': tf.io.FixedLenFeature([], tf.int64),
            'img': tf.io.FixedLenFeature([], tf.string),

            # 'mask_height': tf.io.FixedLenFeature([], tf.int64),
            # 'mask_width': tf.io.FixedLenFeature([], tf.int64),
            # 'mask_raw': tf.io.FixedLenFeature([], tf.string),

            # one_hot
            'mask_height': tf.io.FixedLenFeature([], tf.int64),
            'mask_width': tf.io.FixedLenFeature([], tf.int64),
            'mask_depth': tf.io.FixedLenFeature([], tf.int64),
            'mask': tf.io.FixedLenFeature([], tf.string)
        }

        def _parse_image_function(example_proto):
            """TFRecord Decoder"""
            # Parse the input tf.train.Example proto using the dictionary above
            feature_dict = tf.io.parse_single_example(example_proto, image_feature_description)
            feature_dict['img'] = tf.io.decode_jpeg(feature_dict['img']) / 255  # Decode raw img
            # feature_dict['mask_raw'] = tf.io.decode_png(feature_dict['mask_raw'])

            # one_hot
            feature_dict['mask'] = tf.io.decode_raw(feature_dict['mask'], tf.float32)  # Numpy to Tensor
            shape = [feature_dict['mask_height'], feature_dict['mask_width'], feature_dict['mask_depth']]
            feature_dict['mask'] = tf.reshape(feature_dict['mask'], shape=shape)
            # print(feature_dict['mask'].shape)
            # feature_dict['mask'] = tf.io.decode_png(feature_dict['mask']) / 1
            # feature_dict['mask'] = tf.cast(feature_dict['mask'], dtype=tf.float32)

            return feature_dict['img'], feature_dict['mask']

        # 划分验证集
        if train_set_nums:
            if not isinstance(train_set_nums, int):
                raise ValueError
            train_set = raw_dataset.take(train_set_nums)
            valid_set = raw_dataset.skip(train_set_nums)
            return train_set.map(_parse_image_function), valid_set.map(_parse_image_function)
        # 不划分验证集
        else:
            return raw_dataset.map(_parse_image_function), None


class ResolveError(Exception):
    """无法解析的协议格式"""
    pass


class UnsupportedType(Exception):
    """不支持的编码类型，需要在源码中添加"""
    pass


if __name__ == '__main__':
    save_path = '/home/bmp/ZC/Sperms/dataset/UNet_dataset/training_set/tfrecord'
    img_path = '/home/bmp/ZC/Sperms/dataset/UNet_dataset/training_set/img'
    mask_path = '/home/bmp/ZC/Sperms/dataset/UNet_dataset/training_set/mask'
    TFRecorder(img_path, mask_path, save_path).write_to_tfrecord('train_set', 4)
