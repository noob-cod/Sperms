"""
Load Image data as '.tfrecord' type.

Reference: https://tensorflow.google.cn/tutorials/load_data/tfrecord?hl=en

Functions:
1) _byte_feature(): Encode string or byte data.
2) _float_feature(): Encode float or double byte data.
3) _int64_feature(): Encode int, uint, bool or enum data.

4) img_example(): Define tfrecord protocol, which determines how to write data into tfrecord file.

5) tfrecord_write(): Write images and corresponding one hot coded masks into one tfrecord file.
6) tfrecord_read(): Sparse images and masks from tfrecord file, and return a trainable dataset.

Notice: Numpy array should be written into tfrecord by using _byte_feature() after converted to string type.
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# from one_hot_encode import mask_to_onehot
from utils.one_hot_encode import mask_to_onehot

from tensorflow.keras.losses import binary_crossentropy

"""
tf.train.BytesList: string, byte
tf.train.FloatList: float(float32), double(float64)
tf.train.Int64List: bool, enum, int32, uint32, int64, uint64
"""


def _byte_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double"""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint"""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def img_example(image_string: [str], mask_string: [str]):
    """Define protocol"""
    img_shape = tf.io.decode_jpeg(image_string).shape
    # mask_shape = tf.image.decode_png(mask_string).shape

    # one hot encode
    mask_tensor = tf.io.decode_png(mask_string)
    mask_array = mask_tensor.numpy()  # Change tensor into numpy array
    # mask_array = np.expand_dims(mask_array, axis=2)  # (H, W) to (H, W, 1)
    palette = [[0], [1], [2], [3], [4]]  # Set Color
    mask = mask_to_onehot(mask_array, palette)  # one hot encoding

    # Protocol of encoding
    feature = {
        'img_height': _int64_feature(img_shape[0]),
        'img_width': _int64_feature(img_shape[1]),
        'img_depth': _int64_feature(img_shape[2]),
        'img_raw': _byte_feature(image_string),

        # 'mask_height': _int64_feature(mask_shape[0]),
        # 'mask_width': _int64_feature(mask_shape[1]),
        # 'mask_raw': _byte_feature(mask_string)

        # one_hot
        'mask_height': _int64_feature(mask.shape[0]),
        'mask_width': _int64_feature(mask.shape[1]),
        'mask_depth': _int64_feature(mask.shape[2]),
        'mask': _byte_feature(mask.astype(np.float32).tostring())
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))


def tfrecord_write(record_file: [str], img_path: [str], mask_path: [str]):
    """
    Write tfrecord file

    :param record_file: Name of tfrecord. For example, 'filename.tfrecord'
    :param img_path: Path of data that is going to be written in tfrecord file
    :param mask_path: Path of data that is going to be written in tfrecord file
    :return: None
    """
    img_name_list = os.listdir(img_path)

    with tf.io.TFRecordWriter(record_file) as writer:
        for name in img_name_list:
            image_string = open(os.path.join(img_path, name), 'rb').read()
            mask_string = open(os.path.join(mask_path, name.split('.')[0] + '.png'), 'rb').read()
            tf_example = img_example(image_string, mask_string)
            writer.write(tf_example.SerializeToString())


def tfrecord_read(record_file: [str]):
    """
    Read tfrecord file

    :param record_file: Name of .tfrecord. For example, 'filename.tfrecord'
    :return: dataset. Instance of 'tf.data.Dataset'
    """
    raw_data_set = tf.data.TFRecordDataset(record_file)

    # Protocol of decoding
    image_feature_description = {
        'img_height': tf.io.FixedLenFeature([], tf.int64),
        'img_width': tf.io.FixedLenFeature([], tf.int64),
        'img_depth': tf.io.FixedLenFeature([], tf.int64),
        'img_raw': tf.io.FixedLenFeature([], tf.string),

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
        feature_dict['img_raw'] = tf.io.decode_jpeg(feature_dict['img_raw']) / 255  # Decode raw img
        # feature_dict['mask_raw'] = tf.io.decode_png(feature_dict['mask_raw'])

        # one_hot
        feature_dict['mask'] = tf.io.decode_raw(feature_dict['mask'], tf.float32)  # Numpy to Tensor
        shape = [feature_dict['mask_height'], feature_dict['mask_width'], feature_dict['mask_depth']]
        feature_dict['mask'] = tf.reshape(feature_dict['mask'], shape=shape)

        return feature_dict['img_raw'], feature_dict['mask']

    dataset = raw_data_set.map(_parse_image_function)

    return dataset


if __name__ == '__main__':
    record = '/home/bmp/ZC/Sperms/dataset/UNet_dataset/training_set/training_set.tfrecord'
    img_path = '/home/bmp/ZC/Sperms/dataset/UNet_dataset/training_set/img'
    mask_path = '/home/bmp/ZC/Sperms/dataset/UNet_dataset/training_set/mask'
    tfrecord_write(record, img_path, mask_path)
    # dataset = tfrecord_read(record)

    # i = 0
    # for image, mask in dataset.take(3):
    #     i += 1
    #     plt.title('img')
    #     plt.imshow(image.numpy())
    #     plt.savefig('img_%d' % i)
    #     plt.show()
    #
    #     for i in range(mask.shape[2]):
    #         plt.title('mask')
    #         plt.imshow(mask[:, :, i].numpy().squeeze())
    #         plt.savefig('mask_%d' % i)
    #         plt.show()
    #         print(mask.shape)
