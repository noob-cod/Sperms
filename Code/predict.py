"""
@Date: 2021/9/12 上午10:43
@Author: Chen Zhang
@Brief: 对测试图像进行推理，输出推理结果
"""
import os
import cv2
import csv
import codecs
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from UNet.UNet import UNet
from UNetpp.UNetPP import UNetPP
from TransUNet.TransUNet import TransUnet

from config import cfg


PATTERNS = {
    'segmentation': {
        'UNet': UNet,
        'UNet++': UNetPP,
        'TransUNet': TransUnet
    },

    'detection': {

    }
}


class Predictor:

    @classmethod
    def _load_single_image(cls, img_path, normalization_factor=255, resize_size=None):
        """
        从指定路径读取图像并返回归一化后的结果

        :param img_path: str，图像路径
        :param normalization_factor: int，归一化系数

        :return: 归一化后的图像
        """
        img = tf.io.read_file(img_path)
        res = tf.io.decode_image(img) / normalization_factor

        if resize_size is not None:
            if isinstance(resize_size, int):
                res = tf.image.resize_with_pad(res, resize_size, resize_size)
            elif isinstance(resize_size, list) and len(resize_size) == 2:
                res = tf.image.resize_with_pad(res, resize_size[0], resize_size[1])
            else:
                raise ValueError('Unintelligible target size!')

        return res

    @classmethod
    def _predict_single_img(cls, model, img_path, resize_size=None):
        """
        对给定的单张图像进行预测
        :param model:
        :param img_path:
        :return:
        """
        # 保存文件名
        img_name_components = os.path.basename(img_path).split('.')[:-1]
        img_name = ''.join(img_name_components)

        # 加载图像
        img = Predictor._load_single_image(img_path, resize_size=resize_size)

        # 图像预处理，添加Batch size维度
        img = tf.expand_dims(img, axis=0)

        # 图像预测
        return model.predict(img), img_name

    @classmethod
    def _count_sperms(cls, label, threshold=128, rectangle_size=(5, 5), draw=False):
        """
        :param label: 二维矩阵，单通道灰度图
        :param threshold: 整型值，二值化阈值
        :param rectangle_size: 二元组，形态学操作矩形结构元的尺寸
        :param draw: 布尔值，是否画出二值化和形态学处理后的图像
        :return: 整型值，含背景在内的连通域数量
        """
        label = label.numpy()  # opencv无法处理Tensor类型数据，必须使用转为numpy数组
        label = label.astype(np.uint8)  # 转化为cv2.connectedComponents能够处理的类型

        # 二值化处理
        ret, thresh = cv2.threshold(label, threshold, 255, cv2.THRESH_BINARY)

        # 形态学开操作
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, rectangle_size)  # 矩形结构元
        dst = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)  # 开操作，去噪
        if draw:
            plt.subplot(121)
            plt.imshow(thresh)
            plt.title('Binary')
            plt.subplot(122)
            plt.imshow(dst)
            plt.title('Morphology-Open')
            plt.show()

        # 连通域统计
        num, mask = cv2.connectedComponents(dst)  # cv2自带
        return num, mask

    @classmethod
    def _draw_single_predict_result(cls, predict_result, img_name, save_path, fusion_method='avg'):
        """
        将预测结果绘制并保存到指定位置。
        :param predict_result: 5d-Tensor，预测结果
        :param img_name: 原图的名称
        :param save_path: 保存路径
        :param fusion_method: 设置同类结果的融合方式，仅在网络为UNet++时生效。可在类内直接定义新方法，
                              以接收predict_result列表，输出融合后的单个predict_result张量。
        :return:
        """
        fuse = {
            'avg': Predictor._avg_fusion,
            'max': Predictor._max_fusion
        }

        if not isinstance(predict_result, list):
            predict_result = [predict_result]  # 此处转为列表是为了统一UNet++输出和其他输出的处理方式

        n = len(predict_result)  # 统计预测结果的数量，若>1则为

        # 存放UNet++多个输出的BG结果
        bg = []
        ai = []
        ar = []
        neglect = []
        others = []

        name = img_name
        for i in range(n):
            if n != 1:
                name = img_name + '_out{}'.format(i+1)
            # 图像维度压缩
            result = tf.squeeze(predict_result[i], [0])

            # 图像还原为uint8格式
            result = tf.round(result * 255)
            result = tf.cast(result, dtype=tf.uint8)

            # 分离并保存不同类型的预测结果
            pred_bg = result[:, :, 0]  # 背景
            if n != 1:
                bg.append(pred_bg)
            else:
                bg_num, _ = Predictor._count_sperms(cv2.bitwise_not(pred_bg))
            pred_bg = tf.expand_dims(pred_bg, axis=-1)  # 添加单个通道，存储为灰度图
            pred_bg = tf.io.encode_png(pred_bg)
            tf.io.write_file(os.path.join(save_path, name + '_BG.png'), pred_bg)

            pred_ai = result[:, :, 1]  # 顶体未发生反应
            if n != 1:
                ai.append(pred_ai)
            else:
                ai_num, _ = Predictor._count_sperms(pred_ai)
            pred_ai = tf.expand_dims(pred_ai, axis=-1)
            pred_ai = tf.io.encode_png(pred_ai)
            tf.io.write_file(os.path.join(save_path, name + '_AI.png'), pred_ai)

            pred_ar = result[:, :, 2]  # 顶体发生反应
            if n != 1:
                ar.append(pred_ar)
            else:
                ar_num, _ = Predictor._count_sperms(pred_ar)
            pred_ar = tf.expand_dims(pred_ar, axis=-1)
            pred_ar = tf.io.encode_png(pred_ar)
            tf.io.write_file(os.path.join(save_path, name + '_AR.png'), pred_ar)

            pred_neglect = result[:, :, 3]  # 顶体异常
            if n != 1:
                neglect.append(pred_neglect)
            else:
                neglect_num, _ = Predictor._count_sperms(pred_neglect)
            pred_neglect = tf.expand_dims(pred_neglect, axis=-1)
            pred_neglect = tf.io.encode_png(pred_neglect)
            tf.io.write_file(os.path.join(save_path, name + '_Neglect.png'), pred_neglect)

            pred_others = result[:, :, 4]  # 其他
            if n != 1:
                others.append(pred_others)
            else:
                others_num, _ = Predictor._count_sperms(pred_others)
            pred_others = tf.expand_dims(pred_others, axis=-1)
            pred_others = tf.io.encode_png(pred_others)
            tf.io.write_file(os.path.join(save_path, name + '_Others.png'), pred_others)

        # 若网络为UNet++，则对结果进行融合
        if n != 1:
            final_bg = fuse[fusion_method](bg)
            bg_num, _ = Predictor._count_sperms(final_bg)
            final_bg = tf.cast(final_bg, tf.uint8)
            final_bg = tf.expand_dims(final_bg, axis=-1)
            final_bg = tf.io.encode_png(final_bg)
            tf.io.write_file(os.path.join(save_path, img_name + '_BG_Final.png'), final_bg)

            final_ai = fuse[fusion_method](ai)
            ai_num, _ = Predictor._count_sperms(final_ai)
            final_ai = tf.cast(final_ai, tf.uint8)
            final_ai = tf.expand_dims(final_ai, axis=-1)
            final_ai = tf.io.encode_png(final_ai)
            tf.io.write_file(os.path.join(save_path, img_name + '_AI_Final.png'), final_ai)

            final_ar = fuse[fusion_method](ar)
            ar_num, _ = Predictor._count_sperms(final_ar)
            final_ar = tf.cast(final_ar, tf.uint8)
            final_ar = tf.expand_dims(final_ar, axis=-1)
            final_ar = tf.io.encode_png(final_ar)
            tf.io.write_file(os.path.join(save_path, img_name + '_AR_Final.png'), final_ar)

            final_neglect = fuse[fusion_method](neglect)
            neglect_num, _ = Predictor._count_sperms(final_neglect)
            final_neglect = tf.cast(final_neglect, tf.uint8)
            final_neglect = tf.expand_dims(final_neglect, axis=-1)
            final_neglect = tf.io.encode_png(final_neglect)
            tf.io.write_file(os.path.join(save_path, img_name + '_Neglect_Final.png'), final_neglect)

            final_others = fuse[fusion_method](others)
            others_num, _ = Predictor._count_sperms(final_others)
            final_others = tf.cast(final_others, tf.uint8)
            final_others = tf.expand_dims(final_others, axis=-1)
            final_others = tf.io.encode_png(final_others)
            tf.io.write_file(os.path.join(save_path, img_name + '_Others_Final.png'), final_others)

        # 打印精子统计结果
        info = [ai_num, ar_num, neglect_num]  # (AI, AR, 无法分辨)

        return info

    @classmethod
    def predict(cls, model, dir_path, save_path):
        """使用指定模型预测"""
        img_name_list = os.listdir(dir_path)
        record = [('图片名称', 'AI数量', 'AR数量', '无法分辨的数量')]
        for img_name in img_name_list:
            result, name = cls._predict_single_img(model, os.path.join(dir_path, img_name), resize_size=[896, 1024])
            info = cls._draw_single_predict_result(result, name, save_path)
            info.insert(0, img_name)
            record.append(tuple(info))
        Predictor._write_in_csv(record, save_path)

    @classmethod
    def _avg_fusion(cls, tensor_list):
        """将UNet++的四个输出按加权均值的方式融合"""
        n = len(tensor_list)
        weights = [0.1, 0.2, 0.3, 0.4]
        res = None
        for i in range(n):
            if res is None:
                res = tf.multiply(tf.cast(tensor_list[i], tf.float32), weights[i])
            else:
                res = tf.add(res, tf.multiply(tf.cast(tensor_list[i], tf.float32), weights[i]))
        res = tf.cast(res, tf.uint8)
        return res

    @classmethod
    def _max_fusion(cls, tensor_list):
        """效果不如_avg_fusion"""
        n = len(tensor_list)
        res = None
        for i in range(n):
            if res is None:
                res = tensor_list[i]
            else:
                res = tf.maximum(res, tensor_list[i])
        return res

    @staticmethod
    def _write_in_csv(data, save_path):
        """保存各个图片中精子数量的统计结果到csv中"""
        f = codecs.open(os.path.join(save_path, 'statistical_result.csv'), 'w', 'gbk')  # utf-8需要转gbk
        writer = csv.writer(f)
        for i in data:
            writer.writerow(i)
        f.close()


if __name__ == '__main__':
    print('Constructing model ...')
    my_model = PATTERNS[cfg.MODEL.PATTERN][cfg.MODEL.TYPE](input_shape=cfg.TEST.INPUT_SHAPE)
    model = my_model.get_model()
    print('Done!')
    print()

    checkpoint_dir = 'Filters-16_BatchSize-16_Depth-5_Epoch-150_Loss-bcedice_Optimizer-Adam_InitLR-0.01_Time-2021-10-26-11-55'
    model_name = 'Model-epoch_116.h5'

    print('Loading weights ...')
    model.load_weights('/home/bmp/ZC/Sperms/Checkpoint/' + cfg.MODEL.TYPE + '/' +
                       checkpoint_dir + '/'
                       'models/' +
                       model_name)
    print('Done!')
    print()

    first_path = os.path.join(cfg.TEST.SAVE_PATH, checkpoint_dir)  # 基于超参数建立上层文件夹
    second_path = os.path.join(first_path, model_name)  # 基于具体模型建立实际输出文件夹

    Predictor.predict(model, cfg.DATA.TEST_IMG, second_path)
