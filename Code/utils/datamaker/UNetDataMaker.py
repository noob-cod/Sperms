"""
@Date: 2021/9/4 下午2:25
@Author: Chen Zhang
@Brief: 为UNet和UNet++制作数据集

由于输入图像的尺寸较大，因此借鉴了遥感图像数据增强的方法，（思路参考于知乎用户计算机视觉life）。

对图像进行随机切割，随机生成x,y坐标，然后抠出该坐标下256*256的小图，
之后进行数据增强操作：
    1、img和mask分别旋转90、180、270度；
    2、img和mask沿y轴做镜像操作；
    3、对原图做模糊操作；
    4、对原图做光照调整操作；
    5、位原图增加噪声。

制作好的数据集保存在独立的文件夹中。

可能用到的tf方法：
    裁剪：
        tf.image.crop_to_bounding_box()
    旋转：
        tf.image.rot90()
    镜像（翻转）：
        tf.image.flip_left_right()
        tf.image.flip_up_down()
    亮度：
        tf.image.random_brightness
    噪声：
        。。。

"""
import os
import random
import tensorflow as tf


class UNetDataMaker:

    THRESH_HOLD = 100  # 目标特征能够传播到最后一层的最小面积，值越大，裁剪出的图像中包含的带标签的区域越大

    def __init__(self,
                 img_file_path,
                 msk_file_path,
                 patch_save_path,
                 patch_height=256,
                 patch_width=256,
                 patch_nums=100
                 ):
        """
        :param img_file_path: 原图片存放路径，所有图片的格式必须一致
        :param msk_file_path: Mask存放路径，为png格式
        :param patch_height: 截取图像块的高
        :param patch_width: 截取图像块的宽
        :param patch_nums: 单张原图中截取的有效patch数量
        :param patch_save_path: 产生的块保存路径，默认不保存
        """
        self.img_file_path = img_file_path
        self.msk_file_path = msk_file_path
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.patch_nums = patch_nums
        self.patch_save_path = patch_save_path

        # 解析原图像的名称
        # 获得图片文件的名称列表
        img_filenames = os.listdir(img_file_path)

        # 解析文件格式
        self.img_fmt = img_filenames[0].split('.')[-1]

        # 解析文件名
        self.img_names = []
        for name in img_filenames:
            self.img_names.append(name.split('.')[:-1][0])

    def make_dataset(self,
                     rotate=False,
                     flip=False,
                     change_bright=False
                     ):
        print('Making dataset ...')
        self.__processing(rotate=rotate, flip=flip, change_bright=change_bright)
        print('Successfully!')
        print()

    def __processing(self,
                     rotate=False,
                     flip=False,
                     change_bright=False
                     ):
        """
        随机裁剪图像，并对裁剪到的图像进行增强
        """
        for img_name in self.img_names:
            img_path = os.path.join(self.img_file_path, img_name + '.' + self.img_fmt)
            msk_path = os.path.join(self.msk_file_path, img_name + '.png')

            # 解码图像和mask为Tensor
            img_value = tf.io.read_file(img_path)
            msk_value = tf.io.read_file(msk_path)
            img = tf.io.decode_image(img_value)
            msk = tf.io.decode_image(msk_value)

            height, width = img.shape[0], img.shape[1]

            # 随机截self.patch_nums数量的图像
            count = self.patch_nums
            while count:
                h = random.randint(0, height-self.patch_height-1)
                w = random.randint(0, width-self.patch_width-1)
                msk_patch = tf.image.crop_to_bounding_box(msk, h, w, self.patch_height, self.patch_width)
                if tf.math.reduce_sum(msk_patch) > UNetDataMaker.THRESH_HOLD:
                    # 特征能够传播到最后一层，因此对该图像块进行增强操作
                    # tf.io.write_file('test2.jpg', tf.io.encode_jpeg(tf.image.rot90(img, 1)))
                    # 原图
                    img_patch = tf.image.crop_to_bounding_box(img, h, w, self.patch_height, self.patch_width)
                    # 存储目录结构：doc/img/xxx.jpg & doc/mask/xxx.png
                    tf.io.write_file(
                        os.path.join(self.patch_save_path, 'img', img_name+'_{}-{}'.format(h, w)+'.jpg'),
                        tf.io.encode_jpeg(img_patch)
                    )
                    tf.io.write_file(
                        os.path.join(self.patch_save_path, 'mask', img_name + '_{}-{}'.format(h, w) + '.png'),
                        tf.io.encode_png(msk_patch)
                    )

                    # 旋转
                    if rotate:
                        for k in range(1, 4):
                            tf.io.write_file(
                                os.path.join(self.patch_save_path, 'img',
                                             img_name + '_{}-{}_rot{}'.format(h, w, 90 * k) + '.jpg'),
                                tf.io.encode_jpeg(tf.image.rot90(img_patch, k))
                            )
                            tf.io.write_file(
                                os.path.join(self.patch_save_path, 'mask',
                                             img_name + '_{}-{}_rot{}'.format(h, w, 90 * k) + '.png'),
                                tf.io.encode_png(tf.image.rot90(msk_patch, k))
                            )

                    # 镜像
                    if flip:
                        tf.io.write_file(
                            os.path.join(self.patch_save_path, 'img', img_name + '_{}-{}_flip-ud'.format(h, w) + '.jpg'),
                            tf.io.encode_jpeg(tf.image.flip_up_down(img_patch))
                        )
                        tf.io.write_file(
                            os.path.join(self.patch_save_path, 'mask', img_name + '_{}-{}_flip-ud'.format(h, w) + '.png'),
                            tf.io.encode_png(tf.image.flip_up_down(msk_patch))
                        )

                        tf.io.write_file(
                            os.path.join(self.patch_save_path, 'img', img_name + '_{}-{}_flip-lr'.format(h, w) + '.jpg'),
                            tf.io.encode_jpeg(tf.image.flip_left_right(img_patch))
                        )
                        tf.io.write_file(
                            os.path.join(self.patch_save_path, 'mask', img_name + '_{}-{}_flip-lr'.format(h, w) + '.png'),
                            tf.io.encode_png(tf.image.flip_left_right(msk_patch))
                        )

                    # 亮度
                    if change_bright:
                        tf.io.write_file(
                            os.path.join(self.patch_save_path, 'img', img_name + '_{}-{}_bright'.format(h, w) + '.jpg'),
                            tf.io.encode_jpeg(tf.image.random_brightness(img_patch, 10))
                        )
                        tf.io.write_file(
                            os.path.join(self.patch_save_path, 'mask', img_name + '_{}-{}_bright'.format(h, w) + '.png'),
                            tf.io.encode_png(msk_patch)
                        )

                    count -= 1


if __name__ == '__main__':
    # 制作训练集图像
    src_img = '/home/bmp/ZC/Sperms/dataset/UNet_dataset/training_set/src/img'
    src_msk = '/home/bmp/ZC/Sperms/dataset/UNet_dataset/training_set/src/mask'
    save_path = '/home/bmp/ZC/Sperms/dataset/UNet_dataset/training_set'
    result = UNetDataMaker(src_img, src_msk, save_path)
    result.make_dataset(rotate=True, flip=True)
