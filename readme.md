# 基于深度学习的精子分类与计数  

## 起止日期  
	2021.09.01 - 

## 项目概述  
拟使用UNet、UNET++、TransUNet等算法分别实现对图像中的精子进行分割/检测，并进一步实现对图片中不同种类精子数量的统计

## 目录架构
- Checkpoint
    - UNet: UNet模型训练过程中的checkpoint文件
  
- Code  
    - utils  
    	- losses  
    	    - FocalLoss.py: FocalLoss的实现  
    	    - DiceLoss.py: DiceLoss的实现  
    	    - UNetPPLoss.py: Binary Crossentropy loss + Dice loss的实现，用于UNet++深度监督  
    	- datamaker  
    	    - UNetDataMaker.py： 对原始大图像进行随机裁剪，将生成的图像保存至指定文件夹，供后期制作tfrecord  
    	    - Dataset_to_TFRecord.py： 仅仅使用其中的tfrecord_checkout()函数检验tfrecord中的图片是否符合要求  
    	    - one_hot_encode.py：将mask按照类别数量编码成对应数量的二值图像  
    	    - TFRecorder.py：将指定路径下的图像写到.tfrecord文件中，同时向main.py提供解析.tfrecord文件的接口  
    	    - rgb_to_uint8.py: 将mask由rgb转化为uint8  
    	- img_show.py：用于显示Outputs中的预测图像  
    - UNet  
    	- UNet.py: UNet源代码  
    - UNetpp  
    	- UNetpp.py: UNet++源代码，基于循环的思路生成嵌套结构（有bug）  
    	- UNetPP.py: UNet++源代码（借鉴自他人）  
    - config.py: 模型配置文件（需要重新调整内容的结构，在调参时很难找到参数位置）  
    - main.py: 模型组装、配置、训练  
    - predict.py: 预测并保存预测结果  
  
- Outputs  
    - UNet: 训练好的UNet模型的输出
    	- 以主要超参数命名的上层文件夹
    	    - 以具体模型名称命名的下层文件夹：存放具体的预测输出
  
- src_dataset  

- dataset  
    - UNet_dataset(1):老数据集  
    - UNet_dataset:最新获取的数据集  
    	- training_set  
    	    - img: 随机截取后的训练集图像  
    	    - mask: 随机截取后的训练集mask  
    	    - src  
    	        - img: 训练集原始图像  
    	        - mask: 训练集原始mask  
    	        - mask_vis: 训练集原始mask可视化图像  
    	    - tfrecord: 存放训练用的.tfrecord文件，也是Code/utils/datamaker/TFRecorder.py的输出文件夹
    	- test_set  
    	    - img: 随机截取后的测试集图像  
    	    - mask: 随机截取后的测试集mask  
    	    - src:
    	        - img: 测试集原始图像  
    	        - mask: 测试集原始mask  
