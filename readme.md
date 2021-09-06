# 基于深度学习的精子分类与计数  

## 起止日期  
	2021.09.01 - 

## 项目概述  
拟使用UNet、Faster R-CNN、YOLO-v3算法分别实现对图像中的精子进行分割/检测，并进一步实现对图片中不同种类精子数量的统计

## 目录架构
- Checkpoint
    - UNet: UNet模型训练过程中的checkpoint文件
  
- Code  
    - utils  
    	- losses  
    	    - FocalLoss.py: FocalLoss的实现  
    	    - DiceLoss.py: DiceLoss的实现  
    	- datamaker  
    	    - unet_datamaker
    	- Dataset_to_TFRecord.py: 将图片写入tfrecord文件，但目前写入7800张256*256的图片会生成10G的.tfrecord文件，需要分割。在导入模块处存在bug： "from one_hot_encode import" 在main.py中会报错，需要改为"from utils.one_hot_encode import"，而后者又会在直接运行Dataset_to_TFrecord.py时报错，需要改回第一种写法。
    	- one\_hot_encode.py: 独热编码工具。
    - UNet
    	- UNet.py: UNet源代码
    - UNetpp

    	- UNet++.py: UNet++源代码  
    - config.py: 模型配置文件  
    - main.py: 模型组装、配置、训练
  
- Outputs
    - UNet: 训练好的UNet模型的输出
  
- src_dataset  

- dataset  
    - UNet_dataset  
    	- training_set  
    	    - img: 随机截取后的训练集图像  
    	    - mask: 随机截取后的训练集mask  
    	    - src  
    	        - img: 训练集原始图像  
    	        - mask: 训练集原始mask  
    	        - mask_vis: 训练集原始mask可视化图像  
    	- test_set  
    	    - img: 随机截取后的测试集图像  
    	    - mask: 随机截取后的测试集mask  
    	    - src:
    	        - img: 测试集原始图像  
    	        - mask: 测试集原始mask  
