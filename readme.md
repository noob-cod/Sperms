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
    - UNet: UNet&UNet++数据集
