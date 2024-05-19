# 图像超分辨率重建：SRResNet算法

**介绍：**

SRResNet是一种用于图像超分辨率任务的深度学习模型。SRResNet的核心概念是通过残差连接（Residual Connection）和卷积神经网络（Convolutional Neural Networks，CNN）来实现图像超分辨率。当我们对图像进行超分辨率重建时，我们一般有两个目标：一是增强图像质量使图像的细节更加清晰，二是放大图像尺寸以增加图像的像素数量。对于前者，SRResNet使用了基于残差学习的深层网络结构来完成低分辨率到高分辨率的映射，对于后者则使用了子像素卷积来将低分辨率特征图上采样到高分辨率。

![image-20240519212702403](C:\Users\14249\AppData\Roaming\Typora\typora-user-images\image-20240519212702403.png)

​                                                                      模型结构

**数据训练集部分**：

在这里使用的是Urban100数据集来进行训练

![image-20240519212921869](C:\Users\14249\AppData\Roaming\Typora\typora-user-images\image-20240519212921869.png)

在这里使用了经典的Set5图片，其结果为

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201120192120947.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05pa2tpRWx3aW4=,size_16,color_FFFFFF,t_70#pic_center)

**结果**;

![image-20240519213100381](C:\Users\14249\AppData\Roaming\Typora\typora-user-images\image-20240519213100381.png)