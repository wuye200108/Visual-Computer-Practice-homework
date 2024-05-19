
import torch
from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F



from PIL import Image

from model import SRResNet


def imshow(path, outputPath):
    """展示结果"""
    preTransform = transforms.Compose([transforms.ToTensor()])
    img = Image.open(path)
    img = preTransform(img).unsqueeze(0)

    # 使用cpu就行
    net = SRResNet()
    net.load_state_dict(torch.load(f'./model/SRR1.ckpt'))
    net.cpu()
    source = net(img)[0, :, :, :]
    source = source.cpu().detach().numpy()  # 转为numpy
    source = source.transpose((1, 2, 0))  # 切换形状
    source = np.clip(source, 0, 1)  # 修正图片

    img = Image.fromarray(np.uint8(source * 255))
    img.save(outputPath)  # 将数组保存为图片

if __name__ == '__main__':
    path = "./Set5/baby.png"
    outputPath = "./result/baby.png"
    imshow(path,outputPath)