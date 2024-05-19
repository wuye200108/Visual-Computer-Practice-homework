import os

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import Dataset
from model import SRResNet
from PIL import Image


transform = transforms.Compose([transforms.RandomCrop(96),
                            transforms.ToTensor()])
path = './Urban100/original/'

class PreprocessDataset(Dataset):
    """预处理数据集类"""

    def __init__(self,imgPath = path,transforms = transform, ex = 10):
        """初始化预处理数据集类"""
        self.transforms = transform

        for _, _, files in os.walk(imgPath):
            # ex变量是用于扩充数据集的，在这里默认的是扩充十倍
            self.imgs = [imgPath + file for file in files] * ex

        np.random.shuffle(self.imgs)  # 随机打乱

    def __len__(self):
        """获取数据长度"""
        return len(self.imgs)

    def __getitem__(self, index):
        """获取数据"""
        tempImg = self.imgs[index]
        tempImg = Image.open(tempImg)

        sourceImg = self.transforms(tempImg)  # 对原始图像进行处理
        cropImg = torch.nn.MaxPool2d(4)(sourceImg)
        return cropImg, sourceImg





save_path = 'model'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if __name__ == '__main__':
    net = SRResNet()
    net.to(device)


    BATCH = 32
    processDataset = PreprocessDataset()
    trainData = DataLoader(processDataset,batch_size=BATCH)

    optimizer = optim.Adam(net.parameters(),lr=0.001)  #初始化迭代器
    lossF = nn.MSELoss().to(device)   #初始化损失函数

    EPOCHS = 30
    history = []
    for epoch in range(EPOCHS):
        net.train()
        runningLoss = 0.0

        for i, (cropImg, sourceImg) in tqdm(enumerate(trainData, 1)):
            cropImg, sourceImg = cropImg.to(device), sourceImg.to(device)

            # 清空梯度流
            optimizer.zero_grad()

            # 进行训练
            outputs = net(cropImg)
            loss = lossF(outputs, sourceImg)
            loss.backward()  # 反向传播
            optimizer.step()

            runningLoss += loss.item()

        averageLoss = runningLoss / (i + 1)
        history += [averageLoss]
        print('[INFO] Epoch %d loss: %.3f' % (epoch + 1, averageLoss))

        runningLoss = 0.0

    torch.save(net.state_dict(), f"./model/SRR1.ckpt")

    print('[INFO] Finished Training \nWuhu~')


