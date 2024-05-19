import cv2
import numpy as np
import matplotlib.pyplot as plt

def compute_disparity_map(img_left, img_right):
    # 将图像转换为灰度图
    gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    # 创建一个StereoBM对象
    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)

    # 计算视差图
    disparity = stereo.compute(gray_left, gray_right)

    # 归一化视差图以便显示
    disparity = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    disparity = np.uint8(disparity)

    return disparity

# 加载左图和右图

if __name__ == '__main__':
    img_left = cv2.imread("./pic/1.JPG")
    img_right = cv2.imread('./pic/2.JPG')

    # 计算视差图
    disparity_map = compute_disparity_map(img_left, img_right)

    # 显示左图、右图和视差图
    plt.figure(figsize=(10, 7))

    plt.subplot(1, 3, 1)
    plt.title('Left Image')
    plt.imshow(cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB))

    plt.subplot(1, 3, 2)
    plt.title('Right Image')
    plt.imshow(cv2.cvtColor(img_right, cv2.COLOR_BGR2RGB))

    plt.subplot(1, 3, 3)
    plt.title('Disparity Map')
    plt.imshow(disparity_map, cmap='gray')

    plt.show()
