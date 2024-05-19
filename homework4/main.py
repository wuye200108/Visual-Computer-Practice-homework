import cv2
import numpy as np
import matplotlib.pyplot as plt

def find_homography(img1_path, img2_path):
    # 加载图像
    img1 = cv2.imread(img1_path, cv2.IMREAD_COLOR)
    img2 = cv2.imread(img2_path, cv2.IMREAD_COLOR)

    # 转换为灰度图
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 初始化ORB特征检测器
    orb = cv2.ORB_create()

    # 检测并计算特征
    keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)

    # 使用BFMatcher进行特征匹配
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)

    # 按照距离排序
    matches = sorted(matches, key=lambda x: x.distance)

    # 提取匹配点
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # 计算单应性矩阵
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    return H, matches, keypoints1, keypoints2

def warp_image(img_path, H, output_size):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    warped_img = cv2.warpPerspective(img, H, output_size)
    return warped_img

# 路径到您的图像

if __name__ == '__main__':
    img1_path = './pic/1.JPG'
    img2_path = './pic/2.JPG'

    # 计算单应性矩阵
    H, matches, keypoints1, keypoints2 = find_homography(img1_path, img2_path)

    # 输出单应性矩阵
    print('Homography Matrix:')
    print(H)

    # 变换图像
    img1 = cv2.imread(img1_path, cv2.IMREAD_COLOR)
    output_size = (img1.shape[1], img1.shape[0])
    warped_img = warp_image(img2_path, H, output_size)

    # 显示原始图像和变换后的图像
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    plt.title('Image 1')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(warped_img, cv2.COLOR_BGR2RGB))
    plt.title('Warped Image 2')

    plt.show()
