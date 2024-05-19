import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim, structural_similarity, peak_signal_noise_ratio


def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 255.0
    psnr = 20 * np.log10(PIXEL_MAX / np.sqrt(mse))
    return psnr

def calculate_psnr_color(img1, img2):
    return peak_signal_noise_ratio(img1, img2, data_range=255)

def calculate_ssim_color(img1, img2):
    return structural_similarity(img1, img2, multichannel=True, win_size=win_size, channel_axis=-1)

# 示例用法
if __name__ == '__main__':
    img1 = cv2.imread('./result/baby.png')
    img2 = cv2.imread('./Set5/baby.png')
    target_size = (img1.shape[1], img1.shape[0])  # (width, height)
    img2_resized = cv2.resize(img2, target_size, interpolation=cv2.INTER_CUBIC)
    # 确定 win_size，确保它是奇数并且小于等于图像的最小尺寸
    min_size = min(img1.shape[0], img1.shape[1])
    win_size = min(7, min_size)  # 使用 7 或者更小的值
    if win_size % 2 == 0:
        win_size -= 1  # 确保 win_size 是奇数
    psnr_value = calculate_psnr_color(img1, img2_resized)
    ssim_value = calculate_ssim_color(img1, img2_resized)
    print(f'PSNR: {psnr_value} dB')
    print(f'SSIM: {ssim_value}')
