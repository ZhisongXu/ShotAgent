import cv2
import numpy as np
import os

def high_pass_filter(img, kernel_size=5):
    """
    使用高通滤波器提取高频信息
    """
    blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    high_pass = cv2.subtract(img, blurred)
    return high_pass

def high_pass_fusion(img1, img2, alpha=1.0, kernel_size=5):
    """
    提取 img2 的高频信息，并叠加到 img1 上
    :param img1: PSNR/SSIM 高的图像
    :param img2: IQA 评分高的图像
    :param alpha: 高频信息的权重
    :param kernel_size: 高斯模糊核大小，影响高频提取
    :return: 融合后的图像
    """
    high_freq = high_pass_filter(img2, kernel_size)
    blended = cv2.addWeighted(img1, 1.0, high_freq, alpha, 0)
    return np.clip(blended, 0, 255).astype(np.uint8)

def process_images_in_folder(psnr_folder, iqa_folder, output_folder, alpha=1.0, kernel_size=5):
    """
    处理整个文件夹的图像并进行高通滤波融合
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    file_list = sorted(os.listdir(psnr_folder))
    for file_name in file_list:
        psnr_path = os.path.join(psnr_folder, file_name)
        iqa_path = os.path.join(iqa_folder, file_name)
        output_path = os.path.join(output_folder, file_name)
        
        if not os.path.exists(iqa_path):
            print(f"Warning: {file_name} not found in {iqa_folder}")
            continue
        
        img1 = cv2.imread(psnr_path)
        img2 = cv2.imread(iqa_path)
        
        if img1 is None or img2 is None:
            print(f"Error reading {file_name}")
            continue
        
        # 确保两张图像尺寸一致
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
        blended_img = high_pass_fusion(img1, img2, alpha, kernel_size)
        cv2.imwrite(output_path, blended_img)
        print(f"Processed: {file_name}")

# 示例使用
psnr_folder = "/home/data1/NTIRE2025/ShortformUGCSR/submissions/PISASR_pretrained_wavelet_align_psnr/wild"  # 你的 PSNR/SSIM 高的图片文件夹
iqa_folder = "/home/data1/NTIRE2025/ShortformUGCSR/submissions/PISASR_pretrained_wavelet_align/wild"  # 你的 IQA 评分高的图片文件夹
output_folder = "/home/data1/NTIRE2025/ShortformUGCSR/submissions/PISASR_pretrained_wavelet_fusion/wild"  # 输出文件夹
process_images_in_folder(psnr_folder, iqa_folder, output_folder, alpha=1.0, kernel_size=3)
