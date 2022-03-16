# 开发作者   ：Tian.Z.L
# 开发时间   ：2022/3/14  20:23 
# 文件名称   ：move.PY
# 开发工具   ：PyCharm
# 图像平移 (Translation transform)
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

index_1 = 111  # 接着旋转之后直行的索引
index_2 = 61  # 接着旋转之后左转的索引
index_3 = 101  # 接着旋转之后右转的索引
path = "C:/Users/TianZhonglin/Desktop/labPatentProject/TensorFlowLearnning/AutoDriver/after"  # 文件夹目录
files = os.listdir(path)  # 得到文件夹下的所有文件名称
for file in files:
    img = cv2.imread("./after/" + str(file))  # 读取彩色图像(BGR)
    own = str(file).split('_')[0]
    center = (640 // 2, 480 // 2)
    # index = 4
    if own == '1':
        index = index_1
        index_1 += 10
    elif own == '2':
        index = index_2
        index_2 += 10
    else:
        index = index_3
        index_3 += 10
    rows, cols, ch = img.shape
    for i in range(5):
        dx, dy = (i + 1) * 10, 0  # 向右偏移量,
        MAT = np.float32([[1, 0, dx], [0, 1, dy]])  # 构造平移变换矩阵
        # dst = cv2.warpAffine(img, MAT, (cols, rows))  # 默认为黑色填充
        dst = cv2.warpAffine(img, MAT, (cols, rows), borderValue=(255, 255, 255))  # 设置白色填充
        ret, binary = cv2.threshold(dst, 200, 255, cv2.THRESH_BINARY)
        cv2.imwrite('./after/' + own + '_' + str(index).zfill(5) + '.jpg', binary)
        index += 1
    for i in range(5):
        dx, dy = (i + 1) * -10, 0  # 向左偏移量
        MAT = np.float32([[1, 0, dx], [0, 1, dy]])  # 构造平移变换矩阵
        # dst = cv2.warpAffine(img, MAT, (cols, rows))  # 默认为黑色填充
        dst = cv2.warpAffine(img, MAT, (cols, rows), borderValue=(255, 255, 255))  # 设置白色填充
        ret, binary = cv2.threshold(dst, 200, 255, cv2.THRESH_BINARY)
        cv2.imwrite('./after/' + own + '_' + str(index).zfill(5) + '.jpg', binary)
        index += 1
