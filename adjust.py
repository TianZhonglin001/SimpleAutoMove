# 开发作者   ：Tian.Z.L
# 开发时间   ：2022/3/14  20:54 
# 文件名称   ：adjust.PY
# 开发工具   ：PyCharm
import cv2
import os

path = "C:/Users/TianZhonglin/Desktop/labPatentProject/TensorFlowLearnning/AutoDriver/after"  # 文件夹目录
files = os.listdir(path)  # 得到文件夹下的所有文件名称
for file in files:
    img = cv2.imread('after/' + str(file))
    img = cv2.resize(img, (32, 24))
    cv2.imwrite('after/' + str(file), img)
