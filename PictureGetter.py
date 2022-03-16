# 开发作者   ：Tian.Z.L
# 开发时间   ：2022/3/14  14:48 
# 文件名称   ：PictureGetter.PY
# 开发工具   ：PyCharm
import cv2
import os

index_1 = 1  # 直行的图片索引
index_2 = 1  # 左转图片的索引
index_3 = 1  # 右转图片的索引
path = "C:/Users/TianZhonglin/Desktop/labPatentProject/TensorFlowLearnning/AutoDriver/DataSet"  # 文件夹目录
files = os.listdir(path)  # 得到文件夹下的所有文件名称
for file in files:
    img = cv2.imread("./DataSet/" + str(file))  # 读取彩色图像(BGR)
    # 切割文件名称，例如1_1.png为直行，最前面的1为该图片的类别
    own = str(file).split('_')[0]
    center = (640 // 2, 480 // 2)
    if own == '1':
        index = index_1
        index_1 += 10
    elif own == '2':
        index = index_2
        index_2 += 10
    else:
        index = index_3
        index_3 += 10
    for i in range(5):
        M = cv2.getRotationMatrix2D(center, -i, 1.2)
        rotated = cv2.warpAffine(img, M, (640, 480), borderValue=(255, 255, 255))
        ret, binary = cv2.threshold(rotated, 200, 255, cv2.THRESH_BINARY)
        cv2.imwrite('./after/' + own + '_' + str(index).zfill(5) + '.jpg', binary)
        index += 1
    for i in range(5):
        M = cv2.getRotationMatrix2D(center, i, 1.2)
        rotated = cv2.warpAffine(img, M, (640, 480), borderValue=(255, 255, 255))
        ret, binary = cv2.threshold(rotated, 200, 255, cv2.THRESH_BINARY)
        cv2.imwrite('./after/' + own + '_' + str(index).zfill(5) + '.jpg', binary)
        index += 1
print('success')
# cv2.imwrite("./rotated.jpg", rotated)
