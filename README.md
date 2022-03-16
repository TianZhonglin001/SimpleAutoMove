# SimpleAutoMove
基于tensorflow2.x卷积神经网络的寻迹小车实现
## 说明
* 该项目用于智能小车寻迹
* 使用了高度集成的api，对于想了解底层实现的朋友并不合适，适合新手
* 包含数据集创建，神经网络搭建，模型训练，测试
## 使用方法
- 运行PictureGetter.py,对基础数据集进行旋转
- 运行move.py,对数据集进行平移
- 运行adjust.py,对数据集进行调整
- 运行DataSetLable.py,制作数据集
- 运行TiNet.py,加载数据集，训练模型，保存模型
- 运行test.py,进行检测
## 教程
详细教程麻烦移步https://blog.csdn.net/qq_42500340/article/details/123496002?spm=1001.2014.3001.5501
