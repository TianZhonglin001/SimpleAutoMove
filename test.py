# 开发作者   ：Tian.Z.L
# 开发时间   ：2022/3/15  10:00 
# 文件名称   ：test.PY
# 开发工具   ：PyCharm
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os
import numpy as np

# 网络加载
network = keras.models.load_model('AutoDrive.h5')
network.summary()

label = ['直行', '左转', '右转']
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 训练图片的路径
train_dir = 'C:\\Users\\TianZhonglin\\Desktop\\labPatentProject\\TensorFlowLearnning\\AutoDriver\\after'
AUTOTUNE = tf.data.experimental.AUTOTUNE


# 获取图片，存放到对应的列表中，同时贴上标签，存放到label列表中
def get_files(file_dir):
    # 存放图片类别和标签的列表：第0类
    list_0 = []
    label_0 = []
    # 存放图片类别和标签的列表：第1类
    list_1 = []
    label_1 = []
    # 存放图片类别和标签的列表：第2类
    list_2 = []
    label_2 = []

    for file in os.listdir(file_dir):
        # print(file)
        # 拼接出图片文件路径
        image_file_path = os.path.join(file_dir, file)
        for image_name in os.listdir(image_file_path):
            # print('image_name',image_name)
            # 图片的完整路径
            image_name_path = os.path.join(image_file_path, image_name)
            # print('image_name_path',image_name_path)
            # 将图片存放入对应的列表
            if image_file_path[-1:] == '0':
                list_0.append(image_name_path)
                label_0.append(0)
            elif image_file_path[-1:] == '1':
                list_1.append(image_name_path)
                label_1.append(1)
            else:
                list_2.append(image_name_path)
                label_2.append(2)

    # 合并数据
    image_list = np.hstack((list_0, list_1, list_2))
    label_list = np.hstack((label_0, label_1, label_2))
    # 利用shuffle打乱数据
    temp = np.array([image_list, label_list])
    temp = temp.transpose()  # 转置
    np.random.shuffle(temp)

    # 将所有的image和label转换成list
    image_list = list(temp[:, 0])
    image_list = [i for i in image_list]
    label_list = list(temp[:, 1])
    label_list = [int(float(i)) for i in label_list]
    # print(image_list)
    # print(label_list)
    return image_list, label_list


def get_tensor(image_list, label_list):
    ims = []
    for image in image_list:
        # 读取路径下的图片
        x = tf.io.read_file(image)
        # 将路径映射为照片,3通道
        x = tf.image.decode_jpeg(x, channels=1)
        # 修改图像大小
        x = tf.image.resize(x, [32, 24])
        # 将图像压入列表中
        ims.append(x)
    # 将列表转换成tensor类型
    img = tf.convert_to_tensor(ims)
    y = tf.convert_to_tensor(label_list)
    return img, y


def preprocess(x, y):
    # 归一化
    x = tf.cast(x, dtype=tf.float32) / 255.0
    y = tf.cast(y, dtype=tf.int32)
    return x, y

    # 训练图片与标签


image_list, label_list = get_files(train_dir)
x_test, y_test = get_tensor(image_list, label_list)
print('--------------------------------------------------------')
# 载入训练数据集
db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db_test = db_test.map(preprocess).batch(2970)
db_test = next(iter(db_test))
print(db_test[0].shape)
db_test = db_test[0]

# 显示前25张图片
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.imshow(db_test[i], cmap='gray')
plt.show()
# 改变维度
# testImage = tf.reshape(db_train, (10000, 28, 28, 1))
# 结果预测
result = network.predict(db_test)[0:25]
pred = tf.argmax(result, axis=1)
pred_list = []
for item in pred:
    pred_list.append(label[item.numpy()])
print(pred_list)
