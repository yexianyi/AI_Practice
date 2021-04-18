import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, optimizers
# 去除警告


# 1.数据加载
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

# 2.数据的处理（归一化）
x_train, x_test = x_train/255.0, x_test/255.0
# 对于标签设置类型，设置为整数
y_train = tf.cast(y_train, dtype=tf.int32)
y_test = tf.cast(y_test, dtype=tf.int32)
print("x_train:%s, y_train:%s"%(x_train.shape, y_train.shape))
print("x_test:%s, y_test:%s"%(x_test.shape, y_test.shape))

# 3.设置batch为128
# 训练集中的x_train和y_train转换为tf中的集合
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(10000).batch(128)
# 测试集中的x_test和y_test转换为tf中的集合
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
train_dataset = train_dataset.batch(128)

# 4.打印显示
# 生成一个迭代器，作为sample， 来查看每次迭代的batch
db_iter = iter(train_dataset)
sample = next(db_iter)
print("batch:", sample[0].shape, sample[1].shape)
# 显示前9张图
plt.figure()
for i in range(9):
    plt.subplot(3,3, i+1)
    plt.imshow(x_train[i])
    plt.ylabel(y_train[i].numpy())
# plt.show()

# 5.卷积层， 全连接层
model = keras.models.Sequential([
    layers.Flatten(input_shape=(28, 28)), # 一共784个特征
    layers.Dense(512, activation="relu"),  # 卷积层降维找特征
    layers.Dense(256, activation="relu"),  # 卷积层降维找特征
    layers.Dense(128, activation="relu"),  # 卷积层降维找特征
    layers.Dense(64, activation="relu"),  # 卷积层降维找特征
    layers.Dense(32, activation="relu"),  # 卷积层降维找特征
    layers.Dense(10, activation="softmax")  # 全连接层， 数字分为10类，激活函数使用softmax
])
print(model.summary())
# 6.设置优化器
optimizer = optimizers.Adam(lr=0.001)
model.compile(optimizer = optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])



# 7. 模型的训练
model.fit(x_train, y_train, epochs=20)
model.save("mnist.h5")

