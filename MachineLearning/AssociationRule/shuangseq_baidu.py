import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense



# 读取数据
data = pd.read_csv('shuangseqiu.csv')

# 数据预处理
# 假设我们已经将红色球和蓝色球转换为某种形式的数值特征
# 这里我们简单地将红色球和蓝色球作为分类变量处理
# 将红色球和蓝色球转换为独热编码
red_balls = data.iloc[:, 2:8].values.flatten()
red_balls_one_hot = pd.get_dummies(red_balls).values
blue_ball = data.iloc[:, 14].values
blue_ball_one_hot = pd.get_dummies(blue_ball).values

# 合并特征和标签
X = np.hstack((data.iloc[:, 15:35].values, red_balls_one_hot, blue_ball_one_hot))
y = np.zeros((len(data), 1))  # 假设我们预测的是某种形式的概率或分类标签

# 归一化特征
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 构建模型
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_scaled.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # 假设我们进行的是二分类任务，预测某个号码出现的概率

# 编译模型
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 训练模型
model.fit(X_scaled, y, epochs=10, batch_size=32, validation_split=0.2)


# 生成号码（这里只是一个简单的示例，实际生成号码的逻辑需要更复杂）
def generate_numbers(model, scaler, data):
    # 假设我们根据模型的输出概率选择号码
    probabilities = model.predict(scaler.transform(data.iloc[:, 15:35].values))
    # 根据概率选择红色球和蓝色球
    red_balls = np.random.choice(33, 6, replace=False) + 1  # 1到33之间的6个不重复数字
    blue_ball = np.random.choice(16, 1) + 1  # 1到16之间的1个数字
    return red_balls, blue_ball


# 生成两组号码
for _ in range(2):
    red, blue = generate_numbers(model, scaler, data)
    print(f"红色球: {np.sort(red)}, 蓝色球: {blue[0]}")