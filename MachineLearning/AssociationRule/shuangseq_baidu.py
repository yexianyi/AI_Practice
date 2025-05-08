import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# 假设数据文件名为 'ssq_history.csv'
data = pd.read_csv('shuangseqiu.csv')

# 提取特征和目标
# 特征选择：可以使用一些统计特征，如和值、平均值、奇偶个数等
features = data[['R1', 'R2', 'R3', 'R4', 'R5', 'R6',
                 '和值', '平均值', '尾数和值', '奇号个数', '偶号个数',
                 '奇偶偏差', '大号个数', '小号个数', '大小偏差', '尾号组数',
                 'AC值', '连号个数', '连号组数', '首尾差', '最大间距', '同位相同',
                 '重号个数', '斜号个数']].values

# 检查输入数据有效性
print("输入特征范围:", features.min(), features.max())
print("NaN值检查:", np.isnan(features).any())
print("Inf值检查:", np.isinf(features).any())

# 目标：预测下一期的红球和蓝球号码
# 注意：由于红球和蓝球是分类问题，这里我们使用独热编码
# 红球范围1-33，蓝球范围1-16
red_balls = data[['R1', 'R2', 'R3', 'R4', 'R5', 'R6']].values
blue_ball = data['B1'].values

# 独热编码红球和蓝球
red_balls_one_hot = np.zeros((len(data), 33))
for i in range(len(data)):
    for j in range(6):
        red_balls_one_hot[i, red_balls[i, j] - 1] = 1

blue_ball_one_hot = np.zeros((len(data), 16))
for i in range(len(data)):
    blue_ball_one_hot[i, blue_ball[i] - 1] = 1

# 合并特征和目标
X = features
y_red = red_balls_one_hot
y_blue = blue_ball_one_hot

# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 构建模型
model_red = Sequential([
    Dense(64, activation='relu', input_shape=(X.shape[1],)),
    Dense(32, activation='relu'),
    Dense(33, activation='sigmoid')  # 输出红球的概率分布
])

model_blue = Sequential([
    Dense(64, activation='relu', input_shape=(X.shape[1],)),
    Dense(32, activation='relu'),
    Dense(16, activation='sigmoid')  # 输出蓝球的概率分布
])

# 编译模型

model_red.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
model_blue.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# 训练模型
history = model_red.fit(X_scaled, y_red, epochs=50, batch_size=32, validation_split=0.2)
model_blue.fit(X_scaled, y_blue, epochs=50, batch_size=32, validation_split=0.2)

# 检查模型权重状态
# plt.plot(history.history['loss'], label='train_loss')
# plt.plot(history.history['val_loss'], label='val_loss')
# plt.yscale('log')
# plt.legend()
# plt.show()

dummy_input = np.random.rand(1, X.shape[1])  # 生成随机输入
print(model_red.predict(dummy_input))  # 应输出非全1.0结果

# 预测下一期号码
latest_features = X_scaled[0].reshape(1, -1)  # 使用最近一期的特征
red_prob = model_red.predict(latest_features)
blue_prob = model_blue.predict(latest_features)

# 选择概率最高的号码
predicted_red = np.argsort(red_prob[0])[-6:][::-1] + 1
predicted_blue = np.argmax(blue_prob) + 1

print("Predicted Red Balls:", predicted_red)
print("Predicted Blue Ball:", predicted_blue)