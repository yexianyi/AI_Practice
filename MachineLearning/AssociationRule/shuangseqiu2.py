import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# 假设数据文件名为 'ssq_history.csv'
data = pd.read_csv('shuangseqiu.csv')

new_df = data[['R1', 'R2', 'R3', 'R4', 'R5', 'R6']].copy()

# 创建一个空的DataFrame用于存储Y1到Y6列
y_columns = ['Y1', 'Y2', 'Y3', 'Y4', 'Y5', 'Y6']
y_df = pd.DataFrame(columns=y_columns)

# 填充Y1到Y6列，使用上一个样本的R1到R6数据
for i in range(len(new_df)):
    if i > 0:
        y_df.loc[i] = new_df.loc[i - 1].values
    else:
        # 第一个样本的Y1到Y6可以设为NaN或者0
        y_df.loc[i] = [0] * len(y_columns)  # 或者使用 np.nan

# 将Y1到Y6列合并到new_df中
new_df = pd.concat([new_df, y_df], axis=1)
# 删除第一行无效样本
new_df = new_df.drop(index=0)
print(new_df)

# 提取特征和目标
# 特征选择：
features = new_df[['R1', 'R2', 'R3', 'R4', 'R5', 'R6']].values

# 目标：预测下一期的红球和蓝球号码
# 注意：由于红球和蓝球是分类问题，这里我们使用独热编码
# 红球范围1-33，蓝球范围1-16
red_balls = new_df[['Y1', 'Y2', 'Y3', 'Y4', 'Y5', 'Y6']].values

# 独热编码红球和蓝球
red_balls_one_hot = np.zeros((len(new_df), 33))
for i in range(len(new_df)):
    for j in range(6):
        red_balls_one_hot[i, red_balls[i, j] - 1] = 1

# 合并特征和目标
X = features
y_red = red_balls_one_hot

# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 构建模型
model_red = Sequential([
    Dense(64, activation='relu', input_shape=(X.shape[1],)),
    Dense(32, activation='relu'),
    Dense(33, activation='sigmoid')  # 输出红球的概率分布
])

# 编译模型
model_red.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
# 训练模型
history = model_red.fit(X_scaled, y_red, epochs=50, batch_size=32, validation_split=0.2)

# 预测下一期号码
latest_features = X_scaled[-1].reshape(1, -1)  # 使用最近一期的特征
red_prob = model_red.predict(latest_features)

# 选择概率最高的号码
predicted_red = np.argsort(red_prob[0])[-6:][::-1] + 1

print("Predicted Red Balls:", predicted_red)
