from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# 导入保存模型的API
import pandas as pd
import numpy as np

# 1.获取数据
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

# 2.分割数据到训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.25)
# 3.进行标准化数据（特征工程）处理
std_x = StandardScaler()
x_train = std_x.fit_transform(x_train)
x_test = std_x.transform(x_test)

std_y = StandardScaler()
y_train = std_y.fit_transform(y_train.reshape(-1, 1))
y_test = std_y.transform(y_test.reshape(-1, 1))
print(y_train)
print(y_test)

# 创建Ridge模型
estimator = Ridge(alpha=1)
# estimator = RidgeCV(alphas=(0.1, 1, 10))
estimator.fit(x_train, y_train)
y_predict = estimator.predict(x_test)

print("预测值为 :\n", y_predict)
print("模型中的系数为 :\n", estimator.coef_)
print("模型中的偏置为 :\n", estimator.intercept_)

# 5.2 评价 --- 均方误差
error = mean_squared_error(y_test, y_predict)
print("误差为 :\n", error)