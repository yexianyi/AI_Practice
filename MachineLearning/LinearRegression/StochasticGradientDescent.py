from sklearn.linear_model import SGDRegressor
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

# 创建梯度下降模型
sgd = SGDRegressor()
# 训练模型
sgd.fit(x_train, y_train.ravel())
print(sgd.coef_)
# 预测测试集房子价格
y_sgd_predict = std_y.inverse_transform(sgd.predict(x_test).reshape(-1, 1))
print("测试集里面每个房子的预测价格：%s" % y_sgd_predict)
print(
    "梯度下降的均方误差：%s"
    % mean_squared_error(std_y.inverse_transform(y_test), y_sgd_predict)
)
