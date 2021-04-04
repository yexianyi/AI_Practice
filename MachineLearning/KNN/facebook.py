import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# 1.获取数据集
data = pd.read_csv("train.csv")
# 2.基本数据处理
# 2.1 缩小数据范围
data = data.query("x>1.0 & x<1.25 & y>2.5 & y<2.75")
# 2.2 选择时间特征
time_value = pd.to_datetime(data["time"], unit="s")
time_value = pd.DatetimeIndex(time_value)

data['day'] = time_value.day
data['weekday'] = time_value.weekday
data['hour'] = time_value.hour
data = data.drop(["time"], axis=1)
print(data)
# 2.3 去掉签到较少的地方
place_count = data.groupby("place_id").count()  # place_count作为行标签
tf = place_count[place_count.row_id > 3].reset_index()
data = data[data["place_id"].isin(tf.place_id)]
# 2.4 确定特征值和目标值
# 取出数据中的特征值和目标值
y = data["place_id"]
# 删除之前数据中的place_id列
x = data.drop(["place_id"], axis=1)
x = data.drop(["row_id"], axis=1)

# 2.5对数据进行分割，分为训练数据集和测试数据集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
# 3.特征工程--特征预处理(标准化)
# 3.1 实例化一个转换器
sd = StandardScaler()
# # 3.2 调用fit_transform要对测试集和训练集的目标值做标准化
x_train = sd.fit_transform(x_train)
x_test = sd.fit_transform(x_test)

# 4.机器学习--knn+cv
# 4.1 实例化一个估计器

knn = KNeighborsClassifier(n_neighbors=5)
# 进行模型训练
knn.fit(x_train, y_train)
# 利用模型得出训练结果
y_predict = knn.predict(x_test)

print("预测的目标签到位置：", y_predict)
# 获取准确率
print("预测的准确率：", knn.score(x_test, y_test))
