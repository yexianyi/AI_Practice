from sklearn import svm
import matplotlib.pyplot as plt
import numpy as np

x = [[3, 3], [4, 3], [1, 1]]
y = [1, 1, -1]

x_arr = np.array(x)
y_arr = np.array(y)
# C=1.0, kernel='rbf',gamma='auto'
# 'linear', 'poly', 'rbf', 'sigmoid'
model = svm.SVC(kernel="linear")
model.fit(x, y)
# render scatter diagram
plt.scatter(x_arr[:, 0], x_arr[:, 1], c=y_arr)

w1 = model.coef_[0][0]
w2 = model.coef_[0][1]
b = model.intercept_[0]
# 打印w1,w2，b的值
print(w1, w2, b)
# 图表显示
x1 = np.array([0, 5])
x2 = (w1 * x1 + b) / (-w2)
plt.plot(x1, x2, color="red")
plt.plot(x1, x2 - 1 / w2, "k--")
plt.plot(x1, x2 + 1 / w2, "k--")
plt.show()
# 打印支持向量
print(model.support_vectors_)
# 第0个点和第2个点是支持向量
print(model.support_)
# 预测
print(model.predict([[5, 5], [0, 0]]))
