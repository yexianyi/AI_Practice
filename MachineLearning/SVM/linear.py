import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

# 生成模拟数据
np.random.seed(0)
X = np.r_[
    np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]
]  # 生成了一个形状为(40, 2)的数组，其中前20行是原始随机数组向左下方移动（每个元素减去2）的结果，后20行是另一个原始随机数组向右上方移动（每个元素加上2）的结果。
y = [0] * 20 + [1] * 20  # [0, 0, 0, ..., 0, 1, 1, 1, ..., 1]（前20个元素是0，后20个元素是1）

# 可视化数据点
plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)

# 创建一个SVM分类器实例，并设置为线性核
clf = svm.SVC(kernel="linear", C=1000)
# 训练模型
clf.fit(X, y)
# print(clf.predict([[-1,-1]]))

# 绘制决策边界
"""
.coef_：这是scikit-learn中线性模型类的一个属性，用于存储模型的系数（coefficients）。
        对于线性模型来说，系数是模型参数的一部分，它们与输入数据的特征相乘，再加上截距（如果有的话），以产生预测输出。
        在多元线性回归中，.coef_ 是一个数组，其长度等于输入特征的数量，每个元素对应一个特征的系数。
[0]：   当 .coef_ 是一个数组时（这在大多数线性模型中都是如此），[0] 表示我们正在访问这个数组的第一个元素。
        这通常是因为在scikit-learn中，当处理多类分类问题时（如使用逻辑回归进行多类分类），.coef_ 会返回一个二维数组，
        其中每一行对应于模型中的一个类别（除了基准类别外）的系数。因此，clf.coef_[0] 在这种情况下将返回与第一个非基准类别相关联的系数数组。
        然而，在单类分类或回归问题中，.coef_ 可能只是一个一维数组，此时 [0] 仍然可以用来访问第一个系数，但在这种情况下它更多是出于习惯或一致性而写的。
"""
w = clf.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-5, 5)
yy = a * xx - (clf.intercept_[0]) / w[1]

# 绘制支持向量
b = clf.support_vectors_
plt.scatter(b[:, 0], b[:, 1], s=100, facecolors="none", edgecolors="k", marker="s")

# 绘制决策边界和边界线
plt.plot(xx, yy, "k-")
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("SVM with linear kernel")
plt.show()
