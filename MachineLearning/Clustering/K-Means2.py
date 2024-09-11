'''
随机创建不同二维数据集作为训练集，并结合k-means算法将其聚类
'''

# 1. 创建数据集
import matplotlib.pyplot as plt 
from sklearn.datasets._samples_generator import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score

# X 为样本特征， Y为样本簇类别， 共1000 个样本 ，每4个特征，共 4个簇，
# 簇中心在 [-1, -1], [0,0],[1,1], [2,2] 2,2]， 簇方差分别为 [0.4, 0.2, 0.2,
X, y = make_blobs(n_samples=1000, n_features=2, centers=[[-1, -1], [0, 0], [1, 1], [2, 2]],
cluster_std=[0.4, 0.2, 0.2, 0.2],
random_state=9)

# 数据集可视化
plt.scatter(X[:, 0], X[:, 1], marker='o')
plt.show()

# 2. 使用 k-means 进行聚类 ,并使用 CH 方法评估
y_pred = KMeans(n_clusters=2, random_state=9).fit_predict(X)
# 分别尝试 n_cluses=2 2\3\4, 然后查看聚类效果
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.show()
# 用CalinskiCalinski-Harabasz Index 评估的聚类分数
print(calinski_harabasz_score(X, y_pred))