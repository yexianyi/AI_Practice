"""
聚类算法实现学霸与非学霸的划分
"""

from numpy import vstack
from scipy.cluster.vq import kmeans, vq

list1 = [88.0, 64.0, 96.0, 85.0]
list2 = [92.0, 99.0, 95.0, 94.0]
list3 = [91.0, 87.0, 99.0, 95.0]
list4 = [78.0, 99.0, 97.0, 81.0]
list5 = [88.0, 78.0, 98.0, 84.0]
list6 = [100.0, 95.0, 100, 92.0]
data = vstack((list1, list2, list3, list4, list5, list6))  # vstack:堆积成绩数据
centroids, _ = kmeans(
    data, 2
)  # kmeans函数返回的centroids就是聚类中心：2表示分类的类别数目
result, _ = vq(data, centroids)  # vq函数用来获取每一位同学所属的类别
print(result)
