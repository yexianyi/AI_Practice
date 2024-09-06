from sklearn.datasets import load_iris  
from sklearn.model_selection import train_test_split  
from sklearn.naive_bayes import GaussianNB  
from sklearn.metrics import accuracy_score  
  
# 加载鸢尾花数据集  
iris = load_iris()  
X = iris.data  # 特征数据  
y = iris.target  # 目标类别  
  
# 划分训练集和测试集  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)  
  
# 创建并训练高斯朴素贝叶斯分类器  
gnb = GaussianNB()  
gnb.fit(X_train, y_train)  
  
# 使用训练好的模型进行预测  
y_pred = gnb.predict(X_test)  
  
# 计算预测的准确率  
accuracy = accuracy_score(y_test, y_pred)  
print(f'Accuracy: {accuracy:.2f}')  
  
# （可选）查看一些具体的预测结果  
for i in range(5):  # 只显示前5个预测结果作为示例  
    print(f'Predicted class: {iris.target_names[y_pred[i]]}, True class: {iris.target_names[y_test[i]]}')