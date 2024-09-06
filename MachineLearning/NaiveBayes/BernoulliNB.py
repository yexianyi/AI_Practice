from sklearn.datasets import make_classification  
from sklearn.model_selection import train_test_split  
from sklearn.naive_bayes import BernoulliNB  
from sklearn.metrics import accuracy_score  
import matplotlib.pyplot as plt  
  
# 生成模拟的二元特征数据集  
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, n_informative=15, n_redundant=5, random_state=42)  
  
# 由于BernoulliNB期望的是二元特征，我们需要将特征转换为0和1  
# 这里我们简单地通过阈值（例如0.5）来转换，但在实际应用中可能需要更复杂的转换  
X_binary = (X > 0.5).astype(int)  
  
# 划分训练集和测试集  
X_train, X_test, y_train, y_test = train_test_split(X_binary, y, test_size=0.3, random_state=42)  
  
# 创建并训练BernoulliNB分类器  
bnb = BernoulliNB()  
bnb.fit(X_train, y_train)  
  
# 使用训练好的模型进行预测  
y_pred = bnb.predict(X_test)  
  
# 计算预测的准确率  
accuracy = accuracy_score(y_test, y_pred)  
print(f'Accuracy: {accuracy:.2f}')  
  
# （可选）由于数据集是模拟的，并且特征维度较高，直接可视化可能不太直观  
# 但我们可以可视化预测结果的一部分，例如通过混淆矩阵或简单的类别分布  
# 这里我们仅打印出预测结果的前几个样本作为示例  
print("Predicted classes:", y_pred[:10])  
print("True classes:", y_test[:10])  
