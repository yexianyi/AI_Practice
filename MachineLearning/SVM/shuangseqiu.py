from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score 
from sklearn.metrics import accuracy_score  
import pandas as pd
import matplotlib.pyplot as plt
import warnings

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
warnings.filterwarnings("ignore")

# 加载数据
df = pd.read_csv("./AssociationRule/shuangseqiu.csv")
x = df[['R1', 'R2', 'R3', 'R4', 'R5', 'R6']].values
y = df['B1'].values  

# 对数据进行分割，分为训练数据集和测试数据集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

# 创建SVM分类器实例  
# 在这里指定SVM的一些参数，比如kernel='linear'（线性核），'rbf'（径向基函数核）等  
clf = svm.SVC(kernel='linear')  # 这里以线性核为例  

# 定义C值的候选范围  
C_values = [1, 3]  
  
# 设置GridSearchCV，使用交叉验证来评估不同的C值  
# cv参数指定了交叉验证的折数，这里使用5折交叉验证  
grid_search = GridSearchCV(clf, param_grid={'C': C_values}, cv=5, scoring='accuracy')  
  
# 执行网格搜索  
grid_search.fit(x_train, y_train)  

# 输出最佳参数  
print("最佳参数:", grid_search.best_params_)  
# 输出最佳模型在交叉验证集上的平均分数  
print("最佳模型的平均交叉验证分数:", grid_search.best_score_)  
  
# 你可以使用最佳模型进行预测等操作  
best_model = grid_search.best_estimator_  
# 例如，使用最佳模型进行预测  
y_pred = best_model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)  
print(f'Accuracy: {accuracy:.3f}')


# # 训练模型
# clf.fit(x_train, y_train)  
  
# # 使用已经训练好的模型预测测试集的标签  
# y_pred = clf.predict(x_test)  
  
# # 使用y_pred和y_test来评估模型的性能，比如计算准确率  

# accuracy = accuracy_score(y_test, y_pred)  
# print(f'Accuracy: {accuracy:.2f}')