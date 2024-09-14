#癌症分类预测-良／恶性乳腺癌肿瘤预
import pandas as pd
import numpy as np
from  sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
# 读取数据
column = ['Sample code number','Clump Thickness','Uniformity of Cell Size','Uniformity of Cell Shape',
          'Marginal Adhesion','Single Epithelial Cell Size','Bare Nuclei','Bland Chromatin',
          'Normal Nucleoli','Mitoses','Class']

data = pd.read_csv("breast-cancer-wisconsin.data",names=column)
print(data)
data = data.replace(to_replace="?",value =np.nan )
data = data.dropna()
# 数据分割
x_train,x_test,y_train,y_test = train_test_split(data[column[1:10]],data[column[10]],test_size=0.25)
# 标准化处理
std_x = StandardScaler()
x_train = std_x.fit_transform(x_train)
x_test = std_x.transform(x_test)

lg  = LogisticRegression(C=1.0)
lg.fit(x_train,y_train)
print(lg.coef_)
print("准确率：",lg.score(x_test,y_test))