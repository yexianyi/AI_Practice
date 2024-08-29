import pandas as pd
import sklearn as sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

# 1. Load data
data = pd.read_csv('./dataset/iris.data', header=None)
df = pd.DataFrame(data)
df.columns =['sepal length', 'sepal width', 'petal length', 'petal width', 'Species']
# df.groupby('Species').size()
# print(df.sample(5))
# print(df.describe(include='all'))

# 2. Feature engineering
# Optional: normalisation
transfer = MinMaxScaler(feature_range=(0,1))
data = transfer.fit_transform(df.iloc[:, :4])
new_df = pd.DataFrame(data) 
new_df['Species'] = df['Species'] 
# print(df.sample(5))
# print(new_df.describe(include='all'))

# dataset for features
feature_dataset = new_df.iloc[:, :4]
# dataset for categories
label_dataset = new_df['Species'] 

# transform category label to numeric, as KNeighborsClassifier does not accept string labels. 
le = LabelEncoder()
label_dataset = le.fit_transform(label_dataset)

# 3. Spliting dataset into training set and test set
x_train, x_test, y_train, y_test = train_test_split(feature_dataset, label_dataset, test_size = 0.2, random_state = 0)

# 4. Fitting clasifier to the Training set
estimator = KNeighborsClassifier()
param_dict = {"n_neighbors": [1, 3, 5]}
estimator = GridSearchCV(estimator, param_grid=param_dict, cv=3)

# 5. Fitting the model
estimator.fit(x_train, y_train)

# 6、Evaluate estimator
#方法a:比对预测结果和真实值
y_predict=estimator.predict(x_test)
print("比对预测结果和真实值：", y_predict == y_test)

# 方法b:直接计算准确率
score=estimator.score(x_test, y_test)
print("直接计算准确率：", score)

# 评估查看最终选择的结果和交叉验证的结果
print("在交叉验证中验证的最好结果：", estimator.best_score_)
print("最佳模型的k值是:", estimator.best_params_['n_neighbors'])
print("每次交叉验证后的准确率结果：", estimator.cv_results_)
