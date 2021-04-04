from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

iris = load_iris()
print(iris)
# split data set
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=22)
# Feature Engineering: standardization
transfer = StandardScaler()
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)

# KNN Estimator
estimator = KNeighborsClassifier()
# model selection
param_dict = {"n_neighbors": [1, 3, 5]}
estimator = GridSearchCV(estimator, param_grid=param_dict, cv=3)
# train model
estimator.fit(x_train, y_train)

# assess model
y_predict = estimator.predict(x_test)
print("Compare predict result with real result :", y_predict == y_test)
score = estimator.score(x_test, y_test)
print("Precision: ", score)

# Best result
print(estimator.best_estimator_)
print(estimator.best_params_)
print(estimator.best_score_)
