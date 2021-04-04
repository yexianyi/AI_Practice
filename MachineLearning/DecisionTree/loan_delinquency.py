'''
Decision Tree
Predict if it is possible to default on the loan
'''
import numpy as np
from sklearn import tree

data = np.genfromtxt("exercise.csv", delimiter=",")
# get train data set
x_data = data[1:, 1:-1]
# get test data set
y_data = data[1:, -1]

print(x_data)
print(y_data)

# Create decision tree
dtree = tree.DecisionTreeClassifier(min_samples_leaf=5)
dtree.fit(x_data, y_data)
print(dtree.score(x_data, y_data))
