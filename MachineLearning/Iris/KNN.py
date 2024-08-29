import pandas as pd
import sklearn as sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#1. Load data
data = pd.read_csv('./dataset/iris.data', header=None)
df = pd.DataFrame(data)
df.columns =['sepal length', 'sepal width', 'petal length', 'petal width', 'Species']
# df.groupby('Species').size()
# print(df.sample(5))
# print(df.describe(include='all'))

#2. Feature engineering
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

#3. Spliting dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(feature_dataset, label_dataset, test_size = 0.2, random_state = 0)

#4. Fitting clasifier to the Training set
# Instantiate learning model (k = 3)
model = KNeighborsClassifier(n_neighbors=3)

#5. Fitting the model
model.fit(X_train, y_train)

#6. Predicting the Test set results
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)*100
print('Accuracy of our model: ' + str(round(accuracy, 2)) + ' %.')