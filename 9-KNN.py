import numpy as np
import pandas as pd
df = pd.read_csv('iris_data.csv')

df = df.iloc[:120,:]
df.head()
data=df.iloc[:,0:-1].values
target=df.iloc[:,-1].values
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size = 0.2)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
predictions = knn.predict(x_test)
print("Predictions:\n",predictions)
from sklearn.metrics import accuracy_score, confusion_matrix
print("Training accuracy Score is : ", accuracy_score(y_train, knn.predict(x_train)))
print("Testing accuracy Score is : ", accuracy_score(y_test, knn.predict(x_test)))
print("Training Confusion Matrix is : \n", confusion_matrix(y_train, knn.predict(x_train)))
print("Testing Confusion Matrix is : \n", confusion_matrix(y_test, knn.predict(x_test)))
