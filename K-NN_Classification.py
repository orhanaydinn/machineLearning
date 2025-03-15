# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# 1.1 - KNN - Classification Problem

from sklearn.datasets import load_breast_cancer #sklearn = machine learning library. Load the dataset(Breast Cancer) in the sklearn library
import pandas as pd

cancer = load_breast_cancer()
df = pd.DataFrame(data = cancer.data, columns = cancer.feature_names)
# print(df)

df["target"] = cancer.target


# 1.1.2 - Machine Learning Model - KNN (K-Nearest Neighbors)
# 1.1.3 - Train Model


from sklearn.neighbors import KNeighborsClassifier #import KNN library

X =cancer.data # Features Variable
y = cancer.target # Target Variable

#Train Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Scaling
from sklearn.preprocessing import StandardScaler #import Standard Scaling Library

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


knn = KNeighborsClassifier(n_neighbors=3) # Create a KNN model
knn.fit(X_train,y_train) # Train the model # .fit can teach our datas(sample + target)

y_pred = knn.predict(X_test)
# #print(y_pred)

# from sklearn.metrics import accuracy_score #Import accuracy library
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy",accuracy)

from sklearn.metrics import accuracy_score,confusion_matrix #Import accuracy library and confusion matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix")
print(confusion_matrix)

"""
  KNN: Hyperparamter = K
  K: 1,2,3 ... Kn
  Accuracy: %A, %B, %C ....


k = 1
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
accuracy_1 = accuracy_score(y_test, y_pred)
print("Accuracy_1",accuracy)


k = 2
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
accuracy_2 = accuracy_score(y_test, y_pred)
print("Accuracy_2",accuracy)



k = 3
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy",accuracy)



k = 10
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy",accuracy)

"""
accuracy_values = []
k_values = []
for k in range(1,21):
  knn = KNeighborsClassifier(n_neighbors=k)
  knn.fit(X_train,y_train)
  y_pred = knn.predict(X_test)
  accuracy = accuracy_score(y_test, y_pred)
  accuracy_values.append(accuracy)
  k_values.append(k)
  print(k,'.', "Accuracy",accuracy)

import matplotlib.  pyplot as plt
plt.plot(k_values,accuracy_values,marker = "o", linestyle = "-")
plt.title("Accuracy vs K Values")
plt.xlabel("K Values")
plt.ylabel("Accuracy")
plt.xticks(k_values)
plt.grid(True)
plt.show()



