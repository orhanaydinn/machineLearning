# -*- coding: utf-8 -*-
"""
Created on Sat Mar 15 00:04:02 2025

@author: Orhan
"""

from sklearn.datasets import load_iris


iris = load_iris()

X = iris.data
y = iris.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.tree import DecisionTreeClassifier

tree_clf = DecisionTreeClassifier(criterion = "gini", max_depth= 5, random_state=42)
tree_clf.fit(X_train,y_train)

from sklearn.metrics import accuracy_score, confusion_matrix

y_pred = tree_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy",accuracy)
print("Iris dataset control with DT: ", accuracy)

conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix")
print(conf_matrix)

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize = (15, 10))
plot_tree(tree_clf, filled = True, feature_names = iris.feature_names, class_names = list(iris.target_names))
plt.show()

feature_importances = tree_clf.feature_importances_

features_names = iris.feature_names
feature_importances_sorted = sorted(zip(feature_importances, features_names), reverse = True)
for importance, feature_name in feature_importances_sorted:
  print(f"{feature_name}: {importance}")