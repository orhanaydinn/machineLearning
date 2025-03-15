# -*- coding: utf-8 -*-
"""
Created on Sat Mar 15 00:09:38 2025

@author: Orhan
"""

from sklearn.datasets import fetch_olivetti_faces
import matplotlib.pyplot as plt


oli = fetch_olivetti_faces()

'''
    2D (64X64) - > (4096)
'''

plt.figure()
for i in range(2):
  plt.subplot(1, 2, i+1)
  plt.imshow(oli.images[i], cmap = "gray")

plt.show()

from sklearn.model_selection import train_test_split

X = oli.data
y = oli.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)

y_pred = rf_clf.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy",accuracy)

