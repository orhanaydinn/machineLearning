# -*- coding: utf-8 -*-
"""
Created on Sat Mar 15 00:16:01 2025

@author: Orhan
"""

#Upload Dataset digits

from sklearn.datasets import load_digits
import matplotlib.pyplot as plt


digits = load_digits()

fig, axes = plt.subplots(nrows = 2, ncols = 5, figsize = (10, 5), subplot_kw = {"xticks": [], "yticks": []})

for i, ax in enumerate(axes.flat):
  ax.imshow(digits.images[i], cmap = "binary", interpolation = "nearest")
  ax.set_title(digits.target[i])

plt.show()

from sklearn.model_selection import train_test_split

X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Model selection Support Vektor Model

from sklearn.svm import SVC

svm_clf = SVC(kernel = "linear", random_state = 42)
svm_clf.fit(X_train, y_train)

y_pred = svm_clf.predict(X_test)

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))