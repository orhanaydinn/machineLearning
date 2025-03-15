# -*- coding: utf-8 -*-
"""
Created on Sat Mar 15 00:05:58 2025

@author: Orhan
"""

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
diabetes = load_diabetes()

X = diabetes.data
y = diabetes.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(X_train, y_train)

y_pred = tree_reg.predict(X_test)
print(y_pred)

from sklearn.metrics import mean_squared_error
import numpy as np

mse = mean_squared_error(y_test, y_pred)
print("MSE", mse)

rmse = np.sqrt(mse)
print("RMSE", rmse)

from sklearn.tree import DecisionTreeRegressor
import numpy as np
import matplotlib.pyplot as plt

#create own data set

X = np.sort( 5* np.random.rand(80,1), axis = 0)
y = np.sin(X).ravel()
y[::5] += 0.5 * (0.5 - np.random.rand(16)) #create wrong data every 5 steps for regression problem


plt.scatter(X, y)

regr_1 = DecisionTreeRegressor(max_depth = 2)
regr_2 = DecisionTreeRegressor(max_depth = 5)
regr_1.fit(X,y)
regr_2.fit(X,y)

X_test = np.arange(0, 5, 0.05)[:, np.newaxis]
y_pred_1 = regr_1.predict(X_test)
y_pred_2 = regr_2.predict(X_test)

plt.figure()
plt.scatter(X, y, c = "red",  label = "data")
plt.plot(X_test, y_pred_1, color = "blue", label = "max_depth = 2", linewidth = 2)
plt.plot(X_test, y_pred_2, color = "green", label = "max_depth = 5", linewidth = 2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()