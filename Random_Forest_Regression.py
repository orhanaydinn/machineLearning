# -*- coding: utf-8 -*-
"""
Created on Sat Mar 15 00:11:50 2025

@author: Orhan
"""

from sklearn.datasets import fetch_california_housing


california_housing = fetch_california_housing()

X = california_housing.data
y = california_housing.target

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train, y_train)


from sklearn.metrics import mean_squared_error
import numpy as np

y_pred = rf_reg.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("MSE", mse)

rmse = np.sqrt(mse)
print("RMSE", rmse)