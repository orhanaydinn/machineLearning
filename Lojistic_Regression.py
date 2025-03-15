# -*- coding: utf-8 -*-
"""
Created on Sat Mar 15 00:12:47 2025

@author: Orhan
"""

#pip install ucimlrepo #Upload uci repo with python

from ucimlrepo import fetch_ucirepo
import pandas as pd

heart_disease = fetch_ucirepo(name= "heart disease")

df = pd.DataFrame(data = heart_disease.data.features)
df["target"] = heart_disease.data.targets
print(df)

# Drop Missing Value

if df.isna().any().any():
  df.dropna(inplace = True)  #Disgard missing value
  print("Missing Value")


X= df.drop(["target"],axis = 1).values
y = df.target.values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.1, random_state=42)

from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings("ignore")


log_reg = LogisticRegression(penalty = "l2", C=1, solver = "lbfgs", max_iter=100)
log_reg.fit(X_train, y_train)


#Calculate Accuracy
from sklearn.metrics import accuracy_score, confusion_matrix

log_reg.score(X_test, y_test)
print("Accuracy", log_reg.score(X_test, y_test))