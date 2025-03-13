1 - Supervised Learning
1.1 K-NN (K-Nearest Neighbors)
Dataset Exploration
Choosing a machine learning model
Train Model
Evaluation of Results
Hyperparameter Optimization
1.1.1 Data Exploration
KNN Binary classification Problem Solving


# 1.1.1 - Dataset Exploration

from sklearn.datasets import load_breast_cancer #sklearn = machine learning library. Load the dataset(Breast Cancer) in the sklearn library
import pandas as pd

cancer = load_breast_cancer()
df = pd.DataFrame(data = cancer.data, columns = cancer.feature_names)
# print(df)

df["target"] = cancer.target



     
1.1.2 Machine Learning Model - KNN (K-Nearest Neighbors) & 1.1.3 Train Model

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


     
KNeighborsClassifier(n_neighbors=3)
In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook.
On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.
1.1.4 Evaluation of Results

# y_pred = knn.predict(X_test)
# #print(y_pred)

# from sklearn.metrics import accuracy_score #Import accuracy library
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy",accuracy)

from sklearn.metrics import accuracy_score,confusion_matrix #Import accuracy library and confusion matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix")
print(confusion_matrix)
     
Confusion Matrix
[[ 57   6]
 [  2 106]]
1.1.5 - Hyperparameter Optimization

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
     
1 . Accuracy 0.9532163742690059
2 . Accuracy 0.9532163742690059
3 . Accuracy 0.9590643274853801
4 . Accuracy 0.9649122807017544
5 . Accuracy 0.9590643274853801
6 . Accuracy 0.9590643274853801
7 . Accuracy 0.9590643274853801
8 . Accuracy 0.9649122807017544
9 . Accuracy 0.9707602339181286
10 . Accuracy 0.9707602339181286
11 . Accuracy 0.9707602339181286
12 . Accuracy 0.9707602339181286
13 . Accuracy 0.9649122807017544
14 . Accuracy 0.9649122807017544
15 . Accuracy 0.9532163742690059
16 . Accuracy 0.9649122807017544
17 . Accuracy 0.9532163742690059
18 . Accuracy 0.9590643274853801
19 . Accuracy 0.9473684210526315
20 . Accuracy 0.9532163742690059

1.2 K-NN Regration Problems

import numpy as np

X = np.sort(5 *np.random.rand(40,1), axis=0) #features
y = np.sin(X).ravel() #target

# print(X)
# plt.scatter(X,y)

#add noise
y[::5] += 1 * (0.5 - np.random.rand(8))
# plt.scatter(X,y)

from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt



T = np.linspace(0, 5, 500)[:, np.newaxis]

for i, weight in enumerate(["uniform", "distance"]):


  knn = KNeighborsRegressor(n_neighbors=5, weights=weight)
  y_pred = knn.fit(X,y).predict(T)

  plt.subplot(2, 1, i +1 )
  plt.scatter(X, y, color = "green", label = "data")
  plt.plot(T,y_pred, color = "blue", label ="prediction")
  plt.axis("tight")
  plt.legend()
  plt.title("KNN Regressor weight = {}".format(weight))

plt.tight_layout()
plt.show
     
matplotlib.pyplot.show
def show(*args, **kwargs) -> None
Display all open figures.

Parameters
----------
block : bool, optional
    Whether to wait for all figures to be closed before returning.

    If `True` block and run the GUI main loop until all figure windows
    are closed.

    If `False` ensure that all figure windows are displayed and return
    immediately.  In this case, you are responsible for ensuring
    that the event loop is running to have responsive figures.

    Defaults to True in non-interactive mode and to False in interactive
    mode (see `.pyplot.isinteractive`).

See Also
--------
ion : Enable interactive mode, which shows / updates the figure after
      every plotting command, so that calling ``show()`` is not necessary.
ioff : Disable interactive mode.
savefig : Save the figure to an image file instead of showing it on screen.

Notes
-----
**Saving figures to file and showing a window at the same time**

If you want an image file as well as a user interface window, use
`.pyplot.savefig` before `.pyplot.show`. At the end of (a blocking)
``show()`` the figure is closed and thus unregistered from pyplot. Calling
`.pyplot.savefig` afterwards would save a new and thus empty figure. This
limitation of command order does not apply if the show is non-blocking or
if you keep a reference to the figure and use `.Figure.savefig`.

**Auto-show in jupyter notebooks**

The jupyter backends (activated via ``%matplotlib inline``,
``%matplotlib notebook``, or ``%matplotlib widget``), call ``show()`` at
the end of every cell by default. Thus, you usually don't have to call it
explicitly there.
