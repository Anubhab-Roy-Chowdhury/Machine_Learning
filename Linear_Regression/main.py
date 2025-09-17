from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
X = housing.data
y = housing.target
print(housing.data.shape)   # features
print(housing.target.shape) # target
#print(X,y)
lg = linear_model.LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
lg.fit(X_train,y_train)
predictions = lg.predict(X_test)
print("predictions:",predictions)
print("actual values:",y_test)
print("score:",lg.score(X_test,y_test))
print("coefficients:",lg.coef_)
print("intercept:",lg.intercept_)
plt.scatter(X.T[0],y)
plt.show()


