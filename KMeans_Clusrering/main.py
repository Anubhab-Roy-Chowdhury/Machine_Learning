from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import scale
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target
print(y)
X = scale(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = KMeans(n_clusters=2, random_state=0)
model.fit(X_train)
predictions = model.predict(X_test)
print("predictions:",predictions, "length:", len(predictions))
print("actual values:",y_test)
print("score:",accuracy_score(y_test,predictions))
print("centroids:",model.cluster_centers_)
print("labels:",model.labels_, "length:", len(model.labels_))
print("\nThis shows that labels_ is for training data while predictions are for test data")
# print("inertia:",model.inertia_)
# print("n_iter:",model.n_iter_)