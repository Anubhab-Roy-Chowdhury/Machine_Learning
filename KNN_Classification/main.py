import numpy as np
import pandas as pd
from sklearn import metrics,neighbors
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
data = pd.read_csv("KNN_Classification\car.data")
#print(data.head())
X = data[['buying','maint','safety']].values
y = data[['class']]
#print(X,y)
le = LabelEncoder()
for i in range(len(X[0])):
    X[:, i] = le.fit_transform(X[:, i])

#
label_mapping = {
    'unacc':0,
    'acc':1,
    'good':2,
    'vgood':3

}
y = y['class'].map(label_mapping)
y = np.array(y)
print(y.shape)
#
knn = neighbors.KNeighborsClassifier(n_neighbors=25,weights='uniform')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
knn.fit(X_train,y_train)
prediction = knn.predict(X_test)
accuracy = metrics.accuracy_score(y_test,prediction)
print("predictions:",prediction)
print("accuracy:",accuracy)
# predicted_labels = le.inverse_transform(prediction)
reverse_mapping = {v: k for k, v in label_mapping.items()}
predicted_labels = [reverse_mapping[p] for p in prediction]

print("predicted labels:", predicted_labels[:20])
print("Actual Values:", y[1727])
print("Predicted Values:", knn.predict(X)[1727])