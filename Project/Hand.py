import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from tensorflow.keras.datasets import mnist

# Load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Flatten images to 1D vectors
x_train = x_train.reshape((-1, 28*28))
x_test = x_test.reshape((-1, 28*28))
# print(x_train.shape, y_train.shape)
# print(x_test.shape, y_test.shape)
# Normalize pixel values to [0, 1]
x_train = x_train / 255.0
x_test = x_test / 255.0
model =MLPClassifier(solver='adam',activation='relu',hidden_layer_sizes=(128,64))
model.fit(x_train,y_train)
predictions = model.predict(x_test)
print("predictions:",predictions)
print("actual values:",y_test)
print("score:",accuracy_score(y_test,predictions))
print("confusion matrix:\n",confusion_matrix(y_test,predictions))




