#%% import package
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers impo

#%% Reading file
Training = pd.read_csv("/Users/dauku/Desktop/Courses/2019Fall/Machine Learning/FinalProject/TrainingSet.csv")
Testing = pd.read_csv("/Users/dauku/Desktop/Courses/2019Fall/Machine Learning/FinalProject/TestingSet.csv")

x_train = Training.iloc[:, :-2]
y_train_y1 = Training.iloc[:, -2]
y_train_y2 = Training.iloc[:, -1]
x_test = Testing.iloc[:, :-2]
y_test_y1 = Testing.iloc[:, -2]
x_test_y2 = Testing.iloc[:, -1]
 #%%
sns.heatmap(Training.corr(), annot=True)

#%%
classifier = Sequential()

# First Hidden Layer
classifier.add(Dense(76, activation = "relu", kernel_initializer = "random_normal", input_dim = 152))

# Second Hidden Layer
classifier.add(Dense(76, activation = "relu", kernel_initializer = "random_normal"))

# Output Layer
classifier.add(Dense(1, activation = "sigmoid", kernel_initializer = "random_normal"))

# Compiling the neural network
classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

#%%
classifier.fit(x_train, y_train_y1, batch_size = 10, epochs = 100)

#%%
a = np.array([1, 1, 1], [2, 2, 2], [3, 3, 3])