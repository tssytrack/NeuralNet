#%%
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.datasets import mnist
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler

#%% Data Reading
(x_train, y_train), (x_test, y_test) = mnist.load_data()

plt.imshow(x_train[0])
plt.show()

#%% flatten images into one-dimensional vector
num_pixels = x_train.shape[1] * x_train.shape[2]

x_train = x_train.reshape(x_train.shape[0], num_pixels).astype("float32")
x_test = x_test.reshape(x_test.shape[0], num_pixels).astype("float32")

#%% normalize the data
x_train = x_train / x_train.max()
x_test = x_test / x_test.max()

#%% performing one-hot encoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

num_classes = y_test.shape[1]

#%% building a neural net
def classification_model():
    #create model
    model = Sequential()
    model.add(Dense(num_pixels, activation = "relu", input_shape = (num_pixels, )))
    model.add(Dense(100, activation = "relu"))
    model.add(Dense(num_classes, activation = "softmax"))

    # compile model
    model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])

    return model

#%% train and test the network
# build the model
model = classification_model()

# fit the model
model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs = 10, verbose = 2)

# evaluate the model
scores = model.evaluate(x_test, y_test, verbose = 0)

#%% save the model
model.save("/Users/dauku/Desktop/Git/DavidKu_IAA2020/NeuralNet/classfication_neuralnet.h5")

#%% reload the model
from keras.models import load_model

pretrained_model = load_model("/Users/dauku/Desktop/Git/DavidKu_IAA2020/NeuralNet/classfication_neuralnet.h5")