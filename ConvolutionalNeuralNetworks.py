#%% import packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Flatten
from keras.datasets import mnist

#%% import data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# reshape the data to be [samples][pixels][width][height]
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype("float32")
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype("float32")

# normalize predictors
x_train = x_train / 255
x_test = x_test / 255

# change targets to be categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

num_classes = y_test.shape[1]

#%% define function that creates the model
def convolutional_model():

    # create model
    model = Sequential()
    model.add(Conv2D(16, (5, 5), strides = (1, 1), activation = "relu", input_shape = (28, 28, 1)))
    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))

    model.add(Flatten())
    model.add(Dense(100, activation = "relu"))
    model.add(Dense(num_classes, activation = "softmax"))

    # compile model
    model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
    return model

model = convolutional_model()

#%% train and evaluate the model
# fit
model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs = 10, batch_size = 200, verbose = 2)

# evaluate the model
scores = model.evaluate(x_test, y_test, verbose = 0)

#%% convolutional layer with two sets of convolutional and pooling layers
def convolutional_2_model():
    # create model
    model = Sequential()
    model.add(Conv2D(16, (5, 5), activation = "relu", input_shape = (28, 28, 1))) # 16 filters and each filter has a size of 5 by 5
    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))

    model.add(Conv2D(8, (2, 2), activation = "relu"))
    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))

    model.add(Flatten())
    model.add(Dense(100, activation = "relu"))
    model.add(Dense(num_classes, activation = "softmax"))

    # Compile model
    model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])

    return model

model2 = convolutional_2_model()

#%%
model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs = 10, batch_size = 200, verbose = 2)

scores = model.evaluate(x_test, y_test, verbose = 0)
