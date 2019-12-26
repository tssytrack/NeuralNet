#%%
import keras
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler

#%% Data Reading
concreteData = pd.read_csv("https://ibm.box.com/shared/static/svl8tu7cmod6tizo6rk0ke4sbuhtpdfx.csv")

#%% Split data into predictors and target
columns = concreteData.columns

predictors  = concreteData[columns[columns != "Strength"]]
target = concreteData["Strength"]

#%% Standardize the data
scaler = StandardScaler()
scaler.fit(predictors)
Standardized = scaler.transform(predictors)

std_predictors = pd.DataFrame(Standardized, columns = columns[columns != "Strength"])

n_cols = std_predictors.shape[1]

#%% Build a Neural Network
def regression_model():
    # create model
    model = Sequential()
    model.add(Dense(50, activation = "relu", input_shape = (n_cols, )))
    model.add(Dense(50, activation = "relu"))
    model.add(Dense(1))

    # compile model
    model.compile(optimizer = "adam", loss = "mean_squared_error")

    return(model)

#%% Train adn Test the Network
# build the model
model = regression_model()

# fit the model
model.fit(std_predictors, target, validation_split = 0.3, epochs = 1000, verbose = 2)

