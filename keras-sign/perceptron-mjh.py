# A very simple perceptron for classifying american sign language letters
import signdata
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Reshape, Conv2D
from keras.utils import np_utils
import wandb
from wandb.keras import WandbCallback

# logging code
run = wandb.init()
config = run.config
config.loss = "categorical_crossentropy"
config.optimizer = "adam"
config.epochs = 10
config.dropout = 0.4
config.hidden_layer_1_size = 128

# load data
(X_test, y_test) = signdata.load_test_data()
(X_train, y_train) = signdata.load_train_data()

img_width = X_test.shape[1]
img_height = X_test.shape[2]

print("xtest shape {:}".format(X_test.shape))
print("xtrain  shape {:}".format(X_train.shape))

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

num_classes = y_train.shape[1]

# you may want to normalize the data here..

# create model
# model=Sequential()
# model.add(Flatten(input_shape=(img_width, img_height)))
# model.add(Dense(num_classes))
# model.compile(loss=config.loss, optimizer=config.optimizer,
#                 metrics=['accuracy'])


# MJH  create model used in the "fashion" classifier
config.first_layer_convs = 32
config.first_layer_conv_width = 3
config.first_layer_conv_height = 3

model = Sequential()
model.add(Reshape((img_width, img_height, 1), input_shape=(img_width,img_height)))
model.add(Conv2D(32,
    (config.first_layer_conv_width, config.first_layer_conv_height),
    input_shape=(img_width, img_height, 1),
    activation='relu'))
# ^^^ MJH
model.add(Flatten(input_shape=(img_width, img_height)))
model.add(Dropout(config.dropout))
model.add(Dense(config.hidden_layer_1_size, activation='relu'))
model.add(Dropout(config.dropout))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss=config.loss, optimizer=config.optimizer,
              metrics=['accuracy'])
# Fit the model
# model.fit(X_train, y_train, epochs=config.epochs, validation_data=(X_test, y_test),
#           callbacks=[WandbCallback(data_type="image", labels=labels)])


# Fit the model
model.fit(X_train, y_train, epochs=config.epochs, validation_data=(X_test, y_test), callbacks=[WandbCallback(data_type="image", labels=signdata.letters)])
