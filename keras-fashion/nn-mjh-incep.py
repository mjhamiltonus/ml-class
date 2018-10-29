# Try adding max pool after first conv layer
# max pool after each conv layer, sequentially more 

import numpy
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Reshape, Conv2D, MaxPooling2D
from keras.utils import np_utils
import wandb
from wandb.keras import WandbCallback

from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input, decode_predictions
import numpy as np



# logging code
run = wandb.init()
config = run.config
# MJH config.epochs = 100
config.epochs = 50
config.lr = 0.01
# MJH config.layers = 3
# config.layers = 4
config.dropout = 0.4
config.hidden_layer_1_size = 128


# load data
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# Normalize to 0:1 from 0:255 - weights init'ed with that
X_train = X_train / 255.
X_test = X_test / 255.

img_width = X_train.shape[1]
img_height = X_train.shape[2]
labels = ["T-shirt/top", "Trouser", "Pullover", "Dress",
          "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

### vvv MJH ADD 
config.first_layer_convs = 32
config.first_layer_conv_width = 3
config.first_layer_conv_height = 3
config.dense_layer_size = 128
config.img_width = img_width
config.img_height = img_height
### ^^^ MJH ADD 

# one hot encode outputs - prevents assigning value to specific category - converts 1-d array (categories) to list of 10 item arrays. 
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

num_classes = y_train.shape[1]



model = InceptionV3(weights='imagenet')



# # create model
# model = Sequential()
# #  vvv MJH
# model.add(Reshape((img_width, img_height, 1), input_shape=(img_width,img_height)))
# model.add(Conv2D(config.first_layer_convs,
#     (config.first_layer_conv_width, config.first_layer_conv_height),
#     input_shape=(img_width, img_height, 1),
#                  padding = 'same',
#     activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Conv2D(32,
#     (5,5),
#     input_shape=(img_width, img_height, 1),
#                  padding = 'same',
#     activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Conv2D(32,
#     (7,7),
#     input_shape=(img_width, img_height, 1),
#                  padding = 'same',
#     activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# # ^^^ MJH
# model.add(Flatten(input_shape=(img_width, img_height)))
# model.add(Dense(config.hidden_layer_1_size, activation='relu'))


model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])
# Fit the model
model.fit(X_train, y_train, epochs=config.epochs, validation_data=(X_test, y_test),
          callbacks=[WandbCallback(data_type="image", labels=labels)])
