from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
#from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Model
from keras.optimizers import SGD
import numpy as np
import os
import sys
import wandb
from keras.callbacks import Callback
from wandb.keras import WandbCallback


NB_IV3_LAYERS_TO_FREEZE = 172


def save_bottleneck_features():
#     if os.path.exists('bottleneck_features_train.npy') and (len(sys.argv) == 1 or sys.argv[1] != "--force"):
#         print("Using saved features, pass --force to save new features")
#         return
    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    print( "xtrain shape:  {:}   xtrain size: {:} xtrain shape[0]={:}".format(x_train.shape,x_train.size,x_train.shape[0] ) )
    nb_train_samples = x_train.shape[0]
    nb_validation_samples = x_test.shape[0]
    train_generator = datagen.flow(x_train, y_train, batch_size=config.batch_size)
    val_generator = datagen.flow(x_test, y_test, batch_size=config.batch_size)
    
    # build the VGG network
    model = VGG16(include_top=False, weights='imagenet')
    
    print("Predicting bottleneck training features")
    training_labels = []
    training_features = []
    for batch in range(5): #nb_train_samples // config.batch_size):
        data, labels = next(train_generator)
        training_labels.append(labels)
        training_features.append(model.predict(data))
    training_labels = np.concatenate(training_labels)
    training_features = np.concatenate(training_features)
    np.savez(open('bottleneck_features_train.npy', 'wb'),
            features=training_features, labels=training_labels)
    
    print("Predicting bottleneck validation features")
    validation_labels = []
    validation_features = []
    validation_data = []
    for batch in range(nb_validation_samples // config.batch_size):
        data, labels = next(val_generator)
        validation_features.append(model.predict(data))
        validation_labels.append(labels)
        validation_data.append(data)
    validation_labels = np.concatenate(validation_labels)
    validation_features = np.concatenate(validation_features)
    validation_data = np.concatenate(validation_data)
    np.savez(open('bottleneck_features_validation.npy', 'wb'),
            features=training_features, labels=training_labels, data=validation_data)



def train_top_model():
    train = np.load(open('bottleneck_features_train.npy', 'rb'))
    X_train, y_train = (train['features'], train['labels'])
    test = np.load(open('bottleneck_features_validation.npy', 'rb'))
    X_test, y_test, val_data = (test['features'], test['labels'], test['data'])

    model = Sequential()
    model.add(Flatten(input_shape=X_train[0].shape))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='sigmoid'))
    model.compile(optimizer=SGD(lr=0.005, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
   
    class Images(Callback):
        def on_epoch_end(self, epoch, logs):
            base_model = VGG16(include_top=False, weights='imagenet')
            indices = np.random.randint(val_data.shape[0], size=36)
            test_data = val_data[indices]
            features = base_model.predict(np.array([preprocess_input(data) for data in test_data]))
            pred_data = model.predict(features)
#             wandb.log(
#                 {
#                     "examples": [wandb.Image(test_data[i], caption="test") for i, data in enumerate(test_data)]
#                 }, commit=False)

    model.fit(X_train, y_train,
              epochs=config.epochs,
              batch_size=config.batch_size,
              validation_data=(X_test, y_test),
              callbacks=[Images(), WandbCallback(save_model=False)])
    model.save_weights(top_model_weights_path)



wandb.init()
config = wandb.config

# dimensions of our images.
config.dropout = 0.25
config.dense_layer_nodes = 100
config.learn_rate = 0.01
config.fc_size = 1024

config.img_width = 224
config.img_height = 224
config.epochs = 200
config.batch_size = 64

top_model_weights_path = 'bottleneck.h5'

class_names = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']
nb_classes = len(class_names)

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, nb_classes)
y_test = keras.utils.to_categorical(y_test, nb_classes)

save_bottleneck_features()
train_top_model()
