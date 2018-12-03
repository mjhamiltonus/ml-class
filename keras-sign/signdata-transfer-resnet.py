# modified from https://gist.github.com/fchollet/7eb39b44eb9e16e59632d25fb3119975
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense
from keras.layers import Input
import glob
import wandb
from wandb.keras import WandbCallback
import signdata
# from dogcat_data import generators, get_nb_files

run = wandb.init()

config = run.config
config.loss = "categorical_crossentropy"
config.optimizer = "adam"
config.epochs = 50
config.dropout = 0.4
config.hidden_layer_1_size = 128
config.batch_size = 32

# load data
(X_test, y_test) = signdata.load_test_data()
(X_train, y_train) = signdata.load_train_data()

config.img_width = X_test.shape[1]
config.img_height = X_test.shape[2]

inp = Input(shape=(config.img_height, config.img_width, 3), name='input_image')

#TODO: need to reshape the data into X,Y,3 vs. B/W images I currently have for ResNet.

main_model = ResNet50(include_top=False, weights="imagenet")
for layer in main_model.layers:
    layer.trainable=False

main_model = main_model(inp)
main_out = Flatten()(main_model)
main_out = Dense(512, activation='relu', name='fcc_0')(main_out)
main_out = Dense(1, activation='sigmoid', name='class_id')(main_out)

model = Model(inputs=inp, outputs=main_out)
model._is_graph_network = False

# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])


# Fit the model - try to do without generators
model.fit(X_train, y_train, epochs=config.epochs, 
          validation_data=(X_test, y_test), 
          callbacks=[WandbCallback(data_type="image", labels=signdata.letters)])


# train_dir = "-data/train"
# val_dir = "dogcat-data/validation"
# nb_train_samples = get_nb_files(train_dir)
# nb_classes = len(glob.glob(train_dir + "/*"))
# nb_val_samples = get_nb_files(val_dir)

# train_generator, validation_generator = generators(preprocess_input, config.img_width, config.img_height, config.batch_size, binary=True)

# # fine-tune the model
# model.fit_generator(
#     train_generator,
#     epochs=config.epochs,
#     validation_data=validation_generator,
#     callbacks=[WandbCallback(data_type="image", generator=validation_generator, labels=['cat', 'dog'],save_model=False)],
#     workers=2,
#     steps_per_epoch=nb_train_samples * 2 / config.batch_size,
#     validation_steps=nb_train_samples / config.batch_size,
# )
# model.save('transfered.h5')
