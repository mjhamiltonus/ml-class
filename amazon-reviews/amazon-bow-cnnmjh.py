import amazon
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding, LSTM
from keras.layers import Conv1D, Flatten
from keras.preprocessing import text
import wandb
from wandb.keras import WandbCallback

wandb.init()
config = wandb.config
config.vocab_size = 1000
# for CNN
config.embedding_dims = 50
config.maxlen = 1000
config.filters = 250
config.kernel_size = 3
config.hidden_dims = 250
config.epochs = 10
config.batch_size = 256

(train_summary, train_review_text, train_labels), (test_summary, test_review_text, test_labels) = amazon.load_amazon()

tokenizer = text.Tokenizer(num_words=config.vocab_size)
tokenizer.fit_on_texts(train_review_text)
X_train = tokenizer.texts_to_matrix(train_review_text)
X_test = tokenizer.texts_to_matrix(test_review_text)

y_train = train_labels
y_test = test_labels

# MJH - CNN model from imdb-cnn.py
model = Sequential()
model.add(Embedding(config.vocab_size,
                    config.embedding_dims,
                    input_length=config.maxlen))
model.add(Dropout(0.5))
model.add(Conv1D(config.filters,
                 config.kernel_size,
                 padding='valid',
                 activation='relu'))
model.add(Flatten())
model.add(Dense(config.hidden_dims, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, y_train,
          batch_size=config.batch_size,
          epochs=config.epochs,
          validation_data=(X_test, y_test), callbacks=[WandbCallback()])





# Build the model - original
# model = Sequential()
# model.add(Dense(1, activation='softmax', input_shape=(config.vocab_size,)))

# model.compile(loss='binary_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])
# model.fit(X_train, train_labels, epochs=10, validation_data=(X_test, test_labels),
#     callbacks=[WandbCallback()])
