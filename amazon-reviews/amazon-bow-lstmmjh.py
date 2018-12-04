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
config.batch_size = 32

(train_summary, train_review_text, train_labels), (test_summary, test_review_text, test_labels) = amazon.load_amazon()

# Need to do the tokenization 
tokenizer = text.Tokenizer(num_words=config.vocab_size)
tokenizer.fit_on_texts(train_review_text)

### vvvvv  changes for LSTM  vvvvv
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

X_train = sequence.pad_sequences(X_train, maxlen=config.maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=config.maxlen)

embeddings_index = dict()
f = open('glove.6B.100d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

embedding_matrix = np.zeros((config.vocab_size, 100))
for word, index in tokenizer.word_index.items():
    if index > config.vocab_size - 1:
        break
    else:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector
### ^^^^^^ changes for LSTM ^^^^^^^^
### old stuff from BOW
# X_train = tokenizer.texts_to_matrix(train_review_text)
# X_test = tokenizer.texts_to_matrix(test_review_text)
### old stuff 

y_train = train_labels
y_test = test_labels

# MJH - LSTM model from imdb-embedding-bidir-mjh.py
model = Sequential()
model.add(Embedding(config.vocab_size, 100, input_length=config.maxlen, weights=[embedding_matrix], trainable=False))
# 
model.add(Bidirectional(LSTM(50, activation="sigmoid", dropout=0.50, recurrent_dropout=0.50)))

model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
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
