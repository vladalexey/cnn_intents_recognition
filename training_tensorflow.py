import os
import sys
import imutils
import numpy as np 
import matplotlib as plt

import tensorflow
from tensorflow import keras

from tensorflow.keras.layers import Input, Embedding, Dense, Flatten, Dropout, Concatenate, Reshape
from tensorflow.keras.layers import Conv1D, MaxPool1D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn.externals import joblib

from tensorflow.keras.utils import to_categorical
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import text_to_word_sequence

BS = 128
BASE_DIR = 'data/'
DATA_DIR = BASE_DIR + 'processed_data'

EMBEDDING_DIR = BASE_DIR + '/embedding/crawl-300d-2M-subword.vec'
MAX_NB_WORDS = 100000 # vocab size
MAX_SEQUENCE_LENGTH = 50 # length of each sentence for padding
EMBEDDING_DIM = 300 # dimension of fasttext word vectors

VALIDATION_SPLIT = 0.1

train_text = np.load(open(BASE_DIR + "/train_text.npy", 'rb')).tolist()
train_label = np.load(open(BASE_DIR + "/train_label.npy", 'rb')).tolist()
test_text = np.load(open(BASE_DIR + "/test_text.npy", 'rb')).tolist()
test_label = np.load(open(BASE_DIR + "/test_label.npy")).tolist()


train_label_encoder = preprocessing.LabelEncoder()
train_label_encoder.fit(train_label)

joblib.dump(train_label_encoder, DATA_DIR + '/label_encoder.pkl')

train_label = train_label_encoder.transform(train_label)
test_label = train_label_encoder.transform(test_label)

label_dict = dict(zip(list(train_label_encoder.classes_), train_label_encoder.transform(list(train_label_encoder.classes_))))
print('[INFO] Label dict:', label_dict)
tokenizer = Tokenizer(MAX_NB_WORDS)
tokenizer.fit_on_texts(train_text)
sequences = tokenizer.texts_to_sequences(train_text)

word_index = tokenizer.word_index
print('[INFO] Found %s unique word tokens' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

train_label = to_categorical(np.asarray(train_label))
print('[INFO] Shape of data tensor:', data.shape)
print('[INFO] Shape of label tensor:', train_label.shape)

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices] # rearrange train text to shuffled indices
train_label = train_label[indices] # rearrange test text to shuffled indices
num_val = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-num_val]
y_train = train_label[:-num_val]
x_val = data[-num_val:]
y_val = data[-num_val:]

print('[INFO] Indexing word vectors')

embeddings_index = {}
f = open(EMBEDDING_DIR, encoding='utf-8')
for line in f:
    value = line.split()
    word = value[0]
    coefs = np.asarray(value[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('[INFO] Found %s word vectors' % len(embeddings_index))

num_words = min(MAX_NB_WORDS, len(word_index))
embedding_matrix = np.zeros(len(word_index) + 1, EMBEDDING_DIM)
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

input_layer = Input(MAX_SEQUENCE_LENGTH, dtype='int32')
embedding_layer = Embedding(input_dim=len(word_index) + 1, output_dim=EMBEDDING_DIM, weights=[embedding_matrix]
        , input_length=MAX_SEQUENCE_LENGTH)(input_layer)
# reshape_layer = Reshape((MAX_SEQUENCE_LENGTH, EMBEDDING_DIM, 1))(embedding_layer)

filters = [2, 3, 4, 5]
number_filters = 256

conv_list = []
for filter in filters:
    conv_layer = Conv1D(number_filters, filter, activation='relu')(embedding_layer)
    pool_layer = MaxPool1D(5)(conv_layer)
    conv_list.append(pool_layer)

concat_layer = Concatenate(axis=1)(conv_list)
conv_layer5 = Conv1D(number_filters, 5, activation='relu')(concat_layer)
pool_layer5 = MaxPool1D(5)(conv_layer5)
conv_layer6 = Conv1D(number_filters, 5, activation='relu')(pool_layer5)
pool_layer6 = MaxPool1D(30)(conv_layer6)
flatten = Flatten(pool_layer6)
dense = Dense(256, activation='relu')(flatten)
preds = Dense(7, activation='softmax')(dense)

model = Model(inputs=input_layer, outputs=preds)

model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
model.summary()

filepath = "saved_model-{epoch:02d}-{loss:.4f}.hdf5"
callbacks = [ModelCheckpoint(filepath, 
                monitor='val_accuracy', 
                verbose=1, 
                save_best_only=True),
            EarlyStopping(monitor='val_accuracy', patience=6, verbose=1)
]

history = model.fit(
                x_train, 
                y_train, 
                epochs=3, 
                batch_size=BS,
                validation_data= (x_val, y_val),
                callbacks=callbacks)

test_sequences = tokenizer.texts_to_sequences(test_text)
test_data = pad_sequences(test_sequences)

test_predictions = model.predict(test_data)
test_prediction = test_predictions.argmax(axis=-1)

test_intent_predictions = train_label_encoder.inverse_transform(test_prediction)
test_intent_original = train_label_encoder.inverse_transform(test_label)

print('Accuracy:', sum(test_intent_predictions == test_intent_original) / len(test_data))
print('Precision, Recall, F1-Score:\n\n', classification_report(test_intent_original, test_prediction))



