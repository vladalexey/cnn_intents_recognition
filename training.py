import os
import imutils
import numpy as np 
import matplotlib as plt

import plaidml
from plaidml import keras
plaidml.keras.install_backend()

from keras.layers import Input, Embedding, Dense, Flatten, Dropout, Concatenate
from keras.layers import Conv2D, MaxPool2D
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report

from keras.utils import to_categorical
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import text_to_word_sequence

BS = 128
BASE_DIR = 'data'
DATA_DIR = BASE_DIR + 'processed_data'
EMBEDDING_DIR = BASE_DIR + '/embedding'

VALIDATION_SPLIT = 0.1

train_text = np.load("")