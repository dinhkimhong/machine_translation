#import packages
import tensorflow as tf
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Conv1D, Dense, Flatten, Input, MaxPooling1D, GRU, RepeatVector, \
                        Dropout, BatchNormalization, TimeDistributed, LSTM, Dropout
from tensorflow.python.keras.layers.merge import concatenate
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.python.keras.utils.vis_utils import plot_model
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

#set up parameter from teacher-forcing model
en_vocab = 201
max_en_len = 15
fr_se_vocab = 348
max_fr_se_len = 23

hsize = en_vocab
#encoder
en_input = Input(shape=(max_en_len, en_vocab))
en_gru = GRU(en_vocab, return_state=True)
en_out, en_state = en_gru(en_input)

#decoder
de_input = Input(shape=(max_fr_se_len-1, fr_se_vocab))
de_gru = GRU(en_vocab, return_sequences=True)
de_out = de_gru(de_input, initial_state=en_state)

#prediction layer
de_dense = Dense(fr_se_vocab, activation='softmax')
de_dense_time = TimeDistributed(de_dense)
de_pred = de_dense_time(de_out)

#compiling the model
nmt = Model(inputs=[en_input, de_input], outputs=de_pred)

#load weight from teacher-forcing model
nmt.load_weights('../model_weights.h5')

#get the weights from each layer
en_gru_w = en_gru.get_weights()
de_gru_w = de_gru.get_weights()
de_dense_w = de_dense.get_weights()

def en_gru_weight():
    return en_gru_w

def de_gru_weight():
    return de_gru_w

def de_dense_weight():
    return de_dense_w