import pandas as pd
import numpy as np
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.python.keras.utils.vis_utils import plot_model
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Input, GRU
import pickle
import model_weights

# loading tokenizer
with open('../en_token.pickle', 'rb') as handle:
    en_token = pickle.load(handle)
with open('../fr_se_token.pickle', 'rb') as handle:
    fr_se_token = pickle.load(handle)

#index to word and word to index
en_index_to_word = en_token.index_word
fr_se_index_to_word = fr_se_token.index_word

en_word_to_index = en_token.word_index
fr_se_word_to_index = fr_se_token.word_index
#max len
max_en_len = 15
max_fr_se_len = 23
#number of vocabulary
en_vocab = len(en_token.index_word) + 1
fr_se_vocab = len(fr_se_token.index_word) + 1

en_oh = to_categorical(np.arange(0,en_vocab))
fr_se_oh = to_categorical(np.arange(0, fr_se_vocab))

#function sentence to oh
def sent2oh(sentence, language='en', se=False, reverse=False):
    oh = list()
    if language=='en':
        sequence = en_token.texts_to_sequences([sentence])
        sequence = pad_sequences(sequence, padding='post', maxlen=max_en_len) #add padding
        if reverse == True:
            sequence = sequence[:, ::-1]
        for seq in sequence:
            oh.append(en_oh[seq])
    elif language == 'fr' and se == True:
        sequence = fr_se_token.texts_to_sequences([sentence])
        sequence = pad_sequences(sequence, padding='post', maxlen=max_fr_se_len)
        if reverse == True:
            sequence = sequence[:, ::-1]
        for seq in sequence:
            oh.append(fr_se_oh[seq])       
    return np.array(oh)

#function word to oh
def word2oh(word, language='en', se=False):
    oh = ''
    if language == 'en':
        index = en_word_to_index[word]
        oh = en_oh[index]
    elif language =='fr' and se == True:
        index = fr_se_word_to_index[word]
        oh = fr_se_oh[index]        
    return oh

#function oh to word
def oh2word(ohs, language='en', se=False):
    index = np.argmax(ohs[0])
    if index == 0: 
        word = 'unknown'
    else:
        if language=='en':
            word = en_index_to_word[index]
        elif language =='fr' and se == True:
            word = fr_se_index_to_word[index]
    return word 

#Build encoder, decoder############################
hsize = en_vocab
#encoder
en_input = Input(shape=(max_en_len, en_vocab))
en_gru = GRU(en_vocab, return_state=True)
en_out, en_state = en_gru(en_input)
encoder = Model(inputs=en_input, outputs=en_state)
#decoder
de_input = Input(shape=(1, fr_se_vocab))
de_state_in = Input(shape=(hsize,)) #---> = result from encoder.predict

#decoder's interim layers
de_gru = GRU(hsize, return_state=True)
de_out, de_state_out = de_gru(de_input, initial_state=de_state_in)
de_dense = Dense(fr_se_vocab, activation='softmax')
de_pred = de_dense(de_out)
decoder = Model(inputs=[de_input, de_state_in], outputs=[de_pred, de_state_out])

#set weights from teacher-forcing model
en_gru.set_weights(model_weights.en_gru_weight())
de_gru.set_weights(model_weights.de_gru_weight())
de_dense.set_weights(model_weights.de_dense_weight())

###################################################

#function to translate english to french
def translate_to_french(english_sentence):
    #convert english sentence to one hot
    en_sent_oh = sent2oh(english_sentence, language='en', reverse=True)
    #convert word 'sos' to oh
    de_w_seq = word2oh('sos',language='fr',se=True)
    de_w_seq = de_w_seq.reshape(1,1,fr_se_vocab)
    #get the en_state from result in encoder
    de_state = encoder.predict(en_sent_oh)
    
    fr_sent = ''
    for _ in range(max_fr_se_len):
        de_pred, de_state = decoder.predict([de_w_seq, de_state])
        de_word = oh2word(de_pred, language='fr', se=True)
        de_w_seq = word2oh(de_word, language='fr', se=True).reshape(1,1,fr_se_vocab)
        if de_word == 'eos': break
        fr_sent += de_word + ' '
    return fr_sent  

# a = translate_to_french("india is sometimes cold during june , but it is never freezing in september")
# print(a)