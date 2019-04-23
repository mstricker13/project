from kerasImplTimeSeries import data

from numpy import argmax
from random import randint
from pickle import load
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.models import Model
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers import Dropout
from keras.layers import Input
from keras.layers import Flatten
from keras.callbacks import ModelCheckpoint
from nltk.translate.bleu_score import corpus_bleu
import numpy as np
import random

def define_model_Test(features_per_timestep, input_length, output_length):
    name = 'Test'
    n_units = 512
    dropout = 0.5
    input = Input(shape=(input_length, features_per_timestep))

    lstm_layer = LSTM(n_units, return_sequences=True, return_state=True)
    out, h, c = lstm_layer(input)

    s = [h, c]
    k = Input(shape=(output_length, features_per_timestep))

    lstm_layer2 = LSTM(n_units, return_sequences=True)
    out2 = lstm_layer2(k, initial_state=s)
    dense = TimeDistributed(Dense(features_per_timestep))
    denseout= dense(out2)
    model = Model([input, k], denseout)
    return model, name


#the first network
def define_model_1(features_per_timestep, input_length, output_length):
    name = 'LSTM_2u512D5_2u512D5_D5'
    n_units = 512
    dropout = 0.5

    #define the encoder
    encoder_inputs = Input(shape=(input_length, features_per_timestep))

    #TODO remove embedding

    encoder_lstm1 = LSTM(n_units, return_state=True, return_sequences=True, dropout=dropout)
    encoder_outputs_lstm1 = encoder_lstm1(encoder_inputs)
    encoder_lstm2 = LSTM(n_units, return_state=True, dropout=dropout)
    encoder_outputs_lstm2, state_h, state_c = encoder_lstm2(encoder_outputs_lstm1)
    encoder_dropout = Dropout(dropout)
    encoder_outputs_dropout = encoder_dropout(encoder_outputs_lstm2)
    encoder_states = [state_h, state_c]

    #define the decoder
    decoder_inputs = Input(shape=(output_length, features_per_timestep))

    decoder_lstm1 = LSTM(n_units, return_sequences=True, return_state=True, dropout=dropout)
    decoder_outputs_lstm1 = decoder_lstm1(decoder_inputs, initial_state=encoder_states)
    decoder_lstm2 = LSTM(n_units, return_sequences=True, return_state=True, dropout=dropout)
    decoder_outputs_lstm2, _, _ = decoder_lstm2(decoder_outputs_lstm1)
    decoder_dropout = Dropout(dropout)
    decoder_outputs_dropout = decoder_dropout(decoder_outputs_lstm2)
    decoder_dense = TimeDistributed(Dense(features_per_timestep))
    decoder_outputs = decoder_dense(decoder_outputs_dropout)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    return model, name

#remove dropout in recurrent layers because some people in the internet say it doesn't make sense
def define_model_2(features_per_timestep, input_length, output_length):
    name = 'LSTM_2u512_2u512_D5'
    n_units = 512
    dropout = 0.5

    #define the encoder
    encoder_inputs = Input(shape=(input_length, features_per_timestep))

    encoder_lstm1 = LSTM(n_units, return_state=True, return_sequences=True)
    encoder_outputs_lstm1 = encoder_lstm1(encoder_inputs)
    encoder_lstm2 = LSTM(n_units, return_state=True)
    encoder_outputs_lstm2, state_h, state_c = encoder_lstm2(encoder_outputs_lstm1)
    encoder_dropout = Dropout(dropout)
    encoder_outputs_dropout = encoder_dropout(encoder_outputs_lstm2)
    encoder_states = [state_h, state_c]

    #define the decoder
    decoder_inputs = Input(shape=(output_length, features_per_timestep))

    decoder_lstm1 = LSTM(n_units, return_sequences=True, return_state=True)
    decoder_outputs_lstm1 = decoder_lstm1(decoder_inputs, initial_state=encoder_states)
    decoder_lstm2 = LSTM(n_units, return_sequences=True, return_state=True)
    decoder_outputs_lstm2, _, _ = decoder_lstm2(decoder_outputs_lstm1)
    decoder_dropout = Dropout(dropout)
    decoder_outputs_dropout = decoder_dropout(decoder_outputs_lstm2)
    decoder_dense = TimeDistributed(Dense(features_per_timestep))
    decoder_outputs = decoder_dense(decoder_outputs_dropout)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    return model, name

#the first network, but removed dropout completely
def define_model_3(features_per_timestep, input_length, output_length):
    name = 'LSTM_2u512_2u512'
    n_units = 512
    dropout = 0.5

    #define the encoder
    encoder_inputs = Input(shape=(input_length, features_per_timestep))

    #TODO remove embedding

    encoder_lstm1 = LSTM(n_units, return_state=True, return_sequences=True)
    encoder_outputs_lstm1 = encoder_lstm1(encoder_inputs)
    encoder_lstm2 = LSTM(n_units, return_state=True)
    encoder_outputs_lstm2, state_h, state_c = encoder_lstm2(encoder_outputs_lstm1)
    encoder_dropout = Dropout(dropout)
    encoder_outputs_dropout = encoder_dropout(encoder_outputs_lstm2)
    encoder_states = [state_h, state_c]

    #define the decoder
    decoder_inputs = Input(shape=(output_length, features_per_timestep))

    decoder_lstm1 = LSTM(n_units, return_sequences=True, return_state=True)
    decoder_outputs_lstm1 = decoder_lstm1(decoder_inputs, initial_state=encoder_states)
    decoder_lstm2 = LSTM(n_units, return_sequences=True, return_state=True)
    decoder_outputs_lstm2, _, _ = decoder_lstm2(decoder_outputs_lstm1)
    decoder_dense = TimeDistributed(Dense(features_per_timestep))
    decoder_outputs = decoder_dense(decoder_outputs_lstm2)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    return model, name

#try out a deep and thin architecture
def define_model_4(features_per_timestep, input_length, output_length):
    name = 'LSTM_6u32D5_6u32D5_D5'
    n_units = 32
    dropout = 0.5

    #define the encoder
    encoder_inputs = Input(shape=(input_length, features_per_timestep))

    encoder_lstm1 = LSTM(n_units, return_state=True, return_sequences=True, dropout=dropout)
    encoder_outputs_lstm1 = encoder_lstm1(encoder_inputs)
    encoder_lstm2 = LSTM(n_units, return_state=True, return_sequences=True, dropout=dropout)
    encoder_outputs_lstm2 = encoder_lstm2(encoder_outputs_lstm1)
    encoder_lstm3 = LSTM(n_units, return_state=True, return_sequences=True, dropout=dropout)
    encoder_outputs_lstm3 = encoder_lstm3(encoder_outputs_lstm2)
    encoder_lstm4 = LSTM(n_units, return_state=True, return_sequences=True, dropout=dropout)
    encoder_outputs_lstm4 = encoder_lstm4(encoder_outputs_lstm3)
    encoder_lstm5 = LSTM(n_units, return_state=True, return_sequences=True, dropout=dropout)
    encoder_outputs_lstm5 = encoder_lstm5(encoder_outputs_lstm4)
    encoder_lstm6 = LSTM(n_units, return_state=True, dropout=dropout)
    encoder_outputs_lstm6, state_h, state_c = encoder_lstm6(encoder_outputs_lstm5)
    encoder_dropout = Dropout(dropout)
    encoder_outputs_dropout = encoder_dropout(encoder_outputs_lstm6)
    encoder_states = [state_h, state_c]

    #define the decoder
    decoder_inputs = Input(shape=(output_length, features_per_timestep))

    decoder_lstm1 = LSTM(n_units, return_sequences=True, return_state=True, dropout=dropout)
    decoder_outputs_lstm1 = decoder_lstm1(decoder_inputs, initial_state=encoder_states)
    decoder_lstm2 = LSTM(n_units, return_sequences=True, return_state=True, dropout=dropout)
    decoder_outputs_lstm2 = decoder_lstm2(decoder_outputs_lstm1)
    decoder_lstm3 = LSTM(n_units, return_sequences=True, return_state=True, dropout=dropout)
    decoder_outputs_lstm3 = decoder_lstm3(decoder_outputs_lstm2)
    decoder_lstm4 = LSTM(n_units, return_sequences=True, return_state=True, dropout=dropout)
    decoder_outputs_lstm4 = decoder_lstm4(decoder_outputs_lstm3)
    decoder_lstm5 = LSTM(n_units, return_sequences=True, return_state=True, dropout=dropout)
    decoder_outputs_lstm5 = decoder_lstm5(decoder_outputs_lstm4)
    decoder_lstm6 = LSTM(n_units, return_sequences=True, return_state=True, dropout=dropout)
    decoder_outputs_lstm6, _, _ = decoder_lstm6(decoder_outputs_lstm5)
    decoder_dropout = Dropout(dropout)
    decoder_outputs_dropout = decoder_dropout(decoder_outputs_lstm6)
    decoder_dense = TimeDistributed(Dense(features_per_timestep))
    decoder_outputs = decoder_dense(decoder_outputs_dropout)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    return model, name

#try out a deep and thin architecture, no dropout inside layers
def define_model_5(features_per_timestep, input_length, output_length):
    name = 'LSTM_6u32_6u32_D5'
    n_units = 32
    dropout = 0.5

    #define the encoder
    encoder_inputs = Input(shape=(input_length, features_per_timestep))

    encoder_lstm1 = LSTM(n_units, return_state=True, return_sequences=True)
    encoder_outputs_lstm1 = encoder_lstm1(encoder_inputs)
    encoder_lstm2 = LSTM(n_units, return_state=True, return_sequences=True)
    encoder_outputs_lstm2 = encoder_lstm2(encoder_outputs_lstm1)
    encoder_lstm3 = LSTM(n_units, return_state=True, return_sequences=True)
    encoder_outputs_lstm3 = encoder_lstm3(encoder_outputs_lstm2)
    encoder_lstm4 = LSTM(n_units, return_state=True, return_sequences=True)
    encoder_outputs_lstm4 = encoder_lstm4(encoder_outputs_lstm3)
    encoder_lstm5 = LSTM(n_units, return_state=True, return_sequences=True)
    encoder_outputs_lstm5 = encoder_lstm5(encoder_outputs_lstm4)
    encoder_lstm6 = LSTM(n_units, return_state=True, dropout=dropout)
    encoder_outputs_lstm6, state_h, state_c = encoder_lstm6(encoder_outputs_lstm5)
    encoder_dropout = Dropout(dropout)
    encoder_outputs_dropout = encoder_dropout(encoder_outputs_lstm6)
    encoder_states = [state_h, state_c]

    #define the decoder
    decoder_inputs = Input(shape=(output_length, features_per_timestep))

    decoder_lstm1 = LSTM(n_units, return_sequences=True, return_state=True)
    decoder_outputs_lstm1 = decoder_lstm1(decoder_inputs, initial_state=encoder_states)
    decoder_lstm2 = LSTM(n_units, return_sequences=True, return_state=True)
    decoder_outputs_lstm2 = decoder_lstm2(decoder_outputs_lstm1)
    decoder_lstm3 = LSTM(n_units, return_sequences=True, return_state=True)
    decoder_outputs_lstm3 = decoder_lstm3(decoder_outputs_lstm2)
    decoder_lstm4 = LSTM(n_units, return_sequences=True, return_state=True)
    decoder_outputs_lstm4 = decoder_lstm4(decoder_outputs_lstm3)
    decoder_lstm5 = LSTM(n_units, return_sequences=True, return_state=True)
    decoder_outputs_lstm5 = decoder_lstm5(decoder_outputs_lstm4)
    decoder_lstm6 = LSTM(n_units, return_sequences=True, return_state=True)
    decoder_outputs_lstm6, _, _ = decoder_lstm6(decoder_outputs_lstm5)
    decoder_dropout = Dropout(dropout)
    decoder_outputs_dropout = decoder_dropout(decoder_outputs_lstm6)
    decoder_dense = TimeDistributed(Dense(features_per_timestep))
    decoder_outputs = decoder_dense(decoder_outputs_dropout)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    return model, name

#the next models are going less deep but more wide. They try out a version with dropout inside the layers and one without

def define_model_6(features_per_timestep, input_length, output_length):
    name = 'LSTM_5u64D5_5u64D5_D5'
    n_units = 64
    dropout = 0.5

    #define the encoder
    encoder_inputs = Input(shape=(input_length, features_per_timestep))

    encoder_lstm1 = LSTM(n_units, return_state=True, return_sequences=True, dropout=dropout)
    encoder_outputs_lstm1 = encoder_lstm1(encoder_inputs)
    encoder_lstm2 = LSTM(n_units, return_state=True, return_sequences=True, dropout=dropout)
    encoder_outputs_lstm2 = encoder_lstm2(encoder_outputs_lstm1)
    encoder_lstm3 = LSTM(n_units, return_state=True, return_sequences=True, dropout=dropout)
    encoder_outputs_lstm3 = encoder_lstm3(encoder_outputs_lstm2)
    encoder_lstm4 = LSTM(n_units, return_state=True, return_sequences=True, dropout=dropout)
    encoder_outputs_lstm4 = encoder_lstm4(encoder_outputs_lstm3)
    encoder_lstm5 = LSTM(n_units, return_state=True, dropout=dropout)
    encoder_outputs_lstm5, state_h, state_c = encoder_lstm5(encoder_outputs_lstm4)
    encoder_dropout = Dropout(dropout)
    encoder_outputs_dropout = encoder_dropout(encoder_outputs_lstm5)
    encoder_states = [state_h, state_c]

    #define the decoder
    decoder_inputs = Input(shape=(output_length, features_per_timestep))

    decoder_lstm1 = LSTM(n_units, return_sequences=True, return_state=True, dropout=dropout)
    decoder_outputs_lstm1 = decoder_lstm1(decoder_inputs, initial_state=encoder_states)
    decoder_lstm2 = LSTM(n_units, return_sequences=True, return_state=True, dropout=dropout)
    decoder_outputs_lstm2 = decoder_lstm2(decoder_outputs_lstm1)
    decoder_lstm3 = LSTM(n_units, return_sequences=True, return_state=True, dropout=dropout)
    decoder_outputs_lstm3 = decoder_lstm3(decoder_outputs_lstm2)
    decoder_lstm4 = LSTM(n_units, return_sequences=True, return_state=True, dropout=dropout)
    decoder_outputs_lstm4 = decoder_lstm4(decoder_outputs_lstm3)
    decoder_lstm5 = LSTM(n_units, return_sequences=True, return_state=True, dropout=dropout)
    decoder_outputs_lstm5, _, _ = decoder_lstm5(decoder_outputs_lstm4)
    decoder_dropout = Dropout(dropout)
    decoder_outputs_dropout = decoder_dropout(decoder_outputs_lstm5)
    decoder_dense = TimeDistributed(Dense(features_per_timestep))
    decoder_outputs = decoder_dense(decoder_outputs_dropout)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    return model, name

def define_model_7(features_per_timestep, input_length, output_length):
    name = 'LSTM_5u64_5u64_D5'
    n_units = 64
    dropout = 0.5

    #define the encoder
    encoder_inputs = Input(shape=(input_length, features_per_timestep))

    encoder_lstm1 = LSTM(n_units, return_state=True, return_sequences=True)
    encoder_outputs_lstm1 = encoder_lstm1(encoder_inputs)
    encoder_lstm2 = LSTM(n_units, return_state=True, return_sequences=True)
    encoder_outputs_lstm2 = encoder_lstm2(encoder_outputs_lstm1)
    encoder_lstm3 = LSTM(n_units, return_state=True, return_sequences=True)
    encoder_outputs_lstm3 = encoder_lstm3(encoder_outputs_lstm2)
    encoder_lstm4 = LSTM(n_units, return_state=True, return_sequences=True)
    encoder_outputs_lstm4 = encoder_lstm4(encoder_outputs_lstm3)
    encoder_lstm5 = LSTM(n_units, return_state=True, dropout=dropout)
    encoder_outputs_lstm5, state_h, state_c = encoder_lstm5(encoder_outputs_lstm4)
    encoder_dropout = Dropout(dropout)
    encoder_outputs_dropout = encoder_dropout(encoder_outputs_lstm5)
    encoder_states = [state_h, state_c]

    #define the decoder
    decoder_inputs = Input(shape=(output_length, features_per_timestep))

    decoder_lstm1 = LSTM(n_units, return_sequences=True, return_state=True)
    decoder_outputs_lstm1 = decoder_lstm1(decoder_inputs, initial_state=encoder_states)
    decoder_lstm2 = LSTM(n_units, return_sequences=True, return_state=True)
    decoder_outputs_lstm2 = decoder_lstm2(decoder_outputs_lstm1)
    decoder_lstm3 = LSTM(n_units, return_sequences=True, return_state=True)
    decoder_outputs_lstm3 = decoder_lstm3(decoder_outputs_lstm2)
    decoder_lstm4 = LSTM(n_units, return_sequences=True, return_state=True)
    decoder_outputs_lstm4 = decoder_lstm4(decoder_outputs_lstm3)
    decoder_lstm5 = LSTM(n_units, return_sequences=True, return_state=True)
    decoder_outputs_lstm5, _, _ = decoder_lstm5(decoder_outputs_lstm4)
    decoder_dropout = Dropout(dropout)
    decoder_outputs_dropout = decoder_dropout(decoder_outputs_lstm5)
    decoder_dense = TimeDistributed(Dense(features_per_timestep))
    decoder_outputs = decoder_dense(decoder_outputs_dropout)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    return model, name

def define_model_8(features_per_timestep, input_length, output_length):
    name = 'LSTM_4u128D5_4u128D5_D5'
    n_units = 128
    dropout = 0.5

    #define the encoder
    encoder_inputs = Input(shape=(input_length, features_per_timestep))

    encoder_lstm1 = LSTM(n_units, return_state=True, return_sequences=True, dropout=dropout)
    encoder_outputs_lstm1 = encoder_lstm1(encoder_inputs)
    encoder_lstm2 = LSTM(n_units, return_state=True, return_sequences=True, dropout=dropout)
    encoder_outputs_lstm2 = encoder_lstm2(encoder_outputs_lstm1)
    encoder_lstm3 = LSTM(n_units, return_state=True, return_sequences=True, dropout=dropout)
    encoder_outputs_lstm3 = encoder_lstm3(encoder_outputs_lstm2)
    encoder_lstm4 = LSTM(n_units, return_state=True, dropout=dropout)
    encoder_outputs_lstm4, state_h, state_c = encoder_lstm4(encoder_outputs_lstm3)
    encoder_dropout = Dropout(dropout)
    encoder_outputs_dropout = encoder_dropout(encoder_outputs_lstm4)
    encoder_states = [state_h, state_c]

    #define the decoder
    decoder_inputs = Input(shape=(output_length, features_per_timestep))

    decoder_lstm1 = LSTM(n_units, return_sequences=True, return_state=True, dropout=dropout)
    decoder_outputs_lstm1 = decoder_lstm1(decoder_inputs, initial_state=encoder_states)
    decoder_lstm2 = LSTM(n_units, return_sequences=True, return_state=True, dropout=dropout)
    decoder_outputs_lstm2 = decoder_lstm2(decoder_outputs_lstm1)
    decoder_lstm3 = LSTM(n_units, return_sequences=True, return_state=True, dropout=dropout)
    decoder_outputs_lstm3 = decoder_lstm3(decoder_outputs_lstm2)
    decoder_lstm4 = LSTM(n_units, return_sequences=True, return_state=True, dropout=dropout)
    decoder_outputs_lstm4, _, _ = decoder_lstm4(decoder_outputs_lstm3)
    decoder_dropout = Dropout(dropout)
    decoder_outputs_dropout = decoder_dropout(decoder_outputs_lstm4)
    decoder_dense = TimeDistributed(Dense(features_per_timestep))
    decoder_outputs = decoder_dense(decoder_outputs_dropout)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    return model, name

def define_model_9(features_per_timestep, input_length, output_length):
    name = 'LSTM_4u128_4u128_D5'
    n_units = 128
    dropout = 0.5

    #define the encoder
    encoder_inputs = Input(shape=(input_length, features_per_timestep))

    encoder_lstm1 = LSTM(n_units, return_state=True, return_sequences=True)
    encoder_outputs_lstm1 = encoder_lstm1(encoder_inputs)
    encoder_lstm2 = LSTM(n_units, return_state=True, return_sequences=True)
    encoder_outputs_lstm2 = encoder_lstm2(encoder_outputs_lstm1)
    encoder_lstm3 = LSTM(n_units, return_state=True, return_sequences=True)
    encoder_outputs_lstm3 = encoder_lstm3(encoder_outputs_lstm2)
    encoder_lstm4 = LSTM(n_units, return_state=True, dropout=dropout)
    encoder_outputs_lstm4, state_h, state_c = encoder_lstm4(encoder_outputs_lstm3)
    encoder_dropout = Dropout(dropout)
    encoder_outputs_dropout = encoder_dropout(encoder_outputs_lstm4)
    encoder_states = [state_h, state_c]

    #define the decoder
    decoder_inputs = Input(shape=(output_length, features_per_timestep))

    decoder_lstm1 = LSTM(n_units, return_sequences=True, return_state=True)
    decoder_outputs_lstm1 = decoder_lstm1(decoder_inputs, initial_state=encoder_states)
    decoder_lstm2 = LSTM(n_units, return_sequences=True, return_state=True)
    decoder_outputs_lstm2 = decoder_lstm2(decoder_outputs_lstm1)
    decoder_lstm3 = LSTM(n_units, return_sequences=True, return_state=True)
    decoder_outputs_lstm3 = decoder_lstm3(decoder_outputs_lstm2)
    decoder_lstm4 = LSTM(n_units, return_sequences=True, return_state=True)
    decoder_outputs_lstm4, _, _ = decoder_lstm4(decoder_outputs_lstm3)
    decoder_dropout = Dropout(dropout)
    decoder_outputs_dropout = decoder_dropout(decoder_outputs_lstm4)
    decoder_dense = TimeDistributed(Dense(features_per_timestep))
    decoder_outputs = decoder_dense(decoder_outputs_dropout)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    return model, name

def define_model_10(features_per_timestep, input_length, output_length):
    name = 'LSTM_3u256D5_4u256D5_D5'
    n_units = 256
    dropout = 0.5

    #define the encoder
    encoder_inputs = Input(shape=(input_length, features_per_timestep))

    encoder_lstm1 = LSTM(n_units, return_state=True, return_sequences=True, dropout=dropout)
    encoder_outputs_lstm1 = encoder_lstm1(encoder_inputs)
    encoder_lstm2 = LSTM(n_units, return_state=True, return_sequences=True, dropout=dropout)
    encoder_outputs_lstm2 = encoder_lstm2(encoder_outputs_lstm1)
    encoder_lstm3 = LSTM(n_units, return_state=True, dropout=dropout)
    encoder_outputs_lstm3, state_h, state_c = encoder_lstm3(encoder_outputs_lstm2)
    encoder_dropout = Dropout(dropout)
    encoder_outputs_dropout = encoder_dropout(encoder_outputs_lstm3)
    encoder_states = [state_h, state_c]

    #define the decoder
    decoder_inputs = Input(shape=(output_length, features_per_timestep))

    decoder_lstm1 = LSTM(n_units, return_sequences=True, return_state=True, dropout=dropout)
    decoder_outputs_lstm1 = decoder_lstm1(decoder_inputs, initial_state=encoder_states)
    decoder_lstm2 = LSTM(n_units, return_sequences=True, return_state=True, dropout=dropout)
    decoder_outputs_lstm2 = decoder_lstm2(decoder_outputs_lstm1)
    decoder_lstm3 = LSTM(n_units, return_sequences=True, return_state=True, dropout=dropout)
    decoder_outputs_lstm3, _, _ = decoder_lstm3(decoder_outputs_lstm2)
    decoder_dropout = Dropout(dropout)
    decoder_outputs_dropout = decoder_dropout(decoder_outputs_lstm3)
    decoder_dense = TimeDistributed(Dense(features_per_timestep))
    decoder_outputs = decoder_dense(decoder_outputs_dropout)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    return model, name

def define_model_11(features_per_timestep, input_length, output_length):
    name = 'LSTM_3u256_4u256_D5'
    n_units = 256
    dropout = 0.5

    #define the encoder
    encoder_inputs = Input(shape=(input_length, features_per_timestep))

    encoder_lstm1 = LSTM(n_units, return_state=True, return_sequences=True)
    encoder_outputs_lstm1 = encoder_lstm1(encoder_inputs)
    encoder_lstm2 = LSTM(n_units, return_state=True, return_sequences=True)
    encoder_outputs_lstm2 = encoder_lstm2(encoder_outputs_lstm1)
    encoder_lstm3 = LSTM(n_units, return_state=True, dropout=dropout)
    encoder_outputs_lstm3, state_h, state_c = encoder_lstm3(encoder_outputs_lstm2)
    encoder_dropout = Dropout(dropout)
    encoder_outputs_dropout = encoder_dropout(encoder_outputs_lstm3)
    encoder_states = [state_h, state_c]

    #define the decoder
    decoder_inputs = Input(shape=(output_length, features_per_timestep))

    decoder_lstm1 = LSTM(n_units, return_sequences=True, return_state=True)
    decoder_outputs_lstm1 = decoder_lstm1(decoder_inputs, initial_state=encoder_states)
    decoder_lstm2 = LSTM(n_units, return_sequences=True, return_state=True)
    decoder_outputs_lstm2 = decoder_lstm2(decoder_outputs_lstm1)
    decoder_lstm3 = LSTM(n_units, return_sequences=True, return_state=True)
    decoder_outputs_lstm3, _, _ = decoder_lstm3(decoder_outputs_lstm2)
    decoder_dropout = Dropout(dropout)
    decoder_outputs_dropout = decoder_dropout(decoder_outputs_lstm3)
    decoder_dense = TimeDistributed(Dense(features_per_timestep))
    decoder_outputs = decoder_dense(decoder_outputs_dropout)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    return model, name

#2 layers with 512 units are actually network 1 and 2

def define_model_12(features_per_timestep, input_length, output_length):
    name = 'LSTM_1u1024D5_1u1024D5_D5'
    n_units = 1024
    dropout = 0.5

    #define the encoder
    encoder_inputs = Input(shape=(input_length, features_per_timestep))

    encoder_lstm1 = LSTM(n_units, return_state=True, dropout=dropout)
    encoder_outputs_lstm1, state_h, state_c = encoder_lstm1(encoder_inputs)
    encoder_dropout = Dropout(dropout)
    encoder_outputs_dropout = encoder_dropout(encoder_outputs_lstm1)
    encoder_states = [state_h, state_c]

    #define the decoder
    decoder_inputs = Input(shape=(output_length, features_per_timestep))

    decoder_lstm1 = LSTM(n_units, return_sequences=True, return_state=True, dropout=dropout)
    decoder_outputs_lstm1, _, _ = decoder_lstm1(decoder_inputs, initial_state=encoder_states)
    decoder_dropout = Dropout(dropout)
    decoder_outputs_dropout = decoder_dropout(decoder_outputs_lstm1)
    decoder_dense = TimeDistributed(Dense(features_per_timestep))
    decoder_outputs = decoder_dense(decoder_outputs_dropout)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    return model, name

def define_model_13(features_per_timestep, input_length, output_length):
    name = 'LSTM_1u1024_1u1024_D5'
    n_units = 1024
    dropout = 0.5

    #define the encoder
    encoder_inputs = Input(shape=(input_length, features_per_timestep))

    encoder_lstm1 = LSTM(n_units, return_state=True)
    encoder_outputs_lstm1, state_h, state_c = encoder_lstm1(encoder_inputs)
    encoder_dropout = Dropout(dropout)
    encoder_outputs_dropout = encoder_dropout(encoder_outputs_lstm1)
    encoder_states = [state_h, state_c]

    #define the decoder
    decoder_inputs = Input(shape=(output_length, features_per_timestep))

    decoder_lstm1 = LSTM(n_units, return_sequences=True, return_state=True)
    decoder_outputs_lstm1, _, _ = decoder_lstm1(decoder_inputs, initial_state=encoder_states)
    decoder_dropout = Dropout(dropout)
    decoder_outputs_dropout = decoder_dropout(decoder_outputs_lstm1)
    decoder_dense = TimeDistributed(Dense(features_per_timestep))
    decoder_outputs = decoder_dense(decoder_outputs_dropout)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    return model, name

#Rise up! Fall Down
#TODO fix bug in model14
def define_model_14(features_per_timestep, input_length, output_length):
    name = 'LSTM_4u32to256_4u256to32_D5'
    dropout = 0.5

    #define the encoder
    encoder_inputs = Input(shape=(input_length, features_per_timestep))

    n_units = 32
    encoder_lstm1 = LSTM(n_units, return_state=True, return_sequences=True)
    encoder_outputs_lstm1 = encoder_lstm1(encoder_inputs)
    n_units = 64
    encoder_lstm2 = LSTM(n_units, return_state=True, return_sequences=True)
    encoder_outputs_lstm2 = encoder_lstm2(encoder_outputs_lstm1)
    n_units = 128
    encoder_lstm3 = LSTM(n_units, return_state=True, return_sequences=True)
    encoder_outputs_lstm3 = encoder_lstm3(encoder_outputs_lstm2)
    n_units = 256
    encoder_lstm4 = LSTM(n_units, return_state=True, dropout=dropout)
    encoder_outputs_lstm4, state_h, state_c = encoder_lstm4(encoder_outputs_lstm3)
    encoder_dropout = Dropout(dropout)
    encoder_outputs_dropout = encoder_dropout(encoder_outputs_lstm4)
    encoder_states = [state_h, state_c]

    #define the decoder
    decoder_inputs = Input(shape=(output_length, features_per_timestep))

    n_units = 256
    decoder_lstm1 = LSTM(n_units, return_sequences=True, return_state=True)
    decoder_outputs_lstm1 = decoder_lstm1(decoder_inputs, initial_state=encoder_states)
    n_units = 128
    decoder_lstm2 = LSTM(n_units, return_sequences=True, return_state=True)
    decoder_outputs_lstm2 = decoder_lstm2(decoder_outputs_lstm1)
    n_units = 64
    decoder_lstm3 = LSTM(n_units, return_sequences=True, return_state=True)
    decoder_outputs_lstm3 = decoder_lstm3(decoder_outputs_lstm2)
    n_units = 32
    decoder_lstm4 = LSTM(n_units, return_sequences=True, return_state=True)
    decoder_outputs_lstm4, _, _ = decoder_lstm4(decoder_outputs_lstm3)
    decoder_dropout = Dropout(dropout)
    decoder_outputs_dropout = decoder_dropout(decoder_outputs_lstm4)
    decoder_dense = TimeDistributed(Dense(features_per_timestep))
    decoder_outputs = decoder_dense(decoder_outputs_dropout)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    return model, name

#very small model
def define_model_15(features_per_timestep, input_length, output_length):
    name = 'LSTM_1u128_1u128_D5'
    n_units = 128
    dropout = 0.5

    #define the encoder
    encoder_inputs = Input(shape=(input_length, features_per_timestep))

    encoder_lstm1 = LSTM(n_units, return_state=True)
    encoder_outputs_lstm1, state_h, state_c = encoder_lstm1(encoder_inputs)
    encoder_dropout = Dropout(dropout)
    encoder_outputs_dropout = encoder_dropout(encoder_outputs_lstm1)
    encoder_states = [state_h, state_c]

    #define the decoder
    decoder_inputs = Input(shape=(output_length, features_per_timestep))

    decoder_lstm1 = LSTM(n_units, return_sequences=True, return_state=True)
    decoder_outputs_lstm1, _, _ = decoder_lstm1(decoder_inputs, initial_state=encoder_states)
    decoder_dropout = Dropout(dropout)
    decoder_outputs_dropout = decoder_dropout(decoder_outputs_lstm1)
    decoder_dense = TimeDistributed(Dense(features_per_timestep))
    decoder_outputs = decoder_dense(decoder_outputs_dropout)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    return model, name