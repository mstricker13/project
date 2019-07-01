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
from keras.layers import Add
from keras import regularizers
from keras.initializers import glorot_uniform
from keras.layers.normalization import BatchNormalization
from keras.layers import Flatten
from keras.callbacks import ModelCheckpoint
from nltk.translate.bleu_score import corpus_bleu
import numpy as np
import random

def define_model_simple1(features_per_timestep, input_length, output_length):
    name = 'Simple5_3'
    n_units = 8
    dropout = 0.1

    #define the encoder
    encoder_inputs = Input(shape=(input_length, features_per_timestep))

    encoder_lstm1 = LSTM(n_units, return_state=True, return_sequences=True, dropout=dropout)
    encoder_outputs_lstm1, state_h, state_c = encoder_lstm1(encoder_inputs)
    #encoder_lstm2 = LSTM(n_units, return_state=True, dropout=dropout, activation='relu')
    #encoder_outputs_lstm2, state_h, state_c = encoder_lstm2(encoder_outputs_lstm1)
    #encoder_dropout = Dropout(dropout)
    #encoder_outputs_dropout = encoder_dropout(encoder_outputs_lstm2)
    #repeater = RepeatVector(output_length)
    #encoder_outputs = repeater(encoder_outputs_lstm1)
    encoder_states = [state_h, state_c]

    #define the decoder
    decoder_inputs = Input(shape=(output_length, features_per_timestep))

    #internal_rep_add = Add()
    #internal_rep = internal_rep_add([encoder_outputs, decoder_inputs])

    decoder_lstm1 = LSTM(n_units, return_sequences=True, return_state=True, dropout=dropout)
    decoder_outputs_lstm1, _, _ = decoder_lstm1(decoder_inputs, initial_state=encoder_states)
    #decoder_lstm2 = LSTM(n_units, return_sequences=True, return_state=True, dropout=dropout, activation='relu')
    #decoder_outputs_lstm2, _, _ = decoder_lstm2(decoder_outputs_lstm1)
    #decoder_dropout = Dropout(dropout)
    #decoder_outputs_dropout = decoder_dropout(decoder_outputs_lstm2)

    decoder_dense = TimeDistributed(Dense(features_per_timestep))
    decoder_outputs_value = decoder_dense(decoder_outputs_lstm1)

    decoder_add = Add()
    decoder_outputs = decoder_add([decoder_outputs_value, decoder_inputs])

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    return model, name

def define_model_2l_simple(features_per_timestep, input_length, output_length):
    name = 'simpleFINAL'
    n_units = 16
    dropout = 0.1

    #define the encoder
    encoder_inputs = Input(shape=(input_length, features_per_timestep))

    encoder_lstm1 = LSTM(n_units, return_state=True, return_sequences=True, dropout=dropout) #, activation='relu'
    encoder_outputs_lstm1 = encoder_lstm1(encoder_inputs)
    encoder_lstm2 = LSTM(n_units, return_state=True, dropout=dropout) #, activation='relu'
    encoder_outputs_lstm2, state_h, state_c = encoder_lstm2(encoder_outputs_lstm1)
    #encoder_dropout = Dropout(dropout)
    #encoder_outputs_dropout = encoder_dropout(encoder_outputs_lstm2)
    encoder_states = [state_h, state_c]

    #define the decoder
    decoder_inputs = Input(shape=(output_length, features_per_timestep))

    decoder_lstm1 = LSTM(n_units, return_sequences=True, return_state=True, dropout=dropout) #, activation='relu' , activity_regularizer=regularizers.l2(0.03)
    decoder_outputs_lstm1 = decoder_lstm1(decoder_inputs, initial_state=encoder_states)
    decoder_lstm2 = LSTM(n_units, return_sequences=True, return_state=True, dropout=dropout) #, activation='relu'
    decoder_outputs_lstm2, _, _ = decoder_lstm2(decoder_outputs_lstm1)
    #decoder_dropout = Dropout(dropout)
    #decoder_outputs_dropout = decoder_dropout(decoder_outputs_lstm2)

    decoder_dense = TimeDistributed(Dense(features_per_timestep))
    decoder_outputs_value = decoder_dense(decoder_outputs_lstm2)

    decoder_add = Add()
    decoder_outputs = decoder_add([decoder_outputs_value, decoder_inputs])

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    return model, name

def define_model_2lN_simple(features_per_timestep, input_length, output_length):
    name = 'simple6N_1'
    n_units = 64
    dropout = 0.5
    activation = 'tanh'

    #define the encoder
    encoder_inputs = Input(shape=(input_length, features_per_timestep))

    encoder_lstm1 = LSTM(n_units, return_state=True, return_sequences=True, dropout=dropout, activation=activation) #, activation='relu', kernel_initializer=glorot_uniform(1)
    encoder_outputs_lstm1 = encoder_lstm1(encoder_inputs)
    encoder_lstm2 = LSTM(n_units, return_state=True, return_sequences=True, dropout=dropout, activation=activation) #, activation='relu', kernel_initializer=glorot_uniform(1)
    encoder_outputs_lstm2 = encoder_lstm2(encoder_outputs_lstm1)
    encoder_lstm3 = LSTM(n_units, return_state=True, dropout=dropout, activation=activation)  # , activation='relu', kernel_initializer=glorot_uniform(1)
    encoder_outputs_lstm3, state_h, state_c = encoder_lstm3(encoder_outputs_lstm2)
    #encoder_dropout = Dropout(dropout)
    #encoder_outputs_dropout = encoder_dropout(encoder_outputs_lstm2)
    encoder_states = [state_h, state_c]

    #define the decoder
    decoder_inputs = Input(shape=(output_length, features_per_timestep))

    decoder_lstm1 = LSTM(n_units, return_sequences=True, return_state=True, dropout=dropout, activation=activation, activity_regularizer=regularizers.l2(0.03)) #, activation='relu', kernel_initializer=glorot_uniform(1)
    decoder_outputs_lstm1 = decoder_lstm1(decoder_inputs, initial_state=encoder_states)
    decoder_lstm2 = LSTM(n_units, return_sequences=True, return_state=True, dropout=dropout, activation=activation) #, activation='relu', kernel_initializer=glorot_uniform(1)
    decoder_outputs_lstm2 = decoder_lstm2(decoder_outputs_lstm1)
    decoder_lstm3 = LSTM(n_units, return_sequences=True, return_state=True, dropout=dropout, activation=activation)  # , activation='relu', kernel_initializer=glorot_uniform(1)
    decoder_outputs_lstm3, _, _ = decoder_lstm3(decoder_outputs_lstm2)
    #decoder_dropout = Dropout(dropout)
    #decoder_outputs_dropout = decoder_dropout(decoder_outputs_lstm2)

    decoder_dense = TimeDistributed(Dense(features_per_timestep))
    decoder_outputs_value = decoder_dense(decoder_outputs_lstm3)

    decoder_add = Add()
    decoder_outputs = decoder_add([decoder_outputs_value, decoder_inputs])

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    return model, name