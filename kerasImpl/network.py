from kerasImpl import data

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
from keras.callbacks import ModelCheckpoint
from nltk.translate.bleu_score import corpus_bleu
import numpy as np
import random

# define NMT model
def define_model(src_vocab, tar_vocab, src_timesteps, tar_timesteps, n_units, latent_dim, dropout):
    # latent_dim = 256
    # Define an input sequence and process it.
    # encoder_inputs = Input(shape=(None,))
    # x = Embedding(src_vocab, latent_dim)(encoder_inputs)
    # x, state_h, state_c = LSTM(n_units, return_state=True)(x)
    # encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state.
    # decoder_inputs = Input(shape=(None,))
    # x = Embedding(tar_vocab, latent_dim)(decoder_inputs)
    # x = LSTM(n_units, return_sequences=True)(x, initial_state=encoder_states)
    # decoder_outputs = Dense(tar_vocab, activation='softmax')(x)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    # model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    # return model

    #model = Sequential()
    #model.add(Embedding(src_vocab, n_units, input_length=src_timesteps, mask_zero=True))
    #model.add(LSTM(n_units))
    #model.add(RepeatVector(tar_timesteps))
    #model.add(LSTM(n_units, return_sequences=True))
    #model.add(TimeDistributed(Dense(tar_vocab, activation='softmax')))
    #return model

    model = Sequential()
    model.add(Embedding(src_vocab, latent_dim, input_length=src_timesteps, mask_zero=True))
    model.add(LSTM(n_units, return_sequences=True, dropout=dropout))
    model.add(LSTM(n_units, dropout=dropout))
    model.add(Dropout(dropout))
    model.add(RepeatVector(tar_timesteps))
    # model.add(Embedding(tar_vocab, latent_dim, mask_zero=True))
    model.add(LSTM(n_units, return_sequences=True, dropout=dropout))
    model.add(LSTM(n_units, return_sequences=True, dropout=dropout))
    model.add(Dropout(dropout))
    model.add(Dense(tar_vocab, activation='softmax'))
    return model


def define_model_powerful(src_vocab, tar_vocab, src_timesteps, tar_timesteps, n_units, latent_dim, dropout):
    #TODO rename layers and outputs to unique identifiers
    #define the encoder
    encoder_inputs = Input(shape=(src_timesteps,))#Input(shape=(None, src_vocab)) + in data.py add the functionality,
    # to get the one-hot encoded version of trainX, if I don't want to use embedding

    encoder_embedding = Embedding(src_vocab, latent_dim, input_length=src_timesteps, mask_zero=True)
    encoder_embedding = encoder_embedding(encoder_inputs)

    encoder_lstm1 = LSTM(n_units, return_state=True, return_sequences=True, dropout=dropout)
    #encoder_outputs, state_h, state_c = encoder_lstm1(encoder_inputs)
    lstm_output = encoder_lstm1(encoder_embedding) #encoder_inputs
    encoder_lstm2 = LSTM(n_units, return_state=True, dropout=dropout)
    encoder_outputs, state_h, state_c = encoder_lstm2(lstm_output)
    encoder_dropout = Dropout(dropout)
    encoder_outputs = encoder_dropout(encoder_outputs)
    encoder_states = [state_h, state_c]

    #define the decoder
    decoder_inputs = Input(shape=(None, tar_vocab))
    decoder_lstm1 = LSTM(n_units, return_sequences=True, return_state=True, dropout=dropout)
    lstm_output = decoder_lstm1(decoder_inputs, initial_state=encoder_states)
    decoder_lstm2 = LSTM(n_units, return_sequences=True, return_state=True, dropout=dropout)
    decoder_outputs, _, _ = decoder_lstm2(lstm_output)
    decoder_dropout = Dropout(dropout)
    decoder_outputs = decoder_dropout(decoder_outputs)
    decoder_dense = Dense(tar_vocab, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    # define inference encoder
    encoder_model = Model(encoder_inputs, encoder_states)

    # define inference decoder
    decoder_state_input_h = Input(shape=(n_units,))
    decoder_state_input_c = Input(shape=(n_units,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    temporary_output = decoder_lstm1(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_outputs, state_h, state_c = decoder_lstm2(temporary_output)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

    return model, encoder_model, decoder_model

# generate target given source sequence
def predict_sequence(model, tokenizer, source):
    prediction = model.predict(source, verbose=0)[0]
    integers = [argmax(vector) for vector in prediction]
    target = list()
    for i in integers:
        word = data.word_for_id(i, tokenizer)
        if word is None:
            break
        target.append(word)
    return ' '.join(target)

# evaluate the skill of the model
def evaluate_model(model, tokenizer, sources, raw_dataset):
    actual, predicted = list(), list()
    for i, source in enumerate(sources):
        # translate encoded source text
        source = source.reshape((1, source.shape[0]))
        translation = predict_sequence(model, tokenizer, source)
        raw_target, raw_src = raw_dataset[i]
        if i < 10:
            print('src=[%s], target=[%s], predicted=[%s]' % (raw_src, raw_target, translation))
        actual.append(raw_target.split())
        predicted.append(translation.split())
    # calculate BLEU score
    print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
    print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
    print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))


# make architecture same
# failed add teacher enforcing
#evaluate the same

# generate a sequence of random integers
def generate_sequence(length, n_unique):
    return [randint(1, n_unique - 1) for _ in range(length)]

# prepare data for the LSTM
def get_dataset(n_in, n_out, cardinality, cardinality2, n_samples):
    X1, X2, y = list(), list(), list()
    for _ in range(n_samples):
        # generate source sequence
        source = generate_sequence(n_in, cardinality)
        # define padded target sequence
        target = generate_sequence(n_out, cardinality2)
        target.reverse()
        # create padded input target sequence
        target_in = [0] + target[:-1]
        # encode
        src_encoded = to_categorical([source], num_classes=cardinality)
        tar_encoded = to_categorical([target], num_classes=cardinality2)
        tar2_encoded = to_categorical([target_in], num_classes=cardinality2)
        # store
        X1.append(src_encoded)
        X2.append(tar2_encoded)
        y.append(tar_encoded)
    X1 = np.squeeze(array(X1), axis=1)
    X2 = np.squeeze(array(X2), axis=1)
    y = np.squeeze(array(y), axis=1)
    return X1, X2, y