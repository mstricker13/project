from kerasImpl import data

from numpy import argmax

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

    # model = Sequential()
    # model.add(Embedding(src_vocab, n_units, input_length=src_timesteps, mask_zero=True))
    # model.add(LSTM(n_units))
    # model.add(RepeatVector(tar_timesteps))
    # model.add(LSTM(n_units, return_sequences=True))
    # model.add(TimeDistributed(Dense(tar_vocab, activation='softmax')))

    # return model

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
    #TODO
    n_input = src_timesteps
    n_output = tar_timesteps
    # define training encoder
    encoder_inputs = Input(shape=(None, n_input))
    x = Embedding(src_vocab, latent_dim)(encoder_inputs)
    encoder_outputs, state_h, state_c = LSTM(n_units, return_state=True)(x)
    #encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]
    # define training decoder
    decoder_inputs = Input(shape=(None, n_output))
    decoder_lstm = LSTM(n_units, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(n_output, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    # define inference encoder
    encoder_model = Model(encoder_inputs, encoder_states)
    # define inference decoder
    decoder_state_input_h = Input(shape=(n_units,))
    decoder_state_input_c = Input(shape=(n_units,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
    # return all models
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
