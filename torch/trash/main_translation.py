from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import training, network, data

import warnings

import os
import time
import math
import sys

def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    path = os.path.join('data', 'cif_one.csv')  
    window = 7
    horizon = 1
    train_split = 0.7
    teacher_forcing_ratio = 0.5

    sequences = load(path)
    sequence = sequences[0][3:]
    #print(sequences)


    #for sequence in sequences:
    x, y = make_samples(sequence, window, horizon)
    #TODO need to remove overlap in testset
    #TODO normalize correctly
    train, val, test = split(x, y, train_split)
    train, val, test, mean, std = normalize(train, val, test)

    #print(train[0][0])
    #print(train[0].shape)
    x_train = torch.from_numpy(train[0]).type(torch.Tensor)#.view([window, -1, 1])
    y_train = torch.from_numpy(train[1]).type(torch.Tensor)
    x_val = torch.from_numpy(val[0]).type(torch.Tensor)
    y_val = torch.from_numpy(val[1]).type(torch.Tensor)
    x_test = torch.from_numpy(test[0]).type(torch.Tensor)
    y_test = torch.from_numpy(test[1]).type(torch.Tensor)
    print(x_train.size())
    print(y_train.size())
    print(x_test.size())
    print(y_test.size())
    #sys.exit()
    hidden_size = 256
    encoder1 = EncoderRNN(x_train.size(1), hidden_size)#.to(device)
    decoder1 = DecoderRNN(hidden_size, y_train.size(1))#.to(device)


    training = (x_train, y_train)
    #print(x_train.view(-1,1).size())
    #print(y_train.view(-1,1).size())
    #sys.exit()
    trainIters(encoder1, decoder1, training, print_every=10)
    

def load(path):

    sequences = list()
    with open(path) as f:
        content = f.read()
    for line in content.split('\n'):
        if not (line == ''):
            values = line.split(',')
            values = [values[0]] + [int(values[1])] + [values[2]] + [float(val) for val in values[3:]]
            sequences.append(values)
    return sequences


def make_samples(sequence, window, horizon):
    x = list()
    y = list()
    for i in range(len(sequence)-(window+horizon-1)):
        x_tmp = sequence[i:(i+window)]
        y_tmp = sequence[(i+window):(i+window+horizon)]
        x.append(x_tmp)
        y.append(y_tmp)
    return np.array(x), np.array(y)  


def split(x, y, train_split):
    index = int((len(x)-1)*train_split)
    train = [x[:index], y[:index]]
    val = [x[index:-1], y[index:-1]]
    test = [x[-1:], y[-1:]]
    return train, val, test


def normalize(train, val, test):
    mean = np.mean(train[0])
    std = np.mean(train[0])
    train[0] = (train[0]-mean)/std
    train[1] = (train[1]-mean)/std
    val[0] = (val[0]-mean)/std
    val[1] = (val[1]-mean)/std
    test[0] = (test[0]-mean)/std
    test[1] = (test[1]-mean)/std
    return train, val, test, mean, std


######################################################################
# The Encoder
# -----------
#
# The encoder of a seq2seq network is a RNN that outputs some value for
# every word from the input sentence. For every input word the encoder
# outputs a vector and a hidden state, and uses the hidden state for the
# next input word.
#
# .. figure:: /_static/img/seq-seq-images/encoder-network.png
#    :alt:
#
#

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        #self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(input_size, hidden_size)

    def forward(self, input, hidden):
        #embedded = self.embedding(input).view(1, 1, -1)
        #output = embedded
        output, hidden = self.gru(input, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)#, device=device)

######################################################################
# The Decoder
# -----------
#
# The decoder is another RNN that takes the encoder output vector(s) and
# outputs a sequence of words to create the translation.
#


######################################################################
# Simple Decoder
# ^^^^^^^^^^^^^^
#
# In the simplest seq2seq decoder we use only last output of the encoder.
# This last output is sometimes called the *context vector* as it encodes
# context from the entire sequence. This context vector is used as the
# initial hidden state of the decoder.
#
# At every step of decoding, the decoder is given an input token and
# hidden state. The initial input token is the start-of-string ``<SOS>``
# token, and the first hidden state is the context vector (the encoder's
# last hidden state).
#
# .. figure:: /_static/img/seq-seq-images/decoder-network.png
#    :alt:
#
#

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        #self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        #output = self.embedding(input).view(1, 1, -1)
        output = F.relu(input)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)#, device=device)


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size)#, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]])#, device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


######################################################################
# The whole training process looks like this:
#
# -  Start a timer
# -  Initialize optimizers and criterion
# -  Create set of training pairs
# -  Start empty losses array for plotting
#
# Then we call ``train`` many times and occasionally print the progress (%
# of examples, time so far, estimated time) and average loss.
#

def trainIters(encoder, decoder, train_set, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = train_set
    n_iters = training_pairs[0].size(0)

    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion, training_pairs[0].size(1))
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))


def evaluate(encoder, decoder, sentence, max_length):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size)#, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]])#, device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]


if __name__ == '__main__':
    main()
