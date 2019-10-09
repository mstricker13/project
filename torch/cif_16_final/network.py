import torch
import torch.nn as nn
from torch.nn import functional as F
import NRU

import random

import sys

#set random seed to get deterministc results
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


class Encoder(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout = dropout

        self.rnn = nn.LSTM(input_dim, hid_dim[0], n_layers, dropout=dropout)
        self.rnn2 = nn.LSTM(hid_dim[0], hid_dim[1], n_layers, dropout=dropout)

        #self.dropout = nn.Dropout(dropout)

    def forward(self, src):

        # print(src)
        outputs, (hidden, cell) = self.rnn(src)
        outputs, (hidden, cell) = self.rnn2(F.relu(outputs))

        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dropout = dropout

        self.rnn = nn.LSTM(output_dim, hid_dim[1], n_layers, dropout=dropout)
        self.rnn2 = nn.LSTM(hid_dim[1], hid_dim[0], n_layers, dropout=dropout)

        self.out = nn.Linear(hid_dim[0], output_dim)

        #self.dropout = nn.Dropout(dropout)

    def forward(self, input, expert, hidden, cell):

        input = input.unsqueeze(0)
        expert = expert.unsqueeze(0)

        output, (hidden_1, cell_1) = self.rnn(expert, (hidden, cell))
        output, (hidden, cell) = self.rnn2(F.relu(output))

        prediction = self.out(F.relu(output).squeeze(0))

        #TODO change for expert
        result = torch.add(prediction, expert)
        result = prediction.reshape(result.size())

        return result, hidden_1, cell_1


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.hid_dim == decoder.hid_dim, "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, "Encoder and decoder must have equal number of layers!"

    def forward(self, src, trg, expert):
        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)

        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src)
        hidden_o, cell_o = hidden, cell

        # first input to the decoder is the <sos> tokens
        no_expert = src[-1]
        # print(no_expert)
        # print(src)
        # print(no_expert)

        #input = (trg)
        #exp = (expert)
        #output, hidden_o, cell_o = self.decoder(input, exp, hidden, cell)
        #outputs = output

        for t in range(max_len):
            # TODO check if correct or all at once?!
            input = (trg[t])
            exp = (expert[t])
            # TODO use previous cellstates?
            # TODO change for expert
            # output, hidden_o, cell_o = self.decoder(input, exp, hidden, cell)
            # output, hidden_o, cell_o = self.decoder(input, exp, hidden_o, cell_o)
            output, hidden_o, cell_o = self.decoder(input, no_expert, hidden, cell)
            # top1 = output.max(1)[1].view(batch_size, trg_vocab_size).double()
            no_expert = output[0]
            outputs[t] = output

        return outputs


class Seq2Seq_wholeloop(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.hid_dim == decoder.hid_dim, "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, "Encoder and decoder must have equal number of layers!"

    def forward(self, src, trg, expert):
        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)

        # last hidden state of the encoder is used as the initial hidden state of the decoder
        # hidden, cell = self.encoder(src)
        # hidden_o, cell_o = hidden, cell

        # first input to the decoder is the <sos> tokens
        no_expert = src[-1]
        # print(no_expert)
        # print(src)
        # print(no_expert)

        #input = (trg)
        #exp = (expert)
        #output, hidden_o, cell_o = self.decoder(input, exp, hidden, cell)
        #outputs = output

        for t in range(max_len):
            hidden, cell = self.encoder(src)
            hidden_o, cell_o = hidden, cell

            # TODO check if correct or all at once?!
            input = (trg[t])
            exp = (expert[t])
            # TODO use previous cellstates?
            # TODO change for expert
            output, hidden_o, cell_o = self.decoder(input, exp, hidden, cell)
            # output, hidden_o, cell_o = self.decoder(input, exp, hidden_o, cell_o)
            # output, hidden_o, cell_o = self.decoder(input, no_expert, hidden, cell)
            # top1 = output.max(1)[1].view(batch_size, trg_vocab_size).double()
            no_expert = output[0]
            outputs[t] = output
            # remove first entry from src and add output as new last entry
            src = torch.cat((src[1:], output), 0)

        return outputs


class Encoder_gru(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout = dropout

        self.rnn = nn.GRU(input_dim, hid_dim[0], n_layers, dropout=dropout)
        self.rnn2 = nn.GRU(hid_dim[0], hid_dim[1], n_layers, dropout=dropout)

        #self.dropout = nn.Dropout(dropout)

    def forward(self, src):

        outputs, hidden = self.rnn(src)
        outputs, hidden = self.rnn2(F.relu(outputs))

        return hidden


class Decoder_gru(nn.Module):
    def __init__(self, output_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dropout = dropout

        self.rnn = nn.GRU(output_dim, hid_dim[1], n_layers, dropout=dropout)
        self.rnn2 = nn.GRU(hid_dim[1], hid_dim[0], n_layers, dropout=dropout)

        self.out = nn.Linear(hid_dim[0], output_dim)

        #self.dropout = nn.Dropout(dropout)

    def forward(self, input, expert, hidden):

        input = input.unsqueeze(0)
        expert = expert.unsqueeze(0)

        output, hidden = self.rnn(expert, hidden)
        output, hidden = self.rnn2(F.relu(output))

        prediction = self.out(F.relu(output).squeeze(0))

        result = prediction

        # TODO change for expert
        result = torch.add(prediction, expert)
        result = prediction.reshape(result.size())

        return result, hidden


class Seq2Seq_gru(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.hid_dim == decoder.hid_dim, "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, "Encoder and decoder must have equal number of layers!"

    def forward(self, src, trg, expert):
        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)

        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden = self.encoder(src)

        # first input to the decoder is the <sos> tokens
        no_expert = src[-1]
        # print(no_expert)
        # print(src)
        # print(no_expert)

        for t in range(max_len):
            # TODO check if correct or all at once?!
            input = (trg[t])
            exp = (expert[t])
            # TODO use previous cellstates?
            #output, hidden_o = self.decoder(input, exp, hidden)
            output, hidden_o = self.decoder(input, no_expert, hidden)
            # top1 = output.max(1)[1].view(batch_size, trg_vocab_size).double()
            no_expert = output[0]
            outputs[t] = output

        return outputs


class Encoder_all(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout = dropout

        self.rnn = nn.LSTM(input_dim, hid_dim[0], n_layers, dropout=dropout)
        self.rnn2 = nn.LSTM(hid_dim[0], hid_dim[1], n_layers, dropout=dropout)

        #self.dropout = nn.Dropout(dropout)

    def forward(self, src):

        outputs, (hidden, cell) = self.rnn(src)
        outputs, (hidden, cell) = self.rnn2(F.relu(outputs))

        return hidden, cell


class Decoder_all(nn.Module):
    def __init__(self, output_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dropout = dropout

        self.rnn = nn.LSTM(output_dim, hid_dim[1], n_layers, dropout=dropout)
        self.rnn2 = nn.LSTM(hid_dim[1], hid_dim[0], n_layers, dropout=dropout)

        self.out = nn.Linear(hid_dim[0], output_dim)

        #self.dropout = nn.Dropout(dropout)

    def forward(self, input, expert, hidden, cell):

        input = input.unsqueeze(0)
        expert = expert#.unsqueeze(0)
        # expert = expert.squeeze().unsqueeze(0).unsqueeze(0)

        #print(expert)
        #print(expert.dim())
        #sys.exit()

        output, (hidden_1, cell_1) = self.rnn(expert, (hidden, cell))
        output, (hidden, cell) = self.rnn2(F.relu(output))

        prediction = self.out(F.relu(output).squeeze(0))

        #TODO change for expert
        result = torch.add(prediction, expert)
        # result = prediction.reshape(result.size())

        return result, hidden_1, cell_1


class Seq2Seq_all(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.hid_dim == decoder.hid_dim, "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, "Encoder and decoder must have equal number of layers!"

    def forward(self, src, trg, expert):
        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)

        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src)
        hidden_o, cell_o = hidden, cell

        # first input to the decoder is the <sos> tokens
        no_expert = src[-1]

        input = (trg)
        exp = (expert)
        output, hidden_o, cell_o = self.decoder(input, exp, hidden, cell)
        outputs = output

        return outputs


class Encoder_nru(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout = dropout

        self.rnn = NRU.NRU('cuda', 1, 1, num_layers=n_layers)
        self.rnn2 = NRU.NRU('cuda', 1, 1, num_layers=n_layers)

        #self.dropout = nn.Dropout(dropout)

    def forward(self, src):

        outputs, hidden = self.rnn(src)
        outputs, hidden = self.rnn2(F.relu(outputs))

        return hidden


class Decoder_nru(nn.Module):
    def __init__(self, output_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dropout = dropout

        self.rnn = NRU.NRU('cuda', 1, 1, num_layers=n_layers)
        self.rnn2 = NRU.NRU('cuda', 1, 1, num_layers=n_layers)

        self.out = nn.Linear(hid_dim[0], output_dim)

        #self.dropout = nn.Dropout(dropout)

    def forward(self, input, expert, hidden):

        input = input.unsqueeze(0)
        expert = expert.unsqueeze(0)

        output, hidden = self.rnn(expert, hidden)
        output, hidden = self.rnn2(F.relu(output))

        prediction = self.out(F.relu(output).squeeze(0))

        result = prediction

        # TODO change for expert
        result = torch.add(prediction, expert)
        result = prediction.reshape(result.size())

        return result, hidden


class Seq2Seq_nru(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.hid_dim == decoder.hid_dim, "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, "Encoder and decoder must have equal number of layers!"

    def forward(self, src, trg, expert):
        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)

        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden = self.encoder(src)

        # first input to the decoder is the <sos> tokens
        no_expert = src[-1]
        # print(no_expert)
        # print(src)
        # print(no_expert)

        for t in range(max_len):
            # TODO check if correct or all at once?!
            input = (trg[t])
            exp = (expert[t])
            # TODO use previous cellstates?
            #output, hidden_o = self.decoder(input, exp, hidden)
            output, hidden_o = self.decoder(input, no_expert, hidden)
            # top1 = output.max(1)[1].view(batch_size, trg_vocab_size).double()
            no_expert = output[0]
            outputs[t] = output

        return outputs


class Encoder_1(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout = dropout

        self.rnn = nn.LSTM(input_dim, hid_dim[0], n_layers, dropout=dropout)

        #self.dropout = nn.Dropout(dropout)

    def forward(self, src):

        # print(src)
        outputs, (hidden, cell) = self.rnn(src)

        return hidden, cell


class Decoder_1(nn.Module):
    def __init__(self, output_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dropout = dropout

        self.rnn = nn.LSTM(output_dim, hid_dim[0], n_layers, dropout=dropout)

        self.out = nn.Linear(hid_dim[0], output_dim)

        #self.dropout = nn.Dropout(dropout)

    def forward(self, input, expert, hidden, cell):

        input = input.unsqueeze(0)
        expert = expert.unsqueeze(0)

        output, (hidden_1, cell_1) = self.rnn(expert, (hidden, cell))

        prediction = self.out(F.relu(output).squeeze(0))

        #TODO change for expert
        result = torch.add(prediction, expert)
        # result = prediction.reshape(result.size())

        return result, hidden_1, cell_1


class Seq2Seq_1(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.hid_dim == decoder.hid_dim, "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, "Encoder and decoder must have equal number of layers!"

    def forward(self, src, trg, expert):
        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)

        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src)
        hidden_o, cell_o = hidden, cell

        # first input to the decoder is the <sos> tokens
        no_expert = src[-1]
        # print(no_expert)
        # print(src)
        # print(no_expert)

        #input = (trg)
        #exp = (expert)
        #output, hidden_o, cell_o = self.decoder(input, exp, hidden, cell)
        #outputs = output

        for t in range(max_len):
            # TODO check if correct or all at once?!
            input = (trg[t])
            exp = (expert[t])
            # TODO use previous cellstates?
            # TODO change for expert
            output, hidden_o, cell_o = self.decoder(input, exp, hidden, cell)
            # output, hidden_o, cell_o = self.decoder(input, exp, hidden_o, cell_o)
            # output, hidden_o, cell_o = self.decoder(input, no_expert, hidden, cell)
            # top1 = output.max(1)[1].view(batch_size, trg_vocab_size).double()
            no_expert = output[0]
            outputs[t] = output

        return outputs


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
