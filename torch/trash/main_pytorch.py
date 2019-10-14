import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import training, network, data

import os
import math
import sys

def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    path = os.path.join('data', 'cif_one.csv')  
    window = 7
    horizon = 1
    train_split = 0.7

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
    #print(x_train.size())
    #print(x_train[0])
    #print(len(x_train[0]))

    # build the model
    seq = Sequence()
    seq.float()
    criterion = nn.MSELoss()
    # use LBFGS as optimizer since we can load the whole data to train
    optimizer = optim.LBFGS(seq.parameters(), lr=0.8)

    #begin to train
    for i in range(15):
        print('STEP: ', i)
        def closure():
            optimizer.zero_grad()
            out = seq(x_train)
            loss = criterion(out, y_train)
            print('loss:', loss.item())
            loss.backward()
            return loss
        optimizer.step(closure)
        # begin to predict, no need to track gradient here
        with torch.no_grad():
            future = horizon
            pred = seq(x_val, future=future)
            loss = criterion(pred[:, :-future], y_val)
            print('test loss:', loss.item())
            y = pred.detach().numpy()
    pred = seq(x_test, future=future)
    y = pred.detach().numpy()[0][x_train.size(1)]
    print((y*std)+mean)
    print((x_test*std)+mean)
    print((y_test*std)+mean)
    print(x_train.size())
    print(x_test.size())



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
    x, y = np.array(x), np.array(y)
    return x, y


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


class Sequence(nn.Module):
    def __init__(self):
        super(Sequence, self).__init__()
        self.lstm1 = nn.LSTMCell(1, 51)
        self.lstm2 = nn.LSTMCell(51, 51)
        self.linear = nn.Linear(51, 1)

    def forward(self, input, future = 0):
        outputs = []
        h_t = torch.zeros(input.size(0), 51, dtype=torch.float)
        c_t = torch.zeros(input.size(0), 51, dtype=torch.float)
        h_t2 = torch.zeros(input.size(0), 51, dtype=torch.float)
        c_t2 = torch.zeros(input.size(0), 51, dtype=torch.float)

        for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        for i in range(future):# if we should predict the future
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs

if __name__ == '__main__':
    main()
