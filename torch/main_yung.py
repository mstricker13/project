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
    print(train[0].shape)
    x_train = torch.from_numpy(train[0]).type(torch.Tensor).view([window, -1, 1])
    print(x_train.size())
    #sys.exit()
    y_train = torch.from_numpy(train[1]).type(torch.Tensor).view(-1)
    x_val = torch.from_numpy(val[0]).type(torch.Tensor)
    y_val = torch.from_numpy(val[1]).type(torch.Tensor)
    x_test = torch.from_numpy(test[0]).type(torch.Tensor).view([window, -1, 1])
    print(x_test.size())
    y_test = torch.from_numpy(test[1]).type(torch.Tensor).view(-1)
    #print(x_train.size())
    #print(x_train[0])
    #print(len(x_train[0]))


    lstm_input_size = 1
    h1 = 32
    output_dim = horizon
    num_layers = 2
    learning_rate = 1e-3
    num_epochs = 500
    dtype = torch.float

    model = LSTM(lstm_input_size, h1, batch_size=x_train.size(1), output_dim=output_dim, num_layers=num_layers)

    loss_fn = torch.nn.MSELoss(size_average=False)

    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

#####################
# Train model
#####################

    hist = np.zeros(num_epochs)

    for t in range(num_epochs):
        # Initialise hidden state
        # Don't do this if you want your LSTM to be stateful
        model.hidden = model.init_hidden()
    
        # Forward pass
        y_pred = model(x_train)

        loss = loss_fn(y_pred, y_train)
        if t % 100 == 0:
            print("Epoch ", t, "MSE: ", loss.item())
        hist[t] = loss.item()

        # Zero out gradient, else they will accumulate between epochs
        optimiser.zero_grad()

        # Backward pass
        loss.backward()

        # Update parameters
        optimiser.step()
    y_pred = model(x_test)
    print((y_pred.detach().numpy()*std)+mean)#*std)+mean)
    #print(x_test)
    print((y_test.detach().numpy()*std)+mean)



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


class LSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, batch_size, output_dim=1,
                    num_layers=2):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers

        # Define the LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)

        # Define the output layer
        self.linear = nn.Linear(self.hidden_dim, output_dim)

    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def forward(self, input):
        # Forward pass through LSTM layer
        # shape of lstm_out: [input_size, batch_size, hidden_dim]
        # shape of self.hidden: (a, b), where a and b both 
        # have shape (num_layers, batch_size, hidden_dim).
        lstm_out, self.hidden = self.lstm(input.view(len(input), self.batch_size, -1))
        
        # Only take the output from the final timetep
        # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
        y_pred = self.linear(lstm_out[-1].view(self.batch_size, -1))
        return y_pred.view(-1)

if __name__ == '__main__':
    main()
