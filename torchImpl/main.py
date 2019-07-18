from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms, utils

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import training, network, data

import warnings
warnings.filterwarnings("ignore")

import os
import time
import math
import sys

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu' #TODO fix cuda error

def main():

    # Define hyperparameters
    path = os.path.join('data', 'cif_one.csv')
    #path = os.path.join('data', 'cif_dataset_complete.csv')
    window_flag = '7' # '7': window_size = 7, 'T': horizon from csv file + 1, None: user-defined horizon + 1
    horizon = 3 # None to use horizon of csv file
    train_split = 0.7
    batch_size = 32
    shuffle_dataset = True
    random_seed = 42
    N_EPOCHS = 30
    CLIP = 1
    SAVE_DIR = 'models'

    print('Load Data')

    sequences = load(path)
    # TODO For sequence in sequences
    # TODO need to remove overlap in testset
    # TODO correct normalization
    # for now test on only one sequence
    sequence = sequences[0]#[3:]
    # use torch Dataset to load the sequence
    time_series = TimeSeriesDataset(sequence, window_flag=window_flag, horizon=horizon,
                                    transform=transforms.Compose([ToTensor()]))
    #dataloader = DataLoader(time_series, batch_size=batch_size, shuffle=True, num_workers=4)

    # create train, val and test samples
    dataset_size = len(time_series) - 1
    indices = list(range(dataset_size))
    split_index = int(np.floor(train_split * dataset_size))
    # shuffle them randomly, test set will always be last sample before shuffling
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    # create the indicies to access the samples
    train_indices, val_indices, test_indices = indices[:split_index], indices[split_index:], [dataset_size]
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    #setting the iteretors respectively to the indices
    train_iterator = DataLoader(time_series, batch_size=batch_size, sampler=train_sampler, num_workers=1)
    valid_iterator = DataLoader(time_series, batch_size=batch_size, sampler=val_sampler, num_workers=1)
    test_iterator = DataLoader(time_series, batch_size=batch_size, sampler=test_sampler, num_workers=1)

    #get iterators for data and vocabularies
    #BATCH_SIZE = 128
    #train_iterator2, valid_iterator, test_iterator, SRC, TRG = data.startDataProcess(32, device)
    #for t, t2 in zip(train_iterator, train_iterator2):
    #    print(t['input'].size(), t2.src.size())
    #    print(t['output'].size(), t2.trg.size())
    #    sys.exit()

    print('Define Model')

    #define parameters for model architecture
    INPUT_DIM = 7 #TODO adapt to flag
    OUTPUT_DIM = horizon
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    HID_DIM = 512
    N_LAYERS = 2
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5

    #create encoder, decoder and seq2seq model
    enc = network.Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
    dec = network.Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
    model = network.Seq2Seq(enc, dec, device).to(device)
    print(f'The model has {network.count_parameters(model):,} trainable parameters')
    print(model)
    print(enc)
    print(dec)

    #define parameters for training
    optimizer = optim.Adam(model.parameters())
    #pad_idx = TRG.vocab.stoi['<pad>']
    criterion = nn.MSELoss()#CrossEntropyLoss(ignore_index=pad_idx)

    MODEL_SAVE_PATH = os.path.join(SAVE_DIR, 'tut1_model.pt')

    print('Start Training')

    #start training
    training.start_training(SAVE_DIR, MODEL_SAVE_PATH, N_EPOCHS, model, train_iterator, valid_iterator,optimizer,
                            criterion, CLIP)

    print('Evaluate')

    #evaluate trained model
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    test_loss = training.evaluate(model, test_iterator, criterion)
    print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

    #Test Loss after 30 Epochs: 3.331


def load(path):
    """
    Load csv from given path and return a list of list where each inner list corresponds to a time series in the
    csv. Will convert types appropriately, e.g. int to ints and floats to floats
    :param path: String
    :return: [[Varying Types]]
    """

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
    """
    Creates samples from a sequence with the defined window size and horizon
    Will return one numpy array for the window values and one for the horizon values
    The order defines which window belongs to which horizon
    :param sequence: List of list
    :param window: int
    :param horizon: int
    :return: 2 numpy arrays
    """

    x = list()
    y = list()
    for i in range(len(sequence)-(window+horizon-1)):
        x_tmp = sequence[i:(i+window)]
        y_tmp = sequence[(i+window):(i+window+horizon)]
        x.append(x_tmp)
        y.append(y_tmp)
    return np.array(x), np.array(y)  


def split(x, y, train_split):
    """
    Splits the window and horizon in train and val set based on the percentage given in train_split
    Test set will be the last element
    :param x: numpy array
    :param y: numpy array
    :param train_split: float
    :return: 3 tuples, each being an numpy array
    """
    index = int((len(x)-1)*train_split)
    train = (x[:index], y[:index])
    val = (x[index:-1], y[index:-1])
    test = (x[-1:], y[-1:])
    return train, val, test


class TimeSeriesDataset(Dataset):
    """
    Pytorch Dataset realization of a timeseries
    """

    def __init__(self, sequence_par, window_flag=None, horizon=None, transform=None):
        """
        Initialize the dataset with the given sequence based on the parameters of window size and horizon
        :param sequence_par:
        :param window_flag: None or String
        :param horizon: int
        :param transform: Object
        """

        # values of sequence start at position 3
        self.sequence = sequence_par[3:]

        # set the horizon to the user definition or if none was given to the horizon defined in the csv file
        if horizon == None:
            self.horizon = sequence_par[1]
        else:
            self.horizon = horizon

        # define the size of the window based on the given flag
        if window_flag == '7':
            self.window_size = 7
        if window_flag == 'T':
            self.window_size = sequence_par[1] + 1
        elif window_flag == None: # TODO why need to compare to None?!
            self.window_size = self.horizon + 1

        # create input window x and output window y
        self.x, self.y = make_samples(self.sequence, self.window_size, self.horizon)

        self.transform = transform

    def __len__(self):
        """
        Return the number of samples
        :return: int
        """
        return len(self.x)

    def __getitem__(self, idx):
        """
        get one specific sample from the time series
        :param idx: int
        :return: dictionary of input: np array and output: np array
        """
        x, y = self.x[idx], self.y[idx]
        x = np.array([x]).astype('double') #[x]
        x = x.reshape(1, -1)
        y = np.array([y]).astype('double')
        y = y.reshape(1, -1)

        sample = {'input': x, 'output': y}

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):
    """
    Transform an object to a torch tensor
    """

    def __call__(self, sample):
        in_window, out_window = sample['input'], sample['output']
        return {'input': torch.from_numpy(in_window).to(device), 'output': torch.from_numpy(out_window).to(device)}


if __name__ == '__main__':
    main()
