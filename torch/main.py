from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler, Sampler
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


def main():

    # Define hyperparameters
    cif_path = os.path.join('data', 'cif.csv')
    theta_path = os.path.join('data', 'theta_25_horg.csv')
    window_flag = 'T'  # '7': window_size = 7, 'T': horizon from csv file + 1, None: user-defined horizon + 1
    horizon = None  # None: use horizon of csv file
    train_split = 0.7
    percentage = 0.25  # percentage of elements to be removed due to theta model
    transform_flag = 'standard'  # standard: standardization, 'identiry: nothing, log: log
    cif_offset = 3
    batch_size = 16
    shuffle_dataset = False
    random_seed = 42
    N_EPOCHS = 500
    CLIP = 1
    name_prefix = 'test3'
    SAVE_DIR = os.path.join('output', name_prefix)

    # define parameters for model architecture
    INPUT_DIM = 1
    OUTPUT_DIM = 1
    HID_DIM = [32, 16]
    N_LAYERS = 1
    ENC_DROPOUT = 0.1
    DEC_DROPOUT = 0.1

    print('Load Data')

    sequences = load(cif_path)
    expert_sequences = load_expert(theta_path)
    # TODO log normalization
    # TODO Network Architecture with Expert Knowledge
    # TODO Loss curves figures + network graph png
    time_series_id = 1
    mapes = list()
    for sequence, expert_sequence in zip(sequences, expert_sequences):
        sequence = remove_percentage(sequence, percentage, cif_offset)
        # use torch Dataset to load the sequence
        time_series = TimeSeriesDataset(sequence, expert_sequence, window_flag=window_flag, transform_flag=transform_flag,
                                        horizon=horizon, transform=transforms.Compose([ToTensor()]))
        # dataloader = DataLoader(time_series, batch_size=batch_size, shuffle=True, num_workers=4)

        # create train, val and test samples
        dataset_size = len(time_series) - 1
        indices = list(range(dataset_size))
        split_index = int(np.floor(train_split * dataset_size))
        # shuffle them randomly, test set will always be last sample before shuffling
        if shuffle_dataset:
            np.random.seed(random_seed)
            np.random.shuffle(indices)

        # create the indices to access the samples
        train_indices, val_indices, test_indices = indices[:split_index], indices[split_index:], [dataset_size]
        train_sampler = SeqSampler(train_indices)
        val_sampler = SeqSampler(val_indices)
        test_sampler = SeqSampler(test_indices)

        # setting the iteretors respectively to the indices
        # TODO use a more fancy sampler to realize shuffling
        train_iterator = DataLoader(time_series, batch_size=batch_size, sampler=train_sampler, shuffle=False, num_workers=0)
        valid_iterator = DataLoader(time_series, batch_size=batch_size, sampler=val_sampler, shuffle=False, num_workers=0)
        test_iterator = DataLoader(time_series, batch_size=batch_size, sampler=test_sampler, shuffle=False, num_workers=0)

        #for t in valid_iterator:
        #    print(t['input'])
        #    print(t['output'])
        #    print('!!!!!!!!!!!!!!!!!!!!!!!!!')
        #for t in test_iterator:
        #    print(t['input'])
        #    print(t['output'])
        #    print(t['expert'])
        #    print('!!!!!!!!!!!!!!!!!!!!!!!!!')

        print('Define Model')

        #create encoder, decoder and seq2seq model
        enc = network.Encoder(INPUT_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
        dec = network.Decoder(OUTPUT_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
        model = network.Seq2Seq(enc, dec, device).to(device)
        print(f'The model has {network.count_parameters(model):,} trainable parameters')
        print(model)
        print(enc)
        print(dec)

        # define parameters for training
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
        # pad_idx = TRG.vocab.stoi['<pad>']
        criterion = nn.MSELoss()  # CrossEntropyLoss(ignore_index=pad_idx)

        MODEL_SAVE_PATH = os.path.join(SAVE_DIR, str(time_series_id) + '_model.pt')

        print('Start Training')

        # start training
        training.start_training(SAVE_DIR, MODEL_SAVE_PATH, N_EPOCHS, model, train_iterator, valid_iterator, optimizer,
                                criterion, CLIP)

        print('Evaluate')

        # evaluate trained model
        model.load_state_dict(torch.load(MODEL_SAVE_PATH))
        test_loss, output, gt = training.evaluate_result(model, test_iterator, criterion, transform_flag, device)
        smape = calc_smape(output, gt)
        mapes += [smape]
        print(output.tolist())
        print(gt.tolist())
        print(f'| Test Loss: {test_loss:.3f} |')
        print(f'| sMape: {smape:.3f} |')

        res_file_pred = open(os.path.join(SAVE_DIR, 'predictions' + '.csv'), 'a')
        res_file_gt = open(os.path.join(SAVE_DIR, 'gt' + '.csv'), 'a')
        res_file_pred.write('ts_' + str(time_series_id))
        res_file_gt.write('ts_' + str(time_series_id))
        for out, ground_truth in zip(output.tolist(), gt.tolist()):
            res_file_pred.write(',' + str(out[0]))
            res_file_gt.write(',' + str(ground_truth))
        res_file_pred.write('\n')
        res_file_gt.write('\n')
        res_file_pred.close()
        res_file_gt.close()

        time_series_id += 1

    smape_file = open(os.path.join(SAVE_DIR, 'mean_sMape.txt'), 'w')
    smape_file.write('Mean over all sMapes = ' + str(np.mean(mapes)) + '\n')
    i = 1
    for value in mapes:
        smape_file.write('sMape_' + str(i) + ' = ' + str(value) + '\n')
        i += 1
    smape_file.close()


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
            values = [values[0]] + [int(values[1])] + [values[2]] + [float(val) for val in values[3:] if val != '']
            sequences.append(values)
    return sequences


def load_expert(path):
    """
    Load csv from given path and return a list of list where each inner list corresponds to a time series in the
    csv. Will convert types appropriately, e.g. int to ints and floats to floats
    :param path: String
    :return: [[Float]]
    """

    sequences = list()
    with open(path) as f:
        content = f.read()
    for line in content.split('\n')[2:]:
        if not (line == ''):
            values = line.split(',')
            values = [float(val) for val in values[1:] if val != 'NA']
            sequences.append(values)
    return sequences


def remove_percentage(sequence, percentage, cif_offset):
    """
    Removes the first percentage many elements from the sequence
    :param sequence: list
    :param percentage: float
    :return: list
    """
    value_count = len(sequence) - cif_offset  # don't ignore the first cif_offset many metadata
    ignore_first_values = int(value_count * percentage)  # number of calues to be ignored
    del sequence[cif_offset:(ignore_first_values + cif_offset)]

    return sequence


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
    x, y = remove_overlap(x, y, horizon)
    return np.array(x), np.array(y)


def make_expert_sample(expert_seq, window_size, horizon):
    """
    creates the expert samples, similar to make_samples
    :param expert_seq:
    :param window_size:
    :param horizon:
    :return:
    """

    z = list()
    for i in range(len(expert_seq)-(window_size+horizon-1)):
        z_tmp = expert_seq[(i+window_size):(i+window_size+horizon)]
        z.append(z_tmp)
    del z[-horizon:-1]
    return np.array(z)


def remove_overlap(x, y, horizon):
    """
    Removes the overlap from the test set with the other set
    :param x: list of list containing the inputs
    :param y: list of list containing the outputs
    :param horizon: int defining the horizon
    :return: cleaned x and y list
    """

    # num_removes = horizon - 1  # we need to remove that many samples
    del x[-horizon:-1]
    del y[-horizon:-1]

    return x, y


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


def standardize(x, y, expert):
    """

    :param x:
    :param y:
    :param expert:
    :return:
    """
    mu_list = list()
    std_list = list()
    new_x = list()
    new_y = list()
    new_expert = list()
    for x_element, y_element, expert_element in zip(x[:-1], y[:-1], expert[:-1]):
        tmp = np.concatenate((x_element, y_element, expert_element))
        mean = np.mean(tmp)
        std = np.std(tmp)
        x_element = (x_element - mean)/std
        y_element = (y_element - mean) / std
        expert_element = (expert_element - mean) / std
        new_x.append(x_element)
        new_y.append(y_element)
        new_expert.append(expert_element)
        mu_list.append(mean)
        std_list.append(std)
    tmp = np.concatenate((x[-1], expert[-1]))
    mean = np.mean(tmp)
    std = np.std(tmp)
    x_element = (x[-1] - mean) / std
    expert_element = (expert[-1] - mean) / std
    new_x.append(x_element)
    new_y.append(y[-1])
    new_expert.append(expert_element)
    mu_list.append(mean)
    std_list.append(std)
    return np.array(new_x), np.array(new_y), np.array(new_expert), mu_list, std_list


def make_placeholder(x):
    """

    :param x:
    :return:
    """
    size = len(x)
    return [0]*size, [0]*size


def calc_smape(prediction, ground_truth):
    horizon = len(ground_truth)
    prediction = prediction.tolist()
    prediction = [value[0] for value in prediction]
    ground_truth = ground_truth.tolist()
    add = 0
    for pred, gt in zip(prediction, ground_truth):
        add += (abs(gt - pred) / ((abs(gt) + abs(pred)) / 2))
    smape = (add / horizon) * 100
    return smape


class TimeSeriesDataset(Dataset):
    """
    Pytorch Dataset realization of a timeseries
    """

    def __init__(self, sequence_par, expert_sequence_par, window_flag=None, transform_flag='identity', horizon=None, transform=None):
        """
        Initialize the dataset with the given sequence based on the parameters of window size and horizon
        :param sequence_par:
        :param expert_sequence_par:
        :param window_flag: None or String
        :param horizon: int
        :param transform: Object
        """

        # values of sequence start at position 3
        self.sequence = sequence_par[3:]
        self.expert_seq = expert_sequence_par

        # set the horizon to the user definition or if "None" was given to the horizon defined in the csv file
        if horizon is None:
            self.horizon = sequence_par[1]
        else:
            self.horizon = horizon

        # define the size of the window based on the given flag
        if window_flag == '7':
            self.window_size = 7
        elif window_flag == 'T':
            self.window_size = sequence_par[1] + 1
        elif window_flag is None:  # TODO why need to compare to None?!
            self.window_size = self.horizon + 1

        # create input window x and output window y
        self.x, self.y = make_samples(self.sequence, self.window_size, self.horizon)
        self.expert = make_expert_sample(self.expert_seq, self.window_size, self.horizon)

        if transform_flag == 'identity':
            self.x, self.y, self.expert = self.x, self.y, self.expert
            self.mu, self.std = make_placeholder(self.x)
        elif transform_flag == 'standard':
            self.x, self.y, self.expert, self.mu, self. std = standardize(self.x, self.y, self.expert)

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
        x, y, expert = self.x[idx], self.y[idx], self.expert[idx]
        mu, std = self.mu[idx], self.std[idx]
        x = np.array([x]).astype('double')  # [x]
        x = x.reshape(-1, 1)
        y = np.array([y]).astype('double')
        y = y.reshape(-1, 1)
        expert = np.array([expert]).astype('double')
        expert = expert.reshape(-1, 1)

        sample = {'input': x, 'output': y, 'expert': expert, 'mean': mu, 'std': std}

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):
    """
    Transform an object to a torch tensor
    """

    def __call__(self, sample):
        in_window, out_window, expert_window = sample['input'], sample['output'], sample['expert']
        mean, std = sample['mean'], sample['std']
        return {'input': torch.from_numpy(in_window).to(device), 'output': torch.from_numpy(out_window).to(device),
                'expert': torch.from_numpy(expert_window).to(device),
                'mean': mean, 'std': std}


class SeqSampler(Sampler):
    """
    Samples elements from a given list of indices.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


if __name__ == '__main__':
    main()
