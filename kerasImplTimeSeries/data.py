import os
import string
import re
from pickle import dump, load
from unicodedata import normalize
from numpy import array, append
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np

def create_Pkl_CIF_File(location, saving_location, percentage, split_ratio):
    """
    creates pkl files for training , test and validation based on the file defined by the location parameter. It will
    also clean the text.
    :param location: path to the CIF csv file
    :param saving_location: path to where the pkl files should be saved
    :param percentage: how many of the first rows are ignored (given in percentage of all lines), because these have
    have been used to train Theta.
    """
    #TODO check if pkl files already exist, if yes, skip process

    print('Create Pkl files')

    #load the document
    text = load_doc(location)

    #ignore the first $percentage% of rows
    text = ignore_first_percentage(text, percentage)

    #create pairs of values known to network and and the according values which need to be predicted
    text = to_pairs(text)

    #divide data in train, validation and testset
    train_pair, val_pair, test_pair = divide_data(text, split_ratio)

    #saves data as pkl files on system
    save_data(test_pair, os.path.join(saving_location, 'test.pkl'))
    save_data(train_pair, os.path.join(saving_location, 'train.pkl'))
    save_data(val_pair, os.path.join(saving_location, 'val.pkl'))
    save_data(text, os.path.join(saving_location, 'allData.pkl'))

    print('Pkl files created')

def load_doc(filename):
    """
    loads a document
    :param filename: name of document to be loaded
    :return: text of document
    """
    file = open(filename, mode='rt', encoding='utf-8')
    text = file.read()
    file.close()
    return text

def ignore_first_percentage(text, percentage):
    """
    ignores the first percentage rows
    :param text: From this text the first rows are to be ignored
    :param percentage: how many rows to ignore in percent
    :return: text without the first rows
    """
    #return '\n'.join(text.split('\n')[int(len(text.split('\n')) * percentage):])
    lines = text.split('\n')
    size = len(lines)
    ignore_first_rows = int(size * percentage)
    return '\n'.join(lines[ignore_first_rows:])[:-1]

def to_pairs(text):
    """
    divides the sequence in a pair of 'values known to network' and 'values to be predicted'
    :param text: lines containing the values
    :return: array of these pairs
    """
    #TODO maybe add parameter dataset and make switch case orders depending on the dataset if they have different structures

    lines = text.split('\n')
    pairs = list()
    for line in lines:
        values = line.split(',')
        values = [value for value in values if value != ''] #some lines have a lot of empty values where there are just
                                                            #','. Remove those entries before proceeding
        #horizon is the number of values to be predicted
        horizon = int(values[1]) * (-1) #need it negative for proper slicing
        known_values = values[3:horizon]
        known_values = [float(value) for value in known_values]
        predicted_values = values[horizon:]
        predicted_values = [float(value) for value in predicted_values]
        pairs.append([known_values, predicted_values])
    return array(pairs)

def divide_data(data, split_ratio):
    """
    divides data in train, validation and test set
    :param data: array of dataset to be split
    :param split_ratio: array containing the ratio for train, validation and test
    :return: train, validation and test set
    """
    #define the number of rows per set
    train_split, val_split, test_split = split_ratio[0], split_ratio[1], split_ratio[2]
    length = len(data)
    train_split_abs = int(length * train_split)
    val_split_abs = int(length * val_split)
    test_split_abs = length - train_split_abs - val_split_abs

    train, val, test, = list(), list(), list()
    np.random.shuffle(data)
    i = 0
    while i < train_split_abs:
        train.append(data[i])
        i += 1
    while i < (val_split_abs + train_split_abs):
        val.append(data[i])
        i += 1
    while i < (test_split_abs + val_split_abs + train_split_abs):
        test.append(data[i])
        i += 1
    return array(train), array(val), array(test)

def save_data(pairs, filename):
    """
    saves data on hard disk
    :param pairs: the data to be saved as an array of lists
    :param filename: location and name where it should be saved
    """
    dump(pairs, open(filename, 'wb'))
    print('Saved: %s' % filename)

def load_pkl(filename):
    """
    loads a pkl file
    :param filename: path to file
    :return: the pkl file
    """
    return load(open(filename, 'rb'))



def prepare_Data(train, test, val):
    """
    derives properties from the sequences, pads them to maximal length and creates a teacher enforcing sequence
    :param train:
    :param test:
    :param val:
    :return: size of features, in- and output length and sequences for training/testing/validation. ___X will be the
    input and ___Y is the intended output. ___shifted is the target shifted for teacher enforcing
    """
    #get properties over all sequences
    input_length, output_length, feature_size = get_sequence_properties(train)

    # prepare training data
    trainX = pad_sequences(input_length, train[:, 0])
    trainY = pad_sequences(output_length, train[:, 1])
    trainY_shifted = shift(trainY, output_length)
    #for now no one-hot encoding
    #trainX = one_hot_encode(trainX, feature_size)
    #trainY = one_hot_encode(trainY, feature_size)
    #trainY_shifted = one_hot_encode(trainY_shifted, feature_size)

    # prepare testing data
    testX = pad_sequences(input_length, test[:, 0])
    testY = pad_sequences(output_length, test[:, 1])
    testY_shifted = shift(testY, output_length)

    # prepare validation data
    valX = pad_sequences(input_length, val[:, 0])
    valY = pad_sequences(output_length, val[:, 1])
    valY_shifted = shift(valY, output_length)

    lengths = [input_length, output_length]
    all_data = [trainX, trainY, trainY_shifted, testX, testY, testY_shifted, valX, valY, valY_shifted]
    return feature_size, lengths, all_data

def get_sequence_properties(dataset):
    """
    calculates feature size and maximal input/output length
    :param dataset: the set from which it derives the values. To prevent information leakage only take train set
    #TODO right assumptions? make sure that all horizons are in training set?
    :return: integer for feature size, maximal input/output length
    """
    input_length = max_length(dataset[:, 0])
    output_length = max_length(dataset[:, 1])
    feature_size = calculate_features(dataset)
    print('Maximum input length: %d' % input_length)
    print('Maximum output length: %d' % (output_length))
    print('Feature Size: %d' % (feature_size))
    return input_length, output_length, feature_size

def max_length(lines):
    """
    calculates the length of the longest sequence
    :param lines: the sequences to be analyzed
    :return: maximal length
    """
    return max(len(line) for line in lines)

def calculate_features(dataset):
    """
    calculates the number of features inside the dataset
    :param dataset: set to be analyzed, wich is an array, containing array where each array is a tuple where one is the
    input sequence and the other is the output sequence
    :return: number of features
    """
    #calculates the number of unique elements in all sequences given by dataset
    #TODO correct approach? -> No! therefore return 1 to make it correct
    #unique = []
    #for in_out in dataset:
    #    for line in in_out:
    #        for value in line:
    #            if value not in unique:
    #                unique.append(value)
    #return (len(unique)+1)
    return 1

def pad_sequences(length, lines):
    #TODO should I pad or not?!
    """
    pad the sequence at the end to length with 0's
    :param length:
    :param lines:
    :return:
    """
    #pad sequences with 0 values at the end
    padded_list = array([])
    linenumber = 0
    for line in lines:
        padding = line if len(line) == length else line + ([0]*(length-len(line)))
        padded_list = np.append(padded_list, padding, axis=0)
        linenumber += 1
    #print(padded_list)
    padded_list = padded_list.reshape(linenumber, length, 1)
    #print(padded_list)
    return padded_list

def shift(sequences, length):
    """
    Shifts the sequence for teacher enforcing usage
    :param sequence:
    :return:
    """
    shifted_list = array([])
    linenumber = 0
    for sequence in sequences:
        shifted = array([0])
        for value in sequence:
            shifted = np.append(shifted, value[0])
        shifted = shifted[:-1].copy()
        shifted_list = np.append(shifted_list, shifted, axis=0)
        linenumber += 1
    shifted_list = shifted_list.reshape(linenumber, length, 1)
    return shifted_list


def one_hot_encode(sequences, feature_size):
    """
    one hot encode the sequence into the given feature size
    :param sequences:
    :param feature_size:
    :return:
    """
    #TODO should I one hot encode things?!, probably not, also doesnt work
    ylist = list()
    for sequence in sequences:
        encoded = to_categorical(sequence, num_classes=feature_size)
        ylist.append(encoded)
    y = array(ylist)
    y = y.reshape(sequences.shape[0], sequences.shape[1], feature_size)
    return y