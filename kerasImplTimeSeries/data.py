import os
import string
import re
from pickle import dump, load
from unicodedata import normalize
from numpy import array, append
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
import numpy as np
import sys

#each line in csv starts with generell information, like horizon etc. This number denotes how many of those etc. infos there are
cif_offset = 3

def create_Pkl_CIF_File(location, saving_location, percentage, split_ratio, window_size, stepsize, horizon, use_csv_horizon, step_size):
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
    sequences = load_doc(location)

    #print(len(sequences.split('\n')[0].split(',')))
    #print(sequences.split('\n')[0].split(','))
    #print(len(sequences.split('\n')[1].split(',')))
    #print(sequences.split('\n')[1].split(','))

    #ignore the first $percentage% of rows
    #sequences = sequences.split('\n')[1]
    sequences = ignore_first_percentage(sequences, percentage)
    #print(len(sequences.split('\n')[0].split(',')))
    #print(sequences.split('\n')[0].split(','))
    #print(len(sequences.split('\n')[1].split(',')))
    #print(sequences.split('\n')[1].split(','))

    #create pairs of values known to network and and the according values which need to be predicted
    sequences = create_n_tuples(sequences, window_size, stepsize, horizon, use_csv_horizon)
    #divide data in train, validation and testset
    train_pair, val_pair, test_pair = divide_data(sequences, split_ratio, step_size)

    #saves data as pkl files on system
    save_data(test_pair, os.path.join(saving_location, 'test.pkl'))
    save_data(train_pair, os.path.join(saving_location, 'train.pkl'))
    save_data(val_pair, os.path.join(saving_location, 'val.pkl'))
    save_data(sequences, os.path.join(saving_location, 'allData.pkl'))

    print('Pkl files created')

def create_Pkl_Theta_File(location, saving_location, percentage, split_ratio, window_size, stepsize, horizon, use_csv_horizon, step_size):
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
    sequences = load_doc(location)

    #ignore the first $percentage% of rows
    #sequences = ignore_first_percentage(sequences, percentage)

    #print(len(sequences.split('\n')[0].split(',')))
    #print(sequences.split('\n')[0].split(','))
    #print(len(sequences.split('\n')[1].split(',')))
    #print(sequences.split('\n')[1].split(','))
    #sys.exit()

    #create pairs of values known to network and and the according values which need to be predicted
    #sequences = sequences.split('\n')[1]
    sequences = create_n_tuples(sequences, window_size, stepsize, horizon, use_csv_horizon)
    #divide data in train, validation and testset
    train_pair, val_pair, test_pair = divide_data(sequences, split_ratio, step_size)

    #saves data as pkl files on system
    save_data(test_pair, os.path.join(saving_location, 'testTheta.pkl'))
    save_data(train_pair, os.path.join(saving_location, 'trainTheta.pkl'))
    save_data(val_pair, os.path.join(saving_location, 'valTheta.pkl'))
    save_data(sequences, os.path.join(saving_location, 'allDataTheta.pkl'))

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

def ignore_first_percentage(sequences, percentage):
    """
    ignores the first percentage of values in each time series
    :param text: From this text the first rows are to be ignored
    :param percentage: how many values to ignore in percent
    :return: time sequences without the first percentage values
    """
    lines = sequences.split('\n')
    reduced_lines = ''
    for line in lines:
        values = line.split(',')
        # some lines have a lot of empty values where there are just ','. Remove those entries before proceeding
        values = [value for value in values if value != '']
        #for the count of values ignore the first cif_offset values which are not time series values
        value_count = len(values) - cif_offset
        ignore_first_values = int(value_count * percentage)
        reduced_values = ','.join(values[:cif_offset] + values[(ignore_first_values + cif_offset):])
        reduced_lines += reduced_values + '\n'

    # -1 because there is an empty lines in the end
    return reduced_lines[:-1]

def create_n_tuples(sequences, window_size, stepsize, horizon, use_csv_horizon):
    """
    for each sequence create a list of n-tuples. Each n-tuple consists of n lists. (Note that the term tuple was just used to differentiate, the datastructure of the tuple will be a list
    The lists inside the n-tuple represent:
        First list contains window_size many values representing the values based on which the network will make its prediction
        Next n-1 lists all contain the values to be predicted. The number of values is defined by the given horizon and the horizon in the csv
        #TODO when bigger horizon exceeds list it is not anymore included, therefore the last elements may not contain lists for all horizons
    The list of n-tuples contains multiple n-tuples as defined above. The first n-tuple starts at timestep 0, while the second on starts at timestep 0 + stepsize and so on.
    This list of n-tuples is created for each time series/sequence
    :param text: lines containing the values
    :param window_size: integer defining the window size
    :param stepsize: integer defining the stepsize
    :param horizon: list of integers, defining the different horizons. Should be empty if only csv horizon is to be used
    :return: list containing all n-tuples
    """
    #TODO maybe add parameter dataset and make switch case orders depending on the dataset if they have different structures

    lines = sequences.split('\n')

    result_sequences = list()
    horizon_list = list()
    horizon_in = horizon
    for line in lines:
        horizon = horizon_in
        values = line.split(',')
        #check if horizon defined by csv is in defined horizon or not
        horizon_csv = int(values[1])
        #TODO Dynamic Window sized based on horizon
        window_size = 15#horizon_csv + 1
        #TODO figure out the bug, for adding horizons the first row where it changes kills it why?
        #if horizon_csv not in horizon:
        #    horizon += [horizon_csv]
        defined_horizon = int(values[1])
        horizon_list.append(defined_horizon)
        #remove the metadata from the list
        values = values[3:]
        #convert values from string to float
        values = [float(conversion) for conversion in values]
        #TODO rescuer uncomment next two lines
        test_tuple = [values[(-1*(defined_horizon+window_size)):(-1*defined_horizon)], values[(-1*defined_horizon):]]
        values = values[:(-1*defined_horizon)]

        #create list of tuples with [[Learning, PredH1, PredH2,...], [Learning+step, PredH1+step, PredH2+step,...], ...]
        tuple_list = list()
        first_horizon = True
        if use_csv_horizon:
            horizon = [defined_horizon]
        for prediction_size in horizon:
            #start tuple creation from sequence at start value
            start_value = 0
            #end tuple creation from sequence at end value, needs to stop earlier, than the end of the list, so that the last values of prediction are not empty
            end_value = len(values) - (window_size + prediction_size)
            i = 0
            #TODO not matching stepsize might remove some of the last elements, check it and print warning?
            while start_value <= end_value:
                learning_values = values[start_value:(window_size+start_value)]
                prediction_values = values[(window_size+start_value):(window_size+prediction_size+start_value)]
                if first_horizon:
                    tuple = [learning_values, prediction_values]
                    tuple_list.append(tuple)
                else:
                    tuple_list[i].append(prediction_values)
                start_value += stepsize
                i += 1
            first_horizon = False
        #TODO rescuer uncomment next line and remove 2nd next line
        result_sequences.append(tuple_list + [test_tuple])
        #result_sequences.append(tuple_list)
    return (result_sequences, horizon_list)

def divide_data(data_tuple, split_ratio, step_size):
    """
    divides data in train, validation and test set, where test set will be the last sample of window size and horizon
    :param data: array of dataset to be split
    :param split_ratio: array containing the ratio for train, validation and test
    :return: train, validation and test set
    """
    #define the number of rows per set
    train_split, val_split = split_ratio[0], split_ratio[1]
    train, val, test, = list(), list(), list()
    data = data_tuple[0]
    horizons = data_tuple[1]

    #create a train/val/test sequence for each sequence
    for sequence, horizon in zip(data, horizons):
        tmp_train, tmp_val, tmp_test, = list(), list(), list()
        tuple_count = len(sequence)
        train_split_abs = int(tuple_count * train_split)
        val_split_abs = tuple_count - (train_split_abs + 1)
        test_split_abs = 1
        i = 0
        while i < train_split_abs:
            tmp_train.append(sequence[i])
            i += 1
        while i < (val_split_abs + train_split_abs):
            tmp_val.append(sequence[i])
            i += 1
        while i < (test_split_abs + val_split_abs + train_split_abs):
            tmp_test.append(sequence[i])
            i += 1
        #TODO rescuer comment the next line
        #tmp_val, tmp_test = change_test_to_true_horizon(tmp_val, tmp_test, horizon, step_size)
        train.append(tmp_train)
        val.append(tmp_val)
        test.append(tmp_test) #TODO since it looks different make sure that it still works, just don't cast it to array...
    return array(train), array(val), test

def change_test_to_true_horizon(val, test, horizon, stepsize):
    """
    checks if the horizon of the test set matches with the wanted one defined by horizon. If not modifies val and test,
    so that it matches
    :param val:
    :param test:
    :param horizon:
    :return:
    """
    used_horizon = len(test[0][1])
    used_window_size = len(test[0][0])
    #list of all values, since in the tuples values will appear multiple size depending on step size, iterate smartly
    value_list = list()
    #also makes sure that all test values are NOT in validation set
    tmp = val + test
    #have to add all the values from from the first tuple in tmp to the value_list
    value_list = [val for val in tmp[0][0]]
    value_list += [val for val in tmp[0][1]]
    for tuple in tmp[1:]:
        tuple_values = [val for val in tuple[0]]
        tuple_values += [val for val in tuple[1]]
        tmp_adder = tuple_values[len(tuple_values)-stepsize:]
        value_list += tmp_adder

    #create test set by just using the last elements, and remove them from the list from which validation should be created
    test_label = value_list[-horizon:]
    test_knowledge = value_list[len(value_list)-horizon-used_window_size:-horizon]
    make_to_val = value_list[:len(value_list)-horizon]
    #TODO ASK HOW TO SOLVE
    if len(make_to_val) < (used_window_size + used_horizon):
        diff = len(make_to_val) - (used_window_size + used_horizon)
        pad = make_to_val[-1]
        make_to_val += [pad] * abs(diff)
    assert len(make_to_val) >= (used_window_size + used_horizon), 'Validation list to small to recreate it after creating' \
                                                                 ' a non overlapping test set with the horizon defined ' \
                                                                 'by its csv. Change split ratio to have a bigger validation set.' \
                                                                  ' It\'s also possible that the list is too small for this configuration of horizon, window size in general'

    #recreate the validation samples
    new_val = list()

    #start tuple creation from sequence at start value
    start_value = 0
    #end tuple creation from sequence at end value, needs to stop earlier, than the end of the list, so that the last values of prediction are not empty
    end_value = len(make_to_val) - (used_window_size + used_horizon)
    i = 0
    #TODO not matching stepsize might remove some of the last elements, check it and print warning?
    while start_value <= end_value:
        learning_values = make_to_val[start_value:(used_window_size+start_value)]
        prediction_values = make_to_val[(used_window_size+start_value):(used_window_size+used_horizon+start_value)]
        new_val_tuple = [learning_values, prediction_values]
        new_val.append(new_val_tuple)
        start_value += stepsize
        i += 1

    return new_val, [[test_knowledge, test_label]]

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
    #input_length, output_length, feature_size = get_sequence_properties(train) --> useless, feature size is 1, output is horizon and input is window size

    # prepare training data
    trainX, trainY = splitter(train)
    trainY_shifted = shifter(trainY)

    # prepare testing data
    testX, testY = splitter(test)
    testY_shifted = shifter(testY)

    # prepare validation data
    valX, valY = splitter(val)
    valY_shifted = shifter(valY)

    #lengths = [input_length, output_length]
    all_data = [trainX, trainY, trainY_shifted, testX, testY, testY_shifted, valX, valY, valY_shifted]
    #return feature_size, lengths, all_data

    return all_data

def splitter(set):
    """
    splits the set into the values from which the prediction is done the to predicting values
    :param set:
    :return: set to derive from, prediction set
    """
    x = list()
    y = list()
    for sequence in set:
        tmp_x = list()
        tmp_y = list()
        for tuple in sequence:
            tmp_x.append(tuple[0])
            tmp_y.append((tuple[1:]))
        x.append(tmp_x)
        y.append(tmp_y)
    return x, y

def shifter(sequences):
    """
    Shifts the sequences for teacher enforcing usage
    :param sequences:
    :return:
    """
    shifted_list = list()
    for sequence in sequences:
        tmp_shifted_list = list()
        for tuples in sequence:
            tmp = list()
            for values in tuples:
                length = len(values)
                tmp_values = [0]
                i = 0
                while i < (length - 1):
                    tmp_values += [values[i]]
                    i += 1
                tmp.append(tmp_values)
            tmp_shifted_list.append(tmp)
        shifted_list.append(tmp_shifted_list)
    return shifted_list

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