import os
import matplotlib.pyplot as plt

from kerasImplTimeSeries import data, network, create_models, utils, result_compiler
from keras.utils.vis_utils import plot_model
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

from numpy import array
import numpy as np

import sys

if __name__ == '__main__':

    #some configurations to fix an allocation error with cublas --> sometimes the error comes sometimes it doesn't...
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    config.log_device_placement = True  # to log device placement (on which device the operation ran)
    # (nothing gets printed in Jupyter, only if you run it standalone)
    sess = tf.Session(config=config)
    set_session(sess)  # set this TensorFlow session as the default session for Keras


    #definition of Hyperparameters

    #location where the folders are, which are containing the defining csv files
    location = 'data'
    # location of the CIF folder, containing the CIF csv file
    location_CIF = os.path.join(location, 'CIF', 'cif_dataset_complete.csv')
    location_Theta = os.path.join(location, 'Theta-Predictions', 'theta_25_h1.csv')
    location_Theta2 = os.path.join(location, 'Theta-Predictions', 'theta_25_h1_CIF.csv')
    saving_location = os.path.join(location, 'processed')
    #percentage of rows that were used to train Theta and need to be skipped for usage in our networks
    percentage = 0.25
    #defines the ratio of train, validation and test set the whole dataset is split into
    split_ratio = [0.7, 0.1, 0.2] #TODO why a split ratio for test if that are just our 6/12 values defined in the line
    #TODO therefore exclude these horizons and use them as seperate test?
    #the last x values which are to be used for forecasting values
    window_size = 3 #TODO change to good value, for debugging purposes is short now
    #number of steps to move the window #TODO what step size?
    step_size = 1
    #horizon number of values to be predicted in addition to the horizon defined by the
    horizon = [1, 3] #1,3
    #for model
    n_epochs = 3
    batch_size = 128

    # format Theta file
    utils.convert_Theta_to_CIF_format(location_Theta)

    #create the pkl files for training, validating and testing
    data.create_Pkl_CIF_File(location_CIF, saving_location, percentage, split_ratio, window_size, step_size, horizon)

    data.create_Pkl_Theta_File(location_Theta2, saving_location, percentage, split_ratio, window_size, step_size, horizon)

    #load pkl files
    dataset = data.load_pkl(os.path.join(saving_location, 'allData.pkl'))
    train = data.load_pkl(os.path.join(saving_location, 'train.pkl'))
    test = data.load_pkl(os.path.join(saving_location, 'test.pkl'))
    val = data.load_pkl(os.path.join(saving_location, 'val.pkl'))

    dataset_theta = data.load_pkl(os.path.join(saving_location, 'allDataTheta.pkl'))
    train_theta = data.load_pkl(os.path.join(saving_location, 'trainTheta.pkl'))
    test_theta = data.load_pkl(os.path.join(saving_location, 'testTheta.pkl'))
    val_theta = data.load_pkl(os.path.join(saving_location, 'valTheta.pkl'))

    #prepare data for network
    print('Prepare Data')
    #vocab_size, lang_length, all_data, eng_tokenizer = data.prepare_Data(dataset, train, test, val)
    #feature_size, lengths, all_data = data.prepare_Data(train, test, val)
    #input_length, output_length = lengths[0], lengths[1]
    #trainX, trainY, trainY_shifted, testX, testY, testY_shifted, valX, valY, valY_shifted = array(trainX), array(trainY), array(trainY_shifted), array(testX), array(testY), array(testY_shifted), valX, valY, valY_shifted

    all_data = data.prepare_Data(train, test, val)
    trainX, trainY, trainY_shifted, testX, testY, testY_shifted, valX, valY, valY_shifted = all_data[0], all_data[1], \
                                                                                            all_data[2], all_data[3], \
                                                                                            all_data[4], all_data[5], \
                                                                                            all_data[6], all_data[7], \
                                                                                            all_data[8]

    all_data_theta = data.prepare_Data(train_theta, test_theta, val_theta)
    trainX_theta, trainY_theta, trainY_shifted_theta, testX_theta, testY_theta, testY_shifted_theta, valX_theta, valY_theta, valY_shifted_theta = all_data_theta[0], all_data_theta[1], \
                                                                                            all_data_theta[2], all_data_theta[3], \
                                                                                            all_data_theta[4], all_data_theta[5], \
                                                                                            all_data_theta[6], all_data_theta[7], \
                                                                                            all_data_theta[8]

    # define models
    print('Define models')
    feature_size = 1
    input_length = window_size
    new_horizons = [len(predicts) for predicts in trainY[0][0]]
    output_length = new_horizons
    models = create_models.create(feature_size, input_length, output_length)
    for modelInfo in models:
        model = modelInfo[0]
        current_horizon = modelInfo[2]
        theta = '' #TODO check if theta data is used if yes = '_T'
        name = modelInfo[1] + theta + '_' + str(current_horizon) + '_' + str(window_size) + '_' + str(step_size)
        #summarize defined model
        print(model.summary())
        folderpath = os.path.join('output', name)
        try:
           #Create target Directory
           os.mkdir(folderpath)
           print("Directory ", folderpath, " Created ")
        except FileExistsError:
           print("Directory ", folderpath, " already exists")
        fileprefix = os.path.join(folderpath, name)
        plot_model(model, to_file=fileprefix + '.png', show_shapes=True)
        #iterate through each sequence and train model on it
        i = 1
        mapes = []
        #for trainXS, trainYS, trainY_shiftedS, testXS, testYS, testY_shiftedS, valXS, valYS, valY_shiftedS in zip(trainX, trainY, trainY_shifted_theta, testX, testY, testY_shifted_theta, valX, valY, valY_shifted_theta):
        for trainXS, trainYS, trainY_shiftedS, testXS, testYS, testY_shiftedS, valXS, valYS, valY_shiftedS in zip(trainX, trainY, trainY_shifted, testX, testY, testY_shifted, valX, valY, valY_shifted):
            # TODO reshape input arrays somewhere here
            #get the predictions from targets matching with the current horizon
            trainYS = utils.get_matching_predictions(trainYS, current_horizon)
            trainY_shiftedS = utils.get_matching_predictions(trainY_shiftedS, current_horizon)
            testYS = utils.get_matching_predictions(testYS, current_horizon)
            testY_shiftedS = utils.get_matching_predictions(testY_shiftedS, current_horizon)
            valYS = utils.get_matching_predictions(valYS, current_horizon)
            valY_shiftedS = utils.get_matching_predictions(valY_shiftedS, current_horizon)
            #Because the length of the horizon determines the sample number we have to get it from the train sets!
            example_number_train = len(trainYS)
            example_number_test = len(testYS)
            example_number_val = len(valYS)
            #therefore trainXS might have more samples because of the cases where it matched with smaller horizons, take only matching amount!
            trainXS = trainXS[:example_number_train]
            testXS = testXS[:example_number_test]
            valXS = valXS[:example_number_val]

            if current_horizon == 12:
                if i == 49:
                    print(trainXS, trainYS, trainY_shiftedS)
                    print(testXS, testYS, testY_shiftedS)
                    print(valXS, valYS, valY_shiftedS)
                    print(example_number_val, example_number_test, example_number_train)
                    print(len(trainXS), len(testYS), len(valYS))

            #reshape list to match network input
            trainXS = array(trainXS).reshape(example_number_train, window_size, 1)
            trainYS = array(trainYS).reshape(example_number_train, current_horizon, 1)
            trainY_shiftedS = array(trainY_shiftedS).reshape(example_number_train, current_horizon, 1)
            testXS = array(testXS).reshape(example_number_test, window_size, 1)
            testYS = array(testYS).reshape(example_number_test, current_horizon, 1)
            testY_shiftedS = array(testY_shiftedS).reshape(example_number_test, current_horizon, 1)
            valXS = array(valXS).reshape(example_number_val, window_size, 1)
            valYS = array(valYS).reshape(example_number_val, current_horizon, 1)
            valY_shiftedS = array(valY_shiftedS).reshape(example_number_val, current_horizon, 1)

            #fit model
            filename = fileprefix + '_' + str(i) + '.h5'
            checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
            print('Train')
            history = model.fit([trainXS, trainY_shiftedS], trainYS, epochs=n_epochs, batch_size=batch_size, validation_data=([valXS, valY_shiftedS], valYS), callbacks=[checkpoint], verbose=2)
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'val'], loc='upper left')
            plt.savefig(fileprefix + '_' + str(i) + '_fig')

            #evaluate model
            #model = load_model(fileprefix + '.h5')
            print('Test')
            #scores = model.evaluate([testXS, testY_shiftedS], testYS)
            #result = model.predict([testX, testY_shifted], batch_size=batch_size, verbose=0)
            #smape = result_compiler.sMAPE(result[len(testX)-current_horizon], testY[len(testX)-current_horizon], current_horizon)
            #mapes += [smape]
            #print(scores)
            #print(result)
            i += 1
        #print(np.mean(mapes))