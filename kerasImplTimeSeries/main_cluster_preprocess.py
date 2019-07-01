import os
import matplotlib.pyplot as plt

from kerasImplTimeSeries import data, network, create_models, utils, result_compiler, preprocessor
from keras.utils.vis_utils import plot_model
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras import backend as K
from keras import optimizers

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

from numpy import array
import numpy as np
import math

import sys

if __name__ == '__main__':

    # some configurations to fix an allocation error with cublas --> sometimes the error comes sometimes it doesn't...
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
    # csv files are supposed to NOT have an empty last line!!!
    location_CIF = os.path.join(location, 'CIF', 'cif_dataset_small.csv')
    location_Theta = os.path.join(location, 'Theta-Predictions', 'theta_25_horg_small.csv')
    location_Theta2 = os.path.join(location, 'Theta-Predictions', 'theta_25_horg_CIF.csv')
    saving_location = os.path.join(location, 'processed')
    #percentage of rows that were used to train Theta and need to be skipped for usage in our networks
    percentage = 0.25
    #defines the ratio of train, validation and test set the whole dataset is split into
    split_ratio = [0.7, 0.3]#must add up to 1
    #the last x values which are to be used for forecasting values
    window_size = 7
    #number of steps to move the window
    step_size = 1
    #horizon number of values to be predicted in addition to the horizon defined by the
    horizon = [3] #1,3 #TODO taking only last sample makes support for multiple horizon testing weird, therefore only one for now
    #TODO however now it would work, but since I am lazy and had to add the function change_test_to_true_horizon I didnt account for that there
    use_csv_horizon = True #instead of using the defined horizon use the horizon defined by the csv
    #for model
    n_epochs = 100
    batch_size = 32

    #create the pkl files for training, validating and testing
    data.create_Pkl_CIF_File(location_CIF, saving_location, percentage, split_ratio, window_size, step_size, horizon, use_csv_horizon, step_size)

    #format Theta file
    utils.convert_Theta_to_CIF_format(location_Theta, location_Theta2, location_CIF, percentage)
    #create the corresponding Theta file for training val, and test
    data.create_Pkl_Theta_File(location_Theta2, saving_location, percentage, split_ratio, window_size, step_size, horizon, use_csv_horizon, step_size)

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
    new_horizons = list()
    #for series in trainY:
    #    print(series)
    #    new_horizons.append(len(series[0][0]))
    #new_horizons = [len(predicts) for predicts in trainY[0][0]]
    #output_length = new_horizons
    #for horizon in output_length:

    #iterate through each sequence and train model on it
    i = 1
    mapes = []
    #XXX_shiftedS is the expert data, #TODO rename
    for trainXS, trainYS, trainY_ES, testXS, testYS, testY_ES, valXS, valYS, valY_ES, trainX_E, testX_E, valX_E in zip(trainX, trainY, trainY_theta, testX, testY, testY_theta, valX, valY, valY_theta, trainX_theta, testX_theta, valX_theta):
        #print(trainYS)
        #print(trainY_shiftedS)
        #TODO Dynamic Windowsize, next two lines
        input_length = len(trainXS[0])
        window_size = input_length
        horizon = len(testYS[0][0])
        #TODO Stacked input
        model, name = network.define_model_2l_simple(feature_size, input_length, horizon)
        optimizer = optimizers.sgd(lr=0.01, decay=1e-3, clipnorm=1.)#, momentum=0.99, nesterov=True)
        #optimizer = optimizers.sgd(clipnorm=1.)
        model.compile(optimizer=optimizer, loss='mean_squared_error')
        #model.compile(optimizer='adam', loss='mean_squared_error')

        # compile model for each single sequence
        # model = modelInfo[0]
        # model.compile(optimizer='adam', loss='mean_squared_error')
        # summarize defined model
        print(model.summary())
        current_horizon = horizon
        theta = ''  # TODO check if theta data is used if yes = '_T'
        name = name + theta + '_' + str('th') + '_' + str('D') + '_' + str(step_size)
        folderpath = os.path.join('output', name)
        try:
            # Create target Directory
            os.mkdir(folderpath)
            print("Directory ", folderpath, " Created ")
        except FileExistsError:
            print("Directory ", folderpath, " already exists")
        fileprefix = os.path.join(folderpath, name)
        plot_model(model, to_file=fileprefix + '.png', show_shapes=True)

        # get the predictions from targets matching with the current horizon.
        # this was needed for multiple horizon testing, therefore only effect now is to reduce the dimension
        trainYS = utils.get_matching_predictions(trainYS, current_horizon)
        trainY_ES = utils.get_matching_predictions(trainY_ES, current_horizon)
        test_horizon = len(testYS[0][0])  # TODO another reason to not support multiple horizons
        testYS = utils.get_matching_predictions(testYS, test_horizon)  # TODO another reason to not support multiple horizons
        testY_ES = utils.get_matching_predictions(testY_ES, test_horizon)  # TODO another reason to not support multiple horizons
        valYS = utils.get_matching_predictions(valYS, current_horizon)
        valY_ES = utils.get_matching_predictions(valY_ES, current_horizon)

        trainXS, trainYS, trainY_ES, testXS, testYS, testY_ES, valXS, valYS, valY_ES, norm_val, seasonality_list = \
            preprocessor.cluster_process(trainXS, trainYS, trainY_ES, testXS, testYS, testY_ES, valXS, valYS, valY_ES)

        # Because the length of the horizon determines the sample number we have to get it from the train sets!
        example_number_train = len(trainYS)
        example_number_test = len(testYS)
        example_number_val = len(valYS)
        # therefore trainXS might have more samples because of the cases where it matched with smaller horizons, take only matching amount!
        trainXS = trainXS[:example_number_train]
        testXS = testXS[:example_number_test]
        valXS = valXS[:example_number_val]

        trainYS = utils.shape_transformed_toinput(trainYS, example_number_train, current_horizon)
        trainXS = utils.shape_transformed_toinput(trainXS, example_number_train, window_size)
        trainY_ES = utils.shape_transformed_toinput(trainY_ES, example_number_train, current_horizon)
        testXS = utils.shape_transformed_toinput(testXS, example_number_test, window_size)
        testY_ES = utils.shape_transformed_toinput(testY_ES, example_number_test, test_horizon)
        testYS = array(testYS).reshape(example_number_test, test_horizon, 1)
        valXS = utils.shape_transformed_toinput(valXS, example_number_val, window_size)
        valYS = utils.shape_transformed_toinput(valYS, example_number_val, current_horizon)
        valY_ES = utils.shape_transformed_toinput(valY_ES, example_number_val, current_horizon)

        #fit model
        filename = fileprefix + '_' + str(i) + '.h5'
        checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        print('Train')

        #print(trainXS)
        #print(transformer.inverse_transform(trainXS))
        #trainXS = transformer.inverse_transform(trainXS)
        #trainY_ES = transformer.inverse_transform(trainY_ES)
        #trainYS = transformer.inverse_transform(trainYS)
        #valXS = transformer.inverse_transform(valXS)
        #valY_ES = transformer.inverse_transform(valY_ES)
        #valYS = transformer.inverse_transform(valYS)
        #testXS = transformer.inverse_transform(testXS)
        #testY_ES = transformer.inverse_transform(testY_ES)

        history = model.fit([trainXS, trainY_ES], trainYS, epochs=n_epochs, batch_size=batch_size,
                            validation_data=([valXS, valY_ES], valYS), callbacks=[checkpoint], verbose=2)

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig(fileprefix + '_' + str(i) + '_fig')
        plt.close()

        #evaluate model
        #model = load_model(fileprefix + '.h5')
        print('Test')
        result = model.predict([testXS, testY_ES], batch_size=batch_size, verbose=0)
        #result = []
        #iter_count = 0
        #while iter_count < test_horizon:
        #    result_tmp = model.predict([testXS, (array([testY_ES[0][iter_count:(current_horizon + iter_count)]]))], batch_size=batch_size, verbose=0)
        #    iter_count += current_horizon
        #    for result_tmp_value in result_tmp[0]:
        #        result += [result_tmp_value[0]]
        result = array(result).reshape(example_number_test, test_horizon, 1)
        smape = result_compiler.sMAPE_paper(result, testYS, current_horizon, norm_val, fileprefix, i, seasonality_list)
        mapes += [smape]
        i += 1
        tf.reset_default_graph()
        K.clear_session()
    smape_file = open(fileprefix + 'mean_sMape.txt', 'w')
    smape_file.write('Mean over all sMapes = ' + str(np.mean(mapes)) + '\n')
    i = 1
    for value in mapes:
        smape_file.write('sMape_' + str(i) + ' = ' + str(value) + '\n')
        i += 1
    smape_file.close()
    print('done')