import os
import matplotlib.pyplot as plt

from kerasImplTimeSeries import data, network, create_models
from keras.utils.vis_utils import plot_model
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

from numpy import array

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
    saving_location = os.path.join(location, 'processed')
    #percentage of rows that were used to train Theta and need to be skipped for usage in our networks
    percentage = 0.25
    #defines the ratio of train, validation and test set the whole dataset is split into
    split_ratio = [0.7, 0.1, 0.2]
    #for model
    N_UNITS = 512
    DROPOUT = 0.5
    n_epochs = 3
    batch_size = 128

    #create the pkl files for training, validating and testing
    data.create_Pkl_CIF_File(location_CIF, saving_location, percentage, split_ratio)

    #load pkl files
    dataset = data.load_pkl(os.path.join(saving_location, 'allData.pkl'))
    train = data.load_pkl(os.path.join(saving_location, 'train.pkl'))
    test = data.load_pkl(os.path.join(saving_location, 'test.pkl'))
    val = data.load_pkl(os.path.join(saving_location, 'val.pkl'))

    #prepare data for network
    print('Prepare Data')
    #vocab_size, lang_length, all_data, eng_tokenizer = data.prepare_Data(dataset, train, test, val)
    feature_size, lengths, all_data = data.prepare_Data(train, test, val)
    input_length, output_length = lengths[0], lengths[1]
    trainX, trainY, trainY_shifted, testX, testY, testY_shifted, valX, valY, valY_shifted = all_data[0], all_data[1], all_data[2], all_data[3], all_data[4], all_data[5], all_data[6], all_data[7], all_data[8]
    trainX, trainY, trainY_shifted, testX, testY, testY_shifted, valX, valY, valY_shifted = array(trainX), array(trainY), array(trainY_shifted), array(testX), array(testY), array(testY_shifted), valX, valY, valY_shifted

    # define models
    print('Define models')
    models = create_models.create(feature_size, input_length, output_length, N_UNITS, DROPOUT)
    for modelInfo in models:
        model = modelInfo[0]
        name = modelInfo[1]
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

        #fit model
        filename = fileprefix + '.h5'
        checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        print('Train')
        history = model.fit([trainX, trainY_shifted], trainY, epochs=n_epochs, batch_size=batch_size, validation_data=([valX, valY_shifted], valY), callbacks=[checkpoint], verbose=2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig(fileprefix + '_fig')

        #evaluate model
        #model = load_model(fileprefix + '.h5')
        print('Test')
        #scores = model.evaluate([testX, testY_shifted], testY)
        #result = model.predict([testX, testY_shifted], batch_size=batch_size, verbose=0)
        #print(scores)
        #print(result)
