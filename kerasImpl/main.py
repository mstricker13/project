import os
import tensorflow as tf

from kerasImpl import data, network
from keras.utils.vis_utils import plot_model
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

if __name__ == '__main__':

    #sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

    #some configurations to fix an allocation error with cublas --> sometimes the error comes sometimes it doesn't...
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    config.log_device_placement = True  # to log device placement (on which device the operation ran)
    # (nothing gets printed in Jupyter, only if you run it standalone)
    sess = tf.Session(config=config)
    set_session(sess)  # set this TensorFlow session as the default session for Keras

    #location of our data
    location = 'data'
    #create the pkl files for training and testing
    print('Create Pkl files')
    data.createPklFile(location)
    dataset = data.load_clean_sentences(os.path.join(location, 'traintest.pkl'))
    train = data.load_clean_sentences(os.path.join(location, 'train.pkl'))
    test = data.load_clean_sentences(os.path.join(location, 'test.pkl'))
    val = data.load_clean_sentences(os.path.join(location, 'val.pkl'))
    allData = data.load_clean_sentences(os.path.join(location, 'val.pkl'))
    #prepare data for network
    print('Prepare Data')
    vocab_size, lang_length, all_data, eng_tokenizer = data.prepareData(dataset, train, test, val, allData)
    ger_vocab_size, eng_vocab_size = vocab_size[0], vocab_size[1]
    ger_length, eng_length = lang_length[0], lang_length[1]
    trainX, trainY, testX, testY, valX, valY, trainY_shifted, valY_shifted, testY_shifted = \
        all_data[0], all_data[1], all_data[2], all_data[3], all_data[4], all_data[5], all_data[6], all_data[7], all_data[8]

    # define model
    #model parameters
    N_UNITS = 512 #same as hidden dimension in torch
    DROPOUT = 0.5
    LATENT_DIM = 256
    print('Define model')

    #n_features = ger_vocab_size
    #n_features2 = eng_vocab_size
    #n_steps_in = ger_length
    #n_steps_out = eng_length
    #X1, X2, y = network.get_dataset(n_steps_in, n_steps_out, n_features, n_features2, 137)
    #print(X1.shape)
    #print(trainX.shape)
    #print(X1[0])
    #print(trainX[0])

    #model = network.define_model(ger_vocab_size, eng_vocab_size, ger_length, eng_length, N_UNITS, LATENT_DIM, DROPOUT)
    model, enc_inf, dec_inf = network.define_model_powerful(ger_vocab_size, eng_vocab_size, ger_length, eng_length, N_UNITS, LATENT_DIM, DROPOUT)
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    # summarize defined model
    print(model.summary())
    plot_model(model, to_file='model.png', show_shapes=True)
    #trainer, inf_encoder, inf_decoder = network.define_model_powerful(ger_vocab_size, eng_vocab_size, ger_length, eng_length, N_UNITS, LATENT_DIM, DROPOUT)
    #trainer.compile(optimizer='adam', loss='categorical_crossentropy')
    #print(trainer.summary())

    #plot_model(model, to_file='model.png', show_shapes=True)
    # fit model
    filename = 'modelPowerful.h5'
    checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    print('Train')
    #print(len(trainX))
    #print(len(trainY[0]))
    #print('----------------')
    #print(trainY)
    #if trainX has wrong shapes remove one hot encoding for it
    #model.fit(trainX, trainY, epochs=30, batch_size=128, validation_data=(valX, valY), callbacks=[checkpoint], verbose=2)
    model.fit([trainX, trainY_shifted], trainY, epochs=30, batch_size=128, validation_data=([valX, valY_shifted], valY),
              callbacks=[checkpoint], verbose=2)
    #print(X1.shape, X2.shape, y.shape)
    #print(trainX.shape, trainY_shifted.shape, trainY.shape)

    #evaluate model
    #model = load_model('model.h5')
    print('Test')
    #network.evaluate_model(model, eng_tokenizer, testX, test)
    scores = model.evaluate([testX, testY_shifted], testY)
    print(scores)
    #network.evaluate_model(trainer, eng_tokenizer, testX, test)
    #translation = model.predict(source, verbose=0)

    #Test Loss after 30 epochs: 3.6736
    #Test Loss after 30 epochs for powerful model: 1.5128