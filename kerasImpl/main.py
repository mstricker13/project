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
    trainX, trainY, testX, testY, valX, valY = all_data[0], all_data[1], all_data[2], all_data[3], all_data[4], all_data[5]

    # define model
    #model parameters
    N_UNITS = 512 #same as hidden dimension in torch
    DROPOUT = 0.5
    LATENT_DIM = 256
    #13.898.501
    print('Define model')
    model = network.define_model(ger_vocab_size, eng_vocab_size, ger_length, eng_length, N_UNITS, LATENT_DIM, DROPOUT)
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    # summarize defined model
    print(model.summary())
    #trainer, inf_encoder, inf_decoder = network.define_model_powerful(ger_vocab_size, eng_vocab_size, ger_length, eng_length, N_UNITS, LATENT_DIM, DROPOUT)
    #trainer.compile(optimizer='adam', loss='categorical_crossentropy')
    #print(trainer.summary())

    #plot_model(model, to_file='model.png', show_shapes=True)
    # fit model
    filename = 'model.h5'
    #filename = 'modelPowerful.h5'
    checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    print('Train')
    model.fit(trainX, trainY, epochs=30, batch_size=128, validation_data=(valX, valY), callbacks=[checkpoint], verbose=2)
    #trainer.fit([trainX, trainY], epochs=3, batch_size=128, validation_data=(valX, valY), callbacks=[checkpoint], verbose=2)

    #evaluate model
    #model = load_model('model.h5')
    print('Test')
    #network.evaluate_model(model, eng_tokenizer, testX, test)
    scores = model.evaluate(testX, testY)
    print(scores)
    #network.evaluate_model(trainer, eng_tokenizer, testX, test)
    #translation = model.predict(source, verbose=0)

    #Test Loss after 30 epochs: 3.6736


"""
    #get iterators for data and vocabularies
    BATCH_SIZE = 128
    train_iterator, valid_iterator, test_iterator, SRC, TRG = data.startDataProcess(BATCH_SIZE, device)

    #define parameters for model architecture
    INPUT_DIM = len(SRC.vocab)
    OUTPUT_DIM = len(TRG.vocab)
    ENC_EMB_DIM = 256 ok
    DEC_EMB_DIM = 256 ok
    HID_DIM = 512 ok
    N_LAYERS = 2 ok
    ENC_DROPOUT = 0.5 ok
    DEC_DROPOUT = 0.5 ok

    #create encoder, decoder and seq2seq model
    enc = network.Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
    dec = network.Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
    model = network.Seq2Seq(enc, dec, device).to(device)
    print(f'The model has {network.count_parameters(model):,} trainable parameters')

    #define parameters for training
    optimizer = optim.Adam(model.parameters())
    pad_idx = TRG.vocab.stoi['<pad>']
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    N_EPOCHS = 10
    CLIP = 1
    SAVE_DIR = 'models'
    MODEL_SAVE_PATH = os.path.join(SAVE_DIR, 'tut1_model.pt')

    #start training
    #training.start_training(SAVE_DIR, MODEL_SAVE_PATH, N_EPOCHS, model, train_iterator, valid_iterator,optimizer,
                            #criterion, CLIP)

    #evaluate trained model
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    test_loss = training.evaluate(model, test_iterator, criterion)
    print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
"""