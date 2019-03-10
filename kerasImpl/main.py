import os
import tensorflow as tf

from kerasImpl import data, network
from keras.utils.vis_utils import plot_model
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

if __name__ == '__main__':

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

    #some configurations to fix an allocation error with cublas --> sometimes the error comes sometimes it doesn't...
    #config = tf.ConfigProto()
    #config.gpu_options.allow_growth = True
    #session = tf.Session(config=config)

    #location of our data
    location = 'data'
    #create the pkl files for training and testing
    print('Create Pkl files')
    data.createPklFile(location)
    dataset = data.load_clean_sentences(os.path.join(location, 'traintest.pkl'))
    train = data.load_clean_sentences(os.path.join(location, 'train.pkl'))
    test = data.load_clean_sentences(os.path.join(location, 'test.pkl'))
    #prepare data for network
    print('Prepare Data')
    ger_vocab_size, eng_vocab_size, ger_length, eng_length, trainX, trainY, testX, testY, eng_tokenizer = \
        data.prepareData(dataset, train, test)

    # define model
    print('Define model')
    model = network.define_model(ger_vocab_size, eng_vocab_size, ger_length, eng_length, 256)
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    # summarize defined model
    print(model.summary())
    #plot_model(model, to_file='model.png', show_shapes=True)
    # fit model
    filename = 'model.h5'
    checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    print('Train')
    model.fit(trainX, trainY, epochs=3, batch_size=64, validation_data=(testX, testY), callbacks=[checkpoint],
              verbose=2)

    #evaluate model
    #model = load_model('model.h5')
    print('test')
    network.evaluate_model(model, eng_tokenizer, testX, test)
    #translation = model.predict(source, verbose=0)


"""
    #get iterators for data and vocabularies
    BATCH_SIZE = 128
    train_iterator, valid_iterator, test_iterator, SRC, TRG = data.startDataProcess(BATCH_SIZE, device)

    #define parameters for model architecture
    INPUT_DIM = len(SRC.vocab)
    OUTPUT_DIM = len(TRG.vocab)
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