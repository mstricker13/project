import torch
import torch.nn as nn
import torch.optim as optim

from torchImpl import training, network, data

import os
import math

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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