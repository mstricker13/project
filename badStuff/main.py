from __future__ import unicode_literals, print_function, division
import random
from badStuff import network, data
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    print(device)
    #get the data
    input_lang, input_max, output_lang, output_max, pairs = data.prepareData('eng', 'fra', True)
    print(random.choice(pairs))
    abs_max = input_max if input_max > output_max else output_max
    print('Longest Sentence has ' + str(abs_max) + ' words')

    #train the model
    hidden_size = 256
    encoder1 = network.EncoderRNN(input_lang.n_words, hidden_size).to(device)
    decoder1 = network.AttnDecoderRNN(hidden_size, output_lang.n_words, abs_max, dropout_p=0.1).to(device) #
    #75000 5000
    network.trainIters(encoder1, decoder1, 10, pairs, input_lang, output_lang, abs_max, print_every=1)

#TODO Split of train and val
#TODO use lstm