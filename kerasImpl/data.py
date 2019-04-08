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

def createPklFile(location):
    """
    creates pkl files for training , test and validation based on the folders inside the location parameter. It will
    also clean the text.
    """

    trainDeLoc = os.path.join(location, 'training', 'train.de')
    testDeLoc = os.path.join(location, 'test', 'test2016.de')
    valDeLoc = os.path.join(location, 'validation', 'val.de')
    trainEnLoc = os.path.join(location, 'training', 'train.en')
    testEnLoc = os.path.join(location, 'test', 'test2016.en')
    valEnLoc = os.path.join(location, 'validation', 'val.en')

    doctrainDe = load_doc(trainDeLoc)
    doctestDe = load_doc(testDeLoc)
    docvalDe = load_doc(valDeLoc)
    doctrainEn = load_doc(trainEnLoc)
    doctestEn = load_doc(testEnLoc)
    docvalEn = load_doc(valEnLoc)

    testPair = clean_pairs(to_pairs(doctestEn, doctestDe))
    trainPair = clean_pairs(to_pairs(doctrainEn, doctrainDe))
    valPair = clean_pairs(to_pairs(docvalEn, docvalDe))

    traintestPair = append(trainPair, testPair, axis=0)
    allPair = append(trainPair, testPair, axis=0)
    allPair = append(allPair, valPair, axis=0)

    save_clean_data(testPair, os.path.join(location, 'test.pkl'))
    save_clean_data(trainPair, os.path.join(location, 'train.pkl'))
    save_clean_data(traintestPair, os.path.join(location, 'traintest.pkl'))
    save_clean_data(allPair, os.path.join(location, 'allData.pkl'))
    save_clean_data(valPair, os.path.join(location, 'val.pkl'))

    #for i in range(10):
    #    print('[%s] => [%s]' % (traintestPair[i, 0], traintestPair[i, 1]))


def load_doc(filename):
    """
    loads a document specified by filename
    """
    file = open(filename, mode='rt', encoding='utf-8')
    text = file.read()
    file.close()
    return text


def to_pairs(doc1, doc2):
    """
    creates pairs, where a line in doc1 is paired with its corresponding line in doc2
    """
    lines1 = doc1.strip().split('\n')
    lines2 = doc2.strip().split('\n')
    pairs = list()
    for i in range(0, len(lines1)):
        #print(lines1[i])
        #TODO make possible for all lengths, right now sentence length in english is maximum 10
        if len(lines1[i].split(' ')) <= 10: #or len(lines2[i].split(' ')) < 10:
            #reverse german sentence since pytorch implementation does that
            tmp = lines2[i].split(' ')
            tmp.reverse()
            lines2[i] = ' '.join(tmp)
            pairs.append([lines1[i], lines2[i]])
    #pairs = [[lines1[i], lines2[i]] for i in range(0, len(lines1))]
    #print(array(pairs[0]))
    return array(pairs)


def clean_pairs(lines):
    """
    cleans the lines by removing all non-printable characters and all punctuation characters.
    Normalizes all Unicode characters to ASCII (e.g. Latin characters).
    Normalizes uppercase to lowercase.
    Remove any remaining tokens that are not alphabetic.
    """
    cleaned = list()
    # prepare regex for char filtering
    re_print = re.compile('[^%s]' % re.escape(string.printable))
    # prepare translation table for removing punctuation
    table = str.maketrans('', '', string.punctuation)
    for pair in lines:
        clean_pair = list()
        for line in pair:
            # normalize unicode characters
            line = normalize('NFD', line).encode('ascii', 'ignore')
            line = line.decode('UTF-8')
            # tokenize on white space
            line = line.split()
            # convert to lowercase
            line = [word.lower() for word in line]
            # remove punctuation from each token
            line = [word.translate(table) for word in line]
            # remove non-printable chars form each token
            line = [re_print.sub('', w) for w in line]
            # remove tokens with numbers in them
            line = [word for word in line if word.isalpha()]
            # store as string
            clean_pair.append(' '.join(line))
        cleaned.append(clean_pair)
    return array(cleaned)


def save_clean_data(sentences, filename):
    dump(sentences, open(filename, 'wb'))
    print('Saved: %s' % filename)


def load_clean_sentences(filename):
    return load(open(filename, 'rb'))


def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)

    #remove words which appear once, to be close to the pytorch implementation
    # 4717 --> 2558, 6775 --> 2828
    #count_thres = 2
    #low_count_words = [w for w, c in tokenizer.word_counts.items() if c < count_thres]
    #print(tokenizer.texts_to_sequences(lines))
    #for w in low_count_words:
    #    del tokenizer.word_index[w]
    #    del tokenizer.word_docs[w]
    #    del tokenizer.word_counts[w]
    #print(tokenizer.texts_to_sequences(lines))
    return tokenizer

def max_length(lines):
    return max(len(line.split()) for line in lines)


def prepareData(dataset, train, test, val, allData):
    #was previously on allData, but to recreate the pytorch network I am using the training data only. Prevents information leakage
    eng_tokenizer, eng_vocab_size, eng_length, ger_tokenizer, ger_vocab_size, ger_length = tokenize(train)
    # prepare training data
    trainX = encode_sequences(ger_tokenizer, ger_length, train[:, 1])
    trainY = encode_sequences(eng_tokenizer, eng_length, train[:, 0])
    trainY = encode_output(trainY, eng_vocab_size)
    # prepare testing data
    testX = encode_sequences(ger_tokenizer, ger_length, test[:, 1])
    testY = encode_sequences(eng_tokenizer, eng_length, test[:, 0])
    testY = encode_output(testY, eng_vocab_size)
    # prepare validation data
    valX = encode_sequences(ger_tokenizer, ger_length, val[:, 1])
    valY = encode_sequences(eng_tokenizer, eng_length, val[:, 0])
    valY = encode_output(valY, eng_vocab_size)
    vocab_size = [ger_vocab_size, eng_vocab_size]
    lang_length = [ger_length, eng_length]
    all_data = [trainX, trainY, testX, testY, valX, valY]
    return vocab_size, lang_length, all_data, eng_tokenizer


def tokenize(dataset):
    # prepare english tokenizer
    thresh = 2
    #TODO why does it remove things even though if I rename dataset to tmp and do not use it anywhere
    datasetTMP = remove_below_thresh(dataset, thresh)
    eng_tokenizer = create_tokenizer(dataset[:, 0])
    eng_vocab_size = len(eng_tokenizer.word_index) + 1
    eng_length = max_length(dataset[:, 0])
    print('English Vocabulary Size: %d' % eng_vocab_size)
    print('English Max Length: %d' % (eng_length))
    # prepare german tokenizer
    ger_tokenizer = create_tokenizer(dataset[:, 1])
    ger_vocab_size = len(ger_tokenizer.word_index) + 1
    ger_length = max_length(dataset[:, 1])
    print('German Vocabulary Size: %d' % ger_vocab_size)
    print('German Max Length: %d' % (ger_length))
    return eng_tokenizer, eng_vocab_size, eng_length, ger_tokenizer, ger_vocab_size, ger_length

def remove_below_thresh(dataset, thresh):
    eng = dataset[:, 0]
    engDict = {}
    engIgnore = []
    for sentence in eng:
        words = sentence.split(' ')
        for word in words:
            if word in engDict:
                engDict[word] += 1
            else:
                engDict[word] = 1
    for key in engDict:
        if engDict[key] <= thresh:
            engIgnore.append(key)
    i = 0
    while i < len(eng):
        words = eng[i].split(' ')
        for check in engIgnore:
            if check in words:
                eng[i] = eng[i].replace(" " + check + " ", " unk ")
        i += 1
    ger = dataset[:, 1]
    gerDict = {}
    gerIgnore = []
    for sentence in ger:
        words = sentence.split(' ')
        for word in words:
            if word in gerDict:
                gerDict[word] += 1
            else:
                gerDict[word] = 1
    for key in gerDict:
        if gerDict[key] <= thresh:
            gerIgnore.append(key)
    i = 0
    while i < len(ger):
        words = ger[i].split(' ')
        for check in gerIgnore:
            if check in words:
                ger[i] = ger[i].replace(" " + check + " ", " unk ")
        i += 1

    pair = list()
    i = 0
    while i < len(eng):
        pair.append([eng[i], ger[i]])
        i += 1
    result = array(pair)
    return result

# encode and pad sequences
def encode_sequences(tokenizer, length, lines):
    # integer encode sequences
    X = tokenizer.texts_to_sequences(lines)
    # pad sequences with 0 values
    X = pad_sequences(X, maxlen=length, padding='post')
    return X


# one hot encode target sequence
def encode_output(sequences, vocab_size):
    ylist = list()
    for sequence in sequences:
        encoded = to_categorical(sequence, num_classes=vocab_size)
        ylist.append(encoded)
    y = array(ylist)
    y = y.reshape(sequences.shape[0], sequences.shape[1], vocab_size)
    return y


# map an integer to a word
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


# Same preprocessing -> see cleaning -> done
# Use validation set -> done
# TODO change to pairs to accept sentences of any length and resolve the memory error
# reverse input sentence
