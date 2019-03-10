import os
import string
import re
from pickle import dump, load
from unicodedata import normalize
from numpy import array, append
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical


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

    testPair = clean_pairs(to_pairs(doctestDe, doctestEn))
    trainPair = clean_pairs(to_pairs(doctrainDe, doctrainEn))
    valPair = clean_pairs(to_pairs(docvalDe, docvalEn))

    traintestPair = append(trainPair, testPair, axis=0)

    save_clean_data(testPair, os.path.join(location, 'test.pkl'))
    save_clean_data(trainPair, os.path.join(location, 'train.pkl'))
    save_clean_data(traintestPair, os.path.join(location, 'traintest.pkl'))
    save_clean_data(valPair, os.path.join(location, 'val.pkl'))

    for i in range(10):
        print('[%s] => [%s]' % (traintestPair[i, 0], traintestPair[i, 1]))


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
        if len(lines1[i]) < 50 or len(lines2[i]) < 50:
            pairs.append([lines1[i], lines2[i]])
    #pairs = [[lines1[i], lines2[i]] for i in range(0, len(lines1))]
    #print(array(pairs[0]))
    return array(pairs)


def clean_pairs(lines):
    """
    cleans the lines by removing all non-printable charactersa and all punctuation characters.
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
    return tokenizer


def max_length(lines):
    return max(len(line.split()) for line in lines)


def prepareData(dataset, train, test):
    eng_tokenizer, eng_vocab_size, eng_length, ger_tokenizer, ger_vocab_size, ger_length = tokenize(dataset)
    # prepare training data
    trainX = encode_sequences(ger_tokenizer, ger_length, train[:, 1])
    trainY = encode_sequences(eng_tokenizer, eng_length, train[:, 0])
    trainY = encode_output(trainY, eng_vocab_size)
    # prepare validation data
    testX = encode_sequences(ger_tokenizer, ger_length, test[:, 1])
    testY = encode_sequences(eng_tokenizer, eng_length, test[:, 0])
    testY = encode_output(testY, eng_vocab_size)
    return ger_vocab_size, eng_vocab_size, ger_length, eng_length, trainX, trainY, testX, testY, eng_tokenizer


def tokenize(dataset):
    # prepare english tokenizer
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


"TODO Same preprocessing -> see cleaning" \
"Use validation set" \
"change to pairs to accept sentences of any length and resolve the memory error"
