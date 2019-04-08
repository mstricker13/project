import os
from random import shuffle

if __name__ == '__main__':
    filepath = os.path.join('kerasImpl', 'data', 'deu-eng', 'deu.txt')
    filepathTestD = os.path.join('kerasImpl', 'data', 'test', 'test2016S.de')
    filepathTestE = os.path.join('kerasImpl', 'data', 'test', 'test2016S.en')
    filepathTrainD = os.path.join('kerasImpl', 'data', 'training', 'trainS.de')
    filepathTrainE = os.path.join('kerasImpl', 'data', 'training', 'trainS.en')
    filepathValD = os.path.join('kerasImpl', 'data', 'validation', 'valS.de')
    filepathValE = os.path.join('kerasImpl', 'data', 'validation', 'valS.en')

    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')
    shuffle(lines)
    length = len(lines)
    train = int(length*0.6)
    val = int(length*0.2)
    test = length - train - val
    file1 = open(filepathTestD, 'w', encoding='utf-8')
    file2 = open(filepathTestE, 'w', encoding='utf-8')
    i = 0
    #first is english
    while i < test:
        line = lines[i]
        sentences = line.split('\t')
        file2.write(sentences[0]+'\n')
        file1.write(sentences[1]+'\n')
        i += 1
    file1.close()
    file2.close()

    file1 = open(filepathTrainD, 'w', encoding='utf-8')
    file2 = open(filepathTrainE, 'w', encoding='utf-8')
    i = test
    # first is english
    while i < (test + train):
        line = lines[i]
        sentences = line.split('\t')
        file2.write(sentences[0] + '\n')
        file1.write(sentences[1] + '\n')
        i += 1
    file1.close()
    file2.close()

    file1 = open(filepathValD, 'w', encoding='utf-8')
    file2 = open(filepathValE, 'w', encoding='utf-8')
    i = (test + train)
    # first is english
    while i < (test + train + val):
        line = lines[i]
        sentences = line.split('\t')
        file2.write(sentences[0] + '\n')
        file1.write(sentences[1] + '\n')
        i += 1
    file1.close()
    file2.close()