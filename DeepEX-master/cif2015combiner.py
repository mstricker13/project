def main():
    emptyvaluesadder()

def emptyvaluesadder():
    with open('cif2015_complete.csv', 'r') as f:
        text = f.read()
    max = 0

    for line in text.split('\n'):
        length = len(line.split(','))
        if length > max:
            max = length

    with open('cif2015_completeEmptyVal.csv', 'w+') as f:
        for line in text.split('\n'):
            if len(line.split(',')) < max:
                diff = max - len(line.split(','))
                line_write = line
                for i in range(diff):
                    line_write += ','
                f.write(line_write+'\n')
            else:
                f.write(line + '\n')

def combine():
    train = 'cif2015.csv'
    test = 'cif2015_test.csv'

    with open(train) as f:
        train_content = f.read()
    with open(test) as f:
        test_content = f.read()

    with open('cif2015_complete.csv', 'w+') as f:
        for linetrain, linetest in zip(train_content.split('\n'), test_content.split('\n')):
            linetest = ','.join(linetest.split(';')[1:])
            f.write(linetrain+','+linetest+'\n')

if __name__ == '__main__':
    main()