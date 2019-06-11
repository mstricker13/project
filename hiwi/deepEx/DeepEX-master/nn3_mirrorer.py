import os

def main():
    location = 'data'
    in_path = os.path.join('nn3_no_meta.csv')
    outpath_path = os.path.join('nn3_no_meta_mirror.csv')

    new_text = ''
    with open(in_path) as f:
        text = f.read()
    for i in range(0, 111):
        new_text_tmp = list()
        for line in text.split('\n'):
            values = line.split(',')
            new_text_tmp.append(values[i])
        t = ','.join(new_text_tmp)
        new_text += t + '\n'
    with open(outpath_path, 'w+') as f:
        f.write(new_text)

if __name__ == '__main__':
    main()