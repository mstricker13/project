import os
import numpy as np
if __name__ == '__main__':
    p = os.path.join('output', 'NN5run1', 'TestNN5_th_D_1mean_sMape.txt')
    with open(p) as f:
        text = f.read()
    values = list()
    for line in text.split('\n')[1:-1]:
        values.append(float(line.split(' ')[-1]))
    print(values)
    mean = np.mean(values)
    print(mean)