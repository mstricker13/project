import numpy as np
import sys
import math

def sMAPE(prediction, ground_truth, horizon, transformer, fileprefix, series_counter):
    prediction = prediction[0]
    ground_truth = ground_truth[0]
    add = 0
    res_file = open(fileprefix + 'values_' + str(series_counter) + '.txt', 'w')
    for pred, gt in zip(prediction, ground_truth):
        #prediction has been transformed and logarithmed need to reverse
        reversed_prediction = transformer.inverse_transform(pred)
        reversed_prediction = math.exp(reversed_prediction[0])
        add += (abs(gt[0] - reversed_prediction) / ((abs(gt[0]) + abs(reversed_prediction))) / 2)
        #add += (abs(gt[0] - pred[0])/((abs(gt[0])+abs(pred[0])))/2)
        res_file.write(str(reversed_prediction) + ',' + str(gt[0]) + '\n')
    res_file.close()
    smape = (add/horizon) * 100
    return smape