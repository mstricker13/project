import os
import math
import numpy as np

def main():
    gt = 'cif2015_completeEmptyVal.csv'
    theta = 'theta_50_hT_cif15_12.csv'
    horizon = -6
    horizon_index = 2
    mapes = list()
    with open(gt) as f:
        gt_content = f.read()
    with open(theta) as f:
        theta_content = f.read()
    for line, line_theta in zip(gt_content.split('\n')[:-1], theta_content.split('\n')[2:-1]):
        values = line.split(',')
        values_theta = line_theta.split(',')
        #horizon = (-1)*int(values[horizon_index])
        #remove empty values
        values = [val for val in values if val != '']
        values_theta = [val for val in values_theta if val != 'NA']
        gt_values = values[horizon:]
        gt_values = [float(val) for val in gt_values]
        prediction = values_theta[horizon:]
        prediction = [float(val) for val in prediction]
        mape = sMAPE(prediction, gt_values, ((-1)*horizon))
        #print(mape)
        mapes.append(mape)
    print(np.mean(mapes))


def sMAPE(prediction, ground_truth, horizon):
    add = 0
    #print(prediction)
    #print(ground_truth)
	#CIF and M3 calculate different sMAPE equations, NOT NICE!
    for pred, gt in zip(prediction, ground_truth):
        #add += (abs(gt - pred) / ((abs(gt) + abs(pred))) / 2)
        add += (abs(gt - pred) / ((gt + pred) / 2))
    smape = (add/horizon) * 100
    #smape = (add) * 100
    return smape

if __name__ == '__main__':
    main()