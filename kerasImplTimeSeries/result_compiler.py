import numpy as np
import sys
import math

def sMAPE(prediction, ground_truth, horizon, transformer, fileprefix, series_counter):
    #print('----------------------------------------')
    #print(horizon)
    #print(prediction)
    #print(ground_truth)
    prediction = prediction[0]
    ground_truth = ground_truth[0]
    #print(prediction)
    #print(ground_truth)
    add = 0
    #TODO somehow check if first iteration and delete old file
    res_file_pred = open(fileprefix + '_predictions' + '.csv', 'a')
    res_file_gt = open(fileprefix + '_gt' + '.csv', 'a')
    res_file_pred.write('ts' + str(series_counter))
    res_file_gt.write('ts' + str(series_counter))
    for pred, gt in zip(prediction, ground_truth):
        #print('loop')
        #print(pred)
        #print(gt)
        #prediction has been transformed and logarithmed need to reverse
        reversed_prediction = transformer.inverse_transform(pred)
        reversed_prediction = math.exp(reversed_prediction[0])
        #print(reversed_prediction)
        add += (abs(gt[0] - reversed_prediction) / (((abs(gt[0]) + abs(reversed_prediction))) / 2))
        #add += (abs(gt[0] - pred[0])/((abs(gt[0])+abs(pred[0])))/2)
        res_file_pred.write(',' + str(reversed_prediction))
        res_file_gt.write(',' + str(gt[0]))
    res_file_pred.write('\n')
    res_file_gt.write('\n')
    res_file_pred.close()
    res_file_gt.close()
    smape = (add/horizon) * 100
    return smape

def check():
    prediction = [1635.04249869689,1622.0075469684257,1678.596890354093,1678.8506418384104,1728.425345806478,1691.0356127426946,1715.5888835157205,1690.2931287165413,1691.8058510330231,1710.8629477527134,1741.197604654287,1781.7096757734757]
    ground_truth = [1657.0153375585,1624.5372867048,1585.2229874753,1674.1952841102,1674.9550858404,1695.7598431891,1655.8230532861,1672.3271411566,1650.6057347519,1738.6562138054,1710.3780298543,1683.4180215361]
    add = 0
    horizon = 12
    for pred, gt in zip(prediction, ground_truth):
        #prediction has been transformed and logarithmed need to reverse
        #reversed_prediction = transformer.inverse_transform(pred)
        #reversed_prediction = math.exp(reversed_prediction[0])
        #print(reversed_prediction)
        add += (abs(gt - pred) / ((abs(gt) + abs(pred)) / 2))
        #add += (abs(gt[0] - pred[0])/((abs(gt[0])+abs(pred[0])))/2)
        #res_file_pred.write(',' + str(reversed_prediction))
        #res_file_gt.write(',' + str(gt[0]))
    #res_file_pred.write('\n')
    #res_file_gt.write('\n')
    #res_file_pred.close()
    #res_file_gt.close()
    smape = (add/horizon) * 100
    print(smape)

if __name__ == '__main__':
    check()