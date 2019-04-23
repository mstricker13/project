import numpy as np

def sMAPE(prediction, ground_truth, horizon):
    smapes = []
    for sequence_pred, sequence_gt in zip(prediction, ground_truth):
        add = 0
        #print('!!!!!!!')
        #print(sequence_pred)
        #print(sequence_gt)
        for pred, gt in zip(sequence_pred, sequence_gt):
            print(pred, gt)
            add += (abs(gt[0] - pred[0])/((abs(gt[0])+abs(pred[0])))/2)
        smape = add/horizon
        #print(smape)
        smapes += [smape]
    return np.mean(smapes)