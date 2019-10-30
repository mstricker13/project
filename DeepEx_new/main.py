from season_predictor import run as season
from trend_predictor import run as trend
import os
import sys
import numpy as np

def main():
    season()
    trend()
    trend_list, season_list, gt_list = load_data()
    horizon_list = get_horizon()
    mapes = list()
    for trend_val, season_val, gt_val, horizon_val in zip(trend_list, season_list, gt_list, horizon_list):
        prediction = np.exp(np.array(trend_val) + np.array(season_val))
        prediction = prediction[:horizon_val]
        ground_truth = gt_val[-horizon_val:]
        smape = calc_smape(prediction, ground_truth, horizon_val)
        mapes.append(smape)
    # TODO
    smape_file = open(os.path.join('data', 'output', 'm3_other', 'mean_sMape.txt'), 'w')
    smape_file.write('Mean over all sMapes = ' + str(np.mean(mapes)) + '\n')
    i = 1
    for value in mapes:
        smape_file.write('sMape_' + str(i) + ' = ' + str(value) + '\n')
        i += 1
    smape_file.close()


def load_data():
    sequences_season = list()
    # TODO
    with open(os.path.join('data', 'output', 'm3_other', 'final_25_season_horg.csv')) as f:
        content = f.read()
    for line in content.split('\n'):
        if not (line == ''):
            values = line.split(',')
            values = [float(val) for val in values if val != '']
            sequences_season.append(values)

    sequences_trend = list()
    # TODO
    with open(os.path.join('data', 'output', 'm3_other', 'final_25_trend_horg.csv')) as f:
        content = f.read()
    for line in content.split('\n'):
        if not (line == ''):
            values = line.split(',')
            values = [float(val) for val in values if val != '']
            sequences_trend.append(values)

    sequences_gt = list()
    # TODO
    with open(os.path.join('data', 'M3C_other.csv')) as f:
        content = f.read()
    for line in content.split('\n'):
        if not (line == ''):
            values = line.split(',')
            # TODO
            values = [float(val) for val in values[6:] if val != '']
            sequences_gt.append(values)
    return sequences_trend, sequences_season, sequences_gt


def get_horizon():
    sequences_horizon = list()
    # TODO
    with open(os.path.join('data', 'M3C_other.csv')) as f:
        content = f.read()
    for line in content.split('\n'):
        if not (line == ''):
            values = line.split(',')
            # TODO
            horizon = int(values[2])
            sequences_horizon.append(horizon)
    return sequences_horizon


def calc_smape(prediction, ground_truth, horizon):
    add = 0
    for pred, gt in zip(prediction, ground_truth):
        add += (abs(gt - pred) / ((abs(gt) + abs(pred)) / 2))
    smape = (add / horizon) * 100
    return smape

if __name__ == '__main__':
    main()
