from season_predictor_var import run as season
from trend_predictor_var import run as trend
import os
import sys
import numpy as np


def main():
    season(gt_path, expert_path, value_start_index, max_horizon, horizon_index, window_s, freq, out_path)
    trend(gt_path, expert_path, value_start_index, max_horizon, horizon_index, window_s, freq, out_path)
    trend_list, season_list, gt_list = load_data()
    horizon_list = get_horizon()
    mapes = list()
    for trend_val, season_val, gt_val, horizon_val in zip(trend_list, season_list, gt_list, horizon_list):
        prediction = np.exp(np.array(trend_val) + np.array(season_val))
        prediction = prediction[:horizon_val]
        ground_truth = gt_val[-horizon_val:]
        smape = calc_smape(prediction, ground_truth, horizon_val)
        mapes.append(smape)
    smape_file = open(os.path.join(out_path, 'mean_sMape.txt'), 'w')
    smape_file.write('Mean over all sMapes = ' + str(np.mean(mapes)) + '\n')
    i = 1
    for value in mapes:
        smape_file.write('sMape_' + str(i) + ' = ' + str(value) + '\n')
        i += 1
    smape_file.close()


def load_data():
    sequences_season = list()
    with open(os.path.join(out_path, 'final_25_season_horg.csv')) as f:
        content = f.read()
    for line in content.split('\n'):
        if not (line == ''):
            values = line.split(',')
            values = [float(val) for val in values if val != '']
            sequences_season.append(values)

    sequences_trend = list()
    with open(os.path.join(out_path, 'final_25_trend_horg.csv')) as f:
        content = f.read()
    for line in content.split('\n'):
        if not (line == ''):
            values = line.split(',')
            values = [float(val) for val in values if val != '']
            sequences_trend.append(values)

    sequences_gt = list()
    with open(gt_path) as f:
        content = f.read()
    for line in content.split('\n'):
        if not (line == ''):
            values = line.split(',')
            values = [float(val) for val in values[value_start_index:] if val != '']
            sequences_gt.append(values)
    return sequences_trend, sequences_season, sequences_gt


def get_horizon():
    sequences_horizon = list()
    with open(gt_path) as f:
        content = f.read()
    for line in content.split('\n'):
        if not (line == ''):
            values = line.split(',')
            horizon = int(values[horizon_index])
            sequences_horizon.append(horizon)
    return sequences_horizon


def get_maximum_horizon():
    cur_max = -1
    with open(gt_path) as f:
        content = f.read()
    for line in content.split('\n'):
        if not (line == ''):
            values = line.split(',')
            horizon = int(values[horizon_index])
            if horizon > cur_max:
                cur_max = horizon
    return cur_max


def calc_smape(prediction, ground_truth, horizon):
    add = 0
    for pred, gt in zip(prediction, ground_truth):
        add += (abs(gt - pred) / ((abs(gt) + abs(pred)) / 2))
    smape = (add / horizon) * 100
    return smape


if __name__ == '__main__':

    # call like this:
    # python main_var.py --m3_other_1 --M3C_other.csv --theta_25_hT_m3o4.csv --6 --2 --9 --4

    if len(sys.argv) != 8:
        print('Not enough arguments!')
        sys.exit()

    outfolder_name = sys.argv[1]  # e.g. m3_other
    out_path = os.path.join('data', 'output', outfolder_name)
    if os.path.isdir(out_path):
        print('Output folder already exists please specify a different name!')
        sys.exit()
    else:
        os.mkdir(out_path)

    gt_file = sys.argv[2]  # e.g. M3C_other.csv
    gt_path = os.path.join('data', gt_file)
    if not os.path.isfile(gt_path):
        print('GT file does not exist!')
        sys.exit()

    expert_file = sys.argv[3]  # e.g. theta_25_hT_m3o4.csv
    expert_path = os.path.join('data', expert_file)
    if not os.path.isfile(expert_path):
        print('Expert file does not exist!')
        sys.exit()

    # Index, in the ground truth file, at which column meta information stops and observation values begin
    value_start_index = int(sys.argv[4])  # for m3: 6

    # Index, in the ground truth file, at which column the horizon information is stored
    horizon_index = int(sys.argv[5])  # for m3: 2

    max_horizon = get_maximum_horizon()

    window_s = int(sys.argv[6])  # window size e.g. 9

    freq = int(sys.argv[7])  # frequency e.g. 4

    main()
