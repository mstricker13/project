from season_predictor_var import run as season
from trend_predictor_var import run as trend
import os
import sys
import numpy as np


def main():
    season(gt_path, expert_path, value_start_index, max_horizon, horizon_index, window_s, freq, out_path,
           num_epochs_season, bs_season, l2_season, lr_season, num_filters_season, kernel_size_season)
    trend(gt_path, expert_path, value_start_index, max_horizon, horizon_index, window_s, freq, out_path,
          num_epochs_trend, bs_trend, l2_trend, lr_trend, num_filters_trend, kernel_size_trend)
    trend_list, season_list, gt_list = load_data()
    horizon_list = get_horizon()
    skip_list = get_skip_list()
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


def get_skip_list():
    skip_list = list()
    with open(os.path.join(out_path, 'skipped.csv')) as f:
        content = f.read()
    for line in content.split('\n'):
        if not (line == ''):
            value = int(line)
            skip_list.append(value)
    return skip_list


def calc_smape(prediction, ground_truth, horizon):
    add = 0
    for pred, gt in zip(prediction, ground_truth):
        add += (abs(gt - pred) / ((abs(gt) + abs(pred)) / 2))
    smape = (add / horizon) * 100
    return smape


if __name__ == '__main__':

    # call like this:
    # python main_var.py m3_other_1 M3C_other.csv theta_25_hT_m3o4.csv 6 2 9 4 100 100 8 8 0.03 0.05 0.0001 0.0001 64 32 3 3

    if len(sys.argv) != 20:
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

    num_epochs_season = int(sys.argv[8])  # number of epochs e.g. 100

    num_epochs_trend = int(sys.argv[9])  # number of epochs e.g. 100

    bs_season = int(sys.argv[10])  # batch size

    bs_trend = int(sys.argv[11])

    l2_season = float(sys.argv[12])  # l2 regularizer

    l2_trend = float(sys.argv[13])

    lr_season = float(sys.argv[14])  # learning rate

    lr_trend = float(sys.argv[15])

    num_filters_season = int(sys.argv[16])  # number of filters in convolution

    num_filters_trend = int(sys.argv[17])

    kernel_size_season = int(sys.argv[18])  # kernel size of convolution

    kernel_size_trend = int(sys.argv[19])

    main()

    para_file = open(os.path.join(out_path, 'parameters.txt'), 'w')
    para_file.write('window_size = ' + str(window_s) + '\n' +
                    'frequency = ' + str(freq) + '\n' +
                    'num_epochs_season = ' + str(num_epochs_season) + '\n' +
                    'num_epochs_trend = ' + str(num_epochs_trend) + '\n' +
                    'batch_size_season = ' + str(bs_season) + '\n' +
                    'batch_size_trend = ' + str(bs_trend) + '\n' +
                    'l2_season = ' + str(l2_season) + '\n' +
                    'l2_trend = ' + str(l2_trend) + '\n' +
                    'learning_rate_season = ' + str(lr_season) + '\n' +
                    'learning_rate_trend = ' + str(lr_trend) + '\n' +
                    'num_filters_season = ' + str(num_filters_season) + '\n' +
                    'num_filters_trend = ' + str(num_filters_trend) + '\n' +
                    'kernel_size_season = ' + str(kernel_size_season) + '\n' +
                    'kernel_size_trend = ' + str(kernel_size_trend)
                    )
    para_file.close()
