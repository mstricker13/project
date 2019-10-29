from season_predictor import run as season
from trend_predictor import run as trend
import os
import sys
import numpy as np

def main():
    #season()
    #trend()
    trend_list, season_list, expert_list = load_data()
    horizon_list = get_horizon()
    for trend_val, season_val, expert_val, horizon_val in zip(trend_list, season_list, expert_list, horizon_list):
        print(trend_val)
        print(season_val)
        print(np.exp(np.array(trend_val) + np.array(season_val)))
        sys.exit()


def load_data():
    sequences_season = list()
    with open(os.path.join('data', 'output', 'final_25_season_horg.csv')) as f:
        content = f.read()
    for line in content.split('\n'):
        if not (line == ''):
            values = line.split(',')
            values = [float(val) for val in values if val != '']
            sequences_season.append(values)

    sequences_trend = list()
    with open(os.path.join('data', 'output', 'final_25_trend_horg.csv')) as f:
        content = f.read()
    for line in content.split('\n'):
        if not (line == ''):
            values = line.split(',')
            values = [float(val) for val in values if val != '']
            sequences_trend.append(values)

    sequences_expert = list()
    with open(os.path.join('data', 'theta_25_cif_horg.csv')) as f:
        content = f.read()
    for line in content.split('\n')[2:]:
        if not (line == ''):
            values = line.split(',')
            values = [float(val) for val in values[1:] if val != 'NA']
            sequences_expert.append(values)
    return sequences_trend, sequences_season, sequences_expert


def get_horizon():
    sequences_horizon = list()
    with open(os.path.join('data', 'cif.csv')) as f:
        content = f.read()
    for line in content.split('\n'):
        if not (line == ''):
            values = line.split(',')
            horizon = int(values[1])
            sequences_horizon.append(horizon)
    return sequences_horizon


if __name__ == '__main__':
    main()
