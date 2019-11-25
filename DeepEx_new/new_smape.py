import os
import math
import numpy as np
import sys


def main():
    data_path = 'data'
    theta_yearly = os.path.join(data_path, 'theta_yearly_mean_sMape.txt')
    with open(theta_yearly) as f:
        theta_yearly_content = f.read()
    theta_quarterly = os.path.join(data_path, 'theta_quarterly_mean_sMape.txt')
    with open(theta_quarterly) as f:
        theta_quarterly_content = f.read()
    theta_other = os.path.join(data_path, 'theta_other_mean_sMape.txt')
    with open(theta_other) as f:
        theta_other_content = f.read()
    theta_monthly = os.path.join(data_path, 'theta_monthly_mean_sMape.txt')
    with open(theta_monthly) as f:
        theta_monthly_content = f.read()
    output = os.path.join(data_path, 'output')

    better_settings = open(os.path.join(data_path, 'better_settings.txt'), 'w')

    for i, j, y in os.walk(output):
        print(i)
        subfolders = i.split('/')
        if len(subfolders) == 3:
            foldername = subfolders[2]
            subnames = foldername.split('_')
            if len(subnames) == 3:
                category = subnames[1]
                skipped_list = os.path.join(i, 'skipped.csv')
                mean_smape = os.path.join(i, 'mean_sMape.txt')
                if os.path.isfile(skipped_list):
                    with open(skipped_list) as f:
                        skipped_content = f.read()
                    with open(mean_smape) as f:
                        mean_smape_content = f.read()
                    if category == 'yearly':
                        theta_list = theta_yearly_content.split('\n')
                    elif category == 'monthly':
                        theta_list = theta_monthly_content.split('\n')
                    elif category == 'other':
                        theta_list = theta_other_content.split('\n')
                    elif category == 'quarterly':
                        theta_list = theta_quarterly_content.split('\n')
                    else:
                        print('typo')
                        sys.exit()
                    mean_smape_list = mean_smape_content.split('\n')
                    if len(mean_smape_list) != len(theta_list):
                        print('list not same?!')
                        sys.exit()
                    skipped_values = skipped_content.split('\n')[:-1]
                    skipped_values = [int(float(val)) for val in skipped_values]
                    theta_true_mapes = list()
                    true_mapes = list()
                    smape_file = open(os.path.join(i, 'true_result.txt'), 'w')
                    smape_file.write('Theta,Prediction\n')
                    counter = 1
                    for theta_mapes, predicted_mapes in zip(theta_list[1:-1], mean_smape_list[1:-1]):
                        if i not in skipped_values:
                            theta_val = float(theta_mapes.split(' ')[-1])
                            predicted_val = float(predicted_mapes.split(' ')[-1])
                            theta_true_mapes.append(theta_val)
                            true_mapes.append(predicted_val)
                            smape_file.write(str(theta_val) + ',' + str(predicted_val) + '\n')
                        counter += 1
                    smape_file.close()
                    smape_file = open(os.path.join(i, 'simple_true_result.txt'), 'w')
                    smape_file.write('Theta = ' + str(np.mean(theta_true_mapes)) + '\nPredicted = ' + str(np.mean(true_mapes)))
                    smape_file.close()
                    if np.mean(theta_true_mapes) > np.mean(true_mapes):
                        better_settings.write(i + '\n')
    better_settings.close()


if __name__ == '__main__':
    main()
