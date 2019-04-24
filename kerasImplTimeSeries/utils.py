import math
import sys
from numpy import array

def get_matching_predictions(sequence, horizon):
    """
    Gets from a list of tuples the predictions whose length matches with the horizon
    :param sequence:
    :param horizon:
    :return:
    """
    result = list()
    for tuples in sequence:
        for predictions in tuples:
            if len(predictions) == horizon:
                result.append(predictions)
    return result

def convert_Theta_to_CIF_format(location):
    file = open(location, mode='rt', encoding='utf-8')
    text = file.read()
    file.close()
    location_new = location.split('.')[0] + '_CIF.' + 'csv'
    file = open(location_new, mode='w+', encoding='utf-8')
    lines = text.split('\n')
    i = 0
    result_text = ''
    for line in lines:
        values = line.split(',')
        values = [value for value in values if value != 'NA']
        if i >= 2:
            result_line = ','.join(values)
            adder = '12,' + '12,'
            result_line = adder + result_line
            result_text += result_line + '\n'
        i += 1
    result_text = result_text[:-(2 + len(adder))]
    file.write(result_text)
    file.close()

def to_log(values):
    return [math.log(value) for value in values]

def shape_transformed_toinput(transformed, size1, size2):
    outer = list()
    for inner_val in transformed:
        inner = list()
        for value in inner_val:
            inner += value.tolist()
        outer += [inner]
    outer = array(outer).reshape(size1, size2, 1)
    return outer