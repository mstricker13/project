import math
import sys
from numpy import array
from kerasImplTimeSeries.data import ignore_first_percentage

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

def convert_Theta_to_CIF_format(location, location_reference, percentage):
    #ref file is the normal cif file used to determine the horizon
    file_ref = open(location_reference, mode='rt', encoding='utf-8')
    file = open(location, mode='rt', encoding='utf-8')
    text = file.read()
    text_ref = file_ref.read()
    file.close()
    location_new = location.split('.')[0] + '_CIF.' + 'csv'
    file = open(location_new, mode='w+', encoding='utf-8')
    lines = text.split('\n')
    lines_ref = ignore_first_percentage(text_ref, percentage)
    lines_ref = lines_ref.split('\n')
    i = 0
    result_text = ''
    for line in lines:
        values = line.split(',')
        values = [value for value in values if value != 'NA']
        values = values[1:]
        if i >= 2:
            ref_values = lines_ref[i - 2].split(',')
            ref_values = ref_values[3:]
            ref_horizon = lines_ref[i-2].split(',')[1]
            ref_id = lines_ref[i-2].split(',')[0]
            ref_interval = lines_ref[i-2].split(',')[2]
            #TODO why the inconsistency, especially why did it work earlier?!
            if len(ref_values) == len(values):
                result_line = ','.join(values)
            else:
                result_line = ','.join(values[:len(ref_values)])
            adder = ref_id + ',' + str(ref_horizon) + ',' + ref_interval + ','
            result_line = adder + result_line
            result_text += result_line + '\n'
        i += 1
    #result_text = result_text[:-(1 + len(adder))]
    #-1 because of last empty line at the end
    result_text = result_text[:-1]
    file.write(result_text)
    file.close()
    file_ref.close()

def to_log(values):
    result = list()
    for value in values:
        if value == 0.0:
            value = math.log(value+0.000000000000001)
        else:
            value = math.log(value)
        result.append(value)
    return result

def shape_transformed_toinput(transformed, size1, size2):
    outer = list()
    for inner_val in transformed:
        inner = list()
        for value in inner_val:
            inner += value.tolist()
        outer += [inner]
    outer = array(outer).reshape(size1, size2, 1)
    return outer