#https://machinelearningmastery.com/machine-learning-data-transforms-for-time-series-forecasting/
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from numpy import array
import math
import sys
import numpy as np
from stldecompose import decompose
from statsmodels.tsa.seasonal import seasonal_decompose


def standardization(values):
    # use log on values
    tmp_val = list()
    for val in values:
        if val == 0.0:
            val = val+0.000000000000001
        tmp_val.append(val)
    values = tmp_val
    values = [math.log(value) for value in values]
    values = array(values).reshape(len(values), 1)
    transformer = StandardScaler()
    transformer.fit(values)
    return transformer


def rescaler(values):
    # use log on values
    tmp_val = list()
    for val in values:
        if val == 0.0:
            val = val+0.000000000000001
        tmp_val.append(val)
    values = tmp_val
    values = [math.log(value) for value in values]
    values = array(values).reshape(len(values), 1)
    transformer = MinMaxScaler()
    transformer.fit(values)
    return transformer


def identity(values):
    values = array(values).reshape(len(values), 1)
    transformer = StandardScaler(with_mean=False, with_std=False)
    transformer.fit(values)
    return transformer


def cluster_process(trainXS, trainYS, trainY_ES, testXS, testYS, testY_ES, valXS, valYS, valY_ES):
    seasonality_list = list()
    trainXS, trainYS, trainY_ES, valXS, valYS, valY_ES = change_val_size(trainXS, trainYS, trainY_ES, valXS, valYS, valY_ES)
    trainXS, trainYS, trainY_ES, valXS, valYS, valY_ES, testXS, testY_ES, testYSL = log_series(trainXS, trainYS, trainY_ES, valXS, valYS, valY_ES, testXS, testY_ES, testYS)
    #trainXS, trainYS, trainY_ES, testXS, testYSL, testY_ES, valXS, valYS, valY_ES, seasonality_list = stl(trainXS, trainYS, trainY_ES, testXS, testYSL, testY_ES, valXS, valYS, valY_ES)
    trainXS, trainYS, trainY_ES, testXS, testY_ES, valXS, valYS, valY_ES, norm_val = normalize(trainXS, trainYS, trainY_ES, testXS, testY_ES, valXS, valYS, valY_ES)
    trainXS, trainYS, trainY_ES, testXS, testY_ES, valXS, valYS, valY_ES = reshape(trainXS, trainYS, trainY_ES, testXS, testY_ES, valXS, valYS, valY_ES)
    return trainXS, trainYS, trainY_ES, testXS, testYS, testY_ES, valXS, valYS, valY_ES, norm_val, seasonality_list


def change_val_size(trainXS, trainYS, trainY_ES, valXS, valYS, valY_ES):
    """
    changes the validation set to only contain one sample as described in the paper
    """
    trainXS = trainXS + valXS[:-1]
    valXS = valXS[-1:]
    trainYS = trainYS + valYS[:-1]
    valYS = valYS[-1:]
    trainY_ES = trainY_ES + valY_ES[:-1]
    valY_ES = valY_ES[-1:]
    return trainXS, trainYS, trainY_ES, valXS, valYS, valY_ES

def log_series(trainXS, trainYS, trainY_ES, valXS, valYS, valY_ES, testXS, testY_ES, testYS):
    """
    applies logarith on all serieses as described in the paper
    """
    trainXS = [to_log_paper(val) for val in trainXS]
    trainYS = [to_log_paper(val) for val in trainYS]
    trainY_ES = [to_log_paper(val) for val in trainY_ES]
    valXS = [to_log_paper(val) for val in valXS]
    valYS = [to_log_paper(val) for val in valYS]
    valY_ES = [to_log_paper(val) for val in valY_ES]
    testXS = [to_log_paper(val) for val in testXS]
    testY_ES = [to_log_paper(val) for val in testY_ES]
    testYS = [to_log_paper(val) for val in testYS]
    return trainXS, trainYS, trainY_ES, valXS, valYS, valY_ES, testXS, testY_ES, testYS


def to_log_paper(values):
    result = list()
    for value in values:
        if value <= 0.05:
            value += 1
            value = math.log(value)
        else:
            value = math.log(value)
        result.append(value)
    return result


def stl(trainXS, trainYS, trainY_ES, testXS, testYS, testY_ES, valXS, valYS, valY_ES):
    true_series = trainXS[0].copy()
    for k in trainXS[1:]:
        true_series.append(k[-1])
    true_series += trainYS[-1]
    true_series.append(valYS[-1][-1])
    true_series += testYS[-1]
    decomp_true_series = decompose(true_series, period=12)
    new_true_series = list()
    seasonality_list = list()
    for trend, remainder, season in zip(decomp_true_series.trend, decomp_true_series.resid, decomp_true_series.seasonal):
        new_true_series.append(trend+remainder)
        seasonality_list.append(season)

    new_true_series = new_true_series[:-len(testYS[0])]
    seasonality_list = seasonality_list[-len(testYS[0]):]

    expert_series = trainY_ES[0].copy()
    for k in trainY_ES[1:]:
        expert_series.append(k[-1])
    expert_series.append(valY_ES[-1][-1])
    expert_series += testY_ES[-1]

    decomp_expert_series = decompose(expert_series, period=12)
    new_expert_series = list()
    for trend, remainder in zip(decomp_expert_series.trend, decomp_expert_series.resid):
        new_expert_series.append(trend+remainder)
    trainXSf, trainYSf, valXSf, valYSf, testXSf, testYSf = match_deseasonalized(trainXS, trainYS, valXS, valYS, testXS, testYS, new_true_series)
    trainY_ESf, valY_ESf, testY_ESf = match_deseasonalized_expert(trainY_ES, valY_ES, testY_ES, new_expert_series)

    #print(decomp.trend)
    #print(decomp.seasonal)
    #print(decomp.resid)
    #print(true_series[0])
    #print(decomp.trend[0] + decomp.seasonal[0] + decomp.resid[0])
    #decomp = seasonal_decompose(true_series, model='additive', freq=12)
    #print(decomp.trend)
    #print(decomp.seasonal)
    #print(decomp.resid)

    return trainXSf, trainYSf, trainY_ESf, testXSf, testYS, testY_ESf, valXSf, valYSf, valY_ESf, seasonality_list


def match_deseasonalized(trainXS, trainYS, valXS, valYS, testXS, testYS, new_true_series):
    trainXS_o = list()
    i = 0
    for k in trainXS:
        trainXS_o += [new_true_series[i:len(trainXS[i]) + i]]
        i += 1

    trainYS_o = list()
    i = len(trainXS[0])
    for k in trainYS:
        trainYS_o += [new_true_series[i:len(trainYS[i]) + i]]

    val_in_len = len(valXS[0])
    val_out_len = len(valYS[0])
    test_in_len = len(testXS[0])

    valXS_o = [new_true_series[-(val_out_len + val_in_len):-val_out_len]]

    valYS_o = [new_true_series[-val_out_len:]]

    testXS_o = [new_true_series[-test_in_len:]]

    return trainXS_o, trainYS_o, valXS_o, valYS_o, testXS_o, testYS


def match_deseasonalized_expert(trainY_ES, valY_ES, testY_ES, new_expert_series):
    trainY_ES_o = list()
    i = 0
    for k in trainY_ES:
        trainY_ES_o += [new_expert_series[i:len(trainY_ES[i]) + i]]
        i += 1

    val_len = len(valY_ES[0])
    test_len = len(testY_ES[0])

    valYS_o = [new_expert_series[-(val_len + test_len):-test_len]]

    testY_ES_o = [new_expert_series[-test_len:]]

    return trainY_ES_o, valYS_o, testY_ES_o


def normalize(trainXS, trainYS, trainY_ES, testXS, testY_ES, valXS, valYS, valY_ES):
    """
    normalize by substracting last value of input from input and output as described in the paper
    """
    new_trainXS = list()
    new_trainYS = list()
    new_trainY_ES = list()
    for input, output, expert in zip(trainXS, trainYS, trainY_ES):
        norm_val = input[-1]
        input = [val - norm_val for val in input]
        output = [val - norm_val for val in output]
        expert = [val - norm_val for val in expert]
        new_trainXS.append(input)
        new_trainYS.append(output)
        new_trainY_ES.append(expert)

    new_valXS = list()
    new_valYS = list()
    new_valY_ES = list()
    for input, output, expert in zip(valXS, valYS, valY_ES):
        norm_val = input[-1]
        input = [val - norm_val for val in input]
        output = [val - norm_val for val in output]
        expert = [val - norm_val for val in expert]
        new_valXS.append(input)
        new_valYS.append(output)
        new_valY_ES.append(expert)

    new_testXS = list()
    new_testY_ES = list()
    norm_value = 0
    for input, expert in zip(testXS, testY_ES):
        norm_val = input[-1]
        input = [val - norm_val for val in input]
        expert = [val - norm_val for val in expert]
        new_testXS.append(input)
        new_testY_ES.append(expert)
        norm_value = norm_val
    return new_trainXS, new_trainYS, new_trainY_ES, new_testXS, new_testY_ES, new_valXS, new_valYS, new_valY_ES, norm_value

def reshape(trainXS, trainYS, trainY_ES, testXS, testY_ES, valXS, valYS, valY_ES):
    """
    reshapes input to work with following code
    """
    trainXS = [(array(sample).reshape(len(sample), 1)) for sample in trainXS]
    trainYS = [(array(sample).reshape(len(sample), 1)) for sample in trainYS]
    trainY_ES = [(array(sample).reshape(len(sample), 1)) for sample in trainY_ES]
    testXS = [(array(sample).reshape(len(sample), 1)) for sample in testXS]
    testY_ES = [(array(sample).reshape(len(sample), 1)) for sample in testY_ES]
    valXS = [(array(sample).reshape(len(sample), 1)) for sample in valXS]
    valYS = [(array(sample).reshape(len(sample), 1)) for sample in valYS]
    valY_ES = [(array(sample).reshape(len(sample), 1)) for sample in valY_ES]
    return trainXS, trainYS, trainY_ES, testXS, testY_ES, valXS, valYS, valY_ES

