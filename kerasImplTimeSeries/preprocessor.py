#https://machinelearningmastery.com/machine-learning-data-transforms-for-time-series-forecasting/
from sklearn.preprocessing import StandardScaler
from numpy import array
import math

def standardization(values):
    #use log on values
    values = [math.log(value) for value in values]
    values = array(values).reshape(len(values), 1)
    transformer = StandardScaler()
    transformer.fit(values)
    return transformer