# from __future__ import absolute_import, division, print_function

import sys
import os

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from keras.optimizers import Adam
from sklearn.metrics import mean_squared_error as mse
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential,Model
from keras.layers import Input,LSTM, Dense, Flatten, Conv1D, Lambda, Reshape
from keras.layers.merge import concatenate, multiply,add
import tensorflow as tf
from keras import regularizers
from keras.initializers import glorot_uniform
from tqdm import tqdm
import rpy2
import rpy2.robjects.numpy2ri
from stldecompose import decompose
rpy2.robjects.numpy2ri.activate()
from rpy2.robjects.packages import importr
from keras import regularizers

from rpy2.robjects import r, pandas2ri
pandas2ri.activate()
stats = importr('stats')
stl=stats.stl
ts =stats.ts
from statsmodels.tsa.seasonal import seasonal_decompose

d ="data" # path where data file resides
data = pd.read_csv(os.path.join(d, "cif.csv"), index_col=[0, 1, 2], header=None)
predictions = pd.read_csv(os.path.join(d, 'theta_25_cif_horg.csv'), index_col=0, skiprows=[1])

data_length = data.shape[0]

def make_input(data,window_size,horizon=1):
    length=data.shape[0]
    y = np.zeros([length-window_size+1-horizon,horizon])
    output=np.zeros([length-window_size+1-horizon,window_size])
    for i in range(length-window_size-horizon+1):
        output[i:i+1,:]=data[i:i+window_size]
        y[i,:]= data[i+window_size:i+window_size+horizon]
    return output.reshape(output.shape[0],window_size,1), y

def make_k_input(data,window_size,horizon):
    length = data.shape[0]
    output= np.zeros([length-window_size+1-horizon,horizon])
    for i in range(length-window_size-horizon+1):
        output[i:i+1,:]=data[i+window_size:i+window_size+horizon]
    return output.reshape(output.shape[0],horizon,1)


def nonov_make_input(data, window_size, horizon=1):
    length = data.shape[0] - window_size
    loop = length // horizon
    extra = length % horizon

    data = np.append(data, np.zeros([horizon - extra]))

    if extra == 0:
        i_val = loop
    else:
        i_val = loop + 1

    output = np.zeros([i_val, window_size])
    y = np.zeros([i_val, horizon])
    for i in range(i_val):
        output[i:i + 1, :] = data[i * horizon:(i * horizon) + window_size]
        y[i, :] = data[(i * horizon) + window_size:(i * horizon) + window_size + horizon]

    return output.reshape(output.shape[0], window_size, 1), y


def nonov_make_k_input(data, window_size, horizon):
    length = data.shape[0] - window_size
    loop = length // horizon
    extra = length % horizon
    data_app = np.repeat(data[-1], extra)
    data = np.append(data, data_app)

    if extra == 0:
        i_val = loop
    else:
        i_val = loop + 1
    output = np.zeros([i_val, horizon])
    for i in range(i_val):
        output[i:i + 1, :] = data[(i * horizon) + window_size:(i * horizon) + window_size + horizon]
    return output.reshape(output.shape[0], horizon, 1)


def nonov_make_input(data, window_size, horizon=1):
    length = data.shape[0] - window_size
    loop = length // horizon
    extra = length % horizon

    data = np.append(data, np.zeros([horizon - extra]))

    if extra == 0:
        i_val = loop
    else:
        i_val = loop + 1

    output = np.zeros([i_val, window_size])
    y = np.zeros([i_val, horizon])
    for i in range(i_val):
        output[i:i + 1, :] = data[i * horizon:(i * horizon) + window_size]
        y[i, :] = data[(i * horizon) + window_size:(i * horizon) + window_size + horizon]

    return output.reshape(output.shape[0], window_size, 1), y


def nonov_make_k_input(data, window_size, horizon):
    length = data.shape[0] - window_size
    loop = length // horizon
    extra = length % horizon
    data_app = np.repeat(data[-1], extra)
    data = np.append(data, data_app)

    if extra == 0:
        i_val = loop
    else:
        i_val = loop + 1
    output = np.zeros([i_val, horizon])
    for i in range(i_val):
        output[i:i + 1, :] = data[(i * horizon) + window_size:(i * horizon) + window_size + horizon]
    return output.reshape(output.shape[0], horizon, 1)

def run():
    with tqdm(total=data_length) as pbar:
        max_test_samples = 12

        final_predictions = np.zeros([data_length, max_test_samples])
        for y in range(data_length):

            #             n_test = data.iloc[y].values[2] #----cif
            custom_horizon = data.index.get_level_values(1).values[y]
            n_test = custom_horizon  # number of test samples in the data
            horizon = custom_horizon  # can be varied, horizon to be predicted by one input window

            window_size = 7  # number of past values to be used for prediction
            #             if horizon==6:
            #                 window_size=7
            #             else:
            #                 window_size=15

            nn_val = np.asarray(data.iloc[y].dropna().values, dtype=float)  # data in one time series
            rr = nn_val.size
            rr = int(np.floor(rr * .25))
            temp1 = nn_val[rr:]  # if taking 75% of the data, if 50% is required then temp1=nn_val[2*rr:]
            #             temp1=np.asarray(data.loc[y].dropna().values,dtype=float)[300:]

            epsilon = 0.05
            temp1[temp1 < epsilon] = temp1[temp1 < epsilon] + 0.05
            series = np.log(temp1)
            #             series = temp1

            frequency = 7  # should be adjusted according to dataset
            if temp1.size < 2 * frequency:
                frequency = 2

            result = stl(ts(series, frequency=frequency), "periodic")
            temp = pandas2ri.ri2py(result.rx2('time.series'))
            series_1 = temp[:, 0] + temp[:, 2]  # extracting the seasonality and the residual component from the data

            # defining train,test and validation split
            t_v_data = series_1[:-n_test]
            series_length = t_v_data.size
            n_val = int(np.round(series_length * .2))
            if n_val < horizon:
                n_val = horizon
            train = t_v_data[:-n_val]
            if train.size < 11:
                window_size = 3

            test = series_1[-(n_test + window_size):]

            val = t_v_data[-(n_val + window_size):]
            #             resea=temp[:,1][-n_test:]

            # preparing knowledge based predictions

            temp_theta1 = np.asarray(predictions.iloc[y].dropna().values, dtype=float)
            temp_theta1[np.argwhere(temp_theta1 <= 0)] = 0.5
            temp_theta1 = np.log(temp_theta1)
            result_k = stl(ts(temp_theta1, frequency=frequency), "periodic")
            temp_theta = pandas2ri.ri2py(result_k.rx2('time.series'))
            series_k_org = temp_theta[:, 0] + temp_theta[:, 2]

            #             temp_theta= np.log(temp_theta)-temp[:,0]
            temp2 = series_k_org[:-n_test]
            test_theta = series_k_org[-(n_test + window_size):]
            resea = temp[:, 1][-n_test:]
            temp2_train = temp2[:-n_val]
            temp2_val = temp2[-(n_val + window_size):]

            # making rolling window based input vectors\

            train_sequence = make_input(train, window_size, horizon)
            val_sequence = make_input(val, window_size, horizon)
            test_sequence = nonov_make_input(test, window_size, horizon)

            k_train = make_k_input(temp2_train, window_size, horizon)
            k_val = make_k_input(temp2_val, window_size, horizon)
            k_test = nonov_make_k_input(test_theta, window_size, horizon)

            x_train = train_sequence[0]
            y_train = train_sequence[1]
            x_val = val_sequence[0]
            y_val = val_sequence[1]
            x_test = test_sequence[0]
            y_test = test_sequence[1]

            # conditioning input on expert knowledge

            train_input = np.append(x_train, k_train, axis=1)
            val_input = np.append(x_val, k_val, axis=1)
            test_input = np.append(x_test, k_test, axis=1)

            k_train = k_train.reshape(k_train.shape[0], horizon)
            k_val = k_val.reshape(k_val.shape[0], horizon)
            k_test = k_test.reshape(k_test.shape[0], horizon)

            tf.reset_default_graph()
            K.clear_session()

            inputs_n = Input(batch_shape=(None, window_size + horizon, 1), name='input_n')
            inputs_k = Input(batch_shape=(None, horizon), name='input_k')
            branch_0 = Conv1D(64, 3, strides=1, padding='same', activation='relu', kernel_initializer=glorot_uniform(1))(
                inputs_n)
            branch_0 = Conv1D(64, 3, strides=1, padding='same', activation='relu', kernel_initializer=glorot_uniform(1))(
                branch_0)
            branch_0 = Flatten()(branch_0)
            net = Dense(horizon, name='dense_final', activity_regularizer=regularizers.l2(0.03))(
                branch_0)  # activity regularizer value based on data
            net = add([net, inputs_k])

            model = Model(inputs=[inputs_n, inputs_k], outputs=net)
            opt = Adam(lr=0.0001)
            callback = ModelCheckpoint(filepath=os.path.join(d, 'output', str(y) + 's_nn5.h5'), monitor='val_loss', save_best_only=True,
                                       save_weights_only=True)

            model.compile(loss='mean_squared_error', optimizer=opt)

            model.fit({'input_n': train_input, 'input_k': k_train}, y_train, validation_data=[[val_input, k_val], y_val],
                      callbacks=[callback], batch_size=8, shuffle=True, epochs=75, verbose=0)

            model.load_weights(os.path.join(d, 'output', str(y) + 's_nn5.h5'))
            pred = model.predict({'input_n': test_input, 'input_k': k_test})
    
            pred = pred.reshape(pred.size)[:n_test]

            final_predictions[y, :n_test] = pred.reshape(n_test)
            pbar.update(1)
        np.savetxt(os.path.join(d, 'output', 'final_25_season_horg.csv'), final_predictions, fmt='%1.3f', delimiter=',')