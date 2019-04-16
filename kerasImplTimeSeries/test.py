from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import LSTM
import numpy as np

# prepare sequence
length = 5
seq = array([i / float(length) for i in range(length)])
seq = np.append(seq, seq, axis=0)
X = seq.reshape(2, length, 1)
y = seq.reshape(2, length, 1)
print(seq)
print(seq.shape)
print(X)
print(X.shape)
# define LSTM configuration
n_neurons = length
n_batch = 128
n_epoch = 2
# create LSTM
model = Sequential()
model.add(LSTM(n_neurons, input_shape=(length, 1), return_sequences=True))
model.add(TimeDistributed(Dense(1)))
model.compile(loss='mean_squared_error', optimizer='adam')
print(model.summary())
# train LSTM
model.fit(X, y, epochs=n_epoch, batch_size=n_batch, verbose=2)
# evaluate
result = model.predict(X, batch_size=n_batch, verbose=0)
for value in result[0, :, 0]:
    print('%.1f' % value)