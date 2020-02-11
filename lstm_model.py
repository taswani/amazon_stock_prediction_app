from data import DataPreparation, Query, result_df, r_squared
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras import optimizers
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit

np.random.seed(32)

dp = DataPreparation(result_df)
X_train, X_test, y_train, y_test = dp.time_series_split(n=5)
min_max_scaler, X_train, X_test, y_train, y_test = dp.min_max_scaling(X_train, X_test, y_train, y_test)

# Using keras' timeseriesgenerator in order to divide the data into batches
# Putting data into 3D for input to the LSTM
data_gen_train = TimeseriesGenerator(X_train, y_train,
                               length=14, sampling_rate=1,
                               batch_size=160)

data_gen_test = TimeseriesGenerator(X_test, y_test,
                               length=14, sampling_rate=1,
                               batch_size=160)

# Done for Input shape of LSTM
train_X, train_y = data_gen_train[0]
test_X, test_y = data_gen_test[0]

# Begin LSTM
model = Sequential()
# Remove dropout initially to let overfitting happen.

model.add(LSTM(units = 40, return_sequences = False, input_shape = (train_X.shape[1], train_X.shape[2])))
# model.add(Dropout(0.2)) #Drops 20% of layer to prevent overfitting

model.add(Dense(units = 1))

# Scheduled learning_rate (learning rate slows down over epochs)
# Lower learning rate
adam = optimizers.Adam(learning_rate = .0022)
model.compile(optimizer = adam, loss = 'mean_absolute_error', metrics=[r_squared])

# Try model.fit and get history
history = model.fit_generator(data_gen_train, epochs = 200, validation_data = data_gen_test)
score = model.evaluate_generator(data_gen_test, verbose=0)
model.save('lstm_model.h5')
print('Model saved')

predicted_stock_price = model.predict(test_X)
predicted_stock_price = min_max_scaler.inverse_transform(predicted_stock_price)

# MSE: 0.012109583243727684, R-squared: 0.9007326364517212 after 200 epochs, learning_rate = 0.0022, and 160 batch size, step size = 14
print("Mean-absolute-error: ", score[0])
print("R-squared: ", score[1])

# Plot of metrics and diagnostics - specifically of loss/val-loss
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

# Plot of metrics and diagnostics - specifically of r-squared/val-r-squared
plt.plot(history.history['r_squared'], label='train')
plt.plot(history.history['val_r_squared'], label='test')
plt.legend()
plt.show()
