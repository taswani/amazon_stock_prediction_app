from data import DataPreparation, Query, result_df, r_squared
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from keras.callbacks.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator

# Random number seed to get more reproduceable results
np.random.seed(32)

dp = DataPreparation(result_df)
X_train, X_test, y_train, y_test = dp.time_series_split(n=5)
min_max_scaler, X_train, X_test, y_train, y_test = dp.min_max_scaling(X_train, X_test, y_train, y_test)

def model_creation():
    # Begin NN
    model = Sequential()
    model.add(Dense(units = 20, activation = 'relu'))
    model.add(Dense(units = 1))
    adam = optimizers.Adam(learning_rate = .003)
    model.compile(optimizer = adam, loss = 'mean_absolute_error', metrics=[r_squared])
    return model

model = model_creation()
history = model.fit(X_train, y_train, epochs = 200, validation_data = (X_test, y_test))
score = model.evaluate(X_test, y_test, verbose = 0)
model.save('ff_model.h5')
print('Model saved')

# Mean-absolute-error: 0.0052317037178134474, R-squared: 0.9618780612945557, after 200 epochs, lr = .003, n_splits=5
# Choice of better numbers over slight training variability

# Mean-absolute-error: 0.005896079147027599, R-squared: 0.9513616561889648, after 200 epochs, lr = .003, n_splits=5
# Stable with sentiment

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
