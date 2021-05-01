from joblib import load
import numpy as np
import pandas as pd
import xgboost as xgb
from tensorflow.keras.models import load_model
from data_modules.data import DataPreparation, Query, result_df, r_squared
from app.models import Query as Q

def predict(open, high, low, headline, result_df):
    # load classical machine learning model
    classical_model = xgb.XGBRegressor()
    classical_model.load_model('./data_modules/classical_model.json')
    # load feed-forward neural network from keras with custom r-squared metric function
    ff_model = load_model('./data_modules/ff_model.h5', custom_objects={'r_squared': r_squared})
    # Importing lstm model in case for later testing
    # lstm_model = load_model('./data_modules/lstm_model.h5', custom_objects={'r_squared': r_squared}) # Requires 14x5x1 for LSTM model
    dp = DataPreparation(result_df)
    X_train, X_test, y_train, y_test = dp.time_series_split(n=5)
    min_max_scaler = dp.min_max_scaling(X_train, X_test, y_train, y_test, True)
    q = Query(open, high, low, headline, result_df)
    # Min-max scaling queries for model consumption, with reshaping to fit the min_max_scaler
    converted_query = q.convert_data()
    converted_query = np.array(converted_query).reshape(1, -1)
    scaled_query = min_max_scaler.transform(converted_query)
    # Predictions across different models, with reshaping to fit the min_max_scaler
    classical_prediction = classical_model.predict(scaled_query).reshape(1, -1)
    classical_prediction = min_max_scaler.inverse_transform(classical_prediction)
    ff_prediction = ff_model.predict(scaled_query).reshape(1, -1)
    ff_prediction = min_max_scaler.inverse_transform(ff_prediction)
    return classical_prediction[0][0], ff_prediction[0][0]

