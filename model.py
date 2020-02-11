from data import DataPreparation, Query, result_df, r_squared
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from joblib import dump
from mlxtend.regressor import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE


reg = xgb.XGBRegressor(n_estimators=50)
# Random number seed to get more reproduceable results
np.random.seed(32)

dp = DataPreparation(result_df)
X = dp.X
y = dp.y

X_train, X_test, y_train, y_test = dp.time_series_split(n=4)
min_max_scaler, X_train, X_test, y_train, y_test = dp.min_max_scaling(X_train, X_test, y_train, y_test)

reg.fit(X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        early_stopping_rounds=50, #stop if 50 consequent rounds without decrease of error
        verbose=False) # Change verbose to True if you want to see it train

predict = reg.predict(X_test)
print("R-squared value: ", reg.score(X_test, y_test))
# Best R-squared value I have so far is: 0.9866523109744227

xgb.plot_importance(reg)
plt.show()

lr = LinearRegression()

stregr = StackingRegressor(regressors=[lr, reg],
                           meta_regressor=reg)
stregr = stregr.fit(X_train, y_train)
print('Variance Score: %.4f' % stregr.score(X_train, y_train))

dump(stregr, 'classical_model.joblib')

# RFE Classical Model
estimator = reg
selector = RFE(estimator, 3, step=1)
selector = selector.fit(X, y)
print("Feature Ranking: ", selector.ranking_)

# TODO: Serialize classical model so that doesn't retrain,
# then load up and try with a date from the training data to see if it works.
