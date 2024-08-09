from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import pandas as pd
import numpy as np
def calculate_metrics(y_true:pd.Series, y_pred:pd.Series):
    mse = mean_squared_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, mae, mape, r2

def pex(pe, x):
    """

    :param pe: percent error
    :param x: predictions that fill within +-X % of actual
    :return: percentage of observations within actual price
    """
    within_X_percent = np.abs(pe) <= x
    pex = np.mean(within_X_percent) * 100
    return pex
