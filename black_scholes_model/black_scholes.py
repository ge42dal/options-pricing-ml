import pandas as pd
import numpy as np
from scipy import stats


def d1(S, X, r, sigma, T):
    return (np.log(S / X) + ((r + sigma ** 2) * T)) / (sigma * np.sqrt(T))


def d2(d1, sigma, T):
    return d1 - (sigma * np.sqrt(T))


def call_price(row):
    S = row.Underlying_Price
    X = row.strike
    r = row.RF_Rate
    sigma = row.Sigma_20_Days_Annualized
    ttm = row.Time_to_Maturity
    one = d1(S, X, r, sigma, ttm)
    two = d2(one, sigma, ttm)
    return S * stats.norm.cdf(one) - X * (np.e ** -(r * ttm)) * stats.norm.cdf(two)


def put_price(row):
    S = row.Underlying_Price
    X = row.strike
    r = row.RF_Rate
    sigma = row.Sigma_20_Days_Annualized
    ttm = row.Time_to_Maturity
    one = -d1(S, X, r, sigma, ttm)
    two = -d2(one, sigma, ttm)
    return S * stats.norm.cdf(one) - X * (np.e ** -(r * ttm)) * stats.norm.cdf(two)

def calculate_mse(true, predicted):
    return np.mean((predicted - true) ** 2)

def calculate_rmse(true, predicted):
    return np.sqrt(calculate_mse(true, predicted))

def calculate_mae(true, predicted):
    return np.mean(np.abs(predicted - true))

def calculate_mape(true, predicted):
    return np.mean(np.abs(predicted - true) / predicted)

def calculate_r2_score(true, predicted):
    ss_res = np.sum((predicted - true) ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2)
    return 1 - (ss_res / ss_tot)

if __name__ == "__main__":
    options_data = pd.read_csv('../clean_data/options_free_dataset.csv')

    calls = options_data[options_data['OptionType'] == 'c']
    puts = options_data[options_data['OptionType'] == 'p']

    calls['Predicted_Call_price'] = calls.apply(call_price, axis=1)
    puts['Predicted_Put_Price'] = puts.apply(put_price, axis=1)

    calls['mse'] = calculate_mse(calls['bid_eod'], calls['Predicted_Call_price'])
    calls['rmse'] = calculate_rmse(calls['bid_eod'], calls['Predicted_Call_price'])
    calls['mae'] = calculate_mae(calls['bid_eod'], calls['Predicted_Call_price'])
    calls['mape'] = calculate_mape(calls['bid_eod'], calls['Predicted_Call_price'])
    calls['r2'] = calculate_r2_score(calls['bid_eod'], calls['Predicted_Call_price'])

    puts['mse'] = calculate_mse(puts['ask_eod'], puts['Predicted_Put_Price'])
    puts['rmse'] = calculate_rmse(puts['ask_eod'], puts['Predicted_Put_Price'])
    puts['mae'] = calculate_mae(puts['ask_eod'], puts['Predicted_Put_Price'])
    puts['mape'] = calculate_mape(puts['ask_eod'], puts['Predicted_Put_Price'])
    puts['r2'] = calculate_r2_score(puts['ask_eod'], puts['Predicted_Call_price'])






