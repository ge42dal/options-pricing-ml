import json

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from utils.utlis import pex
from scipy import stats

def d1(S, X, r, sigma, T):
    return (np.log(S / X) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))


def d2(d1, sigma, T):
    return d1 - (sigma * np.sqrt(T))


def call_price(row):
    S = row[' Close']
    X = row['strike']
    r = row['rf_rate']
    sigma = row['sigma']
    ttm = row['ttm']
    one = d1(S, X, r, sigma, ttm)
    two = d2(one, sigma, ttm)
    return S * stats.norm.cdf(one) - X * (np.e ** -(r * ttm)) * stats.norm.cdf(two)


def put_price(row):
    S = row[' Close']
    X = row['strike']
    r = row['rf_rate']
    sigma = row['sigma']
    ttm = row['ttm']
    one = d1(S, X, r, sigma, ttm)
    two = d2(one, sigma, ttm)
    return X * np.exp(-r * ttm) * stats.norm.cdf(-two) - S * stats.norm.cdf(-one)




if __name__ == "__main__":
    options_data = pd.read_csv('../clean_data/aapl_preprocessed.csv')
    calls = options_data[options_data['call_put'] == 'Call']
    puts = options_data[options_data['call_put'] == 'Put']

    calls['pred_call'] = calls.apply(call_price, axis=1)
    puts['pred_put'] = puts.apply(put_price, axis=1)

    # calls
    mse = mean_squared_error(calls['equilibrium'], calls['pred_call'])
    pe = (100 * np.array((calls['pred_call'] - calls['equilibrium']) / calls['equilibrium'])).astype('float32')
    pe = np.round(pe, 6)
    bias = np.median(pe)
    ape = np.abs(pe)
    aape = np.mean(ape)
    mape = np.median(ape)
    pe5 = pex(pe, 5)
    pe10 = pex(pe, 10)
    pe20 = pex(pe, 20)

    metrics = {
        'mse': str(mse),
        'bias': str(bias),
        'aape': str(aape),
        'mape': str(mape),
        'pe5': str(pe5),
        'pe10': str(pe10),
        'pe20': str(pe20)
    }

    with open('call_results_bsm.json', 'w') as fp:
        json.dump(metrics, fp)

    print('Call Metrics: \n'
          f'mse: {mse}\n'
          f'bias: {bias}\n'
          f'aape: {aape}\n'
          f'mape: {mape}\n'
          f'pe5: {pe5}\n'
          f'pe10: {pe10}\n'
          f'pe20: {pe20}\n')

    # puts
    mse = mean_squared_error(puts['equilibrium'],puts['pred_put'])
    pe = (100 * np.array((puts['pred_put'] - puts['equilibrium']) / puts['equilibrium'])).astype('float32')
    pe = np.round(pe, 6)
    bias = np.median(pe)
    ape = np.abs(pe)
    aape = np.mean(ape)
    mape = np.median(ape)
    pe5 = pex(pe, 5)
    pe10 = pex(pe, 10)
    pe20 = pex(pe, 20)

    metrics = {
        'mse': str(mse),
        'bias': str(bias),
        'aape': str(aape),
        'mape': str(mape),
        'pe5': str(pe5),
        'pe10': str(pe10),
        'pe20': str(pe20)
    }

    with open('put_results_bsm.json', 'w') as fp:
        json.dump(metrics, fp)

    print('Put Metrics: \n'
          f'mse: {mse}\n'
          f'bias: {bias}\n'
          f'aape: {aape}\n'
          f'mape: {mape}\n'
          f'pe5: {pe5}\n'
          f'pe10: {pe10}\n'
          f'pe20: {pe20}')












