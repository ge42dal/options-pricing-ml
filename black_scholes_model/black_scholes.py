import pandas as pd
import numpy as np
from scipy import stats

from utils.utlis import calculate_metrics

def d1(S, X, r, sigma, T):
    return (np.log(S / X) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))


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
    one = d1(S, X, r, sigma, ttm)
    two = d2(one, sigma, ttm)
    return X * np.exp(-r * ttm) * stats.norm.cdf(-two) - S * stats.norm.cdf(-one)




if __name__ == "__main__":
    options_data = pd.read_csv('../clean_data/options_free_dataset.csv')

    calls = options_data[options_data['OptionType'] == 'c']
    puts = options_data[options_data['OptionType'] == 'p']

    calls['Predicted_Call_price'] = calls.apply(call_price, axis=1)
    puts['Predicted_Put_Price'] = puts.apply(put_price, axis=1)

    cmse, crmse, cmae, cmape, cr2 = calculate_metrics(calls['bid_eod'], calls['Predicted_Call_price'])
    pmse, prmse, pmae, pmape, pr2 = calculate_metrics(puts['ask_eod'], puts['Predicted_Put_Price'])

    metrics = {
        'Metric': ['MSE', 'RMSE', 'MAE', 'MAPE', 'R2'],
        'Call_Options': [cmse, crmse, cmae, cmape, cr2],
        'Put_Options': [pmse, prmse, pmae, pmape, pr2]
    }

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv('option_pricing_metrics.csv', index=False)










