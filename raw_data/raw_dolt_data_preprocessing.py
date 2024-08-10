import pandas as pd

if __name__ == "__main__":
    path_to_data = "aapl_data_dolt.csv"
    data = pd.read_csv(path_to_data)

    # remove unnecessary cols
    data.drop(columns=['vol', 'gamma', 'theta', 'vega', 'rho', 'delta'], inplace=True)

    #convert date cols to datetime
    data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')
    data['expiration'] = pd.to_datetime(data['expiration'], format='%Y-%m-%d')

    # calculate time to maturity in years
    data['ttm'] = (data['expiration'] - data['date']).dt.days / 365
    data.to_csv("aapl_data_with_ttm_clean.csv", index=False)

