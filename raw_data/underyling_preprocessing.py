import datetime
from pathlib import Path
import pandas as pd
import numpy as np

if __name__ == "__main__":
    path_to_data = Path(__file__).parent / 'aapl_underlying_wsj_data.csv'
    aapl_data = pd.read_csv(path_to_data)
    aapl_data['Date'] = pd.to_datetime(aapl_data['Date'], format="%m/%d/%y")
    aapl_data.sort_values('Date', inplace=True)
    aapl_data['sigma'] = aapl_data[' Close'].rolling(20).apply(lambda x: (np.diff(x)/x[:-1]).std())
    aapl_data.dropna(inplace=True)
    aapl_data.to_csv('aapl_data_with_sigma.csv',index=False)




