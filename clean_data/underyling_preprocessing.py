import datetime

import pandas as pd
import numpy as np

if __name__ == "__main__":
    spx_data = pd.read_csv('SPX.csv')
    spx_data['Date'] = pd.to_datetime(spx_data['Date'])
    begin_2018 = datetime.date(2018,9,1)
    begin_2019 = datetime.datetime(2019,8,1)

    date_list_2018 = pd.date_range(begin_2018, periods=31)
    date_list_2019 = pd.date_range(begin_2019, periods=31)

    mask_2018 = spx_data['Date'].isin(date_list_2018)
    spx_data_2018 = spx_data.loc[mask_2018]
    input_data_2018 = spx_data_2018[['Date', 'Close']]
    input_data_2018['sigma'] = input_data_2018['Close'].rolling(20).apply(lambda x: (np.diff(x)/x[:-1]).std())
    input_data_2018.reset_index(inplace=True)

    mask_2019 = spx_data['Date'].isin(date_list_2019)
    spx_data_2019 = spx_data.loc[mask_2019]
    input_data_2019 = spx_data_2019[['Date', 'Close']]
    input_data_2019['sigma'] = input_data_2019['Close'].rolling(20).apply(lambda x: (np.diff(x)/x[:-1]).std())
    input_data_2019.reset_index(inplace=True)

    input_date = pd.concat([input_data_2018,input_data_2019])




