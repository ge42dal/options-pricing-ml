import pandas as pd
from tqdm import tqdm


def find_nearest_date(date, date_list):
    date_list = sorted(date_list)
    nearest_date = min(date_list, key=lambda d: abs(d - date))
    return nearest_date

def map_ttm_to_rate(ttm_temp, row):
    # if quote date is before half of month, use the interest rate from prior month
    if ttm_temp <= 44:
        return row['1 Mo']
    elif ttm_temp <= 74:
        return row['2 Mo']
    elif ttm_temp <= 104:
        return row['3 Mo']
    elif ttm_temp <= 194:
        return row['6 Mo']
    elif ttm_temp <= 379:
        return row['1 Yr']
    elif ttm_temp <= 749:
        return row['2 Yr']
    elif ttm_temp <= 1109:
        return row['3 Yr']
    elif ttm_temp <= 1836:
        return row['5 Yr']
    elif ttm_temp <= 2569:
        return row['7 Yr']
    elif ttm_temp <= 3664:
        return row['10 Yr']
    elif ttm_temp <= 7314:
        return row['20 Yr']
    else:
        return row['30 Yr']


if __name__ == "__main__":
    # import data
    data2019 = pd.read_csv("us_treasury/daily-treasury-rates-2019.csv")
    data2020 = pd.read_csv("us_treasury/daily-treasury-rates-2020.csv")
    data2021 = pd.read_csv("us_treasury/daily-treasury-rates-2021.csv")
    data2022 = pd.read_csv("us_treasury/daily-treasury-rates-2022.csv")
    data2023 = pd.read_csv("us_treasury/daily-treasury-rates-2023.csv")
    data2024 = pd.read_csv("us_treasury/daily-treasury-rates-2024.csv")

    treasury_data = pd.concat([data2019, data2020, data2021, data2022, data2023, data2024], ignore_index=True)
    # format date
    treasury_data['Date'] = pd.to_datetime(treasury_data['Date']).dt.date
    # get all dates
    treasury_dates = treasury_data['Date'].unique()
    treasury_data.set_index("Date", inplace=True)
    # import clean options data
    options_data = pd.read_csv("aapl_data_with_ttm_clean.csv")
    options_data['Date'] = pd.to_datetime(options_data['date']).dt.date
    tqdm.pandas(desc="Mapping to neares date")
    options_data['nearest_date'] = options_data['Date'].progress_apply(lambda x: x if x in treasury_dates else find_nearest_date(x, treasury_dates))
    options_data.set_index("nearest_date", inplace=True)
    # join options data with treasury data
    data = options_data.join(treasury_data, on="nearest_date")
    # create temporary column with ttm in days
    data['ttm_temp'] = data['ttm'] * 365
    tqdm.pandas(desc="Processing risk free rates")
    data['rf_rate'] = data.progress_apply(lambda row: map_ttm_to_rate(row['ttm_temp'], row), axis=1)
    data.rename({'Date': 'date_rf'}, inplace=True, axis=1)
    # cleanup
    data.drop(columns=['ttm_temp', '1 Mo', '2 Mo', '3 Mo','4 Mo', '6 Mo', '1 Yr', '2 Yr', '3 Yr', '5 Yr', '7 Yr',
                       '10 Yr', '20 Yr', '30 Yr'], inplace=True)

    data.to_csv("options_with_treasury.csv")





