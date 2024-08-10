import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from copy import deepcopy
from pathlib import Path

if __name__ == "__main__":
    options_data = pd.read_csv("options_with_treasury.csv")
    options_data = options_data[['date', 'strike', 'call_put', 'bid', 'ask', 'ttm', 'rf_rate']]
    options_data['date'] = pd.to_datetime(options_data['date']).dt.date.astype('str')
    options_data.set_index('date', inplace=True)

    underlying_data = pd.read_csv("aapl_data_with_sigma.csv")
    underlying_data = underlying_data[['Date', ' Close', 'sigma']]
    underlying_data['Date'] = pd.to_datetime(underlying_data['Date']).dt.date.astype('str')
    underlying_data.set_index('Date', inplace=True)

    data = options_data.join(underlying_data, on='date')

    # drop all option quotes for weekend days (NA)
    data = data.dropna(axis=0, how='any')

    # plot figure
    plt.figure(figsize=(16, 8), dpi=150)
    plt.scatter(pd.to_datetime(data.index), data['ask'], label='ask', marker='_', color='orange', alpha=0.6)
    plt.scatter(pd.to_datetime(data.index), data['bid'], label='bid', marker='_', color='green', alpha=0.3)
    plt.scatter(pd.to_datetime(data.index), data['strike'], label='strike', marker='_', color='purple', alpha=0.3)
    plt.plot(pd.to_datetime(data.index), data[' Close'], label='aapl', color='red')

    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1,7)))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

    plt.gcf().autofmt_xdate()
    plt.legend()
    plt.title("AAPL before preprocessing")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.savefig(Path(__file__).parent / 'img' / 'to_be_processed.png')
    plt.clf()

    # data is imbalanced, does not contain enough samples from 10.05.2019 -> not representative of the years,
    # since we are only selecting approximately 4 dates for each month

    date_as_years = deepcopy(data)
    date_as_years['year'] = date_as_years.index
    date_as_years['year'] = pd.to_datetime(date_as_years['year'], format='%Y-%m-%d').dt.year
    plt.hist(date_as_years['year'], bins=[2018,2019, 2020, 2021, 2022, 2023, 2024, 2025], color='blue', rwidth=0.8)
    plt.title("Histogram for count per year")
    plt.xlabel("year")
    plt.ylabel("count")
    plt.savefig(Path(__file__).parent / 'img' / 'imbalanced.png')
    plt.clf()

    date_as_years.drop(date_as_years[date_as_years['year'].astype('str') == '2019'].index, inplace=True)
    data = deepcopy(date_as_years)

    dates_per_month = deepcopy(data)
    dates_per_month['month'] = dates_per_month.index
    dates_per_month['month'] = pd.to_datetime(dates_per_month['month'], format='%Y-%m-%d').dt.month
    dates_per_month['month_year'] = dates_per_month['month'].astype('str') + '-' + dates_per_month['year'].astype('str')

    plt.xticks(size=5,rotation=90)
    plt.hist(dates_per_month.month_year, bins=100, rwidth=0.5)
    plt.title("Histogram for count per month")
    plt.xlabel("month")
    plt.ylabel("count (days)")
    plt.savefig(Path(__file__).parent / 'img' / 'imbalanced_month.png')
    plt.clf()

    # drop rows that are in 01-2020, 02-2020, since they are abnormal

    dates_per_month.drop(dates_per_month[dates_per_month['year'].astype('str') == '2020'].index, inplace=True)
    # plot figure
    data = deepcopy(dates_per_month)
    data.drop('year', axis=1, inplace=True)
    data.drop('month', axis=1, inplace=True)
    data.drop('month_year', axis=1, inplace=True)
    data['equilibrium'] = (data['bid'] + data['ask']) / 2

    # drop inf values
    data.reset_index(inplace=True)
    data.drop(index=[41428, 42966], inplace=True)
    data.to_csv(Path(__file__).parent.parent / 'clean_data' / 'aapl_preprocessed.csv')

    plt.figure(figsize=(16, 8), dpi=150)
    plt.scatter(pd.to_datetime(data['date']), data['ask'], label='ask', marker='_', color='orange', alpha=0.6)
    plt.scatter(pd.to_datetime(data['date']), data['bid'], label='bid', marker='_', color='green', alpha=0.3)
    plt.scatter(pd.to_datetime(data['date']), data['strike'], label='strike', marker='_', color='purple', alpha=0.3)
    plt.plot(pd.to_datetime(data['date']), data[' Close'], label='aapl', color='red')

    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1,7)))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

    plt.gcf().autofmt_xdate()
    plt.legend()
    plt.title("AAPL after preprocessing")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.savefig(Path(__file__).parent / 'img' / 'preprocessed.png')
    plt.clf()






