import pathlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from sklearn.model_selection import train_test_split

from pathlib import Path

import pandas as pd
import numpy as np


def preprocess_data(path_options: pathlib.Path, path_underlying: pathlib.Path) -> (
        pd.DataFrame, pd.DataFrame, pd.DataFrame):
    days_to_roll = 20

    options_data = pd.read_csv(path_options)
    underlying_data = pd.read_csv(path_underlying)

    options_data.drop(['Sigma_20_Days_Annualized', "Underlying_Price", "bid_eod", "ask_eod"], axis=1, inplace=True)

    # creation of 20 day time series (n,m) matrix to feed in to LSTM layer, where n are individual sample prices of
    # the underlying at different dates, and m represent the previous 20 days from most recent to furthest back
    padded = np.insert(underlying_data[" Close"].values, 0,
                       np.array([np.nan] * days_to_roll))
    list_of_rolled = [np.roll(padded, i) for i in range(days_to_roll)]
    stacked = np.column_stack(list_of_rolled)
    stacked = pd.DataFrame(stacked).dropna(axis=0).reset_index(drop=True)
    relevant_dates = underlying_data['Date'].iloc[days_to_roll - 1:underlying_data.size].reset_index(drop=True)
    stacked = pd.concat([relevant_dates, stacked], axis=1)

    stacked = stacked.set_index('Date')
    options_data = options_data.set_index('QuoteDate')
    joined_df = options_data.join(stacked)
    joined_df.dropna(axis=0, inplace=True)
    final_df = joined_df.apply(pd.to_numeric, errors='ignore')

    calls_df = final_df[final_df['OptionType'] == 'c'].drop(columns='OptionType')
    put_df = final_df[final_df['OptionType'] == 'p'].drop(columns='OptionType')

    return final_df, calls_df, put_df


def get_train_test_split(data: pd.DataFrame) -> (np.array, np.array, np.array, np.array):
    X = data.drop(columns=['Option_Average_Price'])
    y = data['Option_Average_Price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

    return X_train, X_test, y_train, y_test


class LSTMTrain(nn.Module):

    def __init__(self, input_size=1, hidden_size=50, num_layers=1, output_size=1, n_features=4, n_neurons=50):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.n_features = n_features
        self.n_neurons = n_neurons

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc_lstm = nn.Linear(hidden_size, output_size)

        self.fc1 = nn.Linear(hidden_size, n_neurons)
        self.fc2 = nn.Linear(n_neurons, n_neurons)
        self.fc3 = nn.Linear(n_neurons, n_neurons)
        self.fc4 = nn.Linear(n_neurons, output_size)

        self.ReLU = nn.ReLU()

    def forward(self, x_lstm, x_dense=0):
        lstm_out = self.lstm_part(x_lstm)

        return None

    def dense_part(self, x):
        pass
    def lstm_part(self, x):
        batch_size = x.size(0)  # 26205 total dates of prices
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)  # (1, 26205, 50)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)  # (1, 26205, 50)

        x = x.unsqueeze(-1) # (26205, 20, 1)
        out, _ = self.lstm(x, (h0, c0))  # (26205, 20, 50) # chose 50 so that lstm could learn more complex patterns
        return self.fc_lstm(out[:, -1, :])  # pass last lstm to fc with output 1, to get 1 time series


def training_loop(epochs: int, model: nn.Module, X_train: pd.DataFrame,
                  y_train: pd.DataFrame):
    lstm_part_tensor = torch.tensor(X_train.drop(columns=['strike', 'Time_to_Maturity', 'RF_Rate']).values,
                                    dtype=torch.float32)
    dense_part = torch.tensor(X_train[['strike', 'Time_to_Maturity', 'RF_Rate']].values, dtype=torch.float32)
    loss_vec = np.zeros(epochs)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        pred = model(lstm_part_tensor, dense_part)
        loss = loss_fn(pred, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_vec[epoch] += loss.item()


if __name__ == "__main__":
    path_to_options_data = Path(__file__).parent.parent / "clean_data" / "options_free_dataset.csv"

    path_to_underlying_data = Path(__file__).parent.parent / "clean_data" / "underlying.csv"

    final_df, calls_df, puts_df = preprocess_data(path_to_options_data, path_to_underlying_data)
    X_train, cX_test, cy_train, cy_test = get_train_test_split(calls_df)

    model = LSTMTrain()
    training_loop(1, model, X_train=X_train, y_train=cy_train)
