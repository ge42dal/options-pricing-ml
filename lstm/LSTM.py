import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from sklearn.model_selection import train_test_split

from pathlib import Path

import pandas as pd
import numpy as np


def get_train_test_tensors() -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
    path_to_data = Path(__file__).parent.parent / "clean_data" / "options_free_dataset.csv"
    data = pd.read_csv(path_to_data)

    X = data.drop(['Option_Average_Price'])
    y = data['Option_Average_Price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor


class LSTMTrain(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, output_size, n_features, n_units):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        self.fc1 = nn.Linear(hidden_size + n_features, n_units)
        self.fc2 = nn.Linear(n_units, output_size)

        self.ReLU = nn.ReLU()

    def forward(self, var, fc_features):
        batch_size = var.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)

        out, _ = self.lstm(input, (h0, c0))
        out = out[:, -1, :]

        x = torch.cat((var, fc_features), dim=1)

        x = self.fc1(out)
        x = self.ReLU(x)
        x = self.fc2(x)
        return x



def training_loop(epochs: int, model: nn.Module, optim: Adam, X_train: torch.Tensor, y_train: torch.Tensor):  # why did we pick adam
    for epoch in range(epochs):
        model.train()
        optim.zero_grad()

        output = model(X_train.unsqueeze(1))
        loss = F.mse_loss(output, y_train)

        loss.backward()
        optim.step()



if __name__ == "__main__":

    path_to_options_data = Path(__file__).parent.parent / "clean_data" / "options_free_dataset.csv"
    options_data = pd.read_csv(path_to_options_data)

    path_to_underlying_data = Path(__file__).parent.parent / "clean_data" / "underlying.csv"
    underlying_data = pd.read_csv(path_to_underlying_data)

    # creation of 20 day time series (n,m) matrix to feed in to LSTM layer, where n are individual sample prices of
    # the underlying at different dates, and m represent the previous 20 days from most recent to furthest back
    days_to_roll = 20
    padded = np.insert(underlying_data[" Close"].values, 0,
                       np.array([np.nan] * days_to_roll))
    list_of_rolled = [np.roll(padded, i) for i in range(days_to_roll)]
    stacked = np.column_stack(list_of_rolled)
    stacked = pd.DataFrame(stacked).dropna(axis=0).reset_index(drop=True)
    relevant_dates = underlying_data['Date'].iloc[days_to_roll-1:underlying_data.size].reset_index(drop=True)
    stacked = pd.concat([relevant_dates,stacked], axis=1)
    stacked = stacked.set_index('Date')
    options_data = options_data.set_index('QuoteDate')
    joined_df = pd.concat([options_data, stacked], axis=1)


    options_data.drop(['QuoteDate'], axis=1, inplace=True)
    options_data.drop(['Sigma_20_Days_Annualized', "Underlying_Price", "bid_eod", "ask_eod"], axis=1, inplace=True)

    underlying_var = underlying_data['Sigma_20_Days_Annualized']
    fc_features = options_data.drop(['Sigma_20_Days_Annualized'], axis=1)

    full_data_set = fc_features.join(underlying_var, on='QuoteDate')


    data_puts = options_data[options_data['OptionType'] == 'p']
    data_calls = options_data[options_data['OptionType'] == 'c']

    X_train, X_test, y_train, y_test = get_train_test_tensors()

    pass
