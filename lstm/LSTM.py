import pathlib
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from tqdm import tqdm

import pandas as pd
import numpy as np

from utils.utlis import calculate_metrics


def pp_get_rel_cols_and_ot(options_data: pd.DataFrame, option_type: str) ->pd.DataFrame:
    """
    Pre-processing: get relevant columns and option type
    :param options_data: options df to be processed
    :param option_type: option type 'c' for calls, 'p' of puts
    :return: a dataframe where sigma, S_option, bid price, ask price, are removed
    """
    options_data.drop(['Sigma_20_Days_Annualized', "Underlying_Price", "bid_eod", "ask_eod"], axis=1,
                      inplace=True)
    options_data = options_data[options_data['OptionType'] == option_type]
    return options_data

def preprocess_data(path_options: pathlib.Path, path_underlying: pathlib.Path, option_type: str) -> (
        pd.DataFrame, pd.DataFrame, pd.DataFrame):
    days_to_roll = 20

    options_data = pd.read_csv(path_options)
    underlying_data = pd.read_csv(path_underlying)
    pp_get_rel_cols_and_ot(options_data, option_type)

    # creation of 20 day time series (n,m) matrix to feed in to LSTM layer, where n are individual sample prices of
    # the underlying at different dates, and m represent the previous 20 days from most recent to furthest back
    padded = np.insert(underlying_data["Close"].values, 0,
                       np.array([np.nan] * days_to_roll))
    list_of_rolled = [np.roll(padded, i) for i in range(days_to_roll)]
    stacked = np.column_stack(list_of_rolled)
    stacked = pd.DataFrame(stacked).dropna(axis=0).reset_index(drop=True)
    relevant_dates = underlying_data['Date'].iloc[days_to_roll - 1:underlying_data.size].reset_index(drop=True)
    stacked = pd.concat([relevant_dates, stacked], axis=1)

    stacked = stacked.set_index('Date')
    options_data = options_data.set_index('QuoteDate')
    joined_df = options_data.join(stacked, on='QuoteDate')
    joined_df.dropna(axis=0, inplace=True)
    final_df = joined_df.apply(pd.to_numeric, errors='ignore')
    final_df = final_df[final_df['OptionType'] == option_type]
    final_df.drop(columns=['OptionType'], inplace=True)
    return final_df

def get_train_test_split(data: pd.DataFrame) -> (np.array, np.array, np.array, np.array):
    X = data.drop(columns=['Option_Average_Price'])
    y = data['Option_Average_Price']


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    return X_train, X_test, y_train, y_test


class LSTMTrain(nn.Module):

    def __init__(self, lstm_input_size=1, lstm_hidden_size=50, lstm_n_layers=4, output_size=1, fc_n_features=4, fc_n_neurons=400):
        super().__init__()
        self.num_layers = lstm_n_layers
        self.hidden_size = lstm_hidden_size
        self.input_size = lstm_input_size
        self.output_size = output_size
        self.n_features = fc_n_features
        self.n_neurons = fc_n_neurons

        self.lstm = nn.LSTM(lstm_input_size, lstm_hidden_size, lstm_n_layers, batch_first=True)

        self.fc_lstm = nn.Linear(lstm_hidden_size, output_size)

        self.fc1 = nn.Linear(fc_n_features, fc_n_neurons)
        self.bn1 = nn.BatchNorm1d(fc_n_neurons)
        self.fc2 = nn.Linear(fc_n_neurons, fc_n_neurons)
        self.bn2 = nn.BatchNorm1d(fc_n_neurons)
        self.fc3 = nn.Linear(fc_n_neurons, fc_n_neurons)
        self.bn3 = nn.BatchNorm1d(fc_n_neurons)
        self.fc4 = nn.Linear(fc_n_neurons, output_size)
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x_lstm, x_dense):
        lstm_out = self.lstm_part(x_lstm)
        dense_input = torch.cat((lstm_out, x_dense), 1)
        dense_out = self.dense_part(dense_input)
        return dense_out

    def dense_part(self, x):
        x = self.leaky_relu(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.leaky_relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.leaky_relu(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.leaky_relu(x)
        x = self.fc4(x)
        return x

    def lstm_part(self, x):
        batch_size = x.size(0)  # 26205 total dates of prices
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)  # (1, 26205, 50)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)  # (1, 26205, 50)

        x = x.unsqueeze(-1) # (26205, 20, 1)
        out, _ = self.lstm(x, (h0, c0))  # (26205, 20, 50) # chose 50 so that lstm could learn more complex patterns
        return self.fc_lstm(out[:, -1, :])  # pass last lstm to fc with output 1, to get 1 time series



def training_loop(epochs: int, model: nn.Module, X_train: pd.DataFrame,
                  y_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame,):

    lstm_train_tensor = torch.tensor(X_train.drop(columns=['strike', 'Time_to_Maturity', 'RF_Rate']).values,
                                    dtype=torch.float32)
    dense_train_tensor = torch.tensor(X_train[['strike', 'Time_to_Maturity', 'RF_Rate']].values, dtype=torch.float32)
    train_targets = torch.tensor(y_train.values, dtype=torch.float32)

    lstm_test_tensor = torch.tensor(X_test.drop(columns=['strike', 'Time_to_Maturity', 'RF_Rate']).values, dtype=torch.float32)
    dense_test_tensor = torch.tensor(X_test[['strike', 'Time_to_Maturity', 'RF_Rate']].values, dtype=torch.float32)
    test_targets = torch.tensor(y_test.values, dtype=torch.float32)

    train_dataset = TensorDataset(lstm_train_tensor, dense_train_tensor,  train_targets)
    test_dataset = TensorDataset(lstm_test_tensor, dense_test_tensor, test_targets)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # 32 has the best time to MSE score payoff
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = torch.nn.MSELoss()

    train_loss_vec = np.zeros(epochs)
    test_loss_vec = np.zeros(epochs)
    model.train()
    for epoch in tqdm(range(epochs), desc='Epochs'):
        adjust_learning_rate(optimizer, epoch)
        train_loss = 0.0
        for lstm_batch, dense_batch, target_batch in train_loader:
            pred = model(lstm_batch, dense_batch)
            loss = loss_fn(pred, target_batch.unsqueeze(1))  # modified shape because threw errors
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.item() * lstm_batch.size(0)

        train_loss /= len(train_loader)
        train_loss_vec[epoch] = loss.item() * lstm_batch.size(0)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for lstm_batch, dense_batch, target_batch in test_loader:
                pred = model(lstm_batch, dense_batch)
                loss = loss_fn(pred, target_batch.unsqueeze(1))
                val_loss += loss.item() * lstm_batch.size(0)

        val_loss /= len(test_loader)
        test_loss_vec[epoch] = loss.item() * lstm_batch.size(0)

        print(f'Epoch {epoch + 1}/{epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), np.log(train_loss_vec), label='Training Loss')
    plt.plot(range(1, epochs + 1), np.log(test_loss_vec), label='Validation Loss')
    plt.title('LSTM Call Loss')
    plt.xlabel('Epoch')
    plt.ylabel('log(MSE)')
    plt.legend()
    plt.show()
    return model

def adjust_learning_rate(optimizer, epoch):
    """adjusts the learning rate according to the specified schedule."""
    if epoch < 10:
        lr = 1e-2
    elif epoch < 15:
        lr = 1e-3
    else:
        lr = 1e-4
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print(f"Epoch {epoch+1}: Learning rate adjusted to {lr}")



if __name__ == "__main__":
    path_to_options_data = Path(__file__).parent.parent / "clean_data" / "options_free_dataset.csv"

    path_to_underlying_data = Path(__file__).parent.parent / "clean_data" / "data_spx_18_19.csv"

    calls_df = preprocess_data(path_to_options_data, path_to_underlying_data, option_type='c')
    cX_train, cX_test, cy_train, cy_test = get_train_test_split(calls_df)

    model = LSTMTrain()
    model_trained = training_loop(20, model, X_train=cX_train, y_train=cy_train, X_test=cX_test, y_test=cy_test)


    torch.save(model_trained.state_dict(), './model_trained.pth')

    lstm_test_tensor = torch.tensor(cX_test.drop(columns=['strike', 'Time_to_Maturity', 'RF_Rate']).values,
                                    dtype=torch.float32)
    dense_test_tensor = torch.tensor(cX_test[['strike', 'Time_to_Maturity', 'RF_Rate']].values, dtype=torch.float32)
    test_targets = torch.tensor(cy_test.values, dtype=torch.float32)
    cy_pred = model_trained(lstm_test_tensor, dense_test_tensor)

    cy_pred = cy_pred.squeeze(1).detach().numpy()

    diff_sq = np.square(cy_test.to_numpy() - cy_pred)
    mse = np.mean(diff_sq)
    print(calculate_metrics(cy_test.to_numpy(), cy_pred))
