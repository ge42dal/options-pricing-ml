# Options Pricing ML

This project aims to price financial options using both the Black-Scholes model and a machine learning approach (LSTM). It includes data preprocessing, model training, and visualization of the results. This project was created within the scope of the course "Python for Engineering and Data Analysis (PyEDA)" offered at the Technical University of Munich.

## Project Structure

The project is organized into the following directories:

- **black_scholes_model/**
  - `black_scholes_aapl.py`: Script to calculate call and put options prices for Apple (AAPL) using the Black-Scholes model.
  - `call_results_bsm.json`: JSON file containing the Black-Scholes model results for call options.
  - `put_results_bsm.json`: JSON file containing the Black-Scholes model results for put options.

- **clean_data/**
  - `aapl_preprocessed.csv`: Preprocessed Apple data used for the analysis.

- **lstm/**
  - `LSTM_call.py`: Script for training an LSTM model to predict call option prices.
  - `LSTM_put.py`: Script for training an LSTM model to predict put option prices.
  - `model_trained_calls.pth`: Saved LSTM model for call options.
  - `model_trained_put.pth`: Saved LSTM model for put options.
  - `call_loss.png`: Loss graph for the LSTM call option model during training.
  - `put_loss.png`: Loss graph for the LSTM put option model during training.
  - `results_call.json`: JSON file containing the LSTM model results for call options.
  - `results_put.json`: JSON file containing the LSTM model results for put options.

- **payoff graph/**
  - `option_payoff.py`: Script to generate payoff graphs for options.
  - `option_payoff.png`: Payoff graph image generated by the script.

- **raw_data/**
  - **dolt/**
    - `aapl_data_dolt.csv`: Raw Apple data from Dolt.
  - **us_treasury/**
    - Contains US Treasury data.
  - **wsj/**
    - `aapl_data_with_sigma.csv`: Apple data with calculated implied volatility (sigma).
    - `aapl_data_with_ttm_clean.csv`: Apple data with time-to-maturity (TTM) calculated.
    - `options_with_treasury.csv`: Combined data with options and treasury information.
    - `final_preprocessing.py`: Script to perform final data preprocessing.
    - `raw_dolt_data_preprocessing.py`: Script for preprocessing Dolt data.
    - `raw_treasury_data_preprocessing.py`: Script for preprocessing treasury data.
    - `underlying_preprocessing.py`: Script for preprocessing underlying data.

- **utils/**
  - `__init__.py`: Initialization file for the utils module.
  - `utlis.py`: Utility functions for the project.

- `README.md`: This file, explaining the project structure and functionality.
- `requirements.txt`: Python dependencies required to run the project.

## How to Run

1. **Install Dependencies**: Make sure all dependencies are installed by running:
   ```bash
   pip install -r requirements.txt
   ```

2. **Preprocess Data**:
   - Run the scripts in the `raw_data/` directory to preprocess the data.
   - Ensure the data is cleaned and stored appropriately in `clean_data/`.

3. **Run Models**:
   - To run the Black-Scholes model, execute the `black_scholes_aapl.py` script.
   - To train and evaluate the LSTM models, run the `LSTM_call.py` and `LSTM_put.py` scripts in the `lstm/` directory.

4. **Visualize Results**:
   - Use the `option_payoff.py` script in the `payoff graph/` directory to generate exemplary payoff graphs.
   - Check the `call_loss.png` and `put_loss.png` images for model training losses, which are automatically generated by the LSTM scripts.



