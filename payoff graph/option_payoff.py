import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    strike_price = 100
    stock_prices = np.linspace(50, 150, 400)

    long_call_payoff = np.maximum(stock_prices - strike_price, 0)
    short_call_payoff = -long_call_payoff

    long_put_payoff = np.maximum(strike_price - stock_prices, 0)
    short_put_payoff = -long_put_payoff

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(stock_prices, long_call_payoff, label='Long Call Payoff', color='blue')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(strike_price, color='gray', linestyle='--', label=f'Strike Price = {strike_price}')
    plt.title('Long Call Option Payoff')
    plt.xlabel('Stock Price at Expiration (USD)')
    plt.ylabel('Payoff (USD)')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(stock_prices, short_call_payoff, label='Short Call Payoff', color='orange')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(strike_price, color='gray', linestyle='--', label=f'Strike Price = {strike_price}')
    plt.title('Short Call Option Payoff')
    plt.xlabel('Stock Price at Expiration (USD)')
    plt.ylabel('Payoff (USD)')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(stock_prices, long_put_payoff, label='Long Put Payoff', color='red')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(strike_price, color='gray', linestyle='--', label=f'Strike Price = {strike_price}')
    plt.title('Long Put Option Payoff')
    plt.xlabel('Stock Price at Expiration (USD)')
    plt.ylabel('Payoff (USD)')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(stock_prices, short_put_payoff, label='Short Put Payoff', color='green')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(strike_price, color='gray', linestyle='--', label=f'Strike Price = {strike_price}')
    plt.title('Short Put Option Payoff')
    plt.xlabel('Stock Price at Expiration (USD)')
    plt.ylabel('Payoff (USD)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig('option_payoff.png')
