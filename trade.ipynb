#import libraries
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from arch import arch_model

#fetch data from yahoo
ticker = yf.Ticker("USDTRY=X")
data = ticker.history(interval="1d", start="2014-09-21", end="2024-09-21")

#Price Analysis
plt.figure(figsize=(12, 6))
plt.plot(data['Close'])
plt.title('USD/TRY Exchange Rate')
plt.xlabel('Date')
plt.ylabel('Price')
plt.grid(True)
plt.show()

#Log Returns Analysis
plt.figure(figsize=(12, 6))
plt.plot(data['Close'].pct_change().apply(lambda x: np.log(1 + x)))
plt.title('USD/TRY Daily Log Returns')
plt.xlabel('Date')
plt.ylabel('Log Returns')
plt.grid(True)
plt.show()

#Correlations
data = data.drop(columns=['Volume', 'Dividends', 'Stock Splits'])
correlations = data.corr()
print("Correlations with log returns:")
print(correlations.sort_values(by=['Close'], ascending=False))

#Historical volatility
y = data['Close'].pct_change().apply(lambda x: np.log(1 + x)) 
plt.figure(figsize=(12, 6))
window_sizes = [20, 50, 100]
for window_size in window_sizes:
    plt.plot(y.rolling(window=window_size).std(), label=f"{window_size}-day")
plt.title('Standard Deviation of Daily Log Returns')
plt.xlabel('Date')
plt.ylabel('Standard Deviation')
plt.legend()
plt.grid(True)
plt.show()

#Decay Factors
decay_factors = [0.9, 0.95, 0.98]
for decay_factor in decay_factors:
    ewma_volatility = y.ewm(span=1 / (1 - decay_factor), adjust=False).std()
    print(f"EWMA Volatility with decay factor {decay_factor}: {ewma_volatility.iloc[-1]:.4f}")
plt.figure(figsize=(12, 6))
for decay_factor in decay_factors:
    ewma_volatility = y.ewm(span=1 / (1 - decay_factor), adjust=False).std()
    plt.plot(ewma_volatility, label=f"Decay Factor {decay_factor}")
plt.title('EWMA Volatility')
plt.xlabel('Date')
plt.ylabel('Volatility')
plt.legend()
plt.grid(True)
plt.show()

#Garch Model
y = data['Close'].pct_change().apply(lambda x: np.log(1 + x)) 
y = y.fillna(0)
model = arch_model(y, mean='constant', vol='GARCH', p=1, o=0, q=1)
results = model.fit()
results.plot()
plt.show()
print("AIC:", results.aic)
print("BIC:", results.bic)
print("Loglikelihood:", results.loglikelihood)

#Volatility Forecasting
realized_volatility = y.rolling(window=30).std() * np.sqrt(252) 
plt.figure(figsize=(12, 6))
plt.plot(realized_volatility.index, realized_volatility, label='Realized Volatility')
plt.title('Forecasted vs. Realized Volatility')
plt.xlabel('Date')
plt.ylabel('Volatility')
plt.legend()
plt.grid(True)
plt.show()

#Trading Strategy
data['volatility'] = data['Close'].pct_change().rolling(window=20).std() * np.sqrt(252)
signals = pd.DataFrame(index=data.index)
signals['position'] = np.where(data['volatility'] > 0.02, 1, -1)
returns = data['Close'].pct_change() * signals['position'].shift(1)
plt.figure(figsize=(12, 6))
plt.plot(data['Close'], label='Price')
plt.plot(signals['position'] * data['Close'], label='Strategy')
plt.title('Volatility Breakout Strategy')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
cumulative_returns = (1 + returns).cumprod() - 1
print("Cumulative Returns:", cumulative_returns.iloc[-1])

#Evaluation
sharpe_ratio = (returns.mean() - 0.02) / returns.std()
print("Sharpe Ratio:", sharpe_ratio)
downside_deviation = returns[returns < 0].std()
sortino_ratio = (returns.mean() - 0.02) / downside_deviation
print("Sortino Ratio:", sortino_ratio)
max_drawdown = (returns.cumsum().max() - returns.cumsum()).max() / returns.cumsum().max()
print("Maximum Drawdown:", max_drawdown)
annualized_volatility = returns.std() * np.sqrt(252)
print("Annualized Volatility:", annualized_volatility)
total_pnl = returns.sum()
print("Total PnL:", total_pnl)
