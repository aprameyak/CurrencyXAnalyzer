# CurrencyX Analyzer

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white&style=for-the-badge)
![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white&style=for-the-badge)
![yfinance](https://img.shields.io/badge/yfinance-FF9900?logo=yfinance&logoColor=white&style=for-the-badge)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?logo=matplotlib&logoColor=white&style=for-the-badge)
![Seaborn](https://img.shields.io/badge/Seaborn-4479A1?logo=seaborn&logoColor=white&style=for-the-badge)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white&style=for-the-badge)
![ARCH](https://img.shields.io/badge/arch-ED1C24?logo=&logoColor=white&style=for-the-badge)

## About

**CurrencyX Analyzer** is a comprehensive Python project that performs in-depth analysis of the USD/TRY exchange rate. The tool fetches historical data from Yahoo Finance and employs various statistical techniques and models—including volatility estimation, EWMA, GARCH modeling, and trading strategy simulation—to provide valuable insights into market behavior.

## Features

- **Data Fetching:** Retrieves historical USD/TRY exchange rate data using the `yfinance` library.
- **Price Analysis:** Visualizes exchange rate trends over time with Matplotlib.
- **Log Returns Analysis:** Computes and plots daily log returns.
- **Correlation Analysis:** Calculates correlations among key financial metrics.
- **Historical Volatility:** Computes standard deviation of log returns over different window sizes.
- **EWMA Volatility:** Estimates volatility using exponential weighted moving averages with various decay factors.
- **GARCH Modeling:** Fits a GARCH(1,1) model to log returns to analyze volatility clustering.
- **Volatility Forecasting:** Compares realized volatility with forecasted values.
- **Trading Strategy Simulation:** Implements a simple volatility breakout strategy and evaluates performance using metrics like Sharpe Ratio, Sortino Ratio, Maximum Drawdown, and Annualized Volatility.

## Technology Stack

- **Language:** Python  
- **Data Processing:** Pandas, NumPy  
- **Data Visualization:** Matplotlib, Seaborn  
- **Financial Data:** yfinance  
- **Statistical Modeling:** ARCH package (for GARCH models)  
- **Evaluation:** scikit-learn (metrics)
