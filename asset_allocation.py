This notebook demonstrates how to evaluate the performance of a simple 1/n portfolio and find the efficient frontier using Monte Carlo simulations and optimization with SciPy.

You can freely change the following:

- **RISKY_ASSETS**: The list of ticker symbols for the assets you want to include in your portfolio.
- **START_DATE** and **END_DATE**: The time period for which you want to download historical price data.
- **N_PORTFOLIOS**: The number of random portfolios to simulate in the Monte Carlo simulation (a higher number will give a more detailed efficient frontier but take longer to run).
- **N_DAYS**: The number of trading days in a year (used for annualizing returns and volatility).

#Evaluating the performance of a basic 1/n portfolio
"""

!pip install --upgrade yfinance

!pip install requests pandas numpy matplotlib statsmodels cufflinks seaborn pyfolio tabulate

import yfinance as yf
import numpy as np
import pandas as pd
import pyfolio as pf
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
from tabulate import tabulate

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

RISKY_ASSETS = ['BDMN.JK', 'BBCA.JK', 'ANTM.JK', 'ISAT.JK']
START_DATE = '2020-07-01'
END_DATE = '2025-06-30'

n_assets = len(RISKY_ASSETS)

prices_df = yf.download(RISKY_ASSETS, start=START_DATE, end=END_DATE)
prices_df

print(f'Downloaded {prices_df.shape[0]} rows of data.')
prices_df['Close'].plot(title='Stock prices of the considered assets')

"""Calculate individual asset returns:"""

returns = prices_df['Close'].pct_change(fill_method=None).dropna()

prices_df.fillna(method='ffill', inplace=True)
returns = prices_df['Close'].pct_change().dropna()

"""Define the Weights"""

portfolio_weights = n_assets * [1 / n_assets]

"""Calculate portfolio returns:"""

portfolio_returns = pd.Series(np.dot(portfolio_weights, returns.T),
                             index=returns.index)

"""Create the tear sheet (simple variant):"""

pip install quantstats

portfolio_returns = pd.Series(portfolio_returns)
cumulative_returns = (1 + portfolio_returns).cumprod() - 1
annualized_return = portfolio_returns.mean() * 252
annualized_volatility = portfolio_returns.std() * np.sqrt(252)
sharpe_ratio = annualized_return / annualized_volatility

max_drawdown = (portfolio_returns.cumsum().min() - portfolio_returns.cumsum().max()) / portfolio_returns.cumsum().max()
calmar_ratio = annualized_return / abs(max_drawdown)
cumulative_returns = (1 + portfolio_returns).cumprod() - 1
max_drawdown = (cumulative_returns.cummax() - cumulative_returns).max()

cumulative_returns = (1 + portfolio_returns).cumprod() - 1
max_drawdown = (cumulative_returns.cummax() - cumulative_returns).max()

downside_returns = portfolio_returns[portfolio_returns < 0]
sortino_ratio = annualized_return / downside_returns.std() * np.sqrt(252)

skew = portfolio_returns.skew()
kurtosis = portfolio_returns.kurtosis()
tail_ratio = portfolio_returns[portfolio_returns < 0].mean() / portfolio_returns[portfolio_returns > 0].mean()
stability = 1 / portfolio_returns.var()
var = np.percentile(portfolio_returns, 5)

# Print out the metrics
print(f"Annualized Return: {annualized_return:.2%}")
print(f"Annualized Volatility: {annualized_volatility:.2%}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Calmar Ratio: {calmar_ratio:.2f}")
print(f"Max Drawdown: {max_drawdown:.2%}")
print(f"Sortino Ratio: {sortino_ratio:.2f}")
print(f"Skew: {skew:.2f}")
print(f"Kurtosis: {kurtosis:.2f}")
print(f"Tail Ratio: {tail_ratio:.2f}")
print(f"Stability: {stability:.2f}")
print(f"Value at Risk (VaR): {var:.2%}")

"""Plot the Portfolio Cumulative Returns"""

plt.figure(figsize=(10, 6))
plt.plot(cumulative_returns, label='Cumulative Return', color='b')
plt.yscale('log')
plt.title('Portfolio Cumulative Returns (Log Scale)', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Cumulative Return (Log Scale)', fontsize=14)
plt.axhline(0, color='black', linewidth=1, linestyle='--')
plt.grid(True)
plt.legend()
plt.show()

"""Rolling Sharpe Ratio:"""

# Define the rolling window (6 months)
window = 6 * 21  # 6 months with 21 trading days per month (approximately 126 days)

rolling_mean = portfolio_returns.rolling(window=window).mean()
rolling_volatility = portfolio_returns.rolling(window=window).std()

rolling_sharpe_ratio = rolling_mean / rolling_volatility * np.sqrt(252)

plt.figure(figsize=(10, 6))
plt.plot(rolling_sharpe_ratio, label='6-Month Rolling Sharpe Ratio', color='b')

plt.title('6-Month Rolling Sharpe Ratio', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Sharpe Ratio', fontsize=14)
plt.axhline(0, color='black', linewidth=1, linestyle='--')
plt.grid(True)
plt.legend()
plt.show()

"""Underwater Plot"""

cumulative_returns = (1 + portfolio_returns).cumprod() - 1
running_max = cumulative_returns.cummax()
drawdown = (cumulative_returns - running_max) / running_max

plt.figure(figsize=(10, 6))
plt.fill_between(drawdown.index, drawdown, color='red', alpha=0.5, label='Underwater')
plt.title('Underwater Plot (Drawdown)', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Drawdown (%)', fontsize=14)
plt.axhline(0, color='black', linewidth=1, linestyle='--')  # Baseline (0% drawdown)
plt.grid(True)
plt.legend()
plt.show()

"""#Finding the Efficient Frontier using Monte Carlo simulations"""

N_PORTFOLIOS = 10 ** 5
N_DAYS = 252

prices_df = yf.download(RISKY_ASSETS, start=START_DATE, end=END_DATE)
print(f'Downloaded {prices_df.shape[0]} rows of data.')

"""Calculate annualized average returns and the corresponding standard deviation:"""

returns_df = prices_df['Close'].pct_change().dropna()

avg_returns = returns_df.mean() * N_DAYS
cov_mat = returns_df.cov() * N_DAYS
returns_df.plot(title='Daily returns of the considered assets');

"""Simulate random portfolio weights:"""

np.random.seed(42)
weights = np.random.random(size=(N_PORTFOLIOS, n_assets))
weights /=  np.sum(weights, axis=1)[:, np.newaxis]

"""Calculate portfolio metrics:"""

portf_rtns = np.dot(weights, avg_returns)

portf_vol = []
for i in range(0, len(weights)):
    portf_vol.append(np.sqrt(np.dot(weights[i].T,
                                   np.dot(cov_mat, weights[i]))))
portf_vol = np.array(portf_vol)
portf_sharpe_ratio = portf_rtns / portf_vol

"""Create a joint DataFrame with all data:"""

portf_results_df = pd.DataFrame({'returns': portf_rtns,
                                'volatility': portf_vol,
                                'sharpe_ratio': portf_sharpe_ratio})

"""Locate the points creating the Efficient Frontier:"""

N_POINTS = 100
portf_vol_ef = []
indices_to_skip = []

portf_rtns_ef = np.linspace(portf_results_df.returns.min(),
                           portf_results_df.returns.max(),
                           N_POINTS)
portf_rtns_ef = np.round(portf_rtns_ef, 2)
portf_rtns = np.round(portf_rtns, 2)

for point_index in range(N_POINTS):
    if portf_rtns_ef[point_index] not in portf_rtns:
        indices_to_skip.append(point_index)
        continue
    matched_ind = np.where(portf_rtns == portf_rtns_ef[point_index])
    portf_vol_ef.append(np.min(portf_vol[matched_ind]))

portf_rtns_ef = np.delete(portf_rtns_ef, indices_to_skip)

"""Plot the Efficient Frontier:"""

MARKS = ['o', 'X', 'd', '*']

fig, ax = plt.subplots()
portf_results_df.plot(kind='scatter', x='volatility',
                     y='returns', c='sharpe_ratio',
                     cmap='RdYlGn', edgecolors='black',
                     ax=ax)
ax.set(xlabel='Volatility',
      ylabel='Expected Returns',
      title='Efficient Frontier')
ax.plot(portf_vol_ef, portf_rtns_ef, 'b--')
for asset_index in range(n_assets):
    ax.scatter(x=np.sqrt(cov_mat.iloc[asset_index, asset_index]),
              y=avg_returns[asset_index],
              marker=MARKS[asset_index],
              s=150,
              color='black',
              label=RISKY_ASSETS[asset_index])
ax.legend()

plt.tight_layout()
plt.show()

"""#Finding the Efficient Frontier using optimization with scipy

Finding the Efficient Frontier using optimization with scipy
"""

import numpy as np
import scipy.optimize as sco

"""Define functions calculating portfolio returns and volatility:"""

def get_portf_rtn(w, avg_rtns):
    return np.sum(avg_rtns * w)

def get_portf_vol(w, avg_rtns, cov_mat):
    return np.sqrt(np.dot(w.T, np.dot(cov_mat, w)))

"""Define the function calculating the efficient frontier:"""

def get_efficient_frontier(avg_rtns, cov_mat, rtns_range):

    efficient_portfolios = []

    n_assets = len(avg_returns)
    args = (avg_returns, cov_mat)
    bounds = tuple((0,1) for asset in range(n_assets))
    initial_guess = n_assets * [1. / n_assets, ]

    for ret in rtns_range:
        constraints = ({'type': 'eq',
                       'fun': lambda x: get_portf_rtn(x, avg_rtns) - ret},
                      {'type': 'eq',
                      'fun': lambda x: np.sum(x) - 1})
        efficient_portfolio = sco.minimize(get_portf_vol, initial_guess,
                                          args=args, method='SLSQP',
                                          constraints=constraints,
                                          bounds=bounds)
        efficient_portfolios.append(efficient_portfolio)

    return efficient_portfolios

"""Define the considered range of returns:"""

# Example: Simulating portfolio returns as a random walk (replace this with your actual portfolio returns)
np.random.seed(42)  # For reproducibility
portfolio_returns = np.random.normal(loc=0.1, scale=0.2, size=1000)  # mean=10%, std=20%

mean_return = np.mean(portfolio_returns)
std_return = np.std(portfolio_returns)

rtns_range_1 = np.linspace(mean_return - 3*std_return, mean_return + 3*std_return, 200)

lower_percentile = np.percentile(portfolio_returns, 5)
upper_percentile = np.percentile(portfolio_returns, 95)
rtns_range_2 = np.linspace(lower_percentile, upper_percentile, 200)

plt.figure(figsize=(10, 6))
plt.hist(portfolio_returns, bins=50, alpha=0.7, label="Portfolio Returns", color='blue')
plt.axvline(mean_return, color='red', linestyle='--', label=f"Mean: {mean_return:.2f}")
plt.axvline(lower_percentile, color='green', linestyle='--', label=f"5th Percentile: {lower_percentile:.2f}")
plt.axvline(upper_percentile, color='green', linestyle='--', label=f"95th Percentile: {upper_percentile:.2f}")
plt.title("Distribution of Portfolio Returns")
plt.xlabel("Returns")
plt.ylabel("Frequency")
plt.legend()
plt.show()

print(f"Mean Return: {mean_return:.2f}")
print(f"Standard Deviation: {std_return:.2f}")
print(f"Range based on mean ± 3*std: {rtns_range_1[0]:.2f} to {rtns_range_1[-1]:.2f}")
print(f"Range based on 5th to 95th percentile: {rtns_range_2[0]:.2f} to {rtns_range_2[-1]:.2f}")

"""I choose to run based on 5th to 95th percentile
(focuses on the central portion of the data and avoids extreme outliers, the purpose is to focus on typical returns (without extreme values), use this range.)
"""

rtns_range = np.linspace(-0.21, 0.44, 200)

"""Calculate the Efficient Frontier:"""

efficient_portfolios = get_efficient_frontier(avg_returns,
                                             cov_mat,
                                             rtns_range)

"""Extract the volatilities of the efficient portfolios:"""

vols_range = [x['fun'] for x in efficient_portfolios]

"""Plot the calculated Efficient Frontier, together with the simulated portfolios:"""

fig, ax = plt.subplots()
portf_results_df.plot(kind='scatter', x='volatility',
                     y='returns', c='sharpe_ratio',
                     cmap='RdYlGn', edgecolors='black',
                     ax=ax)
ax.plot(vols_range, rtns_range, 'b--', linewidth=3)
ax.set(xlabel='Volatility',
      ylabel='Expected Returns',
      title='Efficient Frontier')

plt.tight_layout()
plt.show()

"""Identify the minimum volatility portfolio:"""

min_vol_ind = np.argmin(vols_range)
min_vol_portf_rtn = rtns_range[min_vol_ind]
min_vol_portf_vol = efficient_portfolios[min_vol_ind]['fun']

min_vol_portf = {'Return': min_vol_portf_rtn,
                'Volatility': min_vol_portf_vol,
                'Sharpe Ratio': (min_vol_portf_rtn /
                                min_vol_portf_vol)}

min_vol_portf

"""Print performance summary:"""

print('Minimum Volatility portfolio ----')
print('Performance')

for index, value in min_vol_portf.items():
    print(f'{index}: {100 * value:.2f}% ', end="", flush=True)

print('\nWeights')
for x, y in zip(RISKY_ASSETS, efficient_portfolios[min_vol_ind]['x']):
    print(f'{x}: {100*y:.2f}% ', end="", flush=True)

"""The minimum volatility portfolio is the portfolio on the efficient frontier with the lowest risk (volatility). While it doesn't offer the highest expected return, it provides the best return for the amount of risk taken among all possible portfolios.

In this case:
- **Return:** The expected annualized return for this portfolio is 14.93%.
- **Volatility:** The expected annualized volatility (risk) for this portfolio is 19.56%.
- **Sharpe Ratio:** The Sharpe Ratio of 0.76 indicates that for every unit of risk (volatility) taken, the portfolio is expected to generate 0.76 units of return above the risk-free rate (assuming a risk-free rate of 0 for simplicity in this calculation). A higher Sharpe Ratio is generally preferred.

The weights show the allocation to each asset in this minimum volatility portfolio:
- BDMN.JK: 7.26%
- BBCA.JK: 54.22%
- ANTM.JK: 33.42%
- ISAT.JK: 5.10%

This indicates that to achieve the lowest volatility for this set of assets, the portfolio should be primarily weighted towards BBCA.JK and ANTM.JK.
"""
