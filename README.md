# Project Overview
This project explores fundamental concepts in quantitative finance related to portfolio management. It provides practical examples of how to:
- Evaluate the performance of a basic 1/n (equal-weighted) portfolio.
- Simulate and visualize the efficient frontier using Monte Carlo methods.
- Optimize portfolio weights to find the efficient frontier using SciPy's optimization tools.

## Modern Portfolio Theory Assumptions
This project is built upon the principles of Modern Portfolio Theory (MPT), which operates under the following key assumptions:
- Investors are rational and aim to maximize their returns while minimizing risk.
- Investors share the goal of maximizing their expected returns for a given level of risk.
- All investors have access to the same level of information regarding potential investments.
- Transaction costs, commissions, and taxes are not considered in the models.
- Investors can borrow and lend money (without limits) at a risk-free rate.

## Recipes Covered
The Jupyter Notebook `asset_allocation.ipynb` covers the following topics:
1.  Analysis of an equally-weighted portfolio's performance metrics (e.g., annualized return, volatility, Sharpe ratio, drawdowns).
2.  Generation of thousands of random portfolios to visualize the risk-return trade-off and identify the efficient frontier.
3.  Using numerical optimization techniques to precisely calculate the minimum volatility portfolio and the efficient frontier.

# Usage
1.  **Launch Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```
2.  **Open `asset_allocation.ipynb`**: Navigate to the `asset_allocation.ipynb` file in your Jupyter interface and open it.
3.  **Run the cells**: Execute the cells sequentially to see the analysis and visualizations. You can modify the `RISKY_ASSETS`, `START_DATE`, `END_DATE`, `N_PORTFOLIOS`, and `N_DAYS` variables at the beginning of the notebook to experiment with different assets and timeframes.

Goodluck!
