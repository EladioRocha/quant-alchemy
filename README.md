# Quant Alchemy

Python source for **time-series statistics and portfolio calculations** using pandas, NumPy, and SciPy. The package exports `Timeseries` and `Portfolio`.

## Use the source checkout

Clone the repository and run examples from its root so Python imports the local `quant_alchemy` directory. There is no `setup.py` or `pyproject.toml` in this tree, so `pip install .` is not a supported source-install command.

The core source imports pandas, NumPy, and SciPy:

```sh
python -m pip install numpy pandas scipy
```

This installs available dependencies rather than recreating a pinned historical environment. [requirements.txt](requirements.txt) pins older versions of those libraries and also includes `quant_alchemy>=0.1.7`, which installs a separately distributed package. A package-index release and this source checkout should not be assumed identical.

## Minimal example

```python
import pandas as pd
from quant_alchemy import Timeseries, Portfolio

prices = pd.DataFrame({
    "asset_a": [100.0, 102.0, 101.0, 104.0],
    "asset_b": [50.0, 51.0, 52.0, 51.5],
}, index=pd.date_range("2024-01-01", periods=4))

series = Timeseries(prices)
print(series.returns())
print(series.volatility())

portfolio = Portfolio(series)
print(portfolio.correlation_matrix())
print(portfolio.returns(weights=[0.5, 0.5]))
```

Use numeric price columns, ordered consistently in time. Do not pass a date column as if it were another asset. Return calculations drop missing results from the initial shift; input cleanup and sampling frequency remain the caller's responsibility.

## API map

- [quant_alchemy/timeseries.py](quant_alchemy/timeseries.py): returns, volatility, annualization, distribution statistics, and related metrics.
- [quant_alchemy/portfolio.py](quant_alchemy/portfolio.py): portfolio returns, covariance, correlation, weights, and optimization helpers.
- [quant_alchemy/__init__.py](quant_alchemy/__init__.py): public imports.

Use `help(Timeseries.annualized_return)` or `help(Portfolio.returns)` to inspect signatures and assumptions. In the current implementation, `annualized_return()` compounds the mean periodic return; it is not a date-aware CAGR calculation.

## Validation and contributions

There is no automated test suite or documented numerical benchmark. The small example is intended to verify imports and basic calculations, not validate every metric or optimizer. When reporting a calculation issue, include a minimal price table, sampling frequency, expected result, and package versions in the [issue tracker](https://github.com/EladioRocha/quant-alchemy/issues).
