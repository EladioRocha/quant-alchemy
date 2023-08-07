## Table of contents

- [Table of contents](#table-of-contents)
- [Introduction](#introduction)
- [Installation](#installation)
  - [Dependencies](#dependencies)
- [Usage](#usage)
- [Contributing](#contributing)

## Introduction
`Quant Alchemy` provide a


## Installation
This package requires some dependencies to be installed.

### Dependencies
- [pandas](https://pandas.pydata.org/)
- [numpy](https://numpy.org/)
- [scipy](https://www.scipy.org/)

To install the package, run the following command in your terminal.

```bash
pip install quant_alchemy
```

To install all the dependencies, run the following command in your terminal.

```bash
pip install -r requirements.txt
```

## Usage
A simple example of how to use the package is shown below.

```python
from quant_alchemy import Timeseries, Portfolio
"""
Suppose we have a dataframe with the following columns:
    - date: date of the stock price
    - close: opening price of the stock
"""
df = pd.read_csv("data/stock.csv")

# Create a timeseries object
ts = Timeseries(df)

# To see all the methods available
print([t for t in dir(ts) if not t.startswith('__')])

# To see how to use a method
help(ts.annualized_return)
```

## Contributing

For any bug reports or recommendations, please visit our [issue tracker](https://github.com/EladioRocha/quant_alchemy/issues) and create a new issue. If you're reporting a bug, it would be great if you can provide a minimal reproducible example.

Thank you for your contribution!