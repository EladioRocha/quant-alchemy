import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, jarque_bera

class Timeseries:
    """
    A class to perform financial calculations on time series data.

    Attributes
    ----------
    data : pandas.Series or pandas.DataFrame
        Time series data.

    Methods
    -------
    returns():
        Calculate the simple returns of the time series data.
    
    log_returns():
        Calculate the logarithmic returns of the time series data.
    
    volatility():
        Calculate the volatility (standard deviation) of the returns of the time series data.
    
    periods_in_year(period, trading_days=252, hours_per_day=8):
        Calculate the number of periods in a year based on the given period, trading days and trading hours.
    
    annualized_returns(period='daily', trading_days=252, hours_per_day=8):
        Calculate the annualized returns of the time series data.
    
    annualized_volatility(period='daily', trading_days=252, hours_per_day=8):
        Calculate the annualized volatility of the time series data.
    """

    def __init__(self, data):
        """
        Constructs all the necessary attributes for the Timeseries object.

        Parameters
        ----------
        data : pandas.Series or pandas.DataFrame
            The time series data.
        """
        self.data = data

    def returns(self):
        """
        Calculate the simple returns of the time series data.

        Returns
        -------
        pandas.Series or pandas.DataFrame
            The simple returns of the time series data.
        """
        return self.data.pct_change()

    def log_returns(self):
        """
        Calculate the logarithmic returns of the time series data.

        Returns
        -------
        pandas.Series or pandas.DataFrame
            The logarithmic returns of the time series data.
        """
        return np.log(self.data/self.data.shift(1))
    
    def volatility(self):
        """
        Calculate the volatility (standard deviation) of the returns of the time series data.

        Returns
        -------
        float
            The volatility of the returns of the time series data.
        """
        return self.data.pct_change().std()
    
    def periods_in_year(self, period, trading_days=252, hours_per_day=8):
        """
        Calculate the number of periods in a year based on the given period, trading days and trading hours.

        Parameters
        ----------
        period : str
            The period ('annually', 'semi-annually', 'quarterly', 'monthly', 'weekly', 'daily', '4hourly', 'hourly', '30min', '15min', '5min', or '1min').
        trading_days : int, optional
            The number of trading days in a year (default is 252).
        hours_per_day : int, optional
            The number of trading hours in a day (default is 8).

        Returns
        -------
        int or str
            The number of periods in a year, or an error message if the period is invalid.
        """
        if hours_per_day not in [8, 24]:
            raise ValueError('Invalid number of trading hours. Please choose 8 (for traditional markets) or 24 (for cryptocurrencies and forex)')

        return {
            'annually': 1,
            'semi-annually': 2,
            'quarterly': 4,
            'monthly': 12,
            'weekly': 52 * (hours_per_day / 24),
            'daily': trading_days,
            '4hourly': trading_days * hours_per_day / 4,
            'hourly': trading_days * hours_per_day,
            '30min': trading_days * hours_per_day * 2,
            '15min': trading_days * hours_per_day * 4,
            '5min': trading_days * hours_per_day * 12,
            '1min': trading_days * hours_per_day * 60,
        }.get(period.lower(), "Invalid period. Please choose a valid period")

    def annualized_returns(self, period='daily', trading_days=252, hours_per_day=8):
        """
        Calculate the annualized returns of the time series data.

        Parameters
        ----------
        period : str, optional
            The period of the time series data (default is 'daily'). ('annually', 'semi-annually', 'quarterly', 'monthly', 'weekly', 'daily', '4hourly', 'hourly', '30min', '15min', '5min', or '1min') (default is 'daily').
        trading_days : int, optional
            The number of trading days in a year (default is 252).
        hours_per_day : int, optional
            The number of trading hours in a day (default is 8).

        Returns
        -------
        float
            The annualized returns of the time series data.
        """
        periods = self.periods_in_year(period, trading_days, hours_per_day)
        if isinstance(periods, str):
            raise ValueError(periods)
        return self.data.pct_change().mean() * periods

    def annualized_volatility(self, period='daily', trading_days=252, hours_per_day=8):
        """ 
        Calculates the annualized volatility of the financial data given a specific period, 
        the number of trading days, and the number of trading hours per day. 

        The annualized volatility is calculated as the standard deviation of the percent changes 
        in the data times the square root of the number of periods in a year.

        Parameters: 
            period (str, optional): The period type of the time series data (i.e. 'annually', 'semi-annually', 'quarterly', 
                                    'monthly', 'weekly', 'daily', '4hourly', 'hourly', '30min', 
                                    '15min', '5min', '1min'). Defaults to 'daily'.
            trading_days (int, optional): The number of trading days in a year. Defaults to 252.
            hours_per_day (int, optional): The number of trading hours in a day. Defaults to 8.

        Raises: 
            ValueError: If the given period is not valid.

        Returns: 
            float: The calculated annualized volatility.
        """
        periods = self.periods_in_year(period, trading_days, hours_per_day)
        if isinstance(periods, str):
            raise ValueError(periods)
        return self.data.pct_change().std() * np.sqrt(periods)
    
    def max_drawdown(self):        
        """
        This method calculates and returns the maximum drawdown of the time series data.
        
        The drawdown is the percentage loss from the highest point to the lowest point.
        
        Returns:
            float: The maximum drawdown of the time series data.
        """
        return (self.data / self.data.cummax()).min() - 1
    
    def drawdown(self):
        """
        This method calculates and returns a DataFrame containing the return series,
        cumulative return series, previous peak in the cumulative returns, 
        and the drawdown series.

        The drawdown is the percentage loss from the highest point to the lowest point.

        Returns:
            DataFrame: A DataFrame containing the return series,
                    cumulative return series, previous peak in the cumulative returns, 
                    and the drawdown series.
        """
        return_series = self.returns()
        cumulative_returns = (1 + return_series).cumprod()
        running_max = cumulative_returns.cummax()
        running_max.iloc[0] = 1.0
        drawdown = (cumulative_returns) / running_max - 1.0

        return pd.DataFrame(
            {
                "Return": return_series, 
                "Cumulative Return": cumulative_returns, 
                "Previous Peak": running_max, 
                "Drawdown": drawdown
            }
        )


    def skewness(self):
        """
        This method calculates and returns the skewness of the returns of the time series data.

        Skewness is a measure of the asymmetry of the probability distribution.

        Returns:
            float: The skewness of the returns of the time series data.
        """
        returns = self.returns()
        return skew(returns)
    
    def kurtois(self):
        """
        This method calculates and returns the kurtosis of the returns of the time series data.

        Kurtosis is a statistical measure that defines how heavily the tails of a distribution differ from the tails of a normal distribution.

        Returns:
            float: The kurtosis of the returns of the time series data.
        """
        returns = self.returns()
        return kurtosis(returns)
    
    def is_normal(self, level=0.01):
        """
        This method applies the Jarque-Bera test to determine if the returns are normally distributed.

        The Jarque-Bera test is a goodness-of-fit test of whether sample data have the skewness and kurtosis matching a normal distribution.

        Returns:
            bool: True if the p-value of the test is greater than the significance level indicating that
                the returns could be from a normal distribution, False otherwise.
        """
        returns = self.returns()
        _, p_value = jarque_bera(returns)
        return p_value > level