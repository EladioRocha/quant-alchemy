import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, jarque_bera, norm

class Timeseries:
    """
    A class to perform financial calculations on time series data.
    """
    
    def __init__(self, prices):
        """
        Initializes the Timeseries object.

        Parameters
        ----------
        prices : pandas.Series or pandas.DataFrame
            The time series data of prices.
        """

        self.prices = self.handle_input_data(prices.copy())

    def handle_input_data(self, prices):
        """
        Handles the input data by converting it to a DataFrame, if it is not already.

        Parameters
        ----------
        prices : pandas.Series, pandas.DataFrame, numpy.ndarray, or list
            The time series data of prices.

        Returns
        -------
        pandas.DataFrame
            The time series data of prices, converted to a DataFrame.
        """

        if isinstance(prices, (pd.Series, np.ndarray, list)):
            prices = pd.DataFrame(prices)

        return prices
    
    @property
    def columns(self):
        """
        Returns the column names of the time series data.

        Returns
        -------
        list
            The column names of the time series data.
        """

        return self.prices.columns.tolist()
    
    @property
    def shape(self):
        """
        Returns the shape of the time series data.

        Returns
        -------
        tuple
            The shape of the time series data.
        """

        return self.prices.shape

    def returns(self, use_log_returns=False):
        """
        Calculates the returns of the time series data.

        Parameters
        ----------
        use_log_returns : bool, optional
            Whether to calculate log returns (default is False).

        Returns
        -------
        pandas.DataFrame
            The returns of the time series data.
        """

        if use_log_returns:
            return np.log(self.prices / self.prices.shift(1)).dropna()
            
        return self.prices.pct_change().dropna()

    def volatility(self):
        """
        Calculates the volatility (standard deviation) of the returns of the time series data.

        Returns
        -------
        float
            The volatility of the returns of the time series data. Degrees of freedom = 1.
        """
        returns = self.returns()
        return returns.std(ddof=1)
    
    def periods_in_year(self, period, trading_days=252, hours_per_day=8):
        """
        Calculate the number of periods in a year for a given period based on trading parameters.

        Parameters
        ----------
        period : str
            The granularity of the time period. Valid options are: 'annually', 'semi-annually', 'quarterly',
            'monthly', 'weekly', 'daily', '4hourly', 'hourly', '30min', '15min', '5min', or '1min'.
        trading_days : int, optional
            The average number of trading days in a year (default is 252).
        hours_per_day : int, optional
            The average number of trading hours in a day (default is 8).

        Raises
        ------
        ValueError
            If the number of trading hours is invalid. Only 8 and 24 are acceptable.

        Returns
        -------
        int or str
            The number of periods in a year for the specified period or an error message if the period is invalid.
        """
        if hours_per_day not in [8, 24]:
            raise ValueError('Invalid number of trading hours. Please choose 8 (for traditional markets) or 24 (for cryptocurrencies and forex)')

        return {
            "annually": 1,
            "semi-annually": 2,
            "quarterly": 4,
            "monthly": 12,
            "weekly": 52 * (hours_per_day / 24),
            "daily": trading_days,
            "4hourly": trading_days * hours_per_day / 4,
            "hourly": trading_days * hours_per_day,
            "30min": trading_days * hours_per_day * 2,
            "15min": trading_days * hours_per_day * 4,
            "5min": trading_days * hours_per_day * 12,
            "1min": trading_days * hours_per_day * 60,
        }.get(period.lower(), "Invalid period. Please choose a valid period")

    def annualized_return(self, period="daily", trading_days=252, hours_per_day=8):
        """
        Calculate the annualized returns of the time series data.

        Parameters
        ----------
        period : str, optional
            The granularity of the time period (default is 'daily'). Valid options are: 'annually', 
            'semi-annually', 'quarterly', 'monthly', 'weekly', 'daily', '4hourly', 'hourly', '30min', 
            '15min', '5min', or '1min'.
        trading_days : int, optional
            The average number of trading days in a year (default is 252).
        hours_per_day : int, optional
            The average number of trading hours in a day (default is 8).

        Raises
        ------
        ValueError
            If the given period is not valid.

        Returns
        -------
        float
            The annualized returns of the time series data.
        """
        periods = self.periods_in_year(period, trading_days, hours_per_day)
        if isinstance(periods, str):
            raise ValueError(periods)

        daily_returns = self.returns().mean()
        return (1 + daily_returns) ** periods - 1

    def annualized_volatility(self, period="daily", trading_days=252, hours_per_day=8):
        """ 
        Calculates the annualized volatility of the financial data.

        Parameters
        ----------
        period : str, optional
            The granularity of the time period (default is 'daily'). Valid options are: 'annually', 
            'semi-annually', 'quarterly', 'monthly', 'weekly', 'daily', '4hourly', 'hourly', '30min', 
            '15min', '5min', or '1min'.
        trading_days : int, optional
            The average number of trading days in a year (default is 252).
        hours_per_day : int, optional
            The average number of trading hours in a day (default is 8).

        Raises
        ------
        ValueError
            If the given period is not valid.

        Returns
        ------- 
        float
            The annualized volatility of the time series data.
        """
        periods = self.periods_in_year(period, trading_days, hours_per_day)
        if isinstance(periods, str):
            raise ValueError(periods)
        
        returns = self.returns()

        return returns.std(ddof=1) * np.sqrt(periods)

    def compound_return(self):
        """
        Calculates and returns the compounded return of the time series data.

        Returns
        -------
        float
            The compounded return of the time series data.
        """
        returns = self.returns()
        return np.expm1(np.log1p(returns).sum())
    
    def cumulative_returns(self):
        """
        Calculates and returns the cumulative return of the time series data.

        Returns
        -------
        pd.DataFrame
            The cumulative return of each column in the time series data.
        """
        returns = self.returns()
        return (1 + returns).cumprod() - 1
    
    def max_drawdown(self):
        """
        Calculates the maximum drawdown of the time series data.

        A drawdown is the decline from a historical peak during a specific period of investment. 
        It is often quoted as the percentage between the peak to the trough.

        Returns
        -------
        pandas.Series
            The maximum drawdown of each column in the time series data.
        """
        drawdown = self.drawdowns()
        max_drawdowns = {}

        for col, df in drawdown.items():
            max_drawdowns[col] = df["drawdowns"].min()

        return pd.Series(max_drawdowns)
    
    def drawdowns(self):
        """
        Calculates and returns the drawdown series for the time series data.

        A drawdown is the decline from a historical peak during a specific period of investment. 
        It is often quoted as the percentage between the peak to the trough.

        Returns
        -------
        dict of pandas.DataFrame
            A dictionary where keys are column names and the values are dataframes containing columns: 
            'Returns', 'Cumulative Returns', 'Running Max', and 'Drawdowns'.
        """
        returns = self.returns()
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = cumulative_returns.divide(running_max) - 1.0

        return {
            col: pd.DataFrame({
                "drawdowns": drawdown[col]
            })
            for col in cumulative_returns.columns
        }

    def skewness(self):
        """
        Calculates and returns the skewness of the returns of the time series data.

        Skewness is a measure of the asymmetry of the probability distribution of a real-valued random variable about its mean. 

        Returns
        -------
        pandas.Series
            The skewness of each column in the returns of the time series data.
        """
        returns = self.returns()

        # For each column, calculate the skewness
        pd_series = {}
        for col in returns.columns:
            pd_series[col] = skew(returns[col])

        return pd.Series(pd_series)
    
    def kurtosis(self):
        """
        Calculates and returns the kurtosis of the returns of the time series data.

        Kurtosis is a statistical measure that defines how heavily the tails of a distribution differ from the tails of a normal distribution.

        Returns
        -------
        pandas.Series
            The kurtosis of each column in the returns of the time series data.
        """
        returns = self.returns()

        # For each column, calculate the kurtosis
        pd_series = {}
        for col in returns.columns:
            pd_series[col] = kurtosis(returns[col])

        return pd.Series(pd_series)
    
    def is_normal(self, level=5):
        """
        Applies the Jarque-Bera test to determine if the returns of the time series data are normally distributed.

        The Jarque-Bera test is a goodness-of-fit test of whether sample data have the skewness and kurtosis matching a normal distribution.

        Parameters
        ----------
        level : int, optional
            The significance level under which the null hypothesis of the Jarque-Bera test will be rejected, expressed as a percentile (default is 5). 
            The null hypothesis is that the skewness and kurtosis of the sample data match that of a normal distribution. 
            The 'level' should be a percentile, i.e., between 0 and 100.

        Raises
        ------
        ValueError
            If the 'level' is not a percentile (i.e., not between 0 and 100).

        Returns
        -------
        pandas.Series
            Boolean series where True indicates that the returns for that column are normally distributed at the specified level of significance.
        """
        if not 0 <= level <= 100:
            raise ValueError("The 'level' should be a percentile, i.e., between 0 and 100.")
        
        level = level / 100.0  # convert percentile to decimal
        returns = self.returns()
        pd_series = {}
        for col in returns.columns:
            _, p_value = jarque_bera(returns[col])
            pd_series[col] = p_value > level

        return pd.Series(pd_series)

    
    def sharpe_ratio(self, risk_free_rate=0.0, period="daily", trading_days=252, hours_per_day=8):
        """
        Calculates and returns the Sharpe ratio of the time series data.

        The Sharpe ratio is a measure of risk-adjusted return, defined as the difference between the returns of the investment and the risk-free rate, divided by the standard deviation of the returns.

        Parameters
        ----------
        risk_free_rate : float, optional
            The risk-free rate of return (default is 0.0).
        period : str, optional
            The granularity of the time period (default is 'daily'). Valid options are: 'annually', 'semi-annually', 
            'quarterly', 'monthly', 'weekly', 'daily', '4hourly', 'hourly', '30min', '15min', '5min', or '1min'.
        trading_days : int, optional
            The average number of trading days in a year (default is 252).
        hours_per_day : int, optional
            The average number of trading hours in a day (default is 8).

        Raises
        ------
        ValueError
            If the given period is not valid.

        Returns
        -------
        float
            The Sharpe ratio of the time series data.
        """
        periods = self.periods_in_year(period, trading_days, hours_per_day)
        if isinstance(periods, str):
            raise ValueError(periods)
        
        annualized_return = self.annualized_return(period, trading_days, hours_per_day)
        annualized_volatility = self.annualized_volatility(period, trading_days, hours_per_day)
        return (annualized_return - risk_free_rate) / annualized_volatility

    def semideviation(self, threshold=0.0):
        """
        Calculates semideviation of the returns that are less than the provided threshold.

        Parameters
        ----------
        threshold : float, optional
            The reference value below which a return is considered as underperforming. 
            Defaults to 0.0, indicating any negative return is considered underperforming.

        Returns
        -------
        float
            The semideviation of the returns. Computed as the square root of the average 
            of the squared deviations of the underperforming returns from their mean. The 
            method uses a degrees of freedom (ddof) value of 1, implying that the function
            computes sample standard deviation, an unbiased estimator of the population 
            standard deviation.
        """
        returns = self.returns()
        return returns[returns < threshold].std(ddof=1)

    def var_historic(self, level=5):
        """
        Calculates the historic Value at Risk (VaR) at a specified level.

        VaR is a statistical technique used to measure and quantify the level of financial risk 
        within a firm or investment portfolio over a specific time frame.

        Parameters
        ----------
        level : float, optional
            The percentile level at which to calculate VaR, indicating the level below 
            which the returns would fall with the given level of probability. Defaults to 5.

        Raises
        ------
        ValueError
            If the given level is not a valid percentile (i.e., not between 0 and 100).

        Returns
        -------
        pandas.Series
            The historic VaR at the specified level. This is the return value such that 
            'level' percent of returns fall below this number.
        """
        if not 0 <= level <= 100:
            raise ValueError("The 'level' should be a percentile, i.e., between 0 and 100.")
        
        returns = self.returns()
        pd_series = {}

        for col in returns.columns:
            pd_series[col] = np.percentile(returns[col], level)

        return pd.Series(pd_series)

    def cvar_historic(self, level=5):
        """
        Calculates the historic Conditional Value at Risk (CVaR), also known as Expected Shortfall (ES),
        at a specified level. CVaR quantifies the expected value of loss given that a certain level of 
        loss threshold (VaR) has been exceeded.

        Parameters
        ----------
        level : float, optional
            The percentile level at which to calculate CVaR. This indicates the level below 
            which the returns would fall with the given level of probability. Defaults to 5.

        Raises
        ------
        ValueError
            If the given level is not a valid percentile (i.e., not between 0 and 100).

        Returns
        -------
        pandas.Series
            The historic CVaR at the specified level. This is the average return value such that 
            'level' percent of returns fall below this number.
        """
        if not 0 <= level <= 100:
            raise ValueError("The 'level' should be a percentile, i.e., between 0 and 100.")
        
        returns = self.returns()

        pd_series = {}
        for col in returns.columns:
            is_beyond = returns[col] <= -self.var_historic(level=level)[col]
            pd_series[col] = returns[col][is_beyond].mean()

        return pd.Series(pd_series)

    def var_gaussian(self, level=5):
        """
        Calculates the Gaussian Value at Risk (VaR) at a specified level.

        Parameters
        ----------
        level : float, optional
            The percentile level at which to calculate VaR. This is the level below which 
            the returns would fall with the given level of probability. Defaults to 5.

        Raises
        ------
        ValueError
            If the given level is not a valid percentile (i.e., not between 0 and 100).

        Returns
        -------
        pandas.Series
            The Gaussian VaR at the specified level. This is the return value such that 
            'level' percent of returns fall below this number, assuming returns follow 
            a Gaussian distribution.
        """
        if not 0 <= level <= 100:
            raise ValueError("The 'level' should be a percentile, i.e., between 0 and 100.")
        
        returns = self.returns()
        pd_series = {}

        for col in returns.columns:
            z = norm.ppf(level / 100)
            mean = returns[col].mean()
            var = returns[col].std(ddof=1)
            pd_series[col] = mean + z * var

        return pd.Series(pd_series)
        
    def calculate_monthly_compound_return(self):
        """
        Calculates the monthly compounded return based on the closing prices.

        Returns
        -------
        dict
            A dictionary where the keys are column names and the values contain DataFrames 
            with the monthly compounded returns. Each DataFrame has "Year", "Month", and 
            "Compound Return" columns.
        """
        def calculate_compound_return(returns):
            return np.expm1(np.log1p(returns).sum())

        result = {}

        returns = self.returns()
        returns["year"] = returns.index.year
        returns["month"] = returns.index.month

        # -2 because we don't want to include the Year and Month columns
        for col in returns.columns[:-2]:
            compound_return = returns.groupby(["year", "month"])[col].apply(calculate_compound_return).reset_index()
            compound_return.columns = ["year", "month", "compound_return"]

            result[col] = compound_return

        return result