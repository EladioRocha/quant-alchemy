import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import skew, kurtosis, jarque_bera, norm

from quant_alchemy.timeseries import Timeseries

class Portfolio:
    def __init__(self, timeseries):
        """
        Initializes the Portfolio object with a collection of time series data.

        Parameters
        ----------
        timeseries : Timeseries object, pandas.DataFrame, or other valid input types for Timeseries
            The time series data representing the assets in the portfolio. This can be a Timeseries
            object or any input that can be converted into a Timeseries object by the Timeseries class.
        """
        self.timeseries = self.handle_input_assets(timeseries)

    def handle_input_assets(self, timeseries):
        """
        Handles the input time series data by ensuring it is a Timeseries object.

        If the input is not a Timeseries object, it will be converted to one using the Timeseries class.

        Parameters
        ----------
        timeseries : Timeseries object, pandas.DataFrame, or other valid input types for Timeseries
            The time series data to be handled.

        Returns
        -------
        Timeseries
            The input time series data converted to a Timeseries object, if necessary.
        """
        if isinstance(timeseries, Timeseries):
            return timeseries

        return Timeseries(timeseries)

    def correlation_matrix(self, use_log_returns=False):
        """
        Returns the correlation matrix of the portfolio.

        Returns
        -------
        pandas.DataFrame
            The correlation matrix of the portfolio.
        """

        returns = self.timeseries.returns(use_log_returns=use_log_returns)
        return returns.corr()
    
    def covariance_matrix(self, use_log_returns=False, annualize=True, period="daily", trading_days=252, hours_per_day=8):
        """
        Returns the covariance matrix of the portfolio.

        Returns
        -------
        pandas.DataFrame
            The covariance matrix of the portfolio.
        """

        returns = self.timeseries.returns(use_log_returns=use_log_returns)
        covariance = returns.cov()

        if annualize:
            covariance *= self.timeseries.periods_in_year(period=period, trading_days=trading_days, hours_per_day=hours_per_day)

        return covariance
    
    def returns(self, weights=[], use_log_returns=False):
        """
        Computes the return on a portfolio from constituent returns.

        Parameters
        ----------
        weights : np.ndarray or list, optional
            Custom weights to use. Must sum to 1 after normalization.
        use_log_returns : bool, optional
            Whether to use log returns or not. Defaults to False.

        Returns
        -------
        float
            The return on the portfolio.
        """

        returns = self.timeseries.returns(use_log_returns=use_log_returns)
        weights = self.handle_weights(weights)
        
        return returns.dot(weights)

    def volatility(self, weights=[], use_log_returns=False):
        """
        Computes the volatility on a portfolio from constituent returns.

        Parameters
        ----------
        weights : np.ndarray or list, optional
            Custom weights to use. Must sum to 1 after normalization.
        use_log_returns : bool, optional
            Whether to use log returns or not. Defaults to False.

        Returns
        -------
        float
            The volatility on the portfolio.
        """

        weights = self.handle_weights(weights)
        covariance = self.covariance_matrix(use_log_returns=use_log_returns)

        return np.sqrt(np.dot(weights, np.dot(covariance, weights)))

    def annualized_return(self, strategy="equal", weights=[], period="daily", trading_days=252, hours_per_day=8):
        """
        Computes the return on a portfolio from constituent returns using different weighting strategies.

        Parameters
        ----------
        strategy : str, optional
            The weighting strategy to use. Supported options are 'equal', 'random', and 'custom'. Defaults to 'equal'.
        weights : np.ndarray or list, optional
            Custom weights to use if strategy is 'custom'. Must sum to 1 after normalization.

        Returns
        -------
        dict
            A dictionary containing:
            - 'portfolio_return': float, the weighted return of the portfolio.
            - 'weights': list of tuples, the weights associated with each asset in the portfolio. Each tuple contains a string (the asset name) and a float (the weight).
        """

        def custom_weights():
            return self.handle_weights(weights)  # Normalize custom weights

        strategies = {
            "equal": self.equal_weights,
            "random": self.random_weights,
            "custom": custom_weights
        }

        if strategy not in strategies:
            raise ValueError(f"Unknown strategy: {strategy}. Supported strategies are 'equal', 'random', and 'custom'.")

        weights = strategies[strategy]()
        returns = self.timeseries.annualized_return(period=period, trading_days=trading_days, hours_per_day=hours_per_day)

        return {
            "return": np.dot(weights, returns),
            "weights": list(zip(self.timeseries.columns, weights)),
        }
    
    def annualized_volatility(self, strategy="equal", weights=[], period="daily", trading_days=252, hours_per_day=8):
        """
        Computes the volatility on a portfolio from constituent returns using different weighting strategies.

        Parameters
        ----------
        strategy : str, optional
            The weighting strategy to use. Supported options are 'equal', 'random', and 'custom'. Defaults to 'equal'.
        weights : np.ndarray or list, optional
            Custom weights to use if strategy is 'custom'. Must sum to 1 after normalization.

        Returns
        -------
        dict
            A dictionary containing:
            - 'portfolio_volatility': float, the weighted volatility of the portfolio.
            - 'weights': list of tuples, the weights associated with each asset in the portfolio. Each tuple contains a string (the asset name) and a float (the weight).
        """

        def custom_weights():
            return self.handle_weights(weights)  # Normalize custom weights
        
        strategies = {
            "equal": self.equal_weights,
            "random": self.random_weights,
            "custom": custom_weights
        }

        if strategy not in strategies:
            raise ValueError(f"Unknown strategy: {strategy}. Supported strategies are 'equal', 'random', and 'custom'.")
        
        weights = strategies[strategy]()
        covariance = self.covariance_matrix()

        volatility = np.sqrt(np.dot(weights, np.dot(covariance, weights)))
        annualized_volatility = volatility * np.sqrt(self.timeseries.periods_in_year(period, trading_days, hours_per_day))

        return {
            "volatility": annualized_volatility,
            "weights": list(zip(self.timeseries.columns, weights)),
        }
    
    def compound_return(self, weights=[]):
        returns = self.returns(weights=weights)
        return np.expm1(np.log1p(returns).sum())
    
    def cumulative_returns(self, weights=[]):
        returns = self.returns(weights=weights)
        return (1 + returns).cumprod() - 1

    def drawdowns(self, weights=[]):
        """
        Calculates and returns the drawdown series for the time series data.

        A drawdown is the decline from a historical peak during a specific period of investment. 
        It is often quoted as the percentage between the peak to the trough.

        Returns
        -------
        pandas.DataFrame
            The drawdown series for the time series data.
        """
        returns = self.returns(weights=weights)
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = cumulative_returns.divide(running_max) - 1.0

        return pd.DataFrame({
            "drawdowns": drawdown
        })
    
    def max_drawdown(self, weights=[]):
        """
        Calculates and returns the maximum drawdown for the time series data.

        A drawdown is the decline from a historical peak during a specific period of investment. 
        It is often quoted as the percentage between the peak to the trough.

        Returns
        -------
        float
            The maximum drawdown for the time series data.
        """
        drawdowns = self.drawdowns(weights=weights)
        return drawdowns["drawdowns"].min()
        
    def compound_return(self, weights=[]):
        """
        Calculates and returns the compound return for the time series data.

        Returns
        -------
        float
            The compound return for the time series data.
        """
        returns = self.returns(weights=weights)
        return np.expm1(np.log1p(returns).sum())
    
    def skewness(self, weights=[]):
        """
        Calculates and returns the skewness for the time series data.

        Returns
        -------
        float
            The skewness for the time series data.
        """
        returns = self.returns(weights=weights)
        return skew(returns)
    
    def kurtosis(self, weights=[]):
        """
        Calculates and returns the kurtosis for the time series data.

        Returns
        -------
        float
            The kurtosis for the time series data.
        """
        returns = self.returns(weights=weights)
        return kurtosis(returns)

    def is_normal(self, weights=[], level=5):
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
        bool
            True if the returns of the time series data are normally distributed, False otherwise.
        """
        if not 0 <= level <= 100:
            raise ValueError("The 'level' should be a percentile, i.e., between 0 and 100.")
        
        level /= 100

        returns = self.returns(weights=weights)
        _, p_value = jarque_bera(returns)
        return p_value > level

    def sharpe_ratio(self, weights=[], risk_free_rate=0.0, period="daily", trading_days=252, hours_per_day=8):
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
        periods = self.timeseries.periods_in_year(period, trading_days, hours_per_day)
        if isinstance(periods, str):
            raise ValueError(periods)
        
        annualized_return = self.annualized_return(strategy="custom", weights=weights, period=period, trading_days=trading_days, hours_per_day=hours_per_day)["return"]
        annualized_volatility = self.annualized_volatility(strategy="custom", weights=weights,  period=period, trading_days=trading_days, hours_per_day=hours_per_day)["volatility"]
        return (annualized_return - risk_free_rate) / annualized_volatility
    
    def sortino_ratio(self, weights=[], risk_free_rate=0.0, period="daily", trading_days=252, hours_per_day=8):
        """
        Calculates and returns the Sortino ratio of the time series data.

        The Sortino ratio is a measure of risk-adjusted return, defined as the difference between the returns of the investment and the risk-free rate, divided by the standard deviation of the returns that are less than the risk-free rate.

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
            The Sortino ratio of the time series data.
        """
        periods = self.timeseries.periods_in_year(period, trading_days, hours_per_day)
        if isinstance(periods, str):
            raise ValueError(periods)
        
        annualized_return = self.annualized_return(strategy="custom", weights=weights, period=period, trading_days=trading_days, hours_per_day=hours_per_day)
        semi_deviation = self.semideviation(weights=weights, threshold=risk_free_rate)
        return (annualized_return - risk_free_rate) / semi_deviation

    def semideviation(self, weights=[], threshold=0.0):
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
        returns = self.returns(weights=weights)
        return returns[returns < threshold].std(ddof=1)
    
    def var_historic(self, weights=[], level=5):
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
        float
            The historic Value at Risk (VaR) at a specified level.
        """
        if not 0 <= level <= 100:
            raise ValueError("The 'level' should be a percentile, i.e., between 0 and 100.")
        
        returns = self.returns(weights=weights)
        return -np.percentile(returns, level)
    
    def cvar_hist(self, weights=[], level=5):
        """
        Calculates the historic Conditional Value at Risk (CVaR) at a specified level.

        CVaR is a risk assessment technique often used to reduce the probability that a portfolio will incur large losses.

        Parameters
        ----------
        level : float, optional
            The percentile level at which to calculate CVaR, indicating the level below 
            which the returns would fall with the given level of probability. Defaults to 5.

        Raises
        ------
        ValueError
            If the given level is not a valid percentile (i.e., not between 0 and 100).

        Returns
        -------
        float
            The historic Conditional Value at Risk (CVaR) at a specified level.
        """
        if not 0 <= level <= 100:
            raise ValueError("The 'level' should be a percentile, i.e., between 0 and 100.")
        
        returns = self.returns(weights=weights)
        var = self.var_historic(weights=weights, level=level)
        return -returns[returns <= -var].mean()
    
    def var_gaussian(self, weights=[], level=5):
        """
        Calculates the parametric Gaussian Value at Risk (VaR) at a specified level.

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
        float
            The parametric Gaussian Value at Risk (VaR) at a specified level.
        """
        if not 0 <= level <= 100:
            raise ValueError("The 'level' should be a percentile, i.e., between 0 and 100.")
        
        returns = self.returns(weights=weights)
        z = norm.ppf(level / 100)
        mean = returns.mean()
        var = returns.std(ddof=1)

        return -(mean + z * var)
    
    def calculate_monthly_compound_return(self, weights=[]):
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

        returns = self.returns(weights=weights).to_frame("returns")
        returns["year"] = returns.index.year
        returns["month"] = returns.index.month

        compound_return = returns.groupby(["year", "month"])["returns"].apply(calculate_compound_return).reset_index()
        compound_return.columns = ["year", "month", "compound_return"]

        return compound_return

    def equal_weights(self):
        num_assets = self.timeseries.shape[1]
        return np.full(num_assets, 1 / num_assets)
    
    def random_weights(self):
        num_assets = self.timeseries.shape[1]
        weights = np.random.random(num_assets)
        return self.handle_weights(weights)
    
    def handle_weights(self, weights):
        """
        Normalize the given weights to sum to 1, ensuring they correspond to the
        number of assets in the time series.

        The number of assets is determined from the `timeseries` attribute of the object.

        Parameters
        ----------
        weights : np.ndarray or list
            The weights corresponding to the assets. Must have the same length as
            the number of assets in the `timeseries` attribute.

        Returns
        -------
        np.ndarray
            The normalized weights, as a NumPy array, such that they sum to 1.

        Raises
        ------
        ValueError
            If the length of `weights` does not match the number of assets in the
            `timeseries` attribute.
        """

        num_assets = self.timeseries.shape[1]

        if len(weights) != num_assets:
            raise ValueError(f"Number of weights ({len(weights)}) must match number of assets ({num_assets}).")

        if not isinstance(weights, np.ndarray):
            weights = np.array(weights)

        # Normalize the weights to sum 1
        return weights / weights.sum()

    def efficient_frontier(self, n_portfolios=10):
        weights = self.optimal_weights(n_portfolios=n_portfolios)
        portfolio_returns = [self.annualized_return("custom", weight)["return"] for weight in weights]
        portfolio_volatilities = [self.annualized_volatility("custom", weight)["volatility"] for weight in weights]

        # Convert the weights into a DataFrame
        weight_data = [dict(zip(self.timeseries.columns, weight)) for weight in weights]
        weights_df = pd.DataFrame(weight_data)

        # Combine all the data into a single DataFrame
        result_df = pd.DataFrame({
            "return": portfolio_returns,
            "volatility": portfolio_volatilities
        })

        return pd.concat([result_df, weights_df], axis=1)

    def optimal_weights(self, n_portfolios=10):
        annualized_returns = self.timeseries.annualized_return()
        min_return, max_return = annualized_returns.min(), annualized_returns.max()
        target_returns = np.linspace(min_return, max_return, n_portfolios)
        weights = [self.minimize_volatility(target_return) for target_return in target_returns]
        return weights
    
    def minimize_volatility(self, target_return):

        def wrapper_annualized_return(weights):
            portfolio_return = self.annualized_return(strategy="custom", weights=weights)["return"]
            return target_return - portfolio_return

        def wrapper_annualized_volatility(weights):
            portfolio_volatility = self.annualized_volatility(strategy="custom", weights=weights)["volatility"]
            return portfolio_volatility

        num_assets = self.timeseries.shape[1]
        initial_weights = self.random_weights()

        bounds = ((0, 1.0),) * num_assets
        weights = minimize(
            wrapper_annualized_volatility,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=(
                {
                    "type": "eq",
                    "fun": wrapper_annualized_return
                }, 
                {
                    "type": "eq",
                    "fun": lambda weights: np.sum(weights) - 1
                }
            )
        )

        return np.round(weights.x, 4)

    def optimize_portfolio(self, strategy="max_sharpe_ratio", risk_free_rate=0.0, level=5, allow_short_selling=False):
        if strategy == "max_sharpe_ratio":
            obj_function = self.minimize_sharpe
            args = (risk_free_rate,)
        elif strategy == "min_volatility":
            obj_function = self.portfolio_volatility
            args = ()
        elif strategy == "max_return":
            obj_function = self.minimize_expected_return
            args = ()
        elif strategy == "min_drawdown":
            obj_function = self.minimize_drawdown
            args = ()
        elif strategy == "min_cvar":
            obj_function = self.minimize_cvar
            args = (level,)
        elif strategy == "risk_parity":
            obj_function = self.risk_parity
            args = ()
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        num_assets = self.timeseries.shape[1]
        initial_weights = self.random_weights()
        if allow_short_selling:
            bounds = ((-1.0, 1.0),) * num_assets  # Allow short selling
            # Adjust the sum of weights constraint for short selling
            constraints = ({
                "type": "eq",
                "fun": lambda weights: np.sum(np.abs(weights)) - 1
            })
        else:
            bounds = ((0, 1.0),) * num_assets  # Long only
            constraints = ({
                "type": "eq",
                "fun": lambda weights: np.sum(weights) - 1
            })

        weights = minimize(
            obj_function,
            initial_weights,
            args=args,
            method="SLSQP",
            bounds=bounds,
            # Weights must sum to 1
            constraints=constraints
        )

        return np.round(weights.x, 4)

    def minimize_sharpe(self, weights, risk_free_rate=0.0):
        portfolio_return = self.annualized_return(strategy="custom", weights=weights)["return"]
        portfolio_volatility = self.annualized_volatility(strategy="custom", weights=weights)["volatility"]
        return -(portfolio_return - risk_free_rate) / portfolio_volatility

    def portfolio_volatility(self, weights):
        return self.annualized_volatility(strategy="custom", weights=weights)["volatility"]
    
    def minimize_expected_return(self, weights):
        portfolio_return = self.annualized_return("custom", weights=weights)["return"]
        return -portfolio_return
    
    def minimize_drawdown(self, weights):
        return self.max_drawdown(weights)
    
    def minimize_cvar(self, weights, level=5):
        return self.cvar_hist(weights=weights, level=level)
    
    def risk_contribution(self, weights):
        portfolio_volatility = self.annualized_volatility(strategy="custom", weights=weights)["volatility"]
        covariance_matrix = self.covariance_matrix(annualize=True)
        print(covariance_matrix)
        print(portfolio_volatility)
        marginal_contrib = covariance_matrix.dot(weights) / portfolio_volatility
        print(marginal_contrib)
        risk_contrib = weights * marginal_contrib
        print(risk_contrib)
        return risk_contrib

    def risk_parity(self, weights):
        risk_contributions = self.risk_contribution(weights)
        print(risk_contributions)
        return np.std(risk_contributions)