import pandas as pd

from timeseries import Timeseries

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

        returns = self.timeseries.log_returns() if use_log_returns else self.timeseries.returns()
        return returns.corr()
