import numpy as np
import pandas as pd
import datetime as dt
import os
import sys
import logging
from collections import defaultdict

# Add the project root to the Python path to import from utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.logger import get_logger
from src.utils.exception import CustomException
from src.strategies.pairs_trading import PairsTrading


class BacktestEngine:
    """
    Engine for backtesting pairs trading strategies.
    
    This class provides functionality to backtest pairs trading strategies
    using historical price data, tracking portfolio performance, and
    generating detailed trade statistics.
    """
    
    def __init__(self, initial_capital=100000.0, transaction_cost=0.001, 
                 slippage=0.0005, price_col='Adjusted'):
        """
        Initialize the backtest engine.
        
        Parameters:
        -----------
        initial_capital : float
            Initial capital to start the backtest with.
        transaction_cost : float
            Transaction cost as a percentage of trade value.
        slippage : float
            Slippage as a percentage of trade price.
        price_col : str
            Column name to use for price data (default: 'Adjusted').
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.price_col = price_col
        self.logger = get_logger("BacktestEngine")
        
        self.portfolio_value = []
        self.positions = defaultdict(float)
        self.trades = []
        
        self.logger.info(f"Initialized BacktestEngine with capital={initial_capital}, "
                         f"transaction_cost={transaction_cost}, slippage={slippage}")
    
    def prepare_data(self, price_data, tickers=None):
        """
        Prepare the price data for backtesting.
        
        Parameters:
        -----------
        price_data : pandas.DataFrame
            DataFrame containing price data for multiple tickers.
        tickers : list, optional
            List of tickers to include in the backtest. If None, use all tickers.
            
        Returns:
        --------
        prepared_data : dict
            Dictionary of DataFrames indexed by ticker.
        """
        try:
            self.logger.info("Preparing price data for backtesting")
            
            if 'Date' not in price_data.columns and price_data.index.name != 'Date':
                self.logger.error("Price data must contain a 'Date' column or have 'Date' as index")
                raise CustomException("Price data must contain a 'Date' column or have 'Date' as index", sys)
            
            if 'Ticker' not in price_data.columns:
                self.logger.error("Price data must contain a 'Ticker' column")
                raise CustomException("Price data must contain a 'Ticker' column", sys)
            
            # Convert date to datetime if it's not already
            if 'Date' in price_data.columns:
                price_data['Date'] = pd.to_datetime(price_data['Date'])
                price_data.set_index('Date', inplace=True)
            
            # Filter for specific tickers if provided
            if tickers is not None:
                price_data = price_data[price_data['Ticker'].isin(tickers)]
                self.logger.info(f"Filtered price data for {len(tickers)} tickers")
            
            # Ensure the price column exists
            if self.price_col not in price_data.columns:
                self.logger.warning(f"Price column '{self.price_col}' not found. Available columns: {price_data.columns}")
                raise CustomException(f"Price column '{self.price_col}' not found", sys)
            
            # Create a dictionary of dataframes, one for each ticker
            ticker_data = {}
            for ticker in price_data['Ticker'].unique():
                ticker_df = price_data[price_data['Ticker'] == ticker].copy()
                ticker_df.sort_index(inplace=True)
                ticker_data[ticker] = ticker_df
                
                # Check for missing values
                missing_values = ticker_df[self.price_col].isna().sum()
                if missing_values > 0:
                    self.logger.warning(f"Ticker {ticker} has {missing_values} missing values in {self.price_col}")
            
            self.logger.info(f"Prepared data for {len(ticker_data)} tickers from "
                             f"{price_data.index.min()} to {price_data.index.max()}")
            
            return ticker_data
        
        except Exception as e:
            self.logger.error(f"Error preparing data: {e}")
            raise CustomException(e, sys)
    
    def run_backtest(self, price_data, pairs, strategy_params=None, 
                     start_date=None, end_date=None):
        """
        Run a backtest for multiple pairs.
        
        Parameters:
        -----------
        price_data : pandas.DataFrame or dict
            Price data for multiple tickers or a prepared dictionary from prepare_data.
        pairs : list of tuples
            List of (ticker1, ticker2) tuples representing pairs to trade.
        strategy_params : dict, optional
            Parameters for the pairs trading strategy.
        start_date : str or datetime, optional
            Start date for the backtest.
        end_date : str or datetime, optional
            End date for the backtest.
            
        Returns:
        --------
        results : dict
            Dictionary containing backtest results.
        """
        try:
            self.logger.info(f"Starting backtest with {len(pairs)} pairs")
            
            # Initialize results
            self.portfolio_value = []
            self.positions = defaultdict(float)
            self.trades = []
            
            # Convert start and end dates to datetime
            if start_date is not None:
                start_date = pd.to_datetime(start_date)
            if end_date is not None:
                end_date = pd.to_datetime(end_date)
            
            # Prepare data if it's a DataFrame
            if isinstance(price_data, pd.DataFrame):
                tickers = [ticker for pair in pairs for ticker in pair]
                price_data = self.prepare_data(price_data, tickers)
            
            # Set default strategy parameters if not provided
            if strategy_params is None:
                strategy_params = {
                    'lookback_period': 60,
                    'entry_threshold': 2.0,
                    'exit_threshold': 0.5,
                    'stop_loss': 3.0,
                    'max_holding_period': 20,
                    'position_size': 1.0
                }
            
            # Initialize the pairs trading strategy
            pairs_strategy = PairsTrading(**strategy_params)
            
            # Get the common dates across all tickers
            all_dates = set()
            for ticker, data in price_data.items():
                if start_date is not None:
                    data = data[data.index >= start_date]
                if end_date is not None:
                    data = data[data.index <= end_date]
                all_dates.update(data.index)
            
            all_dates = sorted(all_dates)
            self.logger.info(f"Backtest period: {all_dates[0]} to {all_dates[-1]} ({len(all_dates)} days)")
            
            # Initialize portfolio
            self.portfolio_value = pd.Series(index=all_dates, dtype=float)
            self.portfolio_value.iloc[0] = self.initial_capital
            
            # Run the backtest for each pair
            pair_results = {}
            for pair in pairs:
                ticker1, ticker2 = pair
                
                # Check if both tickers are in the data
                if ticker1 not in price_data or ticker2 not in price_data:
                    self.logger.warning(f"Pair {ticker1}-{ticker2} skipped: One or both tickers not in data")
                    continue
                
                # Get price data for both tickers on common dates
                common_dates = sorted(set(price_data[ticker1].index).intersection(set(price_data[ticker2].index)))
                
                if start_date is not None:
                    common_dates = [d for d in common_dates if d >= start_date]
                if end_date is not None:
                    common_dates = [d for d in common_dates if d <= end_date]
                
                if len(common_dates) < strategy_params['lookback_period'] + 10:
                    self.logger.warning(f"Pair {ticker1}-{ticker2} skipped: Insufficient common data points")
                    continue
                
                # Extract price series
                price1 = price_data[ticker1].loc[common_dates, self.price_col].values
                price2 = price_data[ticker2].loc[common_dates, self.price_col].values
                
                # Run pair backtest
                pair_result = pairs_strategy.backtest_pair(price1, price2, common_dates)
                
                # Update portfolio based on pair result
                self._update_portfolio(pair_result, ticker1, ticker2, common_dates)
                
                pair_results[(ticker1, ticker2)] = pair_result
                self.logger.info(f"Completed backtest for pair {ticker1}-{ticker2}")
            
            # Calculate final results
            self.portfolio_value.fillna(method='ffill', inplace=True)
            
            # Calculate returns
            portfolio_returns = self.portfolio_value.pct_change().dropna()
            
            results = {
                'portfolio_value': self.portfolio_value,
                'portfolio_returns': portfolio_returns,
                'pair_results': pair_results,
                'trades': self.trades,
                'final_return': (self.portfolio_value.iloc[-1] / self.initial_capital) - 1,
                'sharpe_ratio': self._calculate_sharpe_ratio(portfolio_returns),
                'max_drawdown': self._calculate_max_drawdown(self.portfolio_value)
            }
            
            self.logger.info(f"Backtest completed with final return: {results['final_return']:.4f}, "
                             f"Sharpe ratio: {results['sharpe_ratio']:.4f}, "
                             f"Max drawdown: {results['max_drawdown']:.4f}")
            
            return results
        
        except Exception as e:
            self.logger.error(f"Error in backtest: {e}")
            raise CustomException(e, sys)
    
    def _update_portfolio(self, pair_result, ticker1, ticker2, dates):
        """
        Update the portfolio based on the pair trading results.
        
        Parameters:
        -----------
        pair_result : pandas.DataFrame
            Results from backtest_pair method.
        ticker1 : str
            First ticker in the pair.
        ticker2 : str
            Second ticker in the pair.
        dates : list
            List of dates corresponding to the pair result.
        """
        try:
            # Current portfolio allocation for this pair
            allocation = 0.5  # 50% of capital allocated to each pair
            
            # Get signals
            signals = pair_result['Signal'].values
            
            # Extract spreads and hedge ratios
            spreads = pair_result['Spread'].values
            hedge_ratio = np.polyfit(pair_result['Stock2'].values, pair_result['Stock1'].values, 1)[0]
            
            # Loop through each day
            for i in range(1, len(dates)):
                date = dates[i]
                prev_date = dates[i-1]
                
                # Skip if no signal
                if np.isnan(signals[i]):
                    continue
                
                # Get current prices
                price1 = pair_result.loc[i, 'Stock1']
                price2 = pair_result.loc[i, 'Stock2']
                
                # Get previous prices
                prev_price1 = pair_result.loc[i-1, 'Stock1']
                prev_price2 = pair_result.loc[i-1, 'Stock2']
                
                # Check for signal change
                if signals[i] != signals[i-1]:
                    # Close previous position
                    if signals[i-1] != 0:
                        # Calculate P&L from previous position
                        if signals[i-1] > 0:  # Long spread position
                            pnl = (price1 - prev_price1) - hedge_ratio * (price2 - prev_price2)
                        else:  # Short spread position
                            pnl = (prev_price1 - price1) - hedge_ratio * (prev_price2 - price2)
                        
                        # Apply transaction costs
                        cost = (self.transaction_cost * abs(prev_price1) + 
                                self.transaction_cost * abs(prev_price2 * hedge_ratio))
                        pnl -= cost
                        
                        # Record the trade
                        self.trades.append({
                            'entry_date': prev_date,
                            'exit_date': date,
                            'ticker1': ticker1,
                            'ticker2': ticker2,
                            'direction': 'long' if signals[i-1] > 0 else 'short',
                            'entry_price1': prev_price1,
                            'entry_price2': prev_price2,
                            'exit_price1': price1,
                            'exit_price2': price2,
                            'pnl': pnl,
                            'transaction_cost': cost
                        })
                    
                    # Open new position
                    if signals[i] != 0:
                        # Apply slippage
                        adjusted_price1 = price1 * (1 + self.slippage * (1 if signals[i] > 0 else -1))
                        adjusted_price2 = price2 * (1 + self.slippage * (1 if signals[i] < 0 else -1))
                        
                        # Record position opening
                        self.logger.debug(f"Opening {ticker1}-{ticker2} {'long' if signals[i] > 0 else 'short'} "
                                          f"position on {date}")
            
            # Update final portfolio value
            self.portfolio_value[dates[-1]] = self.initial_capital * (1 + pair_result['Cumulative_Return'].iloc[-1] * allocation)
        
        except Exception as e:
            self.logger.error(f"Error updating portfolio: {e}")
            raise CustomException(e, sys)
    
    def _calculate_sharpe_ratio(self, returns, risk_free_rate=0.02, annualization=252):
        """
        Calculate the Sharpe ratio of returns.
        
        Parameters:
        -----------
        returns : pandas.Series
            Series of returns.
        risk_free_rate : float
            Annual risk-free rate.
        annualization : int
            Number of periods in a year.
            
        Returns:
        --------
        sharpe_ratio : float
            Annualized Sharpe ratio.
        """
        excess_returns = returns - (risk_free_rate / annualization)
        if len(excess_returns) > 0:
            sharpe = np.sqrt(annualization) * np.mean(excess_returns) / np.std(excess_returns)
            return sharpe
        return 0.0
    
    def _calculate_max_drawdown(self, equity_curve):
        """
        Calculate the maximum drawdown of an equity curve.
        
        Parameters:
        -----------
        equity_curve : pandas.Series
            Series representing the equity curve.
            
        Returns:
        --------
        max_drawdown : float
            Maximum drawdown as a percentage.
        """
        peak = equity_curve.cummax()
        drawdown = (equity_curve / peak) - 1.0
        return drawdown.min()


if __name__ == "__main__":
    # Test code to verify the functionality
    try:
        # Generate sample price data
        np.random.seed(42)
        
        # Generate dates
        start_date = pd.Timestamp('2022-01-01')
        dates = pd.date_range(start=start_date, periods=252, freq='B')
        
        # Generate tickers
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
        
        # Create a common market factor
        market_factor = np.cumsum(np.random.normal(0, 0.01, len(dates)))
        
        # Create price data DataFrame
        data_list = []
        for ticker in tickers:
            # Create specific factor for this ticker
            specific_factor = np.cumsum(np.random.normal(0, 0.005, len(dates)))
            
            # Create the price series
            prices = 100 * np.exp(market_factor + specific_factor)
            
            # Add some noise to high/low
            high = prices * (1 + np.random.uniform(0, 0.01, len(dates)))
            low = prices * (1 - np.random.uniform(0, 0.01, len(dates)))
            open_price = prices * (1 + np.random.uniform(-0.005, 0.005, len(dates)))
            
            # Create returns
            returns = np.zeros_like(prices)
            returns[1:] = (prices[1:] / prices[:-1]) - 1
            
            # Create volume
            volume = np.random.randint(100000, 1000000, len(dates))
            
            # Add to list
            for i in range(len(dates)):
                data_list.append({
                    'Date': dates[i],
                    'Ticker': ticker,
                    'Open': open_price[i],
                    'High': high[i],
                    'Low': low[i],
                    'Close': prices[i],
                    'Adjusted': prices[i],
                    'Returns': returns[i],
                    'Volume': volume[i]
                })
        
        # Create DataFrame
        price_data = pd.DataFrame(data_list)
        
        # Initialize the backtest engine
        backtest_engine = BacktestEngine(
            initial_capital=100000.0,
            transaction_cost=0.001,
            slippage=0.0005
        )
        
        # Prepare data
        prepared_data = backtest_engine.prepare_data(price_data)
        
        # Define pairs for trading
        pairs = [('AAPL', 'MSFT'), ('GOOGL', 'AMZN')]
        
        # Set strategy parameters
        strategy_params = {
            'lookback_period': 30,
            'entry_threshold': 2.0,
            'exit_threshold': 0.5,
            'stop_loss': 3.0,
            'max_holding_period': 10,
            'position_size': 0.5
        }
        
        # Run backtest
        results = backtest_engine.run_backtest(
            prepared_data,
            pairs,
            strategy_params,
            start_date='2022-01-15',
            end_date='2022-12-31'
        )
        
        # Print results
        print("\nBacktest Results:")
        print(f"Final Portfolio Value: ${results['portfolio_value'].iloc[-1]:.2f}")
        print(f"Total Return: {results['final_return']:.2%}")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.4f}")
        print(f"Maximum Drawdown: {results['max_drawdown']:.2%}")
        print(f"Number of Trades: {len(results['trades'])}")
        
        print("\nBacktest engine implementation successful!")
    
    except Exception as e:
        print(f"Error testing backtest engine: {e}")