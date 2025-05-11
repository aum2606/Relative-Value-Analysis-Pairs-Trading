import numpy as np
import pandas as pd
import logging
import sys
import os

# Add the project root to the Python path to import from utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.logger import get_logger
from src.utils.exception import CustomException

class PairsTrading:
    """
    A class to implement pairs trading strategies based on statistical arbitrage.
    """
    
    def __init__(self, lookback_period=60, entry_threshold=2.0, exit_threshold=0.0, 
                 stop_loss=4.0, max_holding_period=20, position_size=1.0):
        """
        Initialize the pairs trading strategy with parameters.
        
        Parameters:
        -----------
        lookback_period : int
            Number of periods to look back for calculating spread statistics.
        entry_threshold : float
            Z-score threshold to enter a position.
        exit_threshold : float
            Z-score threshold to exit a position.
        stop_loss : float
            Maximum z-score deviation before triggering a stop loss.
        max_holding_period : int
            Maximum number of periods to hold a position.
        position_size : float
            Size of position to take (1.0 = 100% of available capital).
        """
        self.lookback_period = lookback_period
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.stop_loss = stop_loss
        self.max_holding_period = max_holding_period
        self.position_size = position_size
        self.logger = get_logger("PairsTrading")
        self.logger.info("Initialized PairsTrading strategy with parameters: "
                        f"lookback_period={lookback_period}, entry_threshold={entry_threshold}, "
                        f"exit_threshold={exit_threshold}, stop_loss={stop_loss}, "
                        f"max_holding_period={max_holding_period}, position_size={position_size}")
    
    def calculate_spread(self, stock1_prices, stock2_prices, hedge_ratio=None):
        """
        Calculate the spread between two stock price series.
        
        Parameters:
        -----------
        stock1_prices : array-like
            Price series for the first stock.
        stock2_prices : array-like
            Price series for the second stock.
        hedge_ratio : float, optional
            Hedge ratio between the two stocks. If None, it's calculated using OLS regression.
            
        Returns:
        --------
        spread : numpy.ndarray
            The spread series between the two stocks.
        hedge_ratio : float
            The calculated or provided hedge ratio.
        """
        try:
            stock1_prices = np.array(stock1_prices)
            stock2_prices = np.array(stock2_prices)
            
            if hedge_ratio is None:
                # Calculate hedge ratio using OLS regression
                hedge_ratio = np.polyfit(stock2_prices, stock1_prices, 1)[0]
                self.logger.info(f"Calculated hedge ratio: {hedge_ratio}")
            
            # Calculate the spread
            spread = stock1_prices - hedge_ratio * stock2_prices
            self.logger.debug(f"Calculated spread with shape {spread.shape}")
            
            return spread, hedge_ratio
        
        except Exception as e:
            self.logger.error(f"Error calculating spread: {e}")
            raise CustomException(e, sys)
    
    def calculate_zscore(self, spread, lookback_period=None):
        """
        Calculate the z-score of the spread.
        
        Parameters:
        -----------
        spread : array-like
            The spread between two stocks.
        lookback_period : int, optional
            Number of periods to look back for calculating spread statistics.
            If None, uses the instance's lookback_period.
            
        Returns:
        --------
        zscore : numpy.ndarray
            The z-score series for the spread.
        """
        try:
            if lookback_period is None:
                lookback_period = self.lookback_period
                
            # Initialize z-score array
            zscore = np.zeros_like(spread) * np.nan
            
            # Calculate rolling z-score
            for i in range(lookback_period, len(spread)):
                mean = np.mean(spread[i-lookback_period:i])
                std = np.std(spread[i-lookback_period:i])
                if std > 0:  # Avoid division by zero
                    zscore[i] = (spread[i] - mean) / std
            
            self.logger.debug(f"Calculated z-score with shape {zscore.shape}")
            return zscore
        
        except Exception as e:
            self.logger.error(f"Error calculating z-score: {e}")
            raise CustomException(e, sys)
    
    def generate_signals(self, zscore):
        """
        Generate trading signals based on z-score.
        
        Parameters:
        -----------
        zscore : array-like
            The z-score series for the spread.
            
        Returns:
        --------
        signals : numpy.ndarray
            Array of trading signals: 1 for long, -1 for short, 0 for no position.
        """
        try:
            signals = np.zeros_like(zscore)
            position = 0
            entry_price = 0
            holding_period = 0
            
            for i in range(1, len(zscore)):
                if np.isnan(zscore[i]):
                    signals[i] = 0
                    continue
                
                # Update holding period if in a position
                if position != 0:
                    holding_period += 1
                
                # Check for exit conditions
                if position != 0:
                    # Exit if z-score crosses exit threshold
                    if (position == 1 and zscore[i] <= self.exit_threshold) or \
                       (position == -1 and zscore[i] >= -self.exit_threshold):
                        position = 0
                        holding_period = 0
                        self.logger.debug(f"Exit signal at index {i}, zscore: {zscore[i]}")
                    
                    # Exit if stop loss is hit
                    elif (position == 1 and zscore[i] >= self.stop_loss) or \
                         (position == -1 and zscore[i] <= -self.stop_loss):
                        position = 0
                        holding_period = 0
                        self.logger.debug(f"Stop loss triggered at index {i}, zscore: {zscore[i]}")
                    
                    # Exit if max holding period is reached
                    elif holding_period >= self.max_holding_period:
                        position = 0
                        holding_period = 0
                        self.logger.debug(f"Max holding period reached at index {i}")
                
                # Check for entry conditions
                elif position == 0:
                    # Long entry (pair is undervalued)
                    if zscore[i] <= -self.entry_threshold:
                        position = 1
                        self.logger.debug(f"Long entry signal at index {i}, zscore: {zscore[i]}")
                    
                    # Short entry (pair is overvalued)
                    elif zscore[i] >= self.entry_threshold:
                        position = -1
                        self.logger.debug(f"Short entry signal at index {i}, zscore: {zscore[i]}")
                
                signals[i] = position
            
            self.logger.info(f"Generated signals with {np.sum(signals != 0)} active positions")
            return signals
        
        except Exception as e:
            self.logger.error(f"Error generating signals: {e}")
            raise CustomException(e, sys)
    
    def backtest_pair(self, stock1_prices, stock2_prices, dates=None):
        """
        Backtest a pairs trading strategy for given stock prices.
        
        Parameters:
        -----------
        stock1_prices : array-like
            Price series for the first stock.
        stock2_prices : array-like
            Price series for the second stock.
        dates : array-like, optional
            Dates corresponding to the price series.
            
        Returns:
        --------
        results : pandas.DataFrame
            DataFrame containing the backtest results.
        """
        try:
            # Calculate spread and z-score
            spread, hedge_ratio = self.calculate_spread(stock1_prices, stock2_prices)
            zscore = self.calculate_zscore(spread)
            
            # Generate signals
            signals = self.generate_signals(zscore)
            
            # Calculate returns
            stock1_returns = np.zeros_like(stock1_prices)
            stock2_returns = np.zeros_like(stock2_prices)
            stock1_returns[1:] = np.diff(stock1_prices) / stock1_prices[:-1]
            stock2_returns[1:] = np.diff(stock2_prices) / stock2_prices[:-1]
            
            # Calculate strategy returns
            # Long stock1, short stock2 when signal is 1
            # Short stock1, long stock2 when signal is -1
            strategy_returns = signals * (stock1_returns - hedge_ratio * stock2_returns)
            
            # Calculate cumulative returns
            cumulative_returns = np.cumprod(1 + strategy_returns * self.position_size) - 1
            
            # Create results DataFrame
            results = pd.DataFrame({
                'Date': dates if dates is not None else np.arange(len(stock1_prices)),
                'Stock1': stock1_prices,
                'Stock2': stock2_prices,
                'Spread': spread,
                'Z-Score': zscore,
                'Signal': signals,
                'Strategy_Return': strategy_returns,
                'Cumulative_Return': cumulative_returns
            })
            
            self.logger.info(f"Backtest completed with final return: {cumulative_returns[-1]:.4f}")
            return results
        
        except Exception as e:
            self.logger.error(f"Error in backtest: {e}")
            raise CustomException(e, sys)
    
    def calculate_performance_metrics(self, returns):
        """
        Calculate performance metrics for the strategy.
        
        Parameters:
        -----------
        returns : array-like
            Array of strategy returns.
            
        Returns:
        --------
        metrics : dict
            Dictionary of performance metrics.
        """
        try:
            # Remove NaN values
            returns = returns[~np.isnan(returns)]
            
            # Calculate metrics
            total_return = np.prod(1 + returns) - 1
            annual_return = (1 + total_return) ** (252 / len(returns)) - 1
            daily_returns = returns
            
            volatility = np.std(daily_returns) * np.sqrt(252)
            sharpe_ratio = annual_return / volatility if volatility > 0 else 0
            
            # Calculate maximum drawdown
            cum_returns = np.cumprod(1 + daily_returns)
            running_max = np.maximum.accumulate(cum_returns)
            drawdown = (cum_returns / running_max) - 1
            max_drawdown = np.min(drawdown)
            
            # Calculate win rate
            wins = np.sum(daily_returns > 0)
            losses = np.sum(daily_returns < 0)
            win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0
            
            metrics = {
                'Total Return': total_return,
                'Annual Return': annual_return,
                'Volatility': volatility,
                'Sharpe Ratio': sharpe_ratio,
                'Max Drawdown': max_drawdown,
                'Win Rate': win_rate
            }
            
            self.logger.info(f"Performance metrics calculated: Sharpe Ratio: {sharpe_ratio:.2f}, "
                            f"Max Drawdown: {max_drawdown:.2%}, Win Rate: {win_rate:.2%}")
            return metrics
        
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {e}")
            raise CustomException(e, sys)


if __name__ == "__main__":
    # Test code to verify the functionality
    try:
        # Generate sample data
        np.random.seed(42)
        n = 1000  # Number of data points
        
        # Create two cointegrated series
        common_factor = np.cumsum(np.random.normal(0, 1, n))
        stock1 = common_factor + np.random.normal(0, 1, n)
        stock2 = 0.7 * common_factor + np.random.normal(0, 1, n)
        
        # Initialize the pairs trading strategy
        pairs_trader = PairsTrading(
            lookback_period=50,
            entry_threshold=2.0,
            exit_threshold=0.5,
            stop_loss=3.0,
            max_holding_period=15,
            position_size=0.5
        )
        
        # Run backtest
        results = pairs_trader.backtest_pair(stock1, stock2)
        
        # Calculate and print performance metrics
        metrics = pairs_trader.calculate_performance_metrics(results['Strategy_Return'].values)
        print("\nPerformance Metrics:")
        for key, value in metrics.items():
            if key in ['Total Return', 'Annual Return', 'Volatility', 'Max Drawdown', 'Win Rate']:
                print(f"{key}: {value:.2%}")
            else:
                print(f"{key}: {value:.2f}")
        
        print("\nPairs Trading strategy implementation successful!")
    
    except Exception as e:
        print(f"Error testing pairs trading strategy: {e}")