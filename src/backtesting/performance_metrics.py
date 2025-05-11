import numpy as np
import pandas as pd
import scipy.stats as stats
import sys
import os

# Add the project root to the Python path to import from utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.logger import get_logger
from src.utils.exception import CustomException


class PerformanceMetrics:
    """
    Class for calculating various performance metrics for trading strategies.
    """
    
    def __init__(self, annualization_factor=252):
        """
        Initialize the performance metrics calculator.
        
        Parameters:
        -----------
        annualization_factor : int
            Number of periods in a year for annualizing returns.
            Use 252 for daily, 52 for weekly, 12 for monthly.
        """
        self.annualization_factor = annualization_factor
        self.logger = get_logger("PerformanceMetrics")
        self.logger.info(f"Initialized PerformanceMetrics with annualization_factor={annualization_factor}")
    
    def calculate_returns(self, equity_curve):
        """
        Calculate returns from an equity curve.
        
        Parameters:
        -----------
        equity_curve : pandas.Series or numpy.ndarray
            Series or array of portfolio values.
            
        Returns:
        --------
        returns : pandas.Series or numpy.ndarray
            Series or array of returns.
        """
        try:
            if isinstance(equity_curve, pd.Series):
                returns = equity_curve.pct_change().dropna()
            else:
                returns = np.zeros_like(equity_curve)
                returns[1:] = (equity_curve[1:] / equity_curve[:-1]) - 1
                returns = returns[1:]
            
            self.logger.debug(f"Calculated returns with shape {returns.shape}")
            return returns
        
        except Exception as e:
            self.logger.error(f"Error calculating returns: {e}")
            raise CustomException(e, sys)
    
    def sharpe_ratio(self, returns, risk_free_rate=0.0):
        """
        Calculate the Sharpe ratio.
        
        Parameters:
        -----------
        returns : pandas.Series or numpy.ndarray
            Series or array of returns.
        risk_free_rate : float
            Annualized risk-free rate.
            
        Returns:
        --------
        sharpe_ratio : float
            Annualized Sharpe ratio.
        """
        try:
            # Convert to numpy array
            returns_array = np.array(returns)
            
            # Remove NaN values
            returns_array = returns_array[~np.isnan(returns_array)]
            
            if len(returns_array) == 0:
                self.logger.warning("Empty returns array, cannot calculate Sharpe ratio")
                return 0.0
            
            # Calculate excess returns
            daily_risk_free = (1 + risk_free_rate) ** (1 / self.annualization_factor) - 1
            excess_returns = returns_array - daily_risk_free
            
            # Calculate Sharpe ratio
            if np.std(excess_returns) == 0:
                self.logger.warning("Standard deviation of excess returns is zero")
                return 0.0
            
            sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(self.annualization_factor)
            
            self.logger.debug(f"Calculated Sharpe ratio: {sharpe:.4f}")
            return sharpe
        
        except Exception as e:
            self.logger.error(f"Error calculating Sharpe ratio: {e}")
            raise CustomException(e, sys)
    
    def sortino_ratio(self, returns, risk_free_rate=0.0, target_return=0.0):
        """
        Calculate the Sortino ratio, which penalizes only downside deviation.
        
        Parameters:
        -----------
        returns : pandas.Series or numpy.ndarray
            Series or array of returns.
        risk_free_rate : float
            Annualized risk-free rate.
        target_return : float
            Target return, typically risk-free rate.
            
        Returns:
        --------
        sortino_ratio : float
            Annualized Sortino ratio.
        """
        try:
            # Convert to numpy array
            returns_array = np.array(returns)
            
            # Remove NaN values
            returns_array = returns_array[~np.isnan(returns_array)]
            
            if len(returns_array) == 0:
                self.logger.warning("Empty returns array, cannot calculate Sortino ratio")
                return 0.0
            
            # Calculate excess returns
            daily_risk_free = (1 + risk_free_rate) ** (1 / self.annualization_factor) - 1
            excess_returns = returns_array - daily_risk_free
            
            # Calculate downside deviation (only negative returns)
            downside_returns = excess_returns[excess_returns < target_return]
            
            if len(downside_returns) == 0:
                self.logger.warning("No downside returns, Sortino ratio approaching infinity")
                return np.inf
            
            downside_deviation = np.sqrt(np.mean(downside_returns ** 2))
            
            if downside_deviation == 0:
                self.logger.warning("Downside deviation is zero, Sortino ratio approaching infinity")
                return np.inf
            
            # Calculate Sortino ratio
            sortino = np.mean(excess_returns) / downside_deviation * np.sqrt(self.annualization_factor)
            
            self.logger.debug(f"Calculated Sortino ratio: {sortino:.4f}")
            return sortino
        
        except Exception as e:
            self.logger.error(f"Error calculating Sortino ratio: {e}")
            raise CustomException(e, sys)
    
    def maximum_drawdown(self, equity_curve):
        """
        Calculate the maximum drawdown.
        
        Parameters:
        -----------
        equity_curve : pandas.Series or numpy.ndarray
            Series or array of portfolio values.
            
        Returns:
        --------
        max_drawdown : float
            Maximum drawdown as a percentage.
        mdd_details : dict
            Dictionary containing drawdown details (peak, trough, recovery).
        """
        try:
            # Convert to numpy array if Series
            if isinstance(equity_curve, pd.Series):
                dates = equity_curve.index
                equity_array = equity_curve.values
            else:
                dates = np.arange(len(equity_curve))
                equity_array = np.array(equity_curve)
            
            # Find running maximum
            running_max = np.maximum.accumulate(equity_array)
            
            # Calculate drawdown
            drawdown = (equity_array / running_max) - 1
            
            # Find the maximum drawdown
            max_drawdown = np.min(drawdown)
            
            # Find the peak, trough, and recovery
            if max_drawdown < 0:
                trough_idx = np.argmin(drawdown)
                
                # Find the peak before the trough
                peak_idx = np.argmax(equity_array[:trough_idx+1])
                
                # Find the recovery after the trough
                recovery_indices = np.where(equity_array[trough_idx:] >= equity_array[peak_idx])[0]
                
                if len(recovery_indices) > 0:
                    recovery_idx = recovery_indices[0] + trough_idx
                else:
                    recovery_idx = None
                
                # Create details dictionary
                mdd_details = {
                    'max_drawdown': max_drawdown,
                    'peak_date': dates[peak_idx] if recovery_idx is not None else None,
                    'trough_date': dates[trough_idx],
                    'recovery_date': dates[recovery_idx] if recovery_idx is not None else None,
                    'peak_value': equity_array[peak_idx],
                    'trough_value': equity_array[trough_idx],
                    'recovery_value': equity_array[recovery_idx] if recovery_idx is not None else None
                }
            else:
                # No drawdown
                mdd_details = {
                    'max_drawdown': 0.0,
                    'peak_date': None,
                    'trough_date': None,
                    'recovery_date': None,
                    'peak_value': None,
                    'trough_value': None,
                    'recovery_value': None
                }
            
            self.logger.debug(f"Calculated maximum drawdown: {max_drawdown:.4f}")
            return max_drawdown, mdd_details
        
        except Exception as e:
            self.logger.error(f"Error calculating maximum drawdown: {e}")
            raise CustomException(e, sys)
    
    def calmar_ratio(self, returns, equity_curve):
        """
        Calculate the Calmar ratio (annualized return / maximum drawdown).
        
        Parameters:
        -----------
        returns : pandas.Series or numpy.ndarray
            Series or array of returns.
        equity_curve : pandas.Series or numpy.ndarray
            Series or array of portfolio values.
            
        Returns:
        --------
        calmar_ratio : float
            Calmar ratio.
        """
        try:
            # Calculate annualized return
            returns_array = np.array(returns)
            returns_array = returns_array[~np.isnan(returns_array)]
            
            if len(returns_array) == 0:
                self.logger.warning("Empty returns array, cannot calculate Calmar ratio")
                return 0.0
            
            annualized_return = np.mean(returns_array) * self.annualization_factor
            
            # Calculate maximum drawdown
            max_drawdown, _ = self.maximum_drawdown(equity_curve)
            
            # Avoid division by zero
            if max_drawdown == 0:
                self.logger.warning("Maximum drawdown is zero, Calmar ratio approaching infinity")
                return np.inf
            
            # Calculate Calmar ratio
            calmar = annualized_return / abs(max_drawdown)
            
            self.logger.debug(f"Calculated Calmar ratio: {calmar:.4f}")
            return calmar
        
        except Exception as e:
            self.logger.error(f"Error calculating Calmar ratio: {e}")
            raise CustomException(e, sys)
    
    def information_ratio(self, returns, benchmark_returns):
        """
        Calculate the Information ratio (excess return / tracking error).
        
        Parameters:
        -----------
        returns : pandas.Series or numpy.ndarray
            Series or array of strategy returns.
        benchmark_returns : pandas.Series or numpy.ndarray
            Series or array of benchmark returns.
            
        Returns:
        --------
        information_ratio : float
            Information ratio.
        """
        try:
            # Convert to numpy arrays
            returns_array = np.array(returns)
            benchmark_array = np.array(benchmark_returns)
            
            # Align the arrays
            min_length = min(len(returns_array), len(benchmark_array))
            returns_array = returns_array[:min_length]
            benchmark_array = benchmark_array[:min_length]
            
            # Remove NaN values
            valid_indices = ~(np.isnan(returns_array) | np.isnan(benchmark_array))
            returns_array = returns_array[valid_indices]
            benchmark_array = benchmark_array[valid_indices]
            
            if len(returns_array) == 0:
                self.logger.warning("No valid returns, cannot calculate Information ratio")
                return 0.0
            
            # Calculate excess returns
            excess_returns = returns_array - benchmark_array
            
            # Calculate tracking error
            tracking_error = np.std(excess_returns)
            
            if tracking_error == 0:
                self.logger.warning("Tracking error is zero, Information ratio approaching infinity")
                return np.inf
            
            # Calculate Information ratio
            info_ratio = np.mean(excess_returns) / tracking_error * np.sqrt(self.annualization_factor)
            
            self.logger.debug(f"Calculated Information ratio: {info_ratio:.4f}")
            return info_ratio
        
        except Exception as e:
            self.logger.error(f"Error calculating Information ratio: {e}")
            raise CustomException(e, sys)
    
    def win_rate(self, trades):
        """
        Calculate the win rate from a list of trades.
        
        Parameters:
        -----------
        trades : list of dict
            List of trade dictionaries, each with a 'pnl' key.
            
        Returns:
        --------
        win_rate : float
            Percentage of winning trades.
        """
        try:
            if not trades:
                self.logger.warning("Empty trades list, cannot calculate win rate")
                return 0.0
            
            # Count winning trades
            winning_trades = sum(1 for trade in trades if trade['pnl'] > 0)
            
            # Calculate win rate
            win_rate = winning_trades / len(trades)
            
            self.logger.debug(f"Calculated win rate: {win_rate:.4f}")
            return win_rate
        
        except Exception as e:
            self.logger.error(f"Error calculating win rate: {e}")
            raise CustomException(e, sys)
    
    def profit_factor(self, trades):
        """
        Calculate the profit factor (gross profit / gross loss).
        
        Parameters:
        -----------
        trades : list of dict
            List of trade dictionaries, each with a 'pnl' key.
            
        Returns:
        --------
        profit_factor : float
            Profit factor.
        """
        try:
            if not trades:
                self.logger.warning("Empty trades list, cannot calculate profit factor")
                return 0.0
            
            # Calculate gross profit and loss
            gross_profit = sum(trade['pnl'] for trade in trades if trade['pnl'] > 0)
            gross_loss = sum(abs(trade['pnl']) for trade in trades if trade['pnl'] < 0)
            
            # Calculate profit factor
            if gross_loss == 0:
                self.logger.warning("No losing trades, profit factor approaching infinity")
                return np.inf
            
            profit_factor = gross_profit / gross_loss
            
            self.logger.debug(f"Calculated profit factor: {profit_factor:.4f}")
            return profit_factor
        
        except Exception as e:
            self.logger.error(f"Error calculating profit factor: {e}")
            raise CustomException(e, sys)
    
    def expected_shortfall(self, returns, alpha=0.05):
        """
        Calculate the Expected Shortfall (ES) / Conditional Value at Risk (CVaR).
        
        Parameters:
        -----------
        returns : pandas.Series or numpy.ndarray
            Series or array of returns.
        alpha : float
            Significance level (e.g., 0.05 for 95% confidence).
            
        Returns:
        --------
        expected_shortfall : float
            Expected Shortfall.
        """
        try:
            # Convert to numpy array
            returns_array = np.array(returns)
            
            # Remove NaN values
            returns_array = returns_array[~np.isnan(returns_array)]
            
            if len(returns_array) == 0:
                self.logger.warning("Empty returns array, cannot calculate Expected Shortfall")
                return 0.0
            
            # Sort returns
            sorted_returns = np.sort(returns_array)
            
            # Calculate index for VaR
            var_index = int(np.ceil(alpha * len(sorted_returns))) - 1
            
            # Ensure index is valid
            var_index = max(0, min(var_index, len(sorted_returns) - 1))
            
            # Calculate Expected Shortfall
            es = np.mean(sorted_returns[:var_index+1])
            
            self.logger.debug(f"Calculated Expected Shortfall: {es:.4f}")
            return es
        
        except Exception as e:
            self.logger.error(f"Error calculating Expected Shortfall: {e}")
            raise CustomException(e, sys)
    
    def calculate_all_metrics(self, equity_curve, returns=None, benchmark_returns=None, trades=None):
        """
        Calculate all performance metrics.
        
        Parameters:
        -----------
        equity_curve : pandas.Series or numpy.ndarray
            Series or array of portfolio values.
        returns : pandas.Series or numpy.ndarray, optional
            Series or array of returns. If None, calculated from equity_curve.
        benchmark_returns : pandas.Series or numpy.ndarray, optional
            Series or array of benchmark returns.
        trades : list of dict, optional
            List of trade dictionaries for trade-based metrics.
            
        Returns:
        --------
        metrics : dict
            Dictionary of all calculated metrics.
        """
        try:
            self.logger.info("Calculating all performance metrics")
            
            # Calculate returns if not provided
            if returns is None:
                returns = self.calculate_returns(equity_curve)
            
            # Initialize metrics dictionary
            metrics = {}
            
            # Basic return metrics
            if isinstance(equity_curve, pd.Series):
                metrics['total_return'] = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
            else:
                metrics['total_return'] = (equity_curve[-1] / equity_curve[0]) - 1
                
            metrics['annualized_return'] = np.mean(returns) * self.annualization_factor
            metrics['volatility'] = np.std(returns) * np.sqrt(self.annualization_factor)
            
            # Risk-adjusted metrics
            metrics['sharpe_ratio'] = self.sharpe_ratio(returns)
            metrics['sortino_ratio'] = self.sortino_ratio(returns)
            
            # Drawdown metrics
            metrics['max_drawdown'], metrics['drawdown_details'] = self.maximum_drawdown(equity_curve)
            metrics['calmar_ratio'] = self.calmar_ratio(returns, equity_curve)
            
            # Benchmark-relative metrics (if benchmark provided)
            if benchmark_returns is not None:
                metrics['information_ratio'] = self.information_ratio(returns, benchmark_returns)
                
                # Calculate beta and alpha
                cov_matrix = np.cov(returns, benchmark_returns)
                if cov_matrix.shape == (2, 2) and cov_matrix[1, 1] != 0:
                    metrics['beta'] = cov_matrix[0, 1] / cov_matrix[1, 1]
                    metrics['alpha'] = np.mean(returns) - metrics['beta'] * np.mean(benchmark_returns)
                else:
                    metrics['beta'] = np.nan
                    metrics['alpha'] = np.nan
            
            # Trade-based metrics (if trades provided)
            if trades:
                metrics['win_rate'] = self.win_rate(trades)
                metrics['profit_factor'] = self.profit_factor(trades)
                metrics['average_win'] = np.mean([t['pnl'] for t in trades if t['pnl'] > 0]) if any(t['pnl'] > 0 for t in trades) else 0
                metrics['average_loss'] = np.mean([t['pnl'] for t in trades if t['pnl'] < 0]) if any(t['pnl'] < 0 for t in trades) else 0
                
                if metrics['average_loss'] != 0:
                    metrics['win_loss_ratio'] = abs(metrics['average_win'] / metrics['average_loss'])
                else:
                    metrics['win_loss_ratio'] = np.inf
            
            # Risk metrics
            metrics['value_at_risk_95'] = np.percentile(returns, 5)
            metrics['expected_shortfall_95'] = self.expected_shortfall(returns, alpha=0.05)
            metrics['skewness'] = stats.skew(returns)
            metrics['kurtosis'] = stats.kurtosis(returns)
            
            self.logger.info("Successfully calculated all performance metrics")
            return metrics
        
        except Exception as e:
            self.logger.error(f"Error calculating all metrics: {e}")
            raise CustomException(e, sys)


if __name__ == "__main__":
    # Test code to verify the functionality
    try:
        # Generate sample returns data
        np.random.seed(42)
        n_days = 252  # Trading days in a year
        
        # Generate random returns with positive drift
        returns = np.random.normal(0.0005, 0.01, n_days)  # Mean daily return of 0.05%
        
        # Create an equity curve
        equity_curve = 100000 * np.cumprod(1 + returns)
        
        # Generate benchmark returns
        benchmark_returns = np.random.normal(0.0003, 0.012, n_days)  # Lower mean, higher volatility
        
        # Create sample trades
        trades = []
        for i in range(50):
            pnl = np.random.normal(100, 500)
            trades.append({
                'entry_date': pd.Timestamp('2022-01-01') + pd.Timedelta(days=i*5),
                'exit_date': pd.Timestamp('2022-01-01') + pd.Timedelta(days=i*5+3),
                'ticker1': 'AAPL',
                'ticker2': 'MSFT',
                'direction': 'long' if np.random.rand() > 0.5 else 'short',
                'pnl': pnl,
                'transaction_cost': abs(pnl) * 0.001
            })
        
        # Initialize the performance metrics calculator
        metrics_calculator = PerformanceMetrics(annualization_factor=252)
        
        # Calculate all metrics
        all_metrics = metrics_calculator.calculate_all_metrics(equity_curve, returns, benchmark_returns, trades)
        
        # Print the results
        print("\nPerformance Metrics:")
        for key, value in all_metrics.items():
            if key != 'drawdown_details':
                if isinstance(value, float) and not np.isnan(value):
                    if key in ['total_return', 'annualized_return', 'volatility', 'max_drawdown', 'win_rate']:
                        print(f"{key}: {value:.2%}")
                    else:
                        print(f"{key}: {value:.4f}")
                else:
                    print(f"{key}: {value}")
        
        # Print drawdown details
        print("\nDrawdown Details:")
        for key, value in all_metrics['drawdown_details'].items():
            if key != 'max_drawdown':
                print(f"{key}: {value}")
        
        print("\nPerformance metrics implementation successful!")
    
    except Exception as e:
        print(f"Error testing performance metrics: {e}")