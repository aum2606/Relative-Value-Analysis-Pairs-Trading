import numpy as np
import pandas as pd
import statsmodels.api as sm
import logging
import sys
import os

# Add the project root to the Python path to import from utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.logger import get_logger
from src.utils.exception import CustomException

class MeanReversion:
    """
    A class implementing mean reversion strategies and tools.
    """
    
    def __init__(self):
        """
        Initialize the MeanReversion class.
        """
        self.logger = get_logger("MeanReversion")
        self.logger.info("Initialized MeanReversion class")
    
    def calculate_zscore(self, series, window=20):
        """
        Calculate the rolling z-score of a time series.
        
        Parameters:
        -----------
        series : array-like
            Input time series data.
        window : int
            Rolling window size for calculating mean and standard deviation.
            
        Returns:
        --------
        zscore : numpy.ndarray
            Z-score series.
        """
        try:
            # Convert to numpy array
            series = np.array(series)
            
            # Initialize z-score array
            zscore = np.zeros_like(series) * np.nan
            
            # Calculate rolling z-score
            for i in range(window, len(series)):
                rolling_mean = np.mean(series[i-window:i])
                rolling_std = np.std(series[i-window:i])
                
                if rolling_std > 0:  # Avoid division by zero
                    zscore[i] = (series[i] - rolling_mean) / rolling_std
            
            self.logger.debug(f"Calculated z-score with shape {zscore.shape}")
            return zscore
        
        except Exception as e:
            self.logger.error(f"Error calculating z-score: {e}")
            raise CustomException(e, sys)
    
    def calculate_bollinger_bands(self, series, window=20, num_std=2):
        """
        Calculate Bollinger Bands for a time series.
        
        Parameters:
        -----------
        series : array-like
            Input time series data.
        window : int
            Rolling window size for calculating the moving average.
        num_std : float
            Number of standard deviations for the upper and lower bands.
            
        Returns:
        --------
        bollinger : dict
            Dictionary containing 'middle', 'upper', and 'lower' bands.
        """
        try:
            # Convert to numpy array
            series = np.array(series)
            
            # Initialize arrays
            middle_band = np.zeros_like(series) * np.nan
            upper_band = np.zeros_like(series) * np.nan
            lower_band = np.zeros_like(series) * np.nan
            
            # Calculate rolling statistics
            for i in range(window, len(series)):
                rolling_mean = np.mean(series[i-window:i])
                rolling_std = np.std(series[i-window:i])
                
                middle_band[i] = rolling_mean
                upper_band[i] = rolling_mean + (num_std * rolling_std)
                lower_band[i] = rolling_mean - (num_std * rolling_std)
            
            bollinger = {
                'middle': middle_band,
                'upper': upper_band,
                'lower': lower_band
            }
            
            self.logger.debug(f"Calculated Bollinger Bands for window={window}, num_std={num_std}")
            return bollinger
        
        except Exception as e:
            self.logger.error(f"Error calculating Bollinger Bands: {e}")
            raise CustomException(e, sys)
    
    def calculate_half_life(self, series):
        """
        Calculate the half-life of mean reversion for a time series.
        
        Parameters:
        -----------
        series : array-like
            Input time series data.
            
        Returns:
        --------
        half_life : float
            Estimated half-life of mean reversion.
        """
        try:
            # Convert to numpy array and remove NaN values
            series = np.array(series)
            series = series[~np.isnan(series)]
            
            # Create lagged series
            lagged_series = series[:-1]
            delta = np.diff(series)
            
            # Perform linear regression
            X = sm.add_constant(lagged_series)
            model = sm.OLS(delta, X).fit()
            
            # Calculate half-life
            lambda_param = -model.params[1]
            if lambda_param <= 0:
                self.logger.warning("Non-mean reverting series detected")
                return np.inf
            
            half_life = np.log(2) / lambda_param
            self.logger.info(f"Calculated half-life: {half_life:.2f}")
            
            return half_life
        
        except Exception as e:
            self.logger.error(f"Error calculating half-life: {e}")
            raise CustomException(e, sys)
    
    def generate_mean_reversion_signals(self, zscore, entry_threshold=2.0, exit_threshold=0.5):
        """
        Generate mean reversion trading signals based on z-score.
        
        Parameters:
        -----------
        zscore : array-like
            Z-score series.
        entry_threshold : float
            Z-score threshold to enter a position.
        exit_threshold : float
            Z-score threshold to exit a position.
            
        Returns:
        --------
        signals : numpy.ndarray
            Array of trading signals: 1 for long, -1 for short, 0 for no position.
        """
        try:
            # Convert to numpy array
            zscore = np.array(zscore)
            
            # Initialize signals array
            signals = np.zeros_like(zscore)
            position = 0
            
            for i in range(1, len(zscore)):
                if np.isnan(zscore[i]):
                    signals[i] = 0
                    continue
                
                # Check for exit conditions
                if position == 1 and zscore[i] >= -exit_threshold:
                    position = 0
                elif position == -1 and zscore[i] <= exit_threshold:
                    position = 0
                
                # Check for entry conditions (if not in a position)
                elif position == 0:
                    if zscore[i] <= -entry_threshold:
                        position = 1  # Long position (buy when z-score is low)
                    elif zscore[i] >= entry_threshold:
                        position = -1  # Short position (sell when z-score is high)
                
                signals[i] = position
            
            self.logger.info(f"Generated mean reversion signals with {np.sum(signals != 0)} active positions")
            return signals
        
        except Exception as e:
            self.logger.error(f"Error generating mean reversion signals: {e}")
            raise CustomException(e, sys)
    
    def calculate_bollinger_band_signals(self, series, window=20, num_std=2):
        """
        Generate trading signals based on Bollinger Bands.
        
        Parameters:
        -----------
        series : array-like
            Input time series data.
        window : int
            Rolling window size for calculating the moving average.
        num_std : float
            Number of standard deviations for the upper and lower bands.
            
        Returns:
        --------
        signals : numpy.ndarray
            Array of trading signals: 1 for long, -1 for short, 0 for no position.
        bollinger : dict
            Dictionary containing Bollinger Bands.
        """
        try:
            # Calculate Bollinger Bands
            bollinger = self.calculate_bollinger_bands(series, window, num_std)
            
            # Initialize signals array
            signals = np.zeros_like(series)
            position = 0
            
            for i in range(window, len(series)):
                if np.isnan(series[i]) or np.isnan(bollinger['upper'][i]) or np.isnan(bollinger['lower'][i]):
                    signals[i] = 0
                    continue
                
                # Check for exit conditions
                if position == 1 and series[i] >= bollinger['middle'][i]:
                    position = 0
                elif position == -1 and series[i] <= bollinger['middle'][i]:
                    position = 0
                
                # Check for entry conditions (if not in a position)
                elif position == 0:
                    if series[i] <= bollinger['lower'][i]:
                        position = 1  # Long position (buy when price is below lower band)
                    elif series[i] >= bollinger['upper'][i]:
                        position = -1  # Short position (sell when price is above upper band)
                
                signals[i] = position
            
            self.logger.info(f"Generated Bollinger Band signals with {np.sum(signals != 0)} active positions")
            return signals, bollinger
        
        except Exception as e:
            self.logger.error(f"Error generating Bollinger Band signals: {e}")
            raise CustomException(e, sys)
    
    def rsi(self, series, window=14):
        """
        Calculate the Relative Strength Index (RSI) for a time series.
        
        Parameters:
        -----------
        series : array-like
            Input time series data.
        window : int
            Period for RSI calculation.
            
        Returns:
        --------
        rsi : numpy.ndarray
            RSI values.
        """
        try:
            # Convert to numpy array
            series = np.array(series)
            
            # Calculate price changes
            delta = np.zeros_like(series)
            delta[1:] = np.diff(series)
            
            # Separate gains and losses
            gains = np.copy(delta)
            losses = np.copy(delta)
            
            gains[gains < 0] = 0
            losses[losses > 0] = 0
            losses = np.abs(losses)
            
            # Initialize arrays
            avg_gain = np.zeros_like(series)
            avg_loss = np.zeros_like(series)
            rs = np.zeros_like(series)
            rsi = np.zeros_like(series)
            
            # Calculate initial average gain and loss
            if window <= len(series):
                avg_gain[window] = np.mean(gains[1:window+1])
                avg_loss[window] = np.mean(losses[1:window+1])
            
            # Calculate RSI using Wilder's smoothing method
            for i in range(window + 1, len(series)):
                avg_gain[i] = (avg_gain[i-1] * (window-1) + gains[i]) / window
                avg_loss[i] = (avg_loss[i-1] * (window-1) + losses[i]) / window
                
                if avg_loss[i] != 0:
                    rs[i] = avg_gain[i] / avg_loss[i]
                else:
                    rs[i] = 100  # Avoid division by zero
                
                rsi[i] = 100 - (100 / (1 + rs[i]))
            
            self.logger.debug(f"Calculated RSI with window={window}")
            return rsi
        
        except Exception as e:
            self.logger.error(f"Error calculating RSI: {e}")
            raise CustomException(e, sys)
    
    def macd(self, series, fast_period=12, slow_period=26, signal_period=9):
        """
        Calculate Moving Average Convergence Divergence (MACD) for a time series.
        
        Parameters:
        -----------
        series : array-like
            Input time series data.
        fast_period : int
            Fast EMA period.
        slow_period : int
            Slow EMA period.
        signal_period : int
            Signal line EMA period.
            
        Returns:
        --------
        macd_result : dict
            Dictionary containing 'macd', 'signal', and 'histogram'.
        """
        try:
            # Convert to numpy array
            series = np.array(series)
            
            # Function to calculate EMA
            def calculate_ema(prices, period):
                ema = np.zeros_like(prices)
                # First value is SMA
                if period <= len(prices):
                    ema[period-1] = np.mean(prices[:period])
                    
                    # Calculate EMA
                    multiplier = 2 / (period + 1)
                    for i in range(period, len(prices)):
                        ema[i] = (prices[i] - ema[i-1]) * multiplier + ema[i-1]
                
                return ema
            
            # Calculate EMAs
            fast_ema = calculate_ema(series, fast_period)
            slow_ema = calculate_ema(series, slow_period)
            
            # Calculate MACD line
            macd_line = fast_ema - slow_ema
            
            # Calculate signal line (EMA of MACD line)
            signal_line = calculate_ema(macd_line, signal_period)
            
            # Calculate histogram
            histogram = macd_line - signal_line
            
            macd_result = {
                'macd': macd_line,
                'signal': signal_line,
                'histogram': histogram
            }
            
            self.logger.debug(f"Calculated MACD with fast={fast_period}, slow={slow_period}, signal={signal_period}")
            return macd_result
        
        except Exception as e:
            self.logger.error(f"Error calculating MACD: {e}")
            raise CustomException(e, sys)


if __name__ == "__main__":
    # Test code to verify the functionality
    try:
        # Generate sample data
        np.random.seed(42)
        n = 1000  # Number of data points
        
        # Create a mean-reverting series
        x = np.zeros(n)
        x[0] = 0
        for i in range(1, n):
            x[i] = 0.7 * x[i-1] + np.random.normal(0, 1)
        
        # Initialize the mean reversion class
        mean_reversion = MeanReversion()
        
        # Test Z-Score calculation
        zscore = mean_reversion.calculate_zscore(x, window=50)
        valid_zscores = zscore[~np.isnan(zscore)]
        print(f"Z-Score: Mean={np.mean(valid_zscores):.4f}, Std={np.std(valid_zscores):.4f}")
        
        # Test Bollinger Bands calculation
        bollinger = mean_reversion.calculate_bollinger_bands(x, window=50, num_std=2)
        
        # Test Half-Life calculation
        half_life = mean_reversion.calculate_half_life(x)
        print(f"Half-Life of Mean Reversion: {half_life:.2f}")
        
        # Test Mean Reversion signals
        signals = mean_reversion.generate_mean_reversion_signals(zscore, entry_threshold=1.5, exit_threshold=0.5)
        print(f"Generated {np.sum(signals != 0)} active positions out of {len(signals)} data points")
        
        # Test Bollinger Band signals
        bb_signals, _ = mean_reversion.calculate_bollinger_band_signals(x, window=50, num_std=2)
        print(f"Generated {np.sum(bb_signals != 0)} Bollinger Band signals out of {len(bb_signals)} data points")
        
        # Test RSI calculation
        rsi = mean_reversion.rsi(x, window=14)
        valid_rsi = rsi[~np.isnan(rsi)]
        print(f"RSI: Min={np.min(valid_rsi):.2f}, Max={np.max(valid_rsi):.2f}, Mean={np.mean(valid_rsi):.2f}")
        
        # Test MACD calculation
        macd_result = mean_reversion.macd(x, fast_period=12, slow_period=26, signal_period=9)
        
        print("\nMean Reversion implementation successful!")
    
    except Exception as e:
        print(f"Error testing mean reversion: {e}")