import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import seaborn as sns
from datetime import datetime, timedelta
import sys
import os

# Add the project root to the Python path to import from utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.logger import get_logger
from src.utils.exception import CustomException


class VisualizationTools:
    """
    A class providing visualization tools for pairs trading strategies.
    """
    
    def __init__(self, style='seaborn-v0_8-darkgrid', figsize=(12, 8)):
        """
        Initialize the visualization tools.
        
        Parameters:
        -----------
        style : str
            Matplotlib style to use.
        figsize : tuple
            Default figure size.
        """
        self.style = style
        self.figsize = figsize
        self.logger = get_logger("VisualizationTools")
        
        # Set the style
        plt.style.use(self.style)
        
        self.logger.info(f"Initialized VisualizationTools with style={style}, figsize={figsize}")
    
    def plot_equity_curve(self, equity_curve, benchmark=None, title='Portfolio Equity Curve',
                           save_path=None):
        """
        Plot the equity curve of a portfolio.
        
        Parameters:
        -----------
        equity_curve : pandas.Series
            Series of portfolio values indexed by date.
        benchmark : pandas.Series, optional
            Series of benchmark values indexed by date.
        title : str
            Plot title.
        save_path : str, optional
            Path to save the figure.
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            The figure object.
        """
        try:
            self.logger.info(f"Plotting equity curve with title '{title}'")
            
            # Create figure
            fig, ax = plt.subplots(figsize=self.figsize)
            
            # Convert to series if needed
            if not isinstance(equity_curve, pd.Series):
                if isinstance(equity_curve, np.ndarray):
                    equity_curve = pd.Series(equity_curve)
                else:
                    raise ValueError("equity_curve must be a pandas Series or numpy array")
            
            # Normalize to 100
            normalized_equity = 100 * equity_curve / equity_curve.iloc[0]
            
            # Plot equity curve
            ax.plot(normalized_equity.index, normalized_equity, label='Strategy', linewidth=2)
            
            # Plot benchmark if provided
            if benchmark is not None:
                # Ensure same length
                if len(benchmark) > len(equity_curve):
                    benchmark = benchmark[:len(equity_curve)]
                elif len(benchmark) < len(equity_curve):
                    benchmark = pd.Series(benchmark, index=equity_curve.index[:len(benchmark)])
                
                # Normalize benchmark
                normalized_benchmark = 100 * benchmark / benchmark.iloc[0]
                
                # Plot benchmark
                ax.plot(normalized_benchmark.index, normalized_benchmark, label='Benchmark', 
                         linewidth=2, linestyle='--', alpha=0.7)
            
            # Set labels and title
            ax.set_xlabel('Date')
            ax.set_ylabel('Portfolio Value (Normalized to 100)')
            ax.set_title(title)
            
            # Format x-axis dates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45)
            
            # Format y-axis to show percentage
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(100))
            
            # Add grid and legend
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Add performance stats in textbox
            total_return = (normalized_equity.iloc[-1] / 100) - 1
            annualized_return = ((1 + total_return) ** (252 / len(equity_curve))) - 1
            
            stats_text = (
                f"Total Return: {total_return:.2%}\n"
                f"Annualized Return: {annualized_return:.2%}"
            )
            
            # Add text box for stats
            plt.figtext(0.15, 0.15, stats_text, bbox=dict(facecolor='white', alpha=0.7))
            
            plt.tight_layout()
            
            # Save figure if path provided
            if save_path:
                plt.savefig(save_path)
                self.logger.info(f"Saved equity curve plot to {save_path}")
            
            return fig
        
        except Exception as e:
            self.logger.error(f"Error plotting equity curve: {e}")
            raise CustomException(e, sys)
    
    def plot_drawdowns(self, equity_curve, top_n=5, title='Portfolio Drawdowns', save_path=None):
        """
        Plot the drawdowns of a portfolio.
        
        Parameters:
        -----------
        equity_curve : pandas.Series
            Series of portfolio values indexed by date.
        top_n : int
            Number of largest drawdowns to highlight.
        title : str
            Plot title.
        save_path : str, optional
            Path to save the figure.
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            The figure object.
        """
        try:
            self.logger.info(f"Plotting drawdowns with title '{title}'")
            
            # Create figure
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, gridspec_kw={'height_ratios': [3, 1]})
            
            # Convert to series if needed
            if not isinstance(equity_curve, pd.Series):
                if isinstance(equity_curve, np.ndarray):
                    equity_curve = pd.Series(equity_curve)
                else:
                    raise ValueError("equity_curve must be a pandas Series or numpy array")
            
            # Plot equity curve
            ax1.plot(equity_curve.index, equity_curve, label='Portfolio Value', linewidth=2)
            ax1.set_title(title)
            ax1.set_ylabel('Portfolio Value')
            ax1.grid(True, alpha=0.3)
            
            # Calculate drawdowns
            roll_max = equity_curve.cummax()
            drawdowns = (equity_curve / roll_max) - 1
            
            # Plot drawdowns
            ax2.fill_between(drawdowns.index, drawdowns, 0, color='red', alpha=0.3, label='Drawdowns')
            ax2.set_ylabel('Drawdown')
            ax2.set_xlabel('Date')
            ax2.grid(True, alpha=0.3)
            
            # Format x-axis dates
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45)
            
            # Format y-axis to show percentage
            ax2.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
            
            # Find top drawdown periods
            prev_value = 0
            drawdown_periods = []
            current_drawdown = {'start': None, 'end': None, 'depth': 0}
            
            for date, value in drawdowns.items():
                if value < 0 and prev_value >= 0:
                    # Start of drawdown
                    current_drawdown = {'start': date, 'end': None, 'depth': value}
                elif value >= 0 and prev_value < 0:
                    # End of drawdown
                    if current_drawdown['start'] is not None:
                        current_drawdown['end'] = date
                        drawdown_periods.append(current_drawdown)
                        current_drawdown = {'start': None, 'end': None, 'depth': 0}
                elif value < current_drawdown['depth']:
                    # Deeper drawdown
                    current_drawdown['depth'] = value
                
                prev_value = value
            
            # Add last drawdown if still active
            if current_drawdown['start'] is not None:
                current_drawdown['end'] = drawdowns.index[-1]
                drawdown_periods.append(current_drawdown)
            
            # Sort drawdown periods by depth
            drawdown_periods.sort(key=lambda x: x['depth'])
            
            # Highlight top drawdown periods
            colors = plt.cm.tab10.colors
            for i, dd in enumerate(drawdown_periods[:min(top_n, len(drawdown_periods))]):
                color = colors[i % len(colors)]
                
                # Shade the area in equity curve
                ax1.axvspan(dd['start'], dd['end'], alpha=0.2, color=color)
                
                # Find the lowest point
                dd_section = drawdowns[drawdowns.index >= dd['start']]
                dd_section = dd_section[dd_section.index <= dd['end']]
                lowest_idx = dd_section.idxmin()
                
                # Mark the lowest point
                ax2.scatter(lowest_idx, dd_section.min(), color=color, s=50, zorder=5)
                
                # Add text label
                ax2.text(lowest_idx, dd_section.min(), f" {dd_section.min():.1%}", color=color)
            
            plt.tight_layout()
            
            # Save figure if path provided
            if save_path:
                plt.savefig(save_path)
                self.logger.info(f"Saved drawdowns plot to {save_path}")
            
            return fig
        
        except Exception as e:
            self.logger.error(f"Error plotting drawdowns: {e}")
            raise CustomException(e, sys)
    
    def plot_returns_distribution(self, returns, title='Returns Distribution', save_path=None):
        """
        Plot the distribution of returns.
        
        Parameters:
        -----------
        returns : pandas.Series or numpy.ndarray
            Series or array of returns.
        title : str
            Plot title.
        save_path : str, optional
            Path to save the figure.
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            The figure object.
        """
        try:
            self.logger.info(f"Plotting returns distribution with title '{title}'")
            
            # Create figure
            fig, ax = plt.subplots(figsize=self.figsize)
            
            # Convert to numpy array
            if isinstance(returns, pd.Series):
                returns_array = returns.values
            else:
                returns_array = np.array(returns)
            
            # Remove NaN values
            returns_array = returns_array[~np.isnan(returns_array)]
            
            # Plot histogram
            sns.histplot(returns_array, bins=50, kde=True, ax=ax)
            
            # Add normal distribution for comparison
            mean = np.mean(returns_array)
            std = np.std(returns_array)
            x = np.linspace(min(returns_array), max(returns_array), 100)
            y = 1 / (std * np.sqrt(2 * np.pi)) * np.exp(-(x - mean)**2 / (2 * std**2))
            y = y * (len(returns_array) * (max(returns_array) - min(returns_array)) / 50)
            ax.plot(x, y, 'r--', linewidth=2, label='Normal Distribution')
            
            # Plot mean and standard deviation lines
            ax.axvline(mean, color='g', linestyle='--', linewidth=2, label=f'Mean: {mean:.4f}')
            ax.axvline(mean + std, color='r', linestyle=':', linewidth=1.5, label=f'+1 Std Dev: {(mean + std):.4f}')
            ax.axvline(mean - std, color='r', linestyle=':', linewidth=1.5, label=f'-1 Std Dev: {(mean - std):.4f}')
            
            # Set labels and title
            ax.set_xlabel('Return')
            ax.set_ylabel('Frequency')
            ax.set_title(title)
            
            # Format x-axis to show percentage
            ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
            
            # Calculate statistics
            skewness = pd.Series(returns_array).skew()
            kurtosis = pd.Series(returns_array).kurt()
            jarque_bera = (len(returns_array) / 6) * (skewness**2 + (kurtosis**2) / 4)
            
            # Add statistics in textbox
            stats_text = (
                f"Mean: {mean:.4%}\n"
                f"Std Dev: {std:.4%}\n"
                f"Skewness: {skewness:.4f}\n"
                f"Kurtosis: {kurtosis:.4f}\n"
                f"Jarque-Bera: {jarque_bera:.4f}"
            )
            
            # Add text box for stats
            plt.figtext(0.15, 0.75, stats_text, bbox=dict(facecolor='white', alpha=0.7))
            
            ax.legend()
            plt.tight_layout()
            
            # Save figure if path provided
            if save_path:
                plt.savefig(save_path)
                self.logger.info(f"Saved returns distribution plot to {save_path}")
            
            return fig
        
        except Exception as e:
            self.logger.error(f"Error plotting returns distribution: {e}")
            raise CustomException(e, sys)
    
    def plot_pair_analysis(self, pair_data, ticker1, ticker2, title=None, save_path=None):
        """
        Visualize a pairs trading analysis with price, spread, and z-score.
        
        Parameters:
        -----------
        pair_data : pandas.DataFrame
            DataFrame containing pair analysis data.
        ticker1 : str
            First ticker symbol.
        ticker2 : str
            Second ticker symbol.
        title : str, optional
            Plot title.
        save_path : str, optional
            Path to save the figure.
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            The figure object.
        """
        try:
            if title is None:
                title = f'Pairs Trading Analysis: {ticker1} vs {ticker2}'
            
            self.logger.info(f"Plotting pair analysis with title '{title}'")
            
            # Create figure
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(self.figsize[0], self.figsize[1]*1.5), 
                                                sharex=True, gridspec_kw={'height_ratios': [2, 1, 1]})
            
            # Plot prices
            ax1.plot(pair_data.index, pair_data['Stock1'], label=ticker1, linewidth=2)
            ax1.plot(pair_data.index, pair_data['Stock2'], label=ticker2, linewidth=2)
            ax1.set_title(title)
            ax1.set_ylabel('Price')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot spread
            ax2.plot(pair_data.index, pair_data['Spread'], label='Spread', color='purple', linewidth=2)
            ax2.set_ylabel('Spread')
            ax2.grid(True, alpha=0.3)
            
            # Plot Z-Score
            ax3.plot(pair_data.index, pair_data['Z-Score'], label='Z-Score', color='green', linewidth=2)
            
            # Add entry/exit thresholds
            entry_threshold = 2.0  # Assuming standard entry threshold
            exit_threshold = 0.5   # Assuming standard exit threshold
            
            ax3.axhline(y=entry_threshold, color='r', linestyle='--', alpha=0.7, label='Entry (Short)')
            ax3.axhline(y=-entry_threshold, color='g', linestyle='--', alpha=0.7, label='Entry (Long)')
            ax3.axhline(y=exit_threshold, color='orange', linestyle=':', alpha=0.7, label='Exit')
            ax3.axhline(y=-exit_threshold, color='orange', linestyle=':', alpha=0.7)
            ax3.axhline(y=0, color='grey', linestyle='-', alpha=0.5)
            
            ax3.set_ylabel('Z-Score')
            ax3.set_xlabel('Date')
            ax3.grid(True, alpha=0.3)
            ax3.legend()
            
            # Format x-axis dates
            ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45)
            
            # Plot trading signals
            if 'Signal' in pair_data.columns:
                # Find signal changes
                signal_changes = pair_data['Signal'].diff().fillna(0)
                
                # Long entries (Signal changes from 0 to 1)
                long_entries = pair_data[(signal_changes == 1)].index
                for entry in long_entries:
                    ax3.scatter(entry, pair_data.loc[entry, 'Z-Score'], color='g', marker='^', s=100, zorder=5)
                
                # Long exits (Signal changes from 1 to 0)
                long_exits = pair_data[(signal_changes == -1) & (pair_data['Signal'].shift(1) == 1)].index
                for exit in long_exits:
                    ax3.scatter(exit, pair_data.loc[exit, 'Z-Score'], color='black', marker='o', s=100, zorder=5)
                
                # Short entries (Signal changes from 0 to -1)
                short_entries = pair_data[(signal_changes == -1) & (pair_data['Signal'].shift(1) == 0)].index
                for entry in short_entries:
                    ax3.scatter(entry, pair_data.loc[entry, 'Z-Score'], color='r', marker='v', s=100, zorder=5)
                
                # Short exits (Signal changes from -1 to 0)
                short_exits = pair_data[(signal_changes == 1) & (pair_data['Signal'].shift(1) == -1)].index
                for exit in short_exits:
                    ax3.scatter(exit, pair_data.loc[exit, 'Z-Score'], color='black', marker='o', s=100, zorder=5)
            
            plt.tight_layout()
            
            # Save figure if path provided
            if save_path:
                plt.savefig(save_path)
                self.logger.info(f"Saved pair analysis plot to {save_path}")
            
            return fig
        
        except Exception as e:
            self.logger.error(f"Error plotting pair analysis: {e}")
            raise CustomException(e, sys)
    
    def plot_performance_metrics(self, metrics, title='Strategy Performance Metrics', save_path=None):
        """
        Visualize key performance metrics.
        
        Parameters:
        -----------
        metrics : dict
            Dictionary of performance metrics.
        title : str
            Plot title.
        save_path : str, optional
            Path to save the figure.
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            The figure object.
        """
        try:
            self.logger.info(f"Plotting performance metrics with title '{title}'")
            
            # Select key metrics to display
            key_metrics = {
                'Total Return': metrics.get('total_return', 0),
                'Annualized Return': metrics.get('annualized_return', 0),
                'Volatility': metrics.get('volatility', 0),
                'Sharpe Ratio': metrics.get('sharpe_ratio', 0),
                'Sortino Ratio': metrics.get('sortino_ratio', 0),
                'Max Drawdown': metrics.get('max_drawdown', 0),
                'Win Rate': metrics.get('win_rate', 0),
                'Profit Factor': metrics.get('profit_factor', 0)
            }
            
            # Create figure
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize, gridspec_kw={'width_ratios': [1, 1.5]})
            
            # Plot return metrics
            return_metrics = ['Total Return', 'Annualized Return', 'Volatility', 'Max Drawdown']
            return_values = [abs(key_metrics[metric]) for metric in return_metrics]
            
            colors = ['green', 'green', 'red', 'red']
            
            # Adjust colors for negative returns
            if key_metrics['Total Return'] < 0:
                colors[0] = 'red'
            if key_metrics['Annualized Return'] < 0:
                colors[1] = 'red'
            
            # Create horizontal bar chart
            bars = ax1.barh(return_metrics, return_values, color=colors, alpha=0.7)
            
            # Add percentage labels
            for bar, value in zip(bars, return_values):
                label_x = bar.get_width() * 1.01
                ax1.text(label_x, bar.get_y() + bar.get_height()/2, f'{value:.2%}', 
                         va='center', fontweight='bold')
            
            ax1.set_title('Return Metrics')
            ax1.set_xlim(0, max(return_values) * 1.2)
            ax1.grid(True, alpha=0.3, axis='x')
            
            # Plot ratio metrics
            ratio_metrics = ['Sharpe Ratio', 'Sortino Ratio', 'Win Rate', 'Profit Factor']
            ratio_values = [key_metrics[metric] for metric in ratio_metrics]
            
            # Use table for ratio metrics
            cell_text = [[f'{v:.2f}' if m not in ['Win Rate'] else f'{v:.2%}' 
                           for m, v in zip(ratio_metrics, ratio_values)]]
            
            ax2.axis('tight')
            ax2.axis('off')
            
            # Create table
            table = ax2.table(cellText=cell_text, rowLabels=['Value'], colLabels=ratio_metrics,
                              cellLoc='center', loc='center', bbox=[0, 0.6, 1, 0.3])
            
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1, 2)
            
            # Color code the cells
            for i, v in enumerate(ratio_values):
                if i < 2:  # Sharpe and Sortino
                    if v > 1.0:
                        table.get_celld()[0, i].set_facecolor('#d6ffd6')  # Light green
                    elif v > 0.5:
                        table.get_celld()[0, i].set_facecolor('#ffffd6')  # Light yellow
                    else:
                        table.get_celld()[0, i].set_facecolor('#ffd6d6')  # Light red
                elif i == 2:  # Win Rate
                    if v > 0.6:
                        table.get_celld()[0, i].set_facecolor('#d6ffd6')
                    elif v > 0.4:
                        table.get_celld()[0, i].set_facecolor('#ffffd6')
                    else:
                        table.get_celld()[0, i].set_facecolor('#ffd6d6')
                else:  # Profit Factor
                    if v > 1.5:
                        table.get_celld()[0, i].set_facecolor('#d6ffd6')
                    elif v > 1.0:
                        table.get_celld()[0, i].set_facecolor('#ffffd6')
                    else:
                        table.get_celld()[0, i].set_facecolor('#ffd6d6')
            
            ax2.set_title('Ratio Metrics')
            
            # Add summary text - FIX: use proper key names in key_metrics dictionary 
            summary_text = (
                "Strategy Performance Summary:\n"
                f"• {'Profitable' if key_metrics['Total Return'] > 0 else 'Unprofitable'} strategy "
                f"with {abs(key_metrics['Total Return']):.2%} total return\n"
                f"• Risk-adjusted performance is {'good' if key_metrics['Sharpe Ratio'] > 1 else 'moderate' if key_metrics['Sharpe Ratio'] > 0.5 else 'poor'} "
                f"(Sharpe: {key_metrics['Sharpe Ratio']:.2f})\n"
                f"• Maximum loss was {abs(key_metrics['Max Drawdown']):.2%} from peak\n"
                f"• Win rate of {key_metrics['Win Rate']:.2%} with "
                f"{'favorable' if key_metrics['Profit Factor'] > 1.5 else 'acceptable' if key_metrics['Profit Factor'] > 1 else 'unfavorable'} "
                f"profit factor ({key_metrics['Profit Factor']:.2f})"
            )
            
            fig.text(0.5, 0.02, summary_text, ha='center', bbox=dict(facecolor='white', alpha=0.7))
            
            plt.tight_layout(rect=[0, 0.15, 1, 0.95])
            fig.suptitle(title, fontsize=16)
            
            # Save figure if path provided
            if save_path:
                plt.savefig(save_path)
                self.logger.info(f"Saved performance metrics plot to {save_path}")
            
            return fig
        
        except Exception as e:
            self.logger.error(f"Error plotting performance metrics: {e}")
            raise CustomException(e, sys)
    
    def create_all_plots(self, backtest_results, save_dir=None):
        """
        Create all relevant plots for a backtest.
        
        Parameters:
        -----------
        backtest_results : dict
            Dictionary containing backtest results.
        save_dir : str, optional
            Directory to save plots.
            
        Returns:
        --------
        plots : dict
            Dictionary of all created plot figures.
        """
        try:
            self.logger.info("Creating all plots for backtest results")
            
            plots = {}
            
            # Create directory if provided
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir)
                self.logger.info(f"Created directory {save_dir}")
            
            # Equity curve plot
            plots['equity_curve'] = self.plot_equity_curve(
                backtest_results['portfolio_value'],
                title='Portfolio Equity Curve',
                save_path=os.path.join(save_dir, 'equity_curve.png') if save_dir else None
            )
            
            # Drawdowns plot
            plots['drawdowns'] = self.plot_drawdowns(
                backtest_results['portfolio_value'],
                title='Portfolio Drawdowns',
                save_path=os.path.join(save_dir, 'drawdowns.png') if save_dir else None
            )
            
            # Returns distribution plot
            plots['returns_distribution'] = self.plot_returns_distribution(
                backtest_results['portfolio_returns'],
                title='Returns Distribution',
                save_path=os.path.join(save_dir, 'returns_distribution.png') if save_dir else None
            )
            
            # Pair analysis plots
            for pair, pair_result in backtest_results['pair_results'].items():
                ticker1, ticker2 = pair
                plots[f'pair_{ticker1}_{ticker2}'] = self.plot_pair_analysis(
                    pair_result,
                    ticker1,
                    ticker2,
                    title=f'Pairs Trading Analysis: {ticker1} vs {ticker2}',
                    save_path=os.path.join(save_dir, f'pair_{ticker1}_{ticker2}.png') if save_dir else None
                )
            
            # Performance metrics plot
            from src.backtesting.performance_metrics import PerformanceMetrics
            metrics_calculator = PerformanceMetrics()
            all_metrics = metrics_calculator.calculate_all_metrics(
                backtest_results['portfolio_value'],
                backtest_results['portfolio_returns'],
                trades=backtest_results['trades']
            )
            
            plots['performance_metrics'] = self.plot_performance_metrics(
                all_metrics,
                title='Strategy Performance Metrics',
                save_path=os.path.join(save_dir, 'performance_metrics.png') if save_dir else None
            )
            
            self.logger.info(f"Created {len(plots)} plots")
            return plots
        
        except Exception as e:
            self.logger.error(f"Error creating all plots: {e}")
            raise CustomException(e, sys)


if __name__ == "__main__":
    # Test code to verify the functionality
    try:
        # Generate sample data
        np.random.seed(42)
        
        # Generate dates
        start_date = pd.Timestamp('2022-01-01')
        dates = pd.date_range(start=start_date, periods=252, freq='B')
        
        # Create returns with positive drift
        returns = np.random.normal(0.0005, 0.01, len(dates))
        
        # Create equity curve
        equity_curve = pd.Series(100000 * np.cumprod(1 + returns), index=dates)
        
        # Create benchmark with lower returns
        benchmark_returns = np.random.normal(0.0003, 0.012, len(dates))
        benchmark = pd.Series(100000 * np.cumprod(1 + benchmark_returns), index=dates)
        
        # Create trades
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
        
        # Create pair data
        stock1 = 100 * np.cumprod(1 + np.random.normal(0.0006, 0.015, len(dates)))
        stock2 = 100 * np.cumprod(1 + np.random.normal(0.0004, 0.012, len(dates)))
        
        # Add correlation
        common_factor = np.random.normal(0, 0.008, len(dates))
        stock1 = stock1 * np.exp(common_factor)
        stock2 = stock2 * np.exp(common_factor)
        
        # Calculate spread
        hedge_ratio = np.polyfit(stock2, stock1, 1)[0]
        spread = stock1 - hedge_ratio * stock2
        
        # Calculate z-score
        window = 20
        zscore = np.zeros_like(spread) * np.nan
        for i in range(window, len(spread)):
            zscore[i] = (spread[i] - np.mean(spread[i-window:i])) / np.std(spread[i-window:i])
        
        # Generate signals
        signals = np.zeros_like(zscore)
        for i in range(1, len(zscore)):
            if np.isnan(zscore[i]):
                continue
                
            # Check for entry/exit conditions
            if signals[i-1] == 0:
                if zscore[i] < -2:
                    signals[i] = 1  # Long
                elif zscore[i] > 2:
                    signals[i] = -1  # Short
            elif signals[i-1] == 1:
                if zscore[i] > -0.5:
                    signals[i] = 0  # Exit long
                else:
                    signals[i] = 1  # Stay long
            elif signals[i-1] == -1:
                if zscore[i] < 0.5:
                    signals[i] = 0  # Exit short
                else:
                    signals[i] = -1  # Stay short
        
        # Create pair result dataframe
        pair_result = pd.DataFrame({
            'Date': dates,
            'Stock1': stock1,
            'Stock2': stock2,
            'Spread': spread,
            'Z-Score': zscore,
            'Signal': signals
        }).set_index('Date')
        
        # Initialize the visualization tools
        viz_tools = VisualizationTools()
        
        # Test all plots
        print("Testing equity curve plot...")
        fig1 = viz_tools.plot_equity_curve(equity_curve, benchmark)
        
        print("Testing drawdowns plot...")
        fig2 = viz_tools.plot_drawdowns(equity_curve)
        
        print("Testing returns distribution plot...")
        fig3 = viz_tools.plot_returns_distribution(returns)
        
        print("Testing pair analysis plot...")
        fig4 = viz_tools.plot_pair_analysis(pair_result, 'AAPL', 'MSFT')
        
        # Create performance metrics
        from src.backtesting.performance_metrics import PerformanceMetrics
        metrics_calculator = PerformanceMetrics()
        all_metrics = metrics_calculator.calculate_all_metrics(equity_curve, returns, trades=trades)
        
        print("Testing performance metrics plot...")
        fig5 = viz_tools.plot_performance_metrics(all_metrics)
        
        # Create mock backtest results
        backtest_results = {
            'portfolio_value': equity_curve,
            'portfolio_returns': pd.Series(returns, index=dates),
            'pair_results': {('AAPL', 'MSFT'): pair_result},
            'trades': trades,
            'final_return': (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1,
            'sharpe_ratio': metrics_calculator.sharpe_ratio(returns),
            'max_drawdown': metrics_calculator.maximum_drawdown(equity_curve)[0]
        }
        
        print("Testing create_all_plots function...")
        all_plots = viz_tools.create_all_plots(backtest_results)
        
        print("\nVisualization tools implementation successful!")
        plt.close('all')  # Close all plots
    
    except Exception as e:
        print(f"Error testing visualization tools: {e}")