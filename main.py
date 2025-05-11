import os
import sys
import pandas as pd
import numpy as np
import argparse
from datetime import datetime

# Import project modules
from src.utils.logger import get_logger
from src.utils.exception import CustomException
from src.data.data_ingestion import DataIngestion
from src.data.data_preprocessing import DataPreprocessing
from src.analysis.correlation_analysis import CorrelationAnalysis, CorrelationAnalysisConfig
from src.analysis.cointegration_tests import CointegrationTests, CointegrationTestsConfig
from src.strategies.pairs_trading import PairsTrading
from src.strategies.mean_reversion import MeanReversion
from src.backtesting.backtest_engine import BacktestEngine
from src.backtesting.performance_metrics import PerformanceMetrics
from src.backtesting.visualization import VisualizationTools

# Initialize logger
logger = get_logger(__name__)

def create_directory_structure():
    """Create the necessary directory structure for the project."""
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('notebooks', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    logger.info("Directory structure created successfully")

def run_data_pipeline():
    """Load and preprocess the data."""
    try:
        # Initialize data ingestion
        logger.info("Starting data ingestion process")
        data_ingestion = DataIngestion()
        
        # Read the raw data
        raw_df = data_ingestion.read_data()
        
        # Initialize data preprocessing
        logger.info("Starting data preprocessing process")
        data_preprocessor = DataPreprocessing()
        
        # Preprocess the data
        processed_df = data_preprocessor.preprocess_data(raw_df)
        
        # Save the processed data
        data_preprocessor.save_preprocessed_data(processed_df, "preprocessed_data.csv")
        
        logger.info("Data pipeline completed successfully")
        return processed_df
    
    except Exception as e:
        logger.error(f"Error in data pipeline: {e}")
        raise CustomException("Data pipeline failed", sys)

def perform_correlation_analysis(df, min_correlation_threshold=0.7):
    """Perform correlation analysis to identify potential pairs."""
    try:
        logger.info("Starting correlation analysis")
        
        # Initialize correlation analysis with configuration
        config = CorrelationAnalysisConfig(min_correlation_threshold=min_correlation_threshold)
        correlation_analyzer = CorrelationAnalysis(config)
        
        # Calculate correlation matrix
        corr_matrix = correlation_analyzer.calculate_correlation_matrix(df)
        
        # Save correlation matrix
        correlation_analyzer.save_correlation_matrix(corr_matrix)
        
        # Identify highly correlated pairs
        correlated_pairs = correlation_analyzer.identify_highly_correlated_pairs(corr_matrix)
        
        # Save correlated pairs
        if correlated_pairs:
            correlation_analyzer.save_correlated_pairs(correlated_pairs)
            
            # Log top pairs
            logger.info(f"Identified {len(correlated_pairs)} highly correlated pairs")
            for i, (ticker1, ticker2, corr) in enumerate(correlated_pairs[:5]):
                logger.info(f"Pair {i+1}: {ticker1}-{ticker2}, Correlation: {corr:.4f}")
        else:
            logger.info("No highly correlated pairs found")
        
        return correlated_pairs
    
    except Exception as e:
        logger.error(f"Error in correlation analysis: {e}")
        raise CustomException("Correlation analysis failed", sys)

def perform_cointegration_tests(df, correlated_pairs, test_method='engle_granger'):
    """Test cointegration between pairs to filter for mean-reverting pairs."""
    try:
        logger.info(f"Starting cointegration tests using {test_method} method")
        
        # Initialize cointegration tests
        config = CointegrationTestsConfig()
        cointegration_tester = CointegrationTests(config)
        
        # Prepare pairs for testing
        test_pairs = [(pair[0], pair[1]) for pair in correlated_pairs]
        
        # Test cointegration for all pairs
        results_df = cointegration_tester.test_multiple_pairs(
            df, test_pairs, test_method=test_method
        )
        
        # Save cointegration results
        cointegration_tester.save_cointegration_results(results_df)
        
        # Filter cointegrated pairs
        cointegrated_pairs = results_df[results_df['IsCointegrated'] == True]
        
        # Log cointegrated pairs
        logger.info(f"Identified {len(cointegrated_pairs)} cointegrated pairs out of {len(test_pairs)} tested")
        
        # Convert to list of tuples (ticker1, ticker2)
        cointegrated_pairs_list = list(zip(cointegrated_pairs['Ticker1'], cointegrated_pairs['Ticker2']))
        
        return cointegrated_pairs_list
    
    except Exception as e:
        logger.error(f"Error in cointegration tests: {e}")
        raise CustomException("Cointegration tests failed", sys)

def run_backtest(df, pairs, strategy_params=None):
    """Backtest the pairs trading strategy for the selected pairs."""
    try:
        logger.info(f"Starting backtest for {len(pairs)} pairs")
        
        # Set default strategy parameters if not provided
        if strategy_params is None:
            strategy_params = {
                'lookback_period': 60,
                'entry_threshold': 2.0,
                'exit_threshold': 0.5,
                'stop_loss': 3.0,
                'max_holding_period': 20,
                'position_size': 0.5
            }
        
        # Initialize backtest engine
        backtest_engine = BacktestEngine(
            initial_capital=100000.0,
            transaction_cost=0.001,
            slippage=0.0005
        )
        
        # Prepare data for backtesting
        backtest_data = backtest_engine.prepare_data(df)
        
        # Run backtest
        results = backtest_engine.run_backtest(
            backtest_data,
            pairs,
            strategy_params,
            start_date=None,  # Use all available data
            end_date=None
        )
        
        # Calculate performance metrics
        metrics_calculator = PerformanceMetrics()
        all_metrics = metrics_calculator.calculate_all_metrics(
            results['portfolio_value'],
            results['portfolio_returns'],
            trades=results['trades']
        )
        
        # Log key metrics
        logger.info(f"Backtest completed with total return: {all_metrics['total_return']:.2%}")
        logger.info(f"Sharpe ratio: {all_metrics['sharpe_ratio']:.2f}")
        logger.info(f"Max drawdown: {all_metrics['max_drawdown']:.2%}")
        
        return results, all_metrics
    
    except Exception as e:
        logger.error(f"Error in backtest: {e}")
        raise CustomException("Backtest failed", sys)

def visualize_results(backtest_results, metrics, save_dir='reports'):
    """Create visualizations for the backtest results."""
    try:
        logger.info("Creating visualizations")
        
        # Initialize visualization tools
        viz_tools = VisualizationTools()
        
        # Create all plots
        plots = viz_tools.create_all_plots(backtest_results, save_dir=save_dir)
        
        # Create performance metrics plot
        viz_tools.plot_performance_metrics(
            metrics,
            title='Pairs Trading Strategy Performance',
            save_path=os.path.join(save_dir, 'performance_metrics.png')
        )
        
        logger.info(f"Created visualization plots in {save_dir}")
        
        return plots
    
    except Exception as e:
        logger.error(f"Error in visualization: {e}")
        raise CustomException("Visualization failed", sys)

def save_results(backtest_results, metrics, save_dir='reports'):
    """Save the backtest results and metrics to files."""
    try:
        logger.info("Saving results")
        
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Save portfolio values
        backtest_results['portfolio_value'].to_csv(
            os.path.join(save_dir, 'portfolio_value.csv')
        )
        
        # Save portfolio returns
        backtest_results['portfolio_returns'].to_csv(
            os.path.join(save_dir, 'portfolio_returns.csv')
        )
        
        # Save trades
        trades_df = pd.DataFrame(backtest_results['trades'])
        if not trades_df.empty:
            trades_df.to_csv(os.path.join(save_dir, 'trades.csv'), index=False)
        
        # Save metrics
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(os.path.join(save_dir, 'performance_metrics.csv'), index=False)
        
        # Create summary report
        with open(os.path.join(save_dir, 'pairs_trading_strategy_report.txt'), 'w') as f:
            f.write("PAIRS TRADING STRATEGY REPORT\n")
            f.write("=============================\n\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("PERFORMANCE SUMMARY\n")
            f.write("-----------------\n")
            f.write(f"Initial Capital: ${backtest_results['portfolio_value'].iloc[0]:.2f}\n")
            f.write(f"Final Portfolio Value: ${backtest_results['portfolio_value'].iloc[-1]:.2f}\n")
            f.write(f"Total Return: {metrics['total_return']:.2%}\n")
            f.write(f"Annualized Return: {metrics.get('annualized_return', 0):.2%}\n")
            f.write(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n")
            f.write(f"Sortino Ratio: {metrics.get('sortino_ratio', 0):.2f}\n")
            f.write(f"Maximum Drawdown: {metrics['max_drawdown']:.2%}\n")
            f.write(f"Win Rate: {metrics.get('win_rate', 0):.2%}\n")
            f.write(f"Profit Factor: {metrics.get('profit_factor', 0):.2f}\n\n")
            
            f.write("TRADING PAIRS\n")
            f.write("------------\n")
            for pair in backtest_results['pair_results'].keys():
                f.write(f"- {pair[0]} / {pair[1]}\n")
            
            f.write("\nSTRATEGY PARAMETERS\n")
            f.write("------------------\n")
            for name, pair_result in backtest_results['pair_results'].items():
                if hasattr(pair_result, 'attrs') and 'strategy_params' in pair_result.attrs:
                    for param, value in pair_result.attrs['strategy_params'].items():
                        f.write(f"{param}: {value}\n")
                    break
        
        logger.info(f"Results saved to {save_dir}")
    
    except Exception as e:
        logger.error(f"Error saving results: {e}")
        raise CustomException("Saving results failed", sys)

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Relative Value Analysis & Pairs Trading")
    
    parser.add_argument('--lookback', type=int, default=60,
                        help='Lookback period for calculating spread statistics')
    parser.add_argument('--entry', type=float, default=2.0,
                        help='Z-score threshold to enter a position')
    parser.add_argument('--exit', type=float, default=0.5,
                        help='Z-score threshold to exit a position')
    parser.add_argument('--stop_loss', type=float, default=3.0,
                        help='Maximum z-score deviation before triggering a stop loss')
    parser.add_argument('--max_holding', type=int, default=20,
                        help='Maximum number of periods to hold a position')
    parser.add_argument('--position_size', type=float, default=0.5,
                        help='Size of position to take (1.0 = 100% of available capital)')
    parser.add_argument('--min_correlation', type=float, default=0.7,
                        help='Minimum correlation threshold for pair selection')
    parser.add_argument('--top_n_pairs', type=int, default=10,
                        help='Number of top pairs to select for testing')
    parser.add_argument('--test_method', type=str, choices=['adf', 'engle_granger', 'johansen'],
                        default='engle_granger', help='Cointegration test method')
    
    return parser.parse_args()

def main():
    """Main function to run the entire pipeline."""
    try:
        # Parse command-line arguments
        args = parse_arguments()
        
        # Create directory structure
        create_directory_structure()
        
        # Run data pipeline
        logger.info("================ DATA PIPELINE ================")
        df = run_data_pipeline()
        
        # Perform correlation analysis
        logger.info("================ CORRELATION ANALYSIS ================")
        correlated_pairs = perform_correlation_analysis(df, args.min_correlation)
        
        # If no pairs at threshold, try with lower threshold
        if not correlated_pairs:
            lower_threshold = args.min_correlation * 0.7  # Try with 70% of original threshold
            logger.warning(f"No pairs found at correlation threshold {args.min_correlation}. Trying with lower threshold {lower_threshold:.2f}")
            correlated_pairs = perform_correlation_analysis(df, lower_threshold)
            
            # If still no pairs, try with even lower threshold
            if not correlated_pairs:
                lowest_threshold = 0.3  # Lowest acceptable threshold
                logger.warning(f"No pairs found at correlation threshold {lower_threshold:.2f}. Trying with minimum threshold {lowest_threshold}")
                correlated_pairs = perform_correlation_analysis(df, lowest_threshold)
        
        # Select top N pairs
        top_pairs = correlated_pairs[:args.top_n_pairs] if correlated_pairs else []
        
        if not top_pairs:
            logger.error("No pairs found for analysis. Exiting.")
            return
        
        # Perform cointegration tests
        logger.info("================ COINTEGRATION TESTS ================")
        cointegrated_pairs = perform_cointegration_tests(df, top_pairs, args.test_method)
        
        # If no cointegrated pairs, use correlated pairs instead
        if not cointegrated_pairs:
            logger.warning("No cointegrated pairs found. Using correlated pairs instead.")
            cointegrated_pairs = [(pair[0], pair[1]) for pair in top_pairs[:3]]  # Use top 3 correlated pairs
            
        if not cointegrated_pairs:
            logger.error("No suitable pairs found for trading. Exiting.")
            return
        
        # Set strategy parameters
        strategy_params = {
            'lookback_period': args.lookback,
            'entry_threshold': args.entry,
            'exit_threshold': args.exit,
            'stop_loss': args.stop_loss,
            'max_holding_period': args.max_holding,
            'position_size': args.position_size
        }
        
        # Run backtest
        logger.info("================ BACKTESTING ================")
        backtest_results, metrics = run_backtest(df, cointegrated_pairs, strategy_params)
        
        # Visualize results
        logger.info("================ VISUALIZATION ================")
        visualize_results(backtest_results, metrics)
        
        # Save results
        logger.info("================ SAVING RESULTS ================")
        save_results(backtest_results, metrics)
        
        logger.info("================ ANALYSIS COMPLETE ================")
        logger.info(f"Final portfolio value: ${backtest_results['portfolio_value'].iloc[-1]:.2f}")
        logger.info(f"Total return: {metrics['total_return']:.2%}")
        logger.info(f"Sharpe ratio: {metrics['sharpe_ratio']:.2f}")
        logger.info(f"Max drawdown: {metrics['max_drawdown']:.2%}")
        
        return backtest_results, metrics
    
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        raise CustomException("Main function failed", sys)

if __name__ == "__main__":
    main()