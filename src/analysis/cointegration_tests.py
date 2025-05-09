import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union, Optional
from statsmodels.tsa.stattools import adfuller, coint
# Removed johansen import since it's not available
from statsmodels.regression.linear_model import OLS
from src.utils.logger import get_logger
from src.utils.exception import CustomException

# Initialize logger
logger = get_logger(__name__)

@dataclass
class CointegrationTestsConfig:
    """
    Configuration class for cointegration tests.
    
    Attributes:
        adf_significance_level (float): Significance level for ADF test
        eg_significance_level (float): Significance level for Engle-Granger test
        johansen_significance_level (float): Significance level for Johansen test
        output_dir (str): Directory to store cointegration test results
    """
    adf_significance_level: float = 0.05
    eg_significance_level: float = 0.05
    johansen_significance_level: float = 0.05
    output_dir: str = os.path.join('data', 'processed')

class CointegrationTests:
    """
    Class for performing cointegration tests on stock price pairs.
    """
    
    def __init__(self, config=None):
        """
        Initialize CointegrationTests with configuration.
        
        Args:
            config (CointegrationTestsConfig, optional): Configuration for cointegration tests.
                If None, default configuration is used.
        """
        self.config = config if config else CointegrationTestsConfig()
        
        # Create output directory if it doesn't exist
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Check if Johansen test is available
        self.johansen_available = False
        try:
            # Try to import johansen from statsmodels.tsa.vector_ar.johansen
            from statsmodels.tsa.vector_ar.johansen import johansen
            self.johansen = johansen
            self.johansen_available = True
            logger.info("Johansen test is available")
        except ImportError:
            logger.warning("Johansen test is not available in your statsmodels installation. "
                         "Only ADF and Engle-Granger tests will be available.")
    
    def adf_test(self, series: pd.Series, significance_level: float = None) -> Tuple[bool, Dict]:
        """
        Perform Augmented Dickey-Fuller test for stationarity.
        
        Args:
            series (pd.Series): Time series data to test
            significance_level (float, optional): Significance level for the test.
                                                If None, uses the level from config.
                                   
        Returns:
            Tuple[bool, Dict]: Result (True if stationary) and test statistics
            
        Raises:
            CustomException: If an error occurs during the test
        """
        try:
            # Set significance level
            significance_level = significance_level if significance_level is not None else self.config.adf_significance_level
            logger.info(f"Performing ADF test with significance level {significance_level}")
            
            # Perform ADF test
            result = adfuller(series.dropna())
            
            # Extract statistics
            adf_statistic, p_value, _, _, critical_values, _ = result
            
            # Determine if series is stationary
            is_stationary = p_value < significance_level
            
            # Create results dictionary
            test_results = {
                'adf_statistic': adf_statistic,
                'p_value': p_value,
                'critical_values': critical_values,
                'is_stationary': is_stationary
            }
            
            logger.info(f"ADF test result: Stationary = {is_stationary}, p-value = {p_value:.4f}")
            return is_stationary, test_results
            
        except Exception as e:
            logger.error(f"Exception occurred during ADF test: {str(e)}")
            raise CustomException("Error during ADF test", sys)
    
    def engle_granger_test(self, 
                          y: pd.Series, 
                          x: pd.Series, 
                          significance_level: float = None) -> Tuple[bool, Dict]:
        """
        Perform Engle-Granger two-step cointegration test.
        
        Args:
            y (pd.Series): Dependent variable time series
            x (pd.Series): Independent variable time series
            significance_level (float, optional): Significance level for the test.
                                                If None, uses the level from config.
                                   
        Returns:
            Tuple[bool, Dict]: Result (True if cointegrated) and test statistics
            
        Raises:
            CustomException: If an error occurs during the test
        """
        try:
            # Set significance level
            significance_level = significance_level if significance_level is not None else self.config.eg_significance_level
            logger.info(f"Performing Engle-Granger test with significance level {significance_level}")
            
            # Step 1: Cointegrating regression
            # Add constant to x
            x_with_const = pd.Series(np.ones(len(x)), index=x.index)
            x_with_const = pd.concat([x_with_const, x], axis=1)
            
            # OLS regression
            model = OLS(y, x_with_const)
            model_fitted = model.fit()
            
            # Get residuals
            residuals = model_fitted.resid
            
            # Step 2: Test for stationarity of residuals
            is_stationary, adf_results = self.adf_test(residuals, significance_level)
            
            # Create results dictionary
            test_results = {
                'cointegration_coefficient': model_fitted.params[1],
                'cointegration_constant': model_fitted.params[0],
                'adf_statistic': adf_results['adf_statistic'],
                'p_value': adf_results['p_value'],
                'critical_values': adf_results['critical_values'],
                'is_cointegrated': is_stationary
            }
            
            logger.info(f"Engle-Granger test result: Cointegrated = {is_stationary}, p-value = {adf_results['p_value']:.4f}")
            return is_stationary, test_results
            
        except Exception as e:
            logger.error(f"Exception occurred during Engle-Granger test: {str(e)}")
            raise CustomException("Error during Engle-Granger test", sys)
    
    def simplified_engle_granger_test(self, 
                                     y: pd.Series, 
                                     x: pd.Series, 
                                     significance_level: float = None) -> Tuple[bool, Dict]:
        """
        Simplified version of Engle-Granger test using statsmodels coint function.
        
        Args:
            y (pd.Series): Dependent variable time series
            x (pd.Series): Independent variable time series
            significance_level (float, optional): Significance level for the test.
                                                If None, uses the level from config.
                                   
        Returns:
            Tuple[bool, Dict]: Result (True if cointegrated) and test statistics
            
        Raises:
            CustomException: If an error occurs during the test
        """
        try:
            # Set significance level
            significance_level = significance_level if significance_level is not None else self.config.eg_significance_level
            logger.info(f"Performing simplified Engle-Granger test with significance level {significance_level}")
            
            # Perform cointegration test
            t_stat, p_value, critical_values = coint(y, x)
            
            # Determine if series are cointegrated
            is_cointegrated = p_value < significance_level
            
            # Create results dictionary
            test_results = {
                't_statistic': t_stat,
                'p_value': p_value,
                'critical_values': critical_values,
                'is_cointegrated': is_cointegrated
            }
            
            logger.info(f"Simplified Engle-Granger test result: Cointegrated = {is_cointegrated}, p-value = {p_value:.4f}")
            return is_cointegrated, test_results
            
        except Exception as e:
            logger.error(f"Exception occurred during simplified Engle-Granger test: {str(e)}")
            raise CustomException("Error during simplified Engle-Granger test", sys)
    
    def johansen_test(self, 
                     df: pd.DataFrame, 
                     significance_level: float = None,
                     det_order: int = 0, 
                     k_ar_diff: int = 1) -> Tuple[bool, Dict]:
        """
        Perform Johansen cointegration test.
        
        Args:
            df (pd.DataFrame): DataFrame with time series data (each column is a series)
            significance_level (float, optional): Significance level for the test.
                                                If None, uses the level from config.
            det_order (int, optional): Deterministic term. Defaults to 0 (constant).
            k_ar_diff (int, optional): Number of lagged differences. Defaults to 1.
                                   
        Returns:
            Tuple[bool, Dict]: Result (True if cointegrated) and test statistics
            
        Raises:
            CustomException: If an error occurs during the test
        """
        if not self.johansen_available:
            logger.warning("Johansen test is not available. Using simplified Engle-Granger test instead.")
            # If we have exactly 2 series, use the simplified Engle-Granger test
            if df.shape[1] == 2:
                col1, col2 = df.columns
                return self.simplified_engle_granger_test(df[col1], df[col2], significance_level)
            else:
                raise NotImplementedError("Johansen test is not available and alternative test requires exactly 2 series")
        
        try:
            # Set significance level
            significance_level = significance_level if significance_level is not None else self.config.johansen_significance_level
            logger.info(f"Performing Johansen test with significance level {significance_level}")
            
            # Number of series
            n = df.shape[1]
            
            # Perform Johansen test
            result = self.johansen(df.values, det_order=det_order, k_ar_diff=k_ar_diff)
            
            # Extract statistics
            lr1 = result[0]  # Trace statistic
            lr2 = result[1]  # Maximum eigenvalue statistic
            cvt = result[2]  # Critical values (trace)
            cvm = result[3]  # Critical values (max eigenvalue)
            
            # Determine if series are cointegrated and the number of cointegrating vectors
            trace_numvec = self._count_significant_rank(lr1, cvt, significance_level)
            max_numvec = self._count_significant_rank(lr2, cvm, significance_level)
            
            # Create results dictionary
            test_results = {
                'trace_statistic': lr1,
                'max_eigenvalue_statistic': lr2,
                'trace_critical_values': cvt,
                'max_eigenvalue_critical_values': cvm,
                'trace_num_cointegrating_vectors': trace_numvec,
                'max_eigenvalue_num_cointegrating_vectors': max_numvec,
                'is_cointegrated': (trace_numvec > 0 or max_numvec > 0),
                'num_cointegrating_vectors': max(trace_numvec, max_numvec)
            }
            
            logger.info(f"Johansen test result: Cointegrated = {test_results['is_cointegrated']}, " +
                       f"Num. cointegrating vectors = {test_results['num_cointegrating_vectors']}")
            return test_results['is_cointegrated'], test_results
            
        except Exception as e:
            logger.error(f"Exception occurred during Johansen test: {str(e)}")
            raise CustomException("Error during Johansen test", sys)
    
    def _count_significant_rank(self, statistics, critical_values, significance_level):
        """
        Helper function to count number of cointegrating vectors.
        
        Args:
            statistics (np.ndarray): Test statistics
            critical_values (np.ndarray): Critical values for different significance levels
            significance_level (float): Significance level
            
        Returns:
            int: Number of cointegrating vectors
        """
        # Get critical values for the given significance level
        # Critical values are typically given for 10%, 5%, and 1% significance levels
        # Choose the column that corresponds to the desired significance level
        if significance_level <= 0.01:
            cv_idx = 2  # 1% critical values
        elif significance_level <= 0.05:
            cv_idx = 1  # 5% critical values
        else:
            cv_idx = 0  # 10% critical values
        
        # Count number of statistics that exceed critical values
        count = 0
        for i, (stat, cv) in enumerate(zip(statistics, critical_values[:, cv_idx])):
            if stat > cv:
                count += 1
            else:
                break
        
        return count
    
    def test_stock_pair_cointegration(self, 
                                     df: pd.DataFrame, 
                                     ticker1: str, 
                                     ticker2: str,
                                     price_column: str = 'Adjusted',
                                     group_column: str = 'Ticker',
                                     date_column: str = 'Date',
                                     test_method: str = 'engle_granger') -> Tuple[bool, Dict]:
        """
        Test cointegration between two stocks.
        
        Args:
            df (pd.DataFrame): DataFrame containing price data
            ticker1 (str): First ticker symbol
            ticker2 (str): Second ticker symbol
            price_column (str, optional): Column to use for cointegration test. Defaults to 'Adjusted'.
            group_column (str, optional): Column indicating different stocks/tickers. Defaults to 'Ticker'.
            date_column (str, optional): Column containing dates. Defaults to 'Date'.
            test_method (str, optional): Cointegration test method. 
                                       Options: 'adf', 'engle_granger', 'johansen'.
                                       Defaults to 'engle_granger'.
                                   
        Returns:
            Tuple[bool, Dict]: Result (True if cointegrated) and test statistics
            
        Raises:
            CustomException: If an error occurs during the test
        """
        try:
            logger.info(f"Testing cointegration between {ticker1} and {ticker2} using {test_method} method")
            
            # Check if test method is johansen but not available
            if test_method == 'johansen' and not self.johansen_available:
                logger.warning("Johansen test not available. Falling back to Engle-Granger test.")
                test_method = 'engle_granger'
            
            # Filter data for the two tickers
            df_filtered = df[df[group_column].isin([ticker1, ticker2])]
            
            # Pivot the data to have tickers as columns and dates as index
            pivot_df = df_filtered.pivot(index=date_column, columns=group_column, values=price_column)
            
            # Ensure both tickers are in the data
            if ticker1 not in pivot_df.columns or ticker2 not in pivot_df.columns:
                raise ValueError(f"One or both tickers ({ticker1}, {ticker2}) not found in the data")
            
            # Get price series
            price1 = pivot_df[ticker1]
            price2 = pivot_df[ticker2]
            
            # Perform cointegration test based on method
            if test_method == 'adf':
                # Calculate spread
                spread = price1 - price2
                return self.adf_test(spread)
            elif test_method == 'engle_granger':
                return self.simplified_engle_granger_test(price1, price2)
            elif test_method == 'johansen':
                # For Johansen test, use both series
                test_df = pivot_df[[ticker1, ticker2]].dropna()
                return self.johansen_test(test_df)
            else:
                raise ValueError(f"Invalid test method: {test_method}")
            
        except Exception as e:
            logger.error(f"Exception occurred during stock pair cointegration test: {str(e)}")
            raise CustomException("Error during stock pair cointegration test", sys)
    
    def test_multiple_pairs(self, 
                           df: pd.DataFrame, 
                           pairs: List[Tuple[str, str]],
                           price_column: str = 'Adjusted',
                           group_column: str = 'Ticker',
                           date_column: str = 'Date',
                           test_method: str = 'engle_granger') -> pd.DataFrame:
        """
        Test cointegration for multiple stock pairs.
        
        Args:
            df (pd.DataFrame): DataFrame containing price data
            pairs (List[Tuple[str, str]]): List of ticker pairs to test
            price_column (str, optional): Column to use for cointegration test. Defaults to 'Adjusted'.
            group_column (str, optional): Column indicating different stocks/tickers. Defaults to 'Ticker'.
            date_column (str, optional): Column containing dates. Defaults to 'Date'.
            test_method (str, optional): Cointegration test method. Defaults to 'engle_granger'.
                                   
        Returns:
            pd.DataFrame: DataFrame with results for each pair
            
        Raises:
            CustomException: If an error occurs during testing
        """
        try:
            logger.info(f"Testing cointegration for {len(pairs)} pairs using {test_method} method")
            
            # Check if test method is johansen but not available
            if test_method == 'johansen' and not self.johansen_available:
                logger.warning("Johansen test not available. Falling back to Engle-Granger test.")
                test_method = 'engle_granger'
            
            # Initialize list to store results
            results = []
            
            # Test each pair
            for ticker1, ticker2 in pairs:
                try:
                    is_cointegrated, test_stats = self.test_stock_pair_cointegration(
                        df, ticker1, ticker2, price_column, group_column, date_column, test_method
                    )
                    
                    # Create result entry
                    result = {
                        'Ticker1': ticker1,
                        'Ticker2': ticker2,
                        'IsCointegrated': is_cointegrated,
                        'TestMethod': test_method,
                        'PValue': test_stats.get('p_value', None)
                    }
                    
                    # Add other statistics based on test method
                    if test_method == 'adf':
                        result['ADFStatistic'] = test_stats.get('adf_statistic', None)
                    elif test_method == 'engle_granger':
                        result['TStatistic'] = test_stats.get('t_statistic', None)
                    elif test_method == 'johansen':
                        result['NumCointegratingVectors'] = test_stats.get('num_cointegrating_vectors', None)
                    
                    results.append(result)
                    
                except Exception as pair_error:
                    logger.error(f"Error testing pair ({ticker1}, {ticker2}): {str(pair_error)}")
                    results.append({
                        'Ticker1': ticker1,
                        'Ticker2': ticker2,
                        'Error': str(pair_error),
                        'IsCointegrated': False,
                        'TestMethod': test_method
                    })
            
            # Convert results to DataFrame
            results_df = pd.DataFrame(results)
            
            logger.info(f"Completed cointegration tests for {len(pairs)} pairs")
            logger.info(f"Number of cointegrated pairs: {results_df['IsCointegrated'].sum()}")
            
            return results_df
            
        except Exception as e:
            logger.error(f"Exception occurred during multiple pairs testing: {str(e)}")
            raise CustomException("Error during multiple pairs testing", sys)
    
    def save_cointegration_results(self, results_df: pd.DataFrame, 
                                 filename: str = "cointegration_results.csv"):
        """
        Save cointegration test results to file.
        
        Args:
            results_df (pd.DataFrame): DataFrame with cointegration test results
            filename (str, optional): Name of the file to save. Defaults to "cointegration_results.csv".
            
        Returns:
            str: Path to the saved file
            
        Raises:
            CustomException: If an error occurs while saving data
        """
        try:
            file_path = os.path.join(self.config.output_dir, filename)
            logger.info(f"Saving cointegration test results to {file_path}")
            
            # Save the DataFrame
            results_df.to_csv(file_path, index=False)
            
            logger.info(f"Cointegration test results successfully saved to {file_path}")
            return file_path
            
        except Exception as e:
            logger.error("Exception occurred while saving cointegration test results")
            raise CustomException("Error while saving cointegration test results", sys)

# Example usage
if __name__ == "__main__":
    try:
        logger.info("Cointegration testing process started")
        
        # Load sample data
        from src.data.data_ingestion import DataIngestion
        from src.data.data_preprocessing import DataPreprocessing
        from src.analysis.correlation_analysis import CorrelationAnalysis
        
        # Get and preprocess the data
        data_ingestion = DataIngestion()
        raw_df = data_ingestion.read_data()
        
        data_preprocessor = DataPreprocessing()
        processed_df = data_preprocessor.preprocess_data(raw_df)
        
        # First get highly correlated pairs
        correlation_analyzer = CorrelationAnalysis()
        corr_matrix = correlation_analyzer.calculate_correlation_matrix(processed_df)
        correlated_pairs = correlation_analyzer.identify_highly_correlated_pairs(corr_matrix)
        
        if correlated_pairs:
            # Get pairs for testing (top 5 or all if less than 5)
            test_pairs = [(pair[0], pair[1]) for pair in correlated_pairs[:min(5, len(correlated_pairs))]]
            
            # Initialize cointegration tests
            cointegration_tester = CointegrationTests()
            
            # Test the first pair individually
            ticker1, ticker2, _ = correlated_pairs[0]
            is_cointegrated, test_stats = cointegration_tester.test_stock_pair_cointegration(
                processed_df, ticker1, ticker2, test_method='engle_granger'
            )
            
            logger.info(f"Cointegration test result for {ticker1}-{ticker2}: {is_cointegrated}")
            logger.info(f"Test statistics: {test_stats}")
            
            # Test multiple pairs
            results_df = cointegration_tester.test_multiple_pairs(
                processed_df, test_pairs, test_method='engle_granger'
            )
            
            # Print results
            logger.info(f"Cointegration test results:\n{results_df}")
            
            # Save results
            saved_path = cointegration_tester.save_cointegration_results(results_df)
            
            logger.info(f"Cointegration testing completed. Results saved to {saved_path}")
        else:
            logger.info("No correlated pairs to test")
        
    except Exception as e:
        logger.error("Error in cointegration testing process")
        raise CustomException("Cointegration testing process failed", sys)