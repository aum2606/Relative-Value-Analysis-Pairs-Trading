import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union
from src.utils.logger import get_logger
from src.utils.exception import CustomException

# Initialize logger
logger = get_logger(__name__)

@dataclass
class CorrelationAnalysisConfig:
    """
    Configuration class for correlation analysis.
    
    Attributes:
        min_correlation_threshold (float): Minimum correlation threshold for pair selection
        correlation_method (str): Method to use for correlation calculation ('pearson' or 'spearman')
        output_dir (str): Directory to store correlation analysis results
    """
    min_correlation_threshold: float = 0.7
    correlation_method: str = 'pearson'
    output_dir: str = os.path.join('data', 'processed')


class CorrelationAnalysis:
    """
    Class for performing correlation analysis on stock prices
    """
    def __init__(self,config=None):
        """
        Initialize CorrelationAnalysis with configuration.
        
        Args:
            config (CorrelationAnalysisConfig, optional): Configuration for correlation analysis.
                If None, default configuration is used.
        """
        self.config = config if config else CorrelationAnalysisConfig()
        
        #create output directory if it doesn't exists
        os.makedirs(self.config.output_dir,exist_ok=True)
        
    def calculate_correlation_matrix(self,df:pd.DataFrame,
                                     price_column:str = 'Adjusted',
                                     group_column: str = 'Ticker',
                                     date_column:str = 'Date',
                                     method:str = None)->pd.DataFrame:
        """
        Calculate correlation matrix between different tickers based on price or returns.
        
        Args:
            df (pd.DataFrame): DataFrame containing price data
            price_column (str, optional): Column to use for correlation calculation. Defaults to 'Adjusted'.
            group_column (str, optional): Column indicating different stocks/tickers. Defaults to 'Ticker'.
            date_column (str, optional): Column containing dates. Defaults to 'Date'.
            method (str, optional): Correlation method ('pearson' or 'spearman'). 
                                   If None, uses the method from config.
                                   
        Returns:
            pd.DataFrame: Correlation matrix
            
        Raises:
            CustomException: If an error occurs during correlation calculation
        """
        try:
            logger.info(f"Calculating correlation matrix using {price_column} column")
            
            # Set correlation method
            method = method if method else self.config.correlation_method
            logger.info(f"Using {method} correlation method")
            
            # Pivot the data to have tickers as columns and dates as index
            pivot_df = df.pivot(index=date_column, columns=group_column, values=price_column)
            
            # Calculate correlation matrix
            correlation_matrix = pivot_df.corr(method=method)
            
            logger.info(f"Successfully calculated correlation matrix with shape: {correlation_matrix.shape}")
            return correlation_matrix
            
        except Exception as e:
            logger.error(f"Exception occurred during correlation calculation: {str(e)}")
            raise CustomException("Error during correlation calculation", sys)

    def calculate_correlations_from_returns(self, 
                                           df: pd.DataFrame, 
                                           returns_column: str = 'Returns', 
                                           group_column: str = 'Ticker',
                                           date_column: str = 'Date',
                                           method: str = None) -> pd.DataFrame:
        """
        Calculate correlation matrix between different tickers based on returns.
        
        Args:
            df (pd.DataFrame): DataFrame containing returns data
            returns_column (str, optional): Column containing returns. Defaults to 'Returns'.
            group_column (str, optional): Column indicating different stocks/tickers. Defaults to 'Ticker'.
            date_column (str, optional): Column containing dates. Defaults to 'Date'.
            method (str, optional): Correlation method ('pearson' or 'spearman'). 
                                   If None, uses the method from config.
                                   
        Returns:
            pd.DataFrame: Correlation matrix
            
        Raises:
            CustomException: If an error occurs during correlation calculation
        """
        try:
            logger.info(f"Calculating correlation matrix using {returns_column} column")
            
            # Set correlation method
            method = method if method else self.config.correlation_method
            logger.info(f"Using {method} correlation method")
            
            # Pivot the data to have tickers as columns and dates as index
            pivot_df = df.pivot(index=date_column, columns=group_column, values=returns_column)
            
            # Calculate correlation matrix
            correlation_matrix = pivot_df.corr(method=method)
            
            logger.info(f"Successfully calculated correlation matrix with shape: {correlation_matrix.shape}")
            return correlation_matrix
            
        except Exception as e:
            logger.error(f"Exception occurred during correlation calculation: {str(e)}")
            raise CustomException("Error during correlation calculation", sys)

    def identify_highly_correlated_pairs(self, 
                                correlation_matrix: pd.DataFrame, 
                                threshold: float = None) -> List[Tuple[str, str, float]]:
        """
        Identify pairs of tickers with correlation above a threshold.
        
        Args:
            correlation_matrix (pd.DataFrame): Correlation matrix between tickers
            threshold (float, optional): Minimum correlation threshold. 
                                        If None, uses the threshold from config.
                                
        Returns:
            List[Tuple[str, str, float]]: List of ticker pairs with their correlation values
            
        Raises:
            CustomException: If an error occurs during pair identification
        """
        try:
            # Set threshold
            threshold = threshold if threshold is not None else self.config.min_correlation_threshold
            logger.info(f"Indentifying ticker pairs with correlation above {threshold}")
            
            # Get the ticker pairs with correlation above threshold
            correlation_pairs = []
            
            # Get the correlation values from the upper triangle
            for i, ticker1 in enumerate(correlation_matrix.index):
                for j, ticker2 in enumerate(correlation_matrix.columns):
                    if j > i:  # Upper triangle only
                        correlation = correlation_matrix.iloc[i, j]
                        if correlation > threshold:
                            correlation_pairs.append((ticker1, ticker2, correlation))
            
            # Sort by correlation (highest first)
            correlation_pairs.sort(key=lambda x: x[2], reverse=True)
            
            logger.info(f"Identified {len(correlation_pairs)} ticker pairs with correlation above {threshold}")
            return correlation_pairs
            
        except Exception as e:
            logger.error(f"Exception occurred during pair identification: {str(e)}")
            raise CustomException("Error during pair identification", sys)
        
    def calculate_rolling_correlation(self, 
                                     df: pd.DataFrame, 
                                     ticker1: str, 
                                     ticker2: str,
                                     window: int = 30,
                                     price_column: str = 'Adjusted',
                                     group_column: str = 'Ticker',
                                     date_column: str = 'Date',
                                     method: str = None) -> pd.DataFrame:
        """
        Calculate rolling correlation between two tickers.
        
        Args:
            df (pd.DataFrame): DataFrame containing price data
            ticker1 (str): First ticker symbol
            ticker2 (str): Second ticker symbol
            window (int, optional): Rolling window size. Defaults to 30.
            price_column (str, optional): Column to use for correlation calculation. Defaults to 'Adjusted'.
            group_column (str, optional): Column indicating different stocks/tickers. Defaults to 'Ticker'.
            date_column (str, optional): Column containing dates. Defaults to 'Date'.
            method (str, optional): Correlation method ('pearson' or 'spearman'). 
                                   If None, uses the method from config.
                                   
        Returns:
            pd.DataFrame: DataFrame with dates and rolling correlation values
            
        Raises:
            CustomException: If an error occurs during rolling correlation calculation
        """
        try:
            logger.info(f"Calculating rolling correlation between {ticker1} and {ticker2} with window={window}")

            #set correlation method
            method = method if method else self.config.correlation_method
            
            #filter data for the two tickers
            df_filtered = df[df[group_column].isin([ticker1,ticker2])]
            
            #pivot the data to have tickers as columns and dates as index
            pivot_df = df_filtered.pivot(index=date_column, columns=group_column, values=price_column)

            #ensure both tickers are in the data
            if ticker1 not in pivot_df.columns or ticker2 not in pivot_df.columns:
                raise ValueError(f"One or both tickers ({ticker1}, {ticker2}) not found in the data")

            # Calculate rolling correlation
            rolling_corr = pivot_df[ticker1].rolling(window=window).corr(pivot_df[ticker2], method=method)
            
            # Create a DataFrame with dates and correlation values
            rolling_corr_df = pd.DataFrame({
                'Date': pivot_df.index,
                'RollingCorrelation': rolling_corr
            })
            
            logger.info(f"Successfully calculated rolling correlation between {ticker1} and {ticker2}")
            return rolling_corr_df
            
        except Exception as e:
            logger.error(f"Exception occurred during rolling correlation calculation: {str(e)}")
            raise CustomException("Error during rolling correlation calculation", sys)
    
    def save_correlation_matrix(self, correlation_matrix: pd.DataFrame, filename: str = "correlation_matrix.csv"):
        """
        Save correlation matrix to file.
        
        Args:
            correlation_matrix (pd.DataFrame): Correlation matrix to save
            filename (str, optional): Name of the file to save. Defaults to "correlation_matrix.csv".
            
        Returns:
            str: Path to the saved file
            
        Raises:
            CustomException: If an error occurs while saving data
        """
        try:
            file_path = os.path.join(self.config.output_dir, filename)
            logger.info(f"Saving correlation matrix to {file_path}")
            
            # Save the correlation matrix
            correlation_matrix.to_csv(file_path)
            
            logger.info(f"Correlation matrix successfully saved to {file_path}")
            return file_path
            
        except Exception as e:
            logger.error("Exception occurred while saving correlation matrix")
            raise CustomException("Error while saving correlation matrix", sys)
    
    def save_correlated_pairs(self, correlation_pairs: List[Tuple[str, str, float]], 
                             filename: str = "correlated_pairs.csv"):
        """
        Save correlated pairs to file.
        
        Args:
            correlation_pairs (List[Tuple[str, str, float]]): List of ticker pairs with correlation values
            filename (str, optional): Name of the file to save. Defaults to "correlated_pairs.csv".
            
        Returns:
            str: Path to the saved file
            
        Raises:
            CustomException: If an error occurs while saving data
        """
        try:
            file_path = os.path.join(self.config.output_dir, filename)
            logger.info(f"Saving correlated pairs to {file_path}")
            
            # Convert list of tuples to DataFrame
            pairs_df = pd.DataFrame(correlation_pairs, columns=['Ticker1', 'Ticker2', 'Correlation'])
            
            # Save the DataFrame
            pairs_df.to_csv(file_path, index=False)
            
            logger.info(f"Correlated pairs successfully saved to {file_path}")
            return file_path
            
        except Exception as e:
            logger.error("Exception occurred while saving correlated pairs")
            raise CustomException("Error while saving correlated pairs", sys)

# Example usage
if __name__ == "__main__":
    try:
        logger.info("Correlation analysis process started")
        
        # Load sample data
        from src.data.data_ingestion import DataIngestion
        from src.data.data_preprocessing import DataPreprocessing
        
        # Get and preprocess the data
        data_ingestion = DataIngestion()
        raw_df = data_ingestion.read_data()
        
        data_preprocessor = DataPreprocessing()
        processed_df = data_preprocessor.preprocess_data(raw_df)
        
        # Initialize correlation analysis
        correlation_analyzer = CorrelationAnalysis()
        
        # Calculate correlation matrix
        corr_matrix = correlation_analyzer.calculate_correlation_matrix(processed_df)
        
        # Print correlation matrix summary
        logger.info(f"Correlation matrix shape: {corr_matrix.shape}")
        logger.info(f"Sample of correlation matrix:\n{corr_matrix.iloc[:5, :5]}")
        
        # Identify highly correlated pairs
        correlated_pairs = correlation_analyzer.identify_highly_correlated_pairs(corr_matrix)
        
        # Print top correlated pairs
        if correlated_pairs:
            logger.info(f"Top correlated pairs:")
            for ticker1, ticker2, corr in correlated_pairs[:5]:
                logger.info(f"{ticker1} - {ticker2}: {corr:.4f}")
        else:
            logger.info("No pairs found above the correlation threshold")
        
        # Calculate rolling correlation for the first pair (if available)
        if correlated_pairs:
            ticker1, ticker2, _ = correlated_pairs[0]
            rolling_corr_df = correlation_analyzer.calculate_rolling_correlation(
                processed_df, ticker1, ticker2, window=30
            )
            logger.info(f"Sample of rolling correlation:\n{rolling_corr_df.head()}")
        
        # Save results
        saved_matrix_path = correlation_analyzer.save_correlation_matrix(corr_matrix)
        if correlated_pairs:
            saved_pairs_path = correlation_analyzer.save_correlated_pairs(correlated_pairs)
        
        logger.info("Correlation analysis completed successfully")
        
    except Exception as e:
        logger.error("Error in correlation analysis process")
        raise CustomException("Correlation analysis process failed", sys)
