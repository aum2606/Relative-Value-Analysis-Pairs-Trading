import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass
from src.utils.logger import get_logger
from src.utils.exception import CustomException
from src.data.data_ingestion import DataIngestion

# Initialize logger
logger = get_logger(__name__)

@dataclass
class DataPreprocessingConfig:
    """
    Configuration class for data preprocessing.
    
    Attributes:
        processed_data_path (str): Path to processed data
    """
    processed_data_path: str = os.path.join('data', 'processed', 'price_data.csv')

class DataPreprocessing:
    """
    Class for handling data preprocessing operations.
    """
    
    def __init__(self, config=None):
        """
        Initialize DataPreprocessing with configuration.
        
        Args:
            config (DataPreprocessingConfig, optional): Configuration for data preprocessing.
                If None, default configuration is used.
        """
        self.config = config if config else DataPreprocessingConfig()
    
    def load_data(self, file_path=None):
        """
        Load data for preprocessing.
        
        Args:
            file_path (str, optional): Path to the data file.
                If None, the path from the configuration is used.
                
        Returns:
            pd.DataFrame: Loaded DataFrame
            
        Raises:
            CustomException: If an error occurs during data loading
        """
        try:
            data_path = file_path if file_path else self.config.processed_data_path
            logger.info(f"Loading data from {data_path}")
            
            # Read the data
            df = pd.read_csv(data_path)
            
            logger.info(f"Successfully loaded data with shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.error("Exception occurred during data loading")
            raise CustomException("Error during data loading", sys)
    
    def convert_date_format(self, df, date_column='Date'):
        """
        Convert date column to datetime format.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            date_column (str, optional): Name of the date column. Defaults to 'Date'.
            
        Returns:
            pd.DataFrame: DataFrame with converted date format
            
        Raises:
            CustomException: If an error occurs during date conversion
        """
        try:
            logger.info(f"Converting {date_column} to datetime format")
            
            # Convert date column to datetime
            df[date_column] = pd.to_datetime(df[date_column])
            
            logger.info(f"Successfully converted {date_column} to datetime format")
            return df
            
        except Exception as e:
            logger.error(f"Exception occurred during date conversion for column {date_column}")
            raise CustomException(f"Error during date conversion for column {date_column}", sys)
    
    def handle_missing_values(self, df):
        """
        Handle missing values in the DataFrame.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with handled missing values
            
        Raises:
            CustomException: If an error occurs during handling missing values
        """
        try:
            logger.info("Handling missing values")
            
            # Check for missing values
            missing_values = df.isnull().sum()
            logger.info(f"Missing values before handling: {missing_values}")
            
            # Forward fill for numerical data (price, volume)
            numeric_cols = df.select_dtypes(include=['number']).columns
            df[numeric_cols] = df[numeric_cols].fillna(method='ffill')
            
            # Check remaining missing values
            remaining_missing = df.isnull().sum()
            logger.info(f"Missing values after handling: {remaining_missing}")
            
            # Drop any remaining rows with missing values
            if df.isnull().any().any():
                df = df.dropna()
                logger.info(f"Dropped remaining rows with missing values. New shape: {df.shape}")
            
            return df
            
        except Exception as e:
            logger.error("Exception occurred during handling missing values")
            raise CustomException("Error during handling missing values", sys)
    
    def calculate_returns(self, df, price_column='Adjusted', group_column='Ticker'):
        """
        Calculate returns if they are not already in the dataset.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            price_column (str, optional): Name of the price column to use. Defaults to 'Adjusted'.
            group_column (str, optional): Column to group by before calculating returns. Defaults to 'Ticker'.
            
        Returns:
            pd.DataFrame: DataFrame with calculated returns
            
        Raises:
            CustomException: If an error occurs during returns calculation
        """
        try:
            # Check if Returns column already exists
            if 'Returns' in df.columns:
                logger.info("Returns column already exists, skipping calculation")
                return df
            
            logger.info(f"Calculating returns based on {price_column}")
            
            # Sort by ticker and date
            if 'Date' in df.columns:
                df = df.sort_values(by=[group_column, 'Date'])
            
            # Calculate returns for each ticker separately
            df['Returns'] = df.groupby(group_column)[price_column].pct_change()
            
            logger.info("Successfully calculated returns")
            return df
            
        except Exception as e:
            logger.error("Exception occurred during returns calculation")
            raise CustomException("Error during returns calculation", sys)
    
    def preprocess_data(self, df=None):
        """
        Perform complete preprocessing on the data.
        
        Args:
            df (pd.DataFrame, optional): Input DataFrame.
                If None, the data is loaded using the load_data method.
                
        Returns:
            pd.DataFrame: Preprocessed DataFrame
            
        Raises:
            CustomException: If an error occurs during preprocessing
        """
        try:
            logger.info("Starting data preprocessing")
            
            # Load data if not provided
            if df is None:
                df = self.load_data()
            
            # Preprocessing steps
            df = self.convert_date_format(df)
            df = self.handle_missing_values(df)
            df = self.calculate_returns(df)
            
            logger.info("Data preprocessing completed successfully")
            return df
            
        except Exception as e:
            logger.error("Exception occurred during data preprocessing")
            raise CustomException("Error during data preprocessing", sys)
    
    def save_preprocessed_data(self, df, filename="preprocessed_data.csv"):
        """
        Save the preprocessed DataFrame.
        
        Args:
            df (pd.DataFrame): Preprocessed DataFrame to save
            filename (str, optional): Name of the file to save. Defaults to "preprocessed_data.csv".
            
        Returns:
            str: Path to the saved file
            
        Raises:
            CustomException: If an error occurs while saving data
        """
        try:
            save_path = os.path.join('data', 'processed', filename)
            logger.info(f"Saving preprocessed data to {save_path}")
            
            # Save the DataFrame
            df.to_csv(save_path, index=False)
            
            logger.info(f"Preprocessed data successfully saved to {save_path}")
            return save_path
            
        except Exception as e:
            logger.error("Exception occurred while saving preprocessed data")
            raise CustomException("Error while saving preprocessed data", sys)

# Example usage
if __name__ == "__main__":
    try:
        logger.info("Data preprocessing process started")
        
        # Initialize data preprocessor
        data_preprocessor = DataPreprocessing()
        
        # Option 1: Use data directly from DataIngestion
        data_ingestion = DataIngestion()
        raw_df = data_ingestion.read_data()
        
        # Process the data
        processed_df = data_preprocessor.preprocess_data(raw_df)
        
        # Print processed data summary
        logger.info(f"Processed data summary: {processed_df.describe().to_string()}")
        logger.info(f"Processed data columns: {processed_df.columns.tolist()}")
        logger.info(f"First few rows after preprocessing: \n{processed_df.head().to_string()}")
        
        # Save the processed data
        saved_path = data_preprocessor.save_preprocessed_data(processed_df)
        logger.info(f"Data preprocessing completed. Data saved to {saved_path}")
        
    except Exception as e:
        logger.error("Error in data preprocessing process")
        raise CustomException("Data preprocessing process failed", sys)