import os
import sys
import pandas as pd
from dataclasses import dataclass
from src.utils.logger import get_logger
from src.utils.exception import CustomException

# Initialize logger
logger = get_logger(__name__)

@dataclass
class DataIngestionConfig:
    """
    Configuration class for data ingestion.
    
    Attributes:
        raw_data_path (str): Path to raw data
        processed_data_dir (str): Directory to store processed data
    """
    # Find the project root directory
    _notebook_dir = os.path.join(os.getcwd(), 'notebooks')
    project_root = os.getcwd() if not os.path.exists(_notebook_dir) else os.path.dirname(os.getcwd())
    
    # Define paths relative to project root
    raw_data_path: str = os.path.join(project_root, 'data', 'raw', 'Portfolio_prices.csv')
    processed_data_dir: str = os.path.join(project_root, 'data', 'processed')

class DataIngestion:
    """
    Class for handling data ingestion operations.
    """
    
    def __init__(self, config=None):
        """
        Initialize DataIngestion with configuration.
        
        Args:
            config (DataIngestionConfig, optional): Configuration for data ingestion.
                If None, default configuration is used.
        """
        self.config = config if config else DataIngestionConfig()
        
        # Create processed data directory if it doesn't exist
        os.makedirs(self.config.processed_data_dir, exist_ok=True)
        
        # Print path for debugging
        logger.info(f"Raw data path: {self.config.raw_data_path}")
        logger.info(f"File exists: {os.path.exists(self.config.raw_data_path)}")
    
    def read_data(self):
        """
        Read data from the specified raw data path.
        
        Returns:
            pd.DataFrame: DataFrame containing the portfolio prices data
            
        Raises:
            CustomException: If an error occurs during data ingestion
        """
        try:
            logger.info(f"Started reading data from {self.config.raw_data_path}")
            
            # Check if file exists
            if not os.path.exists(self.config.raw_data_path):
                # Try a direct path relative to current working directory
                direct_path = os.path.join('data', 'raw', 'Portfolio_prices.csv')
                if os.path.exists(direct_path):
                    logger.info(f"Using alternative path: {direct_path}")
                    df = pd.read_csv(direct_path)
                else:
                    raise FileNotFoundError(f"Could not find CSV file at {self.config.raw_data_path} or {direct_path}")
            else:
                # Read the CSV file
                df = pd.read_csv(self.config.raw_data_path)
            
            logger.info(f"Successfully read data with shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Exception occurred during data ingestion: {str(e)}")
            raise CustomException("Error during data ingestion", sys)
    
    def save_data(self, df, filename="price_data.csv"):
        """
        Save processed DataFrame to the processed data directory.
        
        Args:
            df (pd.DataFrame): DataFrame to save
            filename (str, optional): Name of the file to save. Defaults to "price_data.csv".
            
        Returns:
            str: Path to the saved file
            
        Raises:
            CustomException: If an error occurs while saving data
        """
        try:
            file_path = os.path.join(self.config.processed_data_dir, filename)
            logger.info(f"Saving data to {file_path}")
            
            # Save the DataFrame
            df.to_csv(file_path, index=False)
            
            logger.info(f"Data successfully saved to {file_path}")
            return file_path
            
        except Exception as e:
            logger.error("Exception occurred while saving data")
            raise CustomException("Error while saving data", sys)

# Example usage
if __name__ == "__main__":
    try:
        logger.info("Data ingestion process started")
        
        # Initialize data ingestion
        data_ingestion = DataIngestion()
        
        # Read data
        df = data_ingestion.read_data()
        
        # Print data summary
        logger.info(f"Data summary: {df.describe().to_string()}")
        logger.info(f"Data columns: {df.columns.tolist()}")
        logger.info(f"First few rows: \n{df.head().to_string()}")
        
        # Save the data
        saved_path = data_ingestion.save_data(df)
        logger.info(f"Data ingestion completed. Data saved to {saved_path}")
        
    except Exception as e:
        logger.error("Error in data ingestion process")
        raise CustomException("Data ingestion process failed", sys)