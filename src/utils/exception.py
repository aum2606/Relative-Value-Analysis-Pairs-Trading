import sys
import os
from src.utils.logger import get_logger

logger = get_logger(__name__)

class CustomException(Exception):
    """
    Custom exception class for detailed error tracking and reporting.
    
    Attributes:
        error_message (str): Error message
        error_detail (str): Detailed error information including filename and line number
    """
    
    def __init__(self, error_message:str, error_detail:sys):
        """
        Initialize the CustomException class with error message and details.
        
        Args:
            error_message (str): Error message
            error_detail (sys): Error details from sys.exc_info()
        """
        super().__init__(error_message)
        self.error_message = error_message
        self.error_detail = self._get_detailed_error_message(error_message, error_detail)
        logger.error(self.error_detail)
    
    def _get_detailed_error_message(self, error_message:str, error_detail:sys) -> str:
        """
        Create a detailed error message with file name and line number.
        
        Args:
            error_message (str): Original error message
            error_detail (sys): Error details from sys.exc_info()
            
        Returns:
            str: Formatted error message with details
        """
        _, _, exc_tb = error_detail.exc_info()
        
        # Get the filename and line number where the error occurred
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
        
        # Create a detailed error message
        error_message = f"Error occurred in Python script [{os.path.basename(file_name)}] at line number [{line_number}]: {error_message}"
        
        return error_message
    
    def __str__(self):
        """
        Return the string representation of the CustomException.
        
        Returns:
            str: Detailed error message
        """
        return self.error_detail
    
    def __repr__(self):
        """
        Return the representation of the CustomException.
        
        Returns:
            str: Class name with error message
        """
        return f"{self.__class__.__name__}: {self.error_message}"

# Example usage
if __name__ == "__main__":
    try:
        # Simulate an error
        a = 1/0
    except Exception as e:
        raise CustomException("Division by zero error occurred", sys)