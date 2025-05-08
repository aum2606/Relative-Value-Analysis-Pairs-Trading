import logging
import os
from datetime import datetime

#create logs directory if it doesn't exist
LOG_DIR = "logs"
os.makedirs(LOG_DIR,exist_ok=True)

#Generate log filename with timestamp
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE)

# Configure logging
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

def get_logger(name=__name__):
    """
    Returns a logger instance with the specified name.
    
    Args:
        name (str): Name for the logger, typically __name__ of the calling module
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create a logger
    logger = logging.getLogger(name)
    
    # Add a console handler if not already added
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter("[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    logger.setLevel(logging.INFO)
    return logger

# Example usage
if __name__ == "__main__":
    logger = get_logger("LoggerTest")
    logger.info("Logger is working correctly!")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    print(f"Log file created at: {os.path.abspath(LOG_FILE_PATH)}")