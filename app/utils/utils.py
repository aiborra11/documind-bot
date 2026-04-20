import re
import sys
import logging
from typing import Union


def get_logger(name: str, level: Union[int, str] = logging.INFO) -> logging.Logger:
    """
    Configures and returns a logger instance with adjustable severity level.
    """
    logger = logging.getLogger(name)
    
    # Configure format and level
    if not logger.handlers:
        logger.setLevel(level)
        
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        
        # Send log to console
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        
        # Ensure the logger does not pass events to the root logger
        logger.propagate = False
        
    return logger


def extract_keywords(text: str) -> set:
    """
    Extracts words with 4 or more characters AND all numbers regardless of length.
    """

    words = re.findall(r'\b(?:[a-z_]{4,}|\d+)\b', text.lower())
    return set(words)
