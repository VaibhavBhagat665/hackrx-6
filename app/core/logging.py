import logging
import sys
from pathlib import Path

def setup_logging():
    """configure application logging"""
    
    # create logs directory
    Path("logs").mkdir(exist_ok=True)
    
    # configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("logs/app.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # configure specific loggers
    logger = logging.getLogger("intelligent_doc_system")
    logger.setLevel(logging.INFO)
    
    return logger

logger = setup_logging()
