import logging
import os
from logging.handlers import RotatingFileHandler


# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)


def setup_logger():
    logger = logging.getLogger("InsightForge")

    # Prevent duplicate handlers when Streamlit reloads
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = RotatingFileHandler(
        "logs/insightforge.log",
        maxBytes=5 * 1024 * 1024,   # 5 MB
        backupCount=3,
        encoding="utf-8"
    )

    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


logger = setup_logger()