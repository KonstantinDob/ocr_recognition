import logging


def create_logger() -> logging.Logger:
    """Create logger.

    Returns:
        logging.Logger: General logger.
    """
    logging.basicConfig(format="%(message)s", level=logging.INFO)
    logger = logging.getLogger("ocr_recognition")
    return logger


LOGGER = create_logger()
