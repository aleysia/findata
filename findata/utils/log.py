import logging
import sys

FORMATTER = logging.Formatter('%(asctime)s — %(name)s — %(levelname)s — %(message)s')


def new_logger(logger_name, level=logging.DEBUG, console=True, log_file=None):
    if not console and log_file is None:
        raise ValueError("No handlers specified!")
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.handlers = []
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(FORMATTER)
        logger.addHandler(console_handler)
    if log_file is not None:
        if not isinstance(log_file, str):
            raise ValueError("'log_file' must be a string!")
        file_handler = logging.handlers.TimedRotatingFileHandler(log_file, when='midnight')
        file_handler.setFormatter(FORMATTER)
        logger.addHandler(file_handler)
    return logger
