import logging
import sys

def create_logger(name: str='aic21', filepath: str=None, level: int=logging.INFO, stdout=False):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if filepath:
        f_handler = logging.FileHandler(filepath)
        f_format = logging.Formatter('%(levelname)s: %(message)s')
        f_handler.setFormatter(f_format)
        logger.addHandler(f_handler)
    
    if stdout:
        logger.addHandler(logging.StreamHandler(sys.stdout))

    return logger
