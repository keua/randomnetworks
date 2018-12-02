"""
This module is used to manage the logging for the applicaiton.
"""
import logging
import os
import yaml


def get_logger(logger_name='main'):

    logger = logging.getLogger(logger_name)
    if logger.handlers:
        return logger
    
    # Creating logs folder if doesn't exist
    log_file = os.path.join("logs/", '{}.log'.format(logger_name))
    if not os.path.exists("logs/"):
        os.makedirs("logs/")

    # Reading the configuration file
    with open("config.yml", 'r') as ymlfile:
        conf = yaml.load(ymlfile)
    
    # Setting up the logger
    formatter = logging.Formatter(conf['logging']['loggformat'])
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.setLevel(conf['logging']['logglevel'])
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger