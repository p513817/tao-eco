#!/usr/bin/python3
import sys
import subprocess
import logging

try:
    import colorlog
except:
    print("CAN'T NOT FOUND　'colorlog', TRYING TO INSTALL PACKAGES")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'colorlog'])
    import colorlog

def default():

    fmt = "{asctime} {log_color}{levelname}{reset}: {message}"
    colorlog.basicConfig(level=logging.DEBUG, style="{", format=fmt, stream=None)
    return logging.getLogger()

def add_custom_levels_samples():

    import logging, colorlog
    TRACE = 5
    logging.addLevelName(TRACE, 'TRACE')
    formatter = colorlog.ColoredFormatter(log_colors={'TRACE': 'yellow'})
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger = logging.getLogger('example')
    logger.addHandler(handler)
    logger.setLevel('TRACE')
    logger.log(TRACE, 'a message using a custom level')

def custom_logger_with_logfile():

    import sys

    logPath = "."
    fileName = "debug"
    mode = logging.DEBUG
    
    logFormatter = colorlog.ColoredFormatter("%(asctime)s %(log_color)s%(levelname)-5.5s%(reset)s %(message)s",
                                            datefmt="%Y-%m-%d %H:%M:%S",)
    rootLogger = logging.getLogger()
    rootLogger.setLevel(mode)

    fileHandler = logging.FileHandler("{0}/{1}.log".format(logPath, fileName))
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)

    return rootLogger


def custom_logger():

    fmt = colorlog.ColoredFormatter(
                    "%(asctime)s %(log_color)s%(levelname)-8s%(reset)s %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    reset=True,
                    log_colors={
                        "DEBUG": "white",
                        "INFO": "green",
                        "WARNING": "yellow",
                        "ERROR": "red",
                        "CRITICAL":"red,bg_white",
                    },
                    secondary_log_colors={},
                    style='%'
                ) 

    handler = logging.StreamHandler()
    handler.setFormatter(fmt)

    logger = logging.getLogger('example')
    logger.addHandler(handler)
    logger.setLevel('DEBUG')

    return logger


if __name__ == '__main__':

    log = custom_logger_with_logfile()

    log.info('information')
    log.debug('debug')
    log.warning('warning')
    log.error('error')
    log.critical('critical')
