#!/usr/bin/python3
import sys
import subprocess
import logging

class bcolors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
    TEST = '\033[1;30;42m'

try:
    import colorlog
except:
    print("CAN'T NOT FOUND　'colorlog', TRYING TO INSTALL PACKAGES")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'colorlog'])
    import colorlog

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
