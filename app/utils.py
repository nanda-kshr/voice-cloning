import logging
import sys
from config.constants import LOGGING_BASE_FORMAT, LOGGING_BASE_FILE, LOGGING_BASE_NAME
import traceback

logging.basicConfig(
    level=logging.INFO, 
    format=LOGGING_BASE_FORMAT,
    handlers=[
        logging.FileHandler(LOGGING_BASE_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(LOGGING_BASE_NAME)


class ResponseHandler:
    @staticmethod
    def success(message="success",status=200):
        message = {"status": status, "message":message}
        logger.info(message)
        return message


    @staticmethod
    def info(message="",status=True):
        if status:
            logger.info(message)
    

    @staticmethod
    def error(message="",status=400):
        print(traceback.format_exc())
        if message is not None:
            logger.error(message)
            message = {"status": status, "message":  "Server is currently busy. Please try again later.","messageType":"predefined"}
        else:
            message = {"status": status, "message": message,"messageType":"custom"}
        return message


