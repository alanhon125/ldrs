from django.apps import AppConfig
from config import *
import os
import socket
import logging

logger = logging.getLogger(__name__)
# logging.basicConfig(
#     filename=LOG_FILEPATH,
#     filemode='a',
#     format='【%(asctime)s】【%(filename)s:%(lineno)d】【%(levelname)-8s】%(message)s',
#     level=os.environ.get("LOGLEVEL", "INFO"),
#     datefmt='%Y-%m-%d %H:%M:%S'
#     )

HOSTADDR = socket.gethostbyname(socket.gethostname())

class TermExtractConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "app.term_extract"
    
    def ready(self):
        dirs = [
            OUTPUT_DOCPARSE_CSV
        ]
        logger.info(f'About to start running on {HOSTADDR}. Checking requirements for keys, clauses and terms extraction app.')
        if DEV_MODE:
            for directory in dirs:
                if not os.path.exists(directory):
                    logger.warning(f'{directory} doesn\'t exists. Creating the directiory.')
                    os.makedirs(directory)
        logger.info('Completed checking requirements for keys, clauses and terms extraction app.')
