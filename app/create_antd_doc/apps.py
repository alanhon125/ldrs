from django.apps import AppConfig
from config import OUTPUT_ANNOT_DOC_DIR, LOG_FILEPATH
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

class CreateAntdDocConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "app.create_antd_doc"
    
    def ready(self):
        dirs = [
            OUTPUT_ANNOT_DOC_DIR
        ]
        logger.info('About to start running on ' + HOSTADDR + '. Checking requirements for annotated term match result document generation app.')
        for directory in dirs:
            if not os.path.exists(directory):
                logger.warning(f'{directory} doesn\'t exists. Creating the directiory.')
                os.makedirs(directory)
        logger.info('Completed checking requirements for annotated term match result document generation app.')