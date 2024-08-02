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

class DocParseConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "app.docparse"

    def ready(self):
        dirs = [
            DOCPARSE_OUTPUT_JSON_DIR,
            DOCPARSE_OUTPUT_JSON_DIR,
            LOG_DIR,
            OUTPUT_ANNOT_PDF_DIR,
            OUTPUT_DOCPARSE_CSV,
            OUTPUT_IMG_DIR,
            OUTPUT_LAYOUTLM_INPUT_DIR,
            OUTPUT_LAYOUTLM_OUTPUT_DIR,
            PDF_DIR
        ]
        logger.info(f'About to start running on {HOSTADDR}. Checking requirements for document parsing app.')
        if DEV_MODE:
            for directory in dirs:
                if not os.path.exists(directory):
                    logger.warning(f'{directory} doesn\'t exists. Creating the directiory.')
                    os.makedirs(directory)
        if not os.path.exists(DOCPARSE_MODELV3_CLAUSE_PATH):
            logger.error(f'Please check if the token classification model layoutlmv3 for FA has been placed into path {DOCPARSE_MODELV3_CLAUSE_PATH}')
        elif not os.path.exists(DOCPARSE_MODELV3_TS_PATH):
            logger.error(f'Please check if the token classification model layoutlmv3 for TS has been placed into path {DOCPARSE_MODELV3_TS_PATH}')
        else:
            logger.info('Completed checking requirements for document parsing app.')
