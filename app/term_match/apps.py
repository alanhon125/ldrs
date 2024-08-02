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

class TermMatchConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "app.term_match"
    
    def ready(self):
        dirs = [
            DOCPARSE_FA_CSV,
            DOCPARSE_TS_CSV,
            OUTPUT_TERM_MATCH_CSV,
        ]
        logger.info(f'About to start running on {HOSTADDR}. Checking requirements for term matching app.')
        if DEV_MODE:
            for directory in dirs:
                if not os.path.exists(directory):
                    logger.warning(f'{directory} doesn\'t exists. Creating the directiory.')
                    os.makedirs(directory)
        if not os.path.exists(SENT_BERT_MODEL_PATH):
            logger.error(f'The model required for sentence embedding and cosine similarity computation between TS and FA context has not been loaded. Please check if the model gte-base model has been placed into path {SENT_BERT_MODEL_PATH}')
        else:
            logger.info('Completed checking requirements for term matching app.')