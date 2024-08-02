from rest_framework import status
from rest_framework.response import Response
from rest_framework.generics import GenericAPIView
from rest_framework.decorators import authentication_classes, permission_classes
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi

from config import *
from app.utils import *
from app.health_check.models import HealthCheck
from app.health_check.serializers import HealthCheckSerializer 

import logging
import os

logger = logging.getLogger(__name__)

response_schema_dict = {
    "200": openapi.Response(
        description="Success",
        examples={
            "application/json": {
            'configuration_status': 'OK',
            'document_parsing_status': 'OK',
            'features_extraction_status': 'OK',
            'term_matching_status': 'OK'
            }
        }
    ),
    "400": openapi.Response(
        description="Error: Bad Request",
        examples={
            "application/json": {
            'configuration_status': 'Not OK! Configuration file "config.py" must be placed at <CONFIG_FILE_PATH>',
            'document_parsing_status': 'Not OK! Models required for both document parsing of TS & FA has not been loaded. Please check if the token classification models layoutlmv3 has been placed into paths <DOCPARSE_MODELV3_PATHS>',
            'features_extraction_status': 'Not OK! file <FILENAME> is required for features extraction and must be placed at <FILE_PATH>',
            'term_matching_status': 'Not OK! The model required for sentence embedding and cosine similarity computation between TS and FA context has not been loaded. Please check if the model gte-base has been placed into path <SENT_BERT_MODEL_PATH>'
            }
        }
    )
    }

class HealthCheckView(GenericAPIView):
    queryset = HealthCheck.objects.all()
    serializer_class = HealthCheckSerializer
    @authentication_classes([])
    @permission_classes([])
    @swagger_auto_schema(
     operation_summary='generate health report for each module',
     operation_description='perform health checking to inspect if every required configuration and document is prepared.',
    responses=response_schema_dict
    )
    def get(self, request):
        content = {
            'configuration_status': 'Not yet checked',
            'document_parsing_status': 'Not yet checked',
            'features_extraction_status': 'Not yet checked',
            'term_matching_status': 'Not yet checked'
            }
        
        CONFIG_FILE_PATH = os.path.join(PROJ_ROOT_DIR,'config.py')
        REGEX_FILE_PATH = os.path.join(PROJ_ROOT_DIR,'regexp.py')
        SECTION_WORDBANK_FILE_PATH = os.path.join(PROJ_ROOT_DIR,'TS_section.csv')
        
        if not os.path.exists(CONFIG_FILE_PATH):
            content['configuration_status'] = f'Not OK! Configuration file "config.py" must be placed at {CONFIG_FILE_PATH}'
        else:
            content['configuration_status'] = 'OK'
        
        dirs = [
            DOCPARSE_OUTPUT_JSON_DIR,
            DOCPARSE_OUTPUT_JSON_DIR,
            LOG_DIR,
            LOG_FILEPATH,
            OUTPUT_ANNOT_PDF_DIR,
            OUTPUT_DOCPARSE_CSV,
            OUTPUT_IMG_DIR,
            OUTPUT_LAYOUTLM_INPUT_DIR,
            OUTPUT_LAYOUTLM_OUTPUT_DIR,
            PDF_DIR
        ]
        logger.info('About to check requirements for document parsing app.')
        if DEV_MODE:
            for directory in dirs:
                if not os.path.exists(directory):
                    logger.warning(f'{directory} doesn\'t exists. Creating the directiory.')
                    os.makedirs(directory)
        
        if not os.path.exists(DOCPARSE_MODELV3_CLAUSE_PATH) and not os.path.exists(DOCPARSE_MODELV3_TS_PATH):
            content['document_parsing_status'] = f'Not OK! Models required for both document parsing of TS & FA has not been loaded. Please check if the token classification models layoutlmv3 has been placed into paths {DOCPARSE_MODELV3_TS_PATH} & {DOCPARSE_MODELV3_CLAUSE_PATH}'
        elif not os.path.exists(DOCPARSE_MODELV3_CLAUSE_PATH):
            content['document_parsing_status'] = f'Not OK! Model required for document parsing of FA has not been loaded. Please check if the token classification model layoutlmv3 has been placed into path {DOCPARSE_MODELV3_CLAUSE_PATH}'
        elif not os.path.exists(DOCPARSE_MODELV3_TS_PATH):
            content['document_parsing_status'] = f'Not OK! Model required for document parsing of TS has not been loaded. Please check if the token classification model layoutlmv3 has been placed into path {DOCPARSE_MODELV3_TS_PATH}'
        else:
            content['document_parsing_status'] = 'OK'
        logger.info('Completed checking requirements for document parsing app.')
        
        
        dirs = [
            OUTPUT_DOCPARSE_CSV
        ]
        logger.info('About to check requirements for keys, clauses and terms extraction app.')
        if DEV_MODE:
            for directory in dirs:
                if not os.path.exists(directory):
                    logger.warning(f'{directory} doesn\'t exists. Creating the directiory.')
                    os.makedirs(directory)
        
        if not os.path.exists(REGEX_FILE_PATH):
            content['features_extraction_status'] = f'Not OK! Regular expressions file "regex.py" is required for features extraction and must be placed at {REGEX_FILE_PATH}'
        elif not os.path.exists(SECTION_WORDBANK_FILE_PATH):
            content['features_extraction_status'] = f'Not OK! Word bank for TS Sections "TS_section.csv" is required for features extraction and must be placed at {SECTION_WORDBANK_FILE_PATH}'
        else:
            content['features_extraction_status'] = 'OK'
                         
        logger.info('Completed checking requirements for keys, clauses and terms extraction app.')
        
        dirs = [
            DOCPARSE_FA_CSV,
            DOCPARSE_TS_CSV,
            OUTPUT_TERM_MATCH_CSV,
        ]
        logger.info('About to check requirements for term matching app.')
        if DEV_MODE:
            for directory in dirs:
                if not os.path.exists(directory):
                    logger.warning(f'{directory} doesn\'t exists. Creating the directiory.')
                    os.makedirs(directory)
        
        if not os.path.exists(SENT_BERT_MODEL_PATH):
            content['term_matching_status'] = f'Not OK! The model required for sentence embedding and cosine similarity computation between TS and FA context has not been loaded. Please check if the model gte-base has been placed into path {SENT_BERT_MODEL_PATH}'
        else:
            content['term_matching_status'] = 'OK'
                    
                    
        logger.info('Completed checking requirements for term matching app.')
        
        if any('Not OK' in v for v in content.values()):
            return Response(content, status.HTTP_400_BAD_REQUEST)
        else:
            return Response(content, status.HTTP_200_OK)