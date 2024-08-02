import json
import traceback
import sys

from rest_framework import status
from rest_framework.response import Response
from rest_framework.renderers import JSONRenderer
from rest_framework.decorators import authentication_classes, permission_classes
from asgiref.sync import sync_to_async
from adrf.views import APIView
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi

from config import *
from app.utils import *
from app.term_extract.term_extract import docparse2table
from app.term_extract.models import TermExtract
from app.term_extract.serializers import TermExtractSerializer

import logging
import os

logger = logging.getLogger(__name__)

sample_dict = {
                'createUser': 'alan',
                'updateUser': 'alan',
                'id': 44,
                'fileName': '1_GL_SYN_FA_mkd_20221215.json',
                'fileType': 'FA',
                }
sample_dict['filePath'] = os.path.join(DOCPARSE_OUTPUT_JSON_DIR + sample_dict['fileType'],sample_dict['fileName'])
sample_dict['outputFolder'] = os.path.join(OUTPUT_DOCPARSE_CSV ,sample_dict['fileType'])
sample_dict['outputFileName'] = sample_dict['fileName'].split('.')[0]+'.csv'
sample_dict['outputFilePath'] = os.path.join(sample_dict['outputFolder'],sample_dict['outputFileName'])

request_schema_dict = {                        
                       'createUser': openapi.Schema(
                            type=openapi.TYPE_STRING,
                            description='username of task creator',
                            default=sample_dict['createUser']
                        ),
                        'updateUser': openapi.Schema(
                            type=openapi.TYPE_STRING,
                            description='username of task updater',
                            default=sample_dict['updateUser']
                        ),
                        'id': openapi.Schema(
                            type=openapi.TYPE_INTEGER,
                            description='*REQUIRED*. document ID (created automatically when file uploaded)',
                            default=sample_dict['id']
                        ),
                        'fileName': openapi.Schema(
                            type=openapi.TYPE_STRING,
                            description='*REQUIRED*. the filename of document that terms and clauses going to be extracted.',
                            default=sample_dict['fileName']
                        ),
                        'filePath': openapi.Schema(
                            type=openapi.TYPE_STRING,
                            description='the absolute file path of document that terms and clauses going to be extracted.',
                            default=sample_dict['filePath']
                        ),
                        'fileType': openapi.Schema(
                            type=openapi.TYPE_STRING,
                            description='*REQUIRED*. document type, either FA for facility agreement or TS for term sheet',
                            default=sample_dict['fileType'],
                            enum=['FA','TS']
                        )
                    }

response_schema_dict = {
    "200": openapi.Response(
        description="Success",
        examples={
            "application/json": [
                {
                    "indexId": 'integer',
                    "textBlockId": 'integer',
                    "pageId": "string",
                    "sectionId": 'integer',
                    "sectionContent": "string",
                    "subSectionId": 'integer',
                    "subSection": "string",
                    "scheduleId": 'integer',
                    "schedule": "string",
                    "partId": 'integer',
                    "part": "string",
                    "textElement": "title",
                    "listId": "string",
                    "definition": "string",
                    "identifier": "string",
                    "textContent": "text",
                    "success": True,
                    "createUser": "alan",
                    "updateUser": "alan",
                    "id": 'integer',
                    "fileId": 44,
                    "fileType": "FA"
                }
            ]
        }
    ),
    "400": openapi.Response(
        description="Error: Bad Request",
        examples={
            "application/json": {
            "success": False,
            "createUser": sample_dict['createUser'],
            "updateUser": sample_dict['updateUser'],
            "id": sample_dict['id'],
            "fileName": sample_dict['outputFileName'],
            "filePath": None,
            "fileType": sample_dict['fileType'],
            "errorMessage": f"<ERROR_MESSAGE_FROM_TERM_EXTRACTION>"
            }
        }
    )
    }

class TermExtractView(APIView):
    queryset = TermExtract.objects.all()
    serializer_class = TermExtractSerializer

    @authentication_classes([])
    @permission_classes([])
    @swagger_auto_schema(
     operation_summary='extract keys, terms and clauses from parsed JSON',
     operation_description='perform keys (clause section, schedule section, part section, ids), terms (definitions, parties) and clauses (clause/schedule paragraphs) extraction from parsed JSON.',
     request_body=openapi.Schema(
        type=openapi.TYPE_OBJECT,
        properties=request_schema_dict
        ),
     responses=response_schema_dict
    )
    async def post(self, request, *args, **krgs):
        data = request.data

        response = await docparse_table(data)
        return response

def erase_cache_files(filePath):
    if os.path.exists(filePath):
        os.remove(filePath)

def variable_string(var):
    '''show variable as a string representation'''
    return [i for i, j in globals().items() if j == var][0]

async def docparse_table(request):
    '''
    Request Body
    [{
        "createUser": "string",
        "updateUser": "string",
        "id": 0,
        "fileName" : "string",
        "filePath" : "string",
        "fileType" : "string"
    },...]

    Response Body
    
    Perform keys, terms and clause extraction from docparse result with structured JSON document and 
    give pure content with clear hierachical structure of sections and schedules

    '''

    async def request_term_extract(request):
        try:
            createUser = request.get("createUser")
        except:
            createUser = None
        try:
            updateUser = request.get("updateUser")
        except:
            updateUser = None
        Id = request.get("id")
        fileName = request.get("fileName") # fileName of document
        filePath = request.get("filePath") # input file path end with .json
        fileType = request.get("fileType") # input file type, either FA or TS
        
        response = {
            "success": True,
            "createUser": createUser,
            "updateUser": updateUser,
            'id': None,
            "fileId": Id,
            "fileType": fileType
        }
        
        # Input value validation for compulsory fields
        invalid_condition = False
        for variable in [Id, fileName, filePath, fileType]:
            if variable is None:
                invalid_msg = f'Invalid request in keys, terms and clauses extraction. Input value of {variable_string(variable)} should not be None.'
                await update_doc_error_status(Id, invalid_msg)
                response['success'] = False
                response['errorMessage'] = invalid_msg
                logger.error(invalid_msg)
                return response
            local_dict = locals()
            variable_name = [name for name in local_dict if local_dict[name] == variable][0] # turn variable name into string
            if variable_name in ['fileName', 'filePath']:
                invalid_condition = not locals()[variable_name].endswith('.json')
                invalid_msg = f"Invalid request in keys, terms and clauses extraction. Input value of {variable_name}:{locals()[variable_name]} should be ends with '.json'. File input must be a parsed document that in .json format. Please check the input file type or perform document parsing beforehand."
            elif variable_name == 'fileType':
                invalid_condition = not locals()[variable_name] in ['FA','TS']
                invalid_msg = f"Invalid request in keys, terms and clauses extraction. Input value of {variable_name}:{locals()[variable_name]} should be either 'FA' or 'TS'."
            elif variable_name == 'Id':
                invalid_condition = not type(locals()[variable_name])==int
                invalid_msg = f"Invalid request in keys, terms and clauses extraction. Input value of {variable_name}:{locals()[variable_name]} should be an integer, not in data type {type(locals()[variable_name])}."
            if invalid_condition:
                await update_doc_error_status(Id, invalid_msg)
                response['success'] = False
                response['errorMessage'] = invalid_msg
                logger.error(invalid_msg)
                return response
        # check filename and filepath consistency
        if not fileName in filePath:
            invalid_msg = 'Invalid request in keys, terms and clauses extraction. Input fileName should be consistent to filePath.'
            await update_doc_error_status(Id, invalid_msg)
            response['success'] = False
            response['errorMessage'] = invalid_msg
            logger.error(invalid_msg)
            return response
        
        docparse_result = request.get("result")
        if docparse_result is None:
            # check file existence
            if not os.path.exists(filePath):
                invalid_msg = f'Invalid request in keys, terms and clauses extraction. {filePath} does not exists. Please check the input file path.'
                await update_doc_error_status(Id, invalid_msg)
                response['success'] = False
                response['errorMessage'] = invalid_msg
                logger.error(invalid_msg)
                return response

            else:            
                docparse_result = filePath
        
        try:
            result = await sync_to_async(docparse2table)((docparse_result, fileType))
            i = 0
            responses = []
            if isinstance(result, dict) and 'success' in result:
                if not result['success']:
                    invalid_msg = result['errorMessage']
                    response['success'] = False
                    response['errorMessage'] = invalid_msg
                    await update_doc_error_status(Id, invalid_msg)
                    logger.error(invalid_msg)
                    return response
            elif isinstance(result, list):
                while i <= len(result)-1:
                    r = result[i]
                    r.update(response.copy())
                    r['id'] = i
                    responses.append(r)
                    i += 1
            # termExtractCSVOutpath = os.path.join(OUTPUT_DOCPARSE_CSV + fileType, fileName.split('.')[0] +'_docparse.csv')
            # erase_cache_files(termExtractCSVOutpath)
            # erase_cache_files(filePath)
            return responses
        except Exception as e:
            exc_type, exc_value, exc_tb = sys.exc_info()
            tb = traceback.TracebackException(exc_type, exc_value, exc_tb)
            invalid_msg = 'In task name: "keys, terms and clauses extraction"\n'+''.join(tb.format())
            response['success'] = False
            response['errorMessage'] = invalid_msg
            logger.error(invalid_msg)
            await update_doc_error_status(Id, invalid_msg)
            return response
    
    try:
        request = json.loads(request.body.decode("utf-8"))
    except:
        pass
    
    flag = True
    content = []
    if isinstance(request, list):
        # for r in request:
        #     response = await request_term_extract(r)
        #     content.extend(response)
        content = await sync_to_async(multiprocess)(request_term_extract, request)
    elif isinstance(request, dict):
        content = await request_term_extract(request)
    
    if isinstance(content, list):
        if any([not i.get('success') for i in content]):
            flag = False
    elif isinstance(content, dict):
        flag = False

    if flag:
        response = Response(content, status=status.HTTP_200_OK)
    else:
        response = Response(content, status=status.HTTP_400_BAD_REQUEST)
        
    response.accepted_renderer = JSONRenderer()
    response.accepted_media_type = "application/json"
    response.renderer_context = {}
    response.render()
    return response