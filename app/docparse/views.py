import json
import os
import re
import traceback
import sys

from rest_framework import status
from rest_framework.response import Response
from rest_framework.renderers import JSONRenderer
from rest_framework.decorators import authentication_classes, permission_classes
from asgiref.sync import sync_to_async
from adrf.views import APIView

from config import *
from app.utils import *
from app.docparse.document_parser import DocParser
from app.docparse.models import DocParse
from app.docparse.serializers import DocParseSerializer 

from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi

import logging

logger = logging.getLogger(__name__)

sample_dict = {
                'createUser': 'alan',
                'updateUser': 'alan',
                'id': 21,
                'fileName': '1_GL_SYN_FA_mkd_20221215.pdf',
                'fileType': 'FA',
                }
sample_dict['filePath'] = os.path.join(PDF_DIR,sample_dict['fileName'])
sample_dict['outputFolder'] = os.path.join(DOCPARSE_OUTPUT_JSON_DIR ,sample_dict['fileType'])
sample_dict['outputFileName'] = sample_dict['fileName'].split('.')[0]+'.json'
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
                            description='*REQUIRED*. the filename of document to be parsed. It must be in pdf format.',
                            default=sample_dict['fileName']
                        ),
                        'filePath': openapi.Schema(
                            type=openapi.TYPE_STRING,
                            description='*REQUIRED*. the absolute file path of document to be parsed. It must be in pdf format.',
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
            "application/json": {
            "success": True,
            "createUser": sample_dict['createUser'],
            "updateUser": sample_dict['updateUser'],
            "id": sample_dict['id'],
            "fileName": sample_dict['outputFileName'],
            "filePath": sample_dict['outputFilePath'],
            "fileType": sample_dict['fileType'],
            "result": []
            }
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
            "errorMessage": f"<ERROR_MESSAGE_FROM_DOCUMENT_PARSING>"
            }
        }
    )
    }

class DocParseView(APIView):
    queryset = DocParse.objects.all()
    serializer_class = DocParseSerializer

    @authentication_classes([])
    @permission_classes([])
    @swagger_auto_schema(
     operation_summary='doc parsing from pdf to json',
     operation_description='perform document parsing into structured JSON using [PyMuPDF](https://pymupdf.readthedocs.io/en/latest/) and NLP model [LayoutLMv3](https://huggingface.co/docs/transformers/model_doc/layoutlmv3).',
     request_body=openapi.Schema(
        type=openapi.TYPE_OBJECT,
        properties=request_schema_dict
        ),
    responses=response_schema_dict
    )
    async def post(self, request, *args, **krgs):
        data = request.data
        response = await docparse2json(data)
        return response

async def document_parser(filepath, 
                    use_model = USE_MODEL, 
                    use_ocr = USE_OCR, 
                    do_annot = DOCPARSE_PDF_ANNOTATE, 
                    document_type = DOCPARSE_DOCUMENT_TYPE):
    '''
    input the filepath to perform document parsing
    '''
    do_segment = True  # True for text block segmentation
 
    if document_type == 'agreement':
        sub_folder_name = 'FA/'
        model_path = DOCPARSE_MODELV3_CLAUSE_PATH
    elif document_type == 'termSheet':
        sub_folder_name = 'TS/'
        model_path = DOCPARSE_MODELV3_TS_PATH
    output_json_folder = os.path.join(DOCPARSE_OUTPUT_JSON_DIR,sub_folder_name)
    parser = DocParser(
        filepath, output_json_folder, OUTPUT_IMG_DIR, use_model, use_ocr ,model_path, OUTPUT_ANNOT_PDF_DIR, do_annot=do_annot, document_type=document_type
    )
    await sync_to_async(parser.process)(do_segment)
    await sync_to_async(parser.save_output)()
    if do_annot:
        await sync_to_async(parser.annot_pdf)(do_segment)
        
    return parser.txt_blocks

def variable_string(var):
    '''show variable as a string representation'''
    return [i for i, j in globals().items() if j == var][0]

async def docparse2json(request):
    '''
    Request Body
    [{
        "createUser": ,
        "updateUser": ,
        "id": 0,
        "fileName" : "",
        "filePath" : "",
        "fileType": ""
    },...]

    Perform or extract document parsing output with filename provided
    '''

    async def request_docparse(request):

        try:
            createUser = request.get("createUser")
        except:
            createUser = None
        try:
            updateUser = request.get("updateUser")
        except:
            updateUser = None
        try:
            use_ocr = request.get("use_ocr")
        except:
            use_ocr = USE_OCR
        # compulsory fields
        Id = request.get("id")
        fileName = request.get("fileName") # fileName of document
        filePath = request.get("filePath") # input file path end with .pdf
        fileType = request.get("fileType") # input file type, either FA or TS

        response = {
            "success": True,
            "createUser": createUser,
            "updateUser": updateUser,
            "id": Id,
            "fileName" : fileName,
            "filePath" : None,
            "fileType": fileType
        }

        # Input value validation for compulsory fields
        invalid_condition = False
        for variable in [Id, fileName, filePath, fileType]:
            if variable is None:
                invalid_msg = f'Invalid request in document parsing. Input value of {variable_string(variable)} should not be None.'
                await update_doc_error_status(Id, invalid_msg)
                response['success'] = False
                response['errorMessage'] = invalid_msg
                return response
            local_dict = locals()
            variable_name = [name for name in local_dict if local_dict[name] == variable][0] # turn variable name into string
            if variable_name in ['fileName', 'filePath']:
                invalid_condition = not locals()[variable_name].endswith('.pdf')
                invalid_msg = f"Invalid request in document parsing. Input value of {variable_name}:{locals()[variable_name]} should be ends with '.pdf'. File must be in .pdf format. Please check the input file type or perform PDF conversion beforehand."
            elif variable_name == 'fileType':
                invalid_condition = not locals()[variable_name] in ['FA','TS']
                invalid_msg = f"Invalid request in document parsing. Input value of {variable_name}:{locals()[variable_name]} should be either 'FA' or 'TS'."
            elif variable_name == 'Id':
                invalid_condition = not type(locals()[variable_name])==int
                invalid_msg = f"Invalid request in document parsing. Input value of {variable_name}:{locals()[variable_name]} should be an integer, not in data type {type(locals()[variable_name])}."
            if invalid_condition:
                await update_doc_error_status(Id, invalid_msg)
                response['success'] = False
                response['errorMessage'] = invalid_msg
                return response
        # check filename and filepath consistency
        if not fileName in filePath:
            invalid_msg = 'Invalid request in document parsing. Input fileName should be consistent to filePath.'
            await update_doc_error_status(Id, invalid_msg)
            response['success'] = False
            response['errorMessage'] = invalid_msg
            return response
        # check file existence
        if not os.path.exists(filePath):
            invalid_msg = f'Invalid request in document parsing. {filePath} does not exists. Please check the input file path.'
            await update_doc_error_status(Id, invalid_msg)
            response['success'] = False
            response['errorMessage'] = invalid_msg
            return response
        
        newFileName = fileName.split('.')[0] + '.json'
        response["fileName"] = newFileName
        
        if fileType == 'TS':
            document_type = 'termSheet'
        elif fileType == 'FA':
            document_type = 'agreement'

        outpath = os.path.join(os.path.join(DOCPARSE_OUTPUT_JSON_DIR, fileType), re.sub(".pdf", ".json", fileName))
        
        if not os.path.exists(outpath) or not DEV_MODE:
            try:
                logger.info(f'Document parsing {fileName} ...')
                result = await document_parser(filePath, use_ocr = use_ocr, document_type = document_type)
                logger.info(f'Complete document parsing {fileName}\n')
                response['filePath'] = outpath
                response['result'] = result
                
                # layoutlmInputOutpath = os.path.join(OUTPUT_LAYOUTLM_INPUT_DIR, outpath)
                # layoutlmOutputOutpath = os.path.join(OUTPUT_LAYOUTLM_OUTPUT_DIR, outpath)
                # erase_cache_files(layoutlmInputOutpath)
                # erase_cache_files(layoutlmOutputOutpath)
                
                return response
            except Exception as e:
                exc_type, exc_value, exc_tb = sys.exc_info()
                tb = traceback.TracebackException(exc_type, exc_value, exc_tb)
                invalid_msg = 'In task name: "document parsing"\n'+''.join(tb.format())
                response['success'] = False
                response['errorMessage'] = invalid_msg
                await update_doc_error_status(Id, invalid_msg)
                logger.error(json.dumps(response,indent=4))
                
                return response
        else:
            with open(outpath, 'r') as f:
                result = json.load(f)
            response['filePath'] = outpath
            response['result'] = result
            return response

    try:
        request = json.loads(request.body.decode("utf-8"))
    except:
        pass
    
    flag = True
    if isinstance(request, list):
        content = []
        for r in request:
            response = await request_docparse(r)
            if not response.get('success'):
                flag = False
            content.append(response)
    elif isinstance(request, dict):
        content = await request_docparse(request)
        if not content.get('success'):
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
