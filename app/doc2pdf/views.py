import logging
import json
import os

from rest_framework import status
from rest_framework.response import Response
from rest_framework.renderers import JSONRenderer
from rest_framework.decorators import authentication_classes, permission_classes
from asgiref.sync import sync_to_async
from adrf.views import APIView
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi

from app.doc2pdf.doc2pdf import libreOffice_convert_to_pdf
from app.doc2pdf.docx_processing import docx_preprocess, lowercase_file_ext
from config import DOC_DIR, PDF_DIR
from app.utils import update_doc_error_status
from app.doc2pdf.serializers import DocConvertSerializer 
from app.doc2pdf.models import DocConvert

logger = logging.getLogger(__name__)

sample_dict = {
                'createUser': 'alan',
                'updateUser': 'alan',
                'id': 21,
                'filePath': f'{DOC_DIR}/FA/1_GL_SYN_FA_mkd_20221215.docx',
                'fileType': 'FA',
                'convertTo':"pdf"
            }
sample_dict['outputFolder'] = PDF_DIR
sample_dict['fileName'] = sample_dict['filePath'].split('/')[-1]
sample_dict['outputFileName'] = sample_dict['fileName'].split('.')[0]+'.'+sample_dict['convertTo']
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
                            description='*REQUIRED*. the filename of document to be converted',
                            default=sample_dict['fileName']
                        ),
                        'filePath': openapi.Schema(
                            type=openapi.TYPE_STRING,
                            description='*REQUIRED*. the absolute file path of document to be converted',
                            default=sample_dict['filePath']
                        ),
                        'fileType': openapi.Schema(
                            type=openapi.TYPE_STRING,
                            description='*REQUIRED*. document type, either FA for facility agreement or TS for term sheet',
                            default=sample_dict['fileType'],
                            enum=['FA','TS']
                        ),
                        'outputFolder': openapi.Schema(
                            type=openapi.TYPE_STRING,
                            description='the desired output folder path',
                            default=sample_dict['outputFolder']
                        ),
                        'convertTo': openapi.Schema(
                            type=openapi.TYPE_STRING,
                            description='[LibreOffice](https://www.libreoffice.org/discover/libreoffice/) converter is used. It only supports file conversion from .doc to .pdf, .html, .txt, input value must be one of the following: html, pdf, txt, docx',
                            default=sample_dict['convertTo'],
                            enum=['pdf', 'html', 'txt']
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
            "fileType": sample_dict['fileType']
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
            "errorMessage": f"Error occurred while converting {sample_dict['filePath']} to {sample_dict['outputFilePath']} as follows:\n"
            }
        }
    )
    }

class DocConvertView(APIView):
    queryset = DocConvert.objects.all()
    serializer_class = DocConvertSerializer
    @authentication_classes([])
    @permission_classes([])
    @swagger_auto_schema(
     operation_summary='doc convert to pdf',
     operation_description='perform doc2pdf (e.g. .docx to .pdf) using [LibreOffice](https://www.libreoffice.org/discover/libreoffice/).',
     request_body=openapi.Schema(
        type=openapi.TYPE_OBJECT,
        properties=request_schema_dict
        ),
     responses=response_schema_dict
    )
    async def post(self, request, *args, **krgs):
        data = request.data
        response = await convert_doc2pdf(data)
        return response

def variable_string(var):
    '''show variable as a string representation'''
    return [i for i, j in globals().items() if j == var][0]

async def convert_doc2pdf(request):
    '''
    Request Body
    [{
        "createUser": ,
        "updateUser": ,
        "id": 0,
        "fileName" : "",
        "filePath" : "",
        "fileType": "",
        "convertTo": "",
        "outputFolder": ""
    },...]
    '''

    async def request_doc2pdf(request):
        try:
            createUser = request.get("createUser")
        except:
            createUser = None
        try:
            updateUser = request.get("updateUser")
        except:
            updateUser = None
            
        # compulsory fields
        Id = request.get("id")
        fileName = request.get("fileName") # fileName of document
        filePath = request.get("filePath") # input file path end with .doc or .docx
        fileType = request.get("fileType") # input file type, either FA or TS
        # Input value validation for compulsory fields
        invalid_condition = False
        
        response = {
                "success": True,
                "createUser": createUser,
                "updateUser": updateUser,
                "id": Id,
                "fileName" : fileName,
                "filePath" : None,
                "fileType": fileType
            }
        
        for variable in [Id, fileName, filePath, fileType]:
            if variable is None:
                invalid_msg = f'Invalid request in converting document to PDF. Input value of {variable_string(variable)} should not be None.'
                logger.error(invalid_msg)
                await update_doc_error_status(Id, invalid_msg)
                response['success'] = False
                response['errorMessage'] = invalid_msg
                return response
            local_dict = locals()
            variable_name = [name for name in local_dict if local_dict[name] == variable][0] # turn variable name into string
            if variable_name in ['fileName', 'filePath']:
                invalid_condition = not locals()[variable_name].endswith(('.doc', '.docx','.DOC','.DOCX','.pdf'))
                invalid_msg = f"Invalid request in converting document to PDF. Input value of {variable_name}:{locals()[variable_name]} should be ends with '.doc', '.docx','.DOC','.DOCX','.pdf' but not '.{not locals()[variable_name].split('.')[-1]}'"
            elif variable_name == 'fileType':
                invalid_condition = not locals()[variable_name] in ['FA','TS']
                invalid_msg = f"Invalid request in converting document to PDF. Input value of {variable_name}:{locals()[variable_name]} should be either 'FA' or 'TS'."
            elif variable_name == 'Id':
                invalid_condition = not type(locals()[variable_name])==int
                invalid_msg = f"Invalid request in converting document to PDF. Input value of {variable_name}:{locals()[variable_name]} should be an integer, not in data type {type(locals()[variable_name])}."
            if invalid_condition:
                await update_doc_error_status(Id, invalid_msg)
                response['success'] = False
                response['errorMessage'] = invalid_msg
                logger.error(invalid_msg)
                return response
        # check filename and filepath consistency
        if not fileName in filePath:
            invalid_msg = 'Invalid request in converting document to PDF. Input fileName should be consistent to filePath.'
            logger.error(invalid_msg)
            await update_doc_error_status(Id, invalid_msg)
            response['success'] = False
            response['errorMessage'] = invalid_msg
            return response
        # check file existence
        if not os.path.exists(filePath):
            invalid_msg = f'Invalid request in converting document to PDF. {filePath} does not exists. Please check the input file path.'
            logger.error(invalid_msg)
            await update_doc_error_status(Id, invalid_msg)
            response['success'] = False
            response['errorMessage'] = invalid_msg
            return response
        try:
            outputFolder = request.get("outputFolder") # output folder path
        except:
            outputFolder = PDF_DIR
        try:
            convertTo = request.get("convertTo")
        except:
            convertTo = 'pdf'
            
        if outputFolder is None:
            outputFolder = PDF_DIR
        if convertTo is None:
            convertTo = 'pdf'
        newFilename = fileName.split('.')[0] + '.' + convertTo
        response["fileName"] = newFilename
        outFilePath = os.path.join(outputFolder, newFilename)
        oriFileType = fileName.split('.')[-1]

        if not os.path.exists(filePath):
            invalid_msg = f'Invalid request in converting document to PDF since {filePath} does not exists. Please check the input file path for conversion.'
            logger.error(invalid_msg)
            response["success"] = False
            response["errorMessage"] = invalid_msg
            await update_doc_error_status(Id, invalid_msg)
            return response
        
        if oriFileType == convertTo:
            response['filePath'] = filePath
            return response
        try:
            if oriFileType.isupper():
                filePath = lowercase_file_ext(filePath)
            if fileName.endswith(('.docx','.DOCX')):
                await sync_to_async(docx_preprocess)(filePath)
            await sync_to_async(libreOffice_convert_to_pdf)(outputFolder, filePath, fileExtension=convertTo)
            response['filePath'] = outFilePath
            return response
        except Exception as e:
            invalid_msg = f"Error occurred while converting document {filePath} to {os.path.join(outputFolder, newFilename)} as follows:\n" + str(e)
            logger.error(invalid_msg)
            response['success'] = False
            response['errorMessage'] = invalid_msg
            await update_doc_error_status(Id, invalid_msg)
            return response
    
    try:
        request = json.loads(request.body.decode("utf-8"))
    except:
        pass
    
    flag = True
    if isinstance(request, list):
        content = []
        for r in request:
            response = await request_doc2pdf(r)
            if not response.get('success'):
                flag = False
            content.append(response)
    elif isinstance(request, dict):
        content = await request_doc2pdf(request)
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