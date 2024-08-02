import json
import os
from copy import deepcopy
import time

from rest_framework import status
from rest_framework.response import Response
from rest_framework.renderers import JSONRenderer
from rest_framework.decorators import authentication_classes, permission_classes
from asgiref.sync import sync_to_async
from adrf.views import APIView

from app.doc2pdf.views import convert_doc2pdf
from app.docparse.views import docparse2json
from app.term_extract.views import docparse_table
from app.doc_process.models import DocProcess
from app.doc_process.serializers import DocProcessSerializer
from config import *
from app.utils import POST, update_doc_error_status

from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi

import logging

logger = logging.getLogger(__name__)

request_schema_dict = {
                        'createUser': openapi.Schema(
                            type=openapi.TYPE_STRING,
                            description='username of task creator',
                            default='alan'
                        ),
                        'updateUser': openapi.Schema(
                            type=openapi.TYPE_STRING,
                            description='username of task updater',
                            default='alan'
                        ),
                        'id': openapi.Schema(
                            type=openapi.TYPE_INTEGER,
                            description='*REQUIRED*. Document ID (created automatically when file uploaded)',
                            default=21
                        ),
                        'fileName': openapi.Schema(
                            type=openapi.TYPE_STRING,
                            description='*REQUIRED*. The filename of document to be converted (if not in pdf format), docparsing and term extracting.',
                            default="1_GL_SYN_FA_mkd_20221215.docx"
                        ),
                        'filePath': openapi.Schema(
                            type=openapi.TYPE_STRING,
                            description='*REQUIRED*. The absolute file path of document to be converted (if not in pdf format), docparsed and term extracted.',
                            default=os.path.join(DOC_DIR,"1_GL_SYN_FA_mkd_20221215.docx")
                        ),
                        'fileType': openapi.Schema(
                            type=openapi.TYPE_STRING,
                            description='*REQUIRED*. Document type, either FA for facility agreement or TS for term sheet',
                            default="FA",
                            enum=['FA','TS']
                        ),
                        'outputFolder': openapi.Schema(
                            type=openapi.TYPE_STRING,
                            description='the desired output folder path',
                            default=PDF_DIR
                        ),
                        'convertTo': openapi.Schema(
                            type=openapi.TYPE_STRING,
                            description='[LibreOffice](https://www.libreoffice.org/discover/libreoffice/) converter is used. It only supports file conversion from .doc to .pdf, .html, .txt, input value must be one of the following: html, pdf, txt, docx',
                            default="pdf",
                            enum=['pdf', 'html', 'txt']
                        )
                    }

def validate_docparse(responses, doc_type):
    
    def validate(item, field_name, dtype, length_limit=None, can_null=True):
        if length_limit is not None and can_null:
            condition = [(isinstance(i[field_name],dtype) and len(i[field_name])<=length_limit) or i[field_name] is None for i in item]
        elif length_limit is not None and not can_null:
            condition = [isinstance(i[field_name],dtype) and len(i[field_name])<=length_limit for i in item]
        elif length_limit is None and can_null:
            condition = [isinstance(i[field_name],dtype) or i[field_name] is None for i in item]
        elif length_limit is None and not can_null:
            condition = [isinstance(i[field_name],dtype) for i in item]

        if all(condition):
            return True, None
        else:
            extra_msg = ''
            indices, values = list(zip(*[(i,item[i][field_name]) for i,v in enumerate(condition) if v is False]))
            if length_limit:
                extra_msg += f' with length limit {length_limit}'
            if can_null:
                extra_msg += ', accept null'
            msg = f'Data field name "{field_name}" required data type to be {dtype}{extra_msg}, but there are values {str(values)} in indices {str(indices)}.'

            return False, msg
    
    validated = {}
    
    # case "int", no length limit, cannot be null
    for fieldname in ["fileId", "indexId", "textBlockId", "phraseId"]:
        validated[fieldname], validated[fieldname + '_msg'] = validate(responses, fieldname, int, length_limit=None, can_null=False)
    
    # case "str", length limit 16, cannot be null
    for fieldname in ["pageId"]:
        validated[fieldname], validated[fieldname + '_msg'] = validate(responses, fieldname, str, length_limit=16, can_null=False)
    
    # case "str", length limit 512, can be null
    for fieldname in ["textElement", "listId"]:
        validated[fieldname], validated[fieldname + '_msg'] = validate(responses, fieldname, str, length_limit=512, can_null=True)
        
    # case "str", no length limit, can be null
    for fieldname in ["sectionContent", "textContent", "parentCaption", "parentList"]:
        validated[fieldname], validated[fieldname + '_msg'] = validate(responses, fieldname, str, length_limit=None, can_null=True)
    
    # case "list", no length limit, can be null
    for fieldname in ["parentListIds"]:
        validated[fieldname], validated[fieldname + '_msg'] = validate(responses, fieldname, list, length_limit=None, can_null=True)
    
    if doc_type=='FA':
        
        # case "str", length limit 512, can be null
        for fieldname in ["sectionId", "subSectionId", "scheduleId", "partId"]:
            validated[fieldname], validated[fieldname + '_msg'] = validate(responses, fieldname, str, length_limit=512, can_null=True)
            
        # case "str", no length limit, can be null
        for fieldname in ["subSection", "schedule", "part", "definition", "identifier"]:
            validated[fieldname], validated[fieldname + '_msg'] = validate(responses, fieldname, str, length_limit=None, can_null=True)
        
        # case "list", length limit 4, can be null
        for fieldname in ["bbox"]:
            validated[fieldname], validated[fieldname + '_msg'] = validate(responses, fieldname, list, length_limit=4, can_null=True)
        
    for k, v in validated.items():
        if '_msg' in k and v is not None:
            return False, v
    
    return True, None

class DocProcessView(APIView):
    queryset = DocProcess.objects.all()
    serializer_class = DocProcessSerializer

    @authentication_classes([])
    @permission_classes([])
    @swagger_auto_schema(
     operation_summary='integrated doc convert, docparse & term extract',
     operation_description='perform doc2pdf, document parsing & key, term and clauses extraction.\nIf all of the above tasks run successfully, automatically insert FA/TS Data records into DB table `tbl_fa_data` or `tbl_ts_data`',
     request_body=openapi.Schema(
        type=openapi.TYPE_OBJECT,
        properties=request_schema_dict
        )
     )
    async def post(self, request, *args, **krgs):
        data = request.data
        response = await docparse_term_extract(data)
        return response

async def docparse_term_extract(request):
    '''
    request = 
    {
        "id": 0,
        "fileName": "5_AF_SYN_TS_mkd_20221213.docx", 
        "filePath": "/home/data/ldrs_analytics/data/doc/TS/5_AF_SYN_TS_mkd_20221213.docx", 
        "fileType": "TS"
    }
    response = 
    [
        {
            "createUser": null,
            "updateUser": null,
            "id": 135,
            "fileId": 0,
            "indexId": 92,
            "textBlockId": 33,
            "pageId": "6",
            "phraseId": 0,
            "sectionContent": "Appendix A",
            "textElement": "section",
            "listId": null,
            "textContent": "Appendix A"
        }, ...
    ]
    '''
    start_time = time.perf_counter()
    all_responses = deepcopy(request)
    
    request1 = await convert_doc2pdf(request)
    request1 = json.loads(request1.content.decode('utf-8'))
    
    if isinstance(request1, list):
        tmp_response1 = [i for i in deepcopy(request1)]
        for i, tmp in enumerate(tmp_response1):
            logger.info(f"processing file: {tmp['fileName']}")
            tmp['PdfFileName'] = tmp.pop('fileName')
            tmp['PdfFilePath'] = tmp.pop('filePath')
            all_responses[i].update(tmp)
    elif isinstance(request1, dict):
        if not request1['success']:
            logger.info(f'response1 [convert_doc2pdf]: \n{json.dumps(request1,indent=4)}')
            await update_doc_error_status(request1["id"], request1["errorMessage"])
            response = Response(request1, status=status.HTTP_400_BAD_REQUEST)
            response.accepted_renderer = JSONRenderer()
            response.accepted_media_type = "application/json"
            response.renderer_context = {}
            response.render()
            return response
        else:
            tmp_response1 = deepcopy(request1)
            logger.info(f"processing file: {tmp_response1['fileName']}")
            tmp_response1['PdfFileName'] = tmp_response1.pop('fileName')
            tmp_response1['PdfFilePath'] = tmp_response1.pop('filePath')
            all_responses.update(tmp_response1)
    del tmp_response1
    
    request2 = await docparse2json(request1)
    request2 = json.loads(request2.content.decode('utf-8'))
    
    if isinstance(request2, list):
        tmp_response2 = [i for i in deepcopy(request2)]
        for i, tmp in enumerate(tmp_response2):
            tmp.pop('result')
            tmp['DocparseFileName'] = tmp.pop('fileName')
            tmp['DocparseFilePath'] = tmp.pop('filePath')
            all_responses[i].update(tmp)
    elif isinstance(request2, dict):
        if not request2['success']:
            logger.info(f'response2 [docparse2json]: \n{json.dumps(request2,indent=4)}')
            await update_doc_error_status(request2["id"], request2["errorMessage"])
            response = Response(request2, status=status.HTTP_400_BAD_REQUEST)
            response.accepted_renderer = JSONRenderer()
            response.accepted_media_type = "application/json"
            response.renderer_context = {}
            response.render()
            return response
        else:
            tmp_response2 = deepcopy(request2)
            tmp_response2['DocparseFileName'] = tmp_response2.pop('fileName')
            tmp_response2['DocparseFilePath'] = tmp_response2.pop('filePath')
            all_responses.update(tmp_response2)
    del tmp_response2
    
    all_response2 = dict()
    if isinstance(all_responses,list):
        for item in all_responses:
            all_response2[item['id']] = {k:v for k, v in item.items() if k != 'id'}
    elif isinstance(all_responses,dict):
        all_response2[all_responses['id']] = {k:v for k, v in all_responses.items() if k != 'id'}
    
    request3 = await docparse_table(request2)
    request3 = json.loads(request3.content.decode('utf-8'))
    
    if isinstance(request3, list):
        responses = []
        for tmp in request3:
            tmp.pop('success')
            responses.append(tmp)
    elif isinstance(request3, dict):
        if not request3['success']:
            logger.info(f'response3 [docparse2table]: \n{json.dumps(request3,indent=4)}')
            await update_doc_error_status(request3["fileId"], request3["errorMessage"])
            response = Response(request3, status=status.HTTP_400_BAD_REQUEST)
            response.accepted_renderer = JSONRenderer()
            response.accepted_media_type = "application/json"
            response.renderer_context = {}
            response.render()
            return response
        else:
            request3.pop('success')
            responses = deepcopy(request3)
    
    fileId2RecordCount = dict()
    fileId2FARecordCount = dict()
    fileId2TSRecordCount = dict()

    if isinstance(responses, list):
        for record in responses:
            fileId = record['fileId']
            fileType = record['fileType']
            if fileId not in fileId2RecordCount:
                fileId2RecordCount[fileId] = 0
            if fileType == 'FA' and fileId not in fileId2FARecordCount:
                fileId2FARecordCount[fileId] = 0
            if fileType == 'TS' and fileId not in fileId2TSRecordCount:
                fileId2TSRecordCount[fileId] = 0
            if fileType == 'FA':
                fileId2FARecordCount[fileId] += 1
            elif fileType == 'TS':
                fileId2TSRecordCount[fileId] += 1
            fileId2RecordCount[fileId] += 1
 
    FAresponses = [i for i in responses if i['fileType']=='FA']    
    TSresponses = [i for i in responses if i['fileType']=='TS']
    FA_ids = list(fileId2FARecordCount.keys())
    TS_ids = list(fileId2TSRecordCount.keys())
    
    batch_id = 1
    final_responses = []
    post_success = True
    for responses, url, ids, doc_type in [(FAresponses,INSERT_FA_URL,FA_ids,'FA'),(TSresponses,INSERT_TS_URL,TS_ids,'TS')]:
        i = 0
        if len(responses)<1:
            continue
        while True:
            # logger.info(f'input to backend {url} with records from {i} to {i+INSERT_DATA_BATCH_SIZE}: \n{json.dumps(responses[i:i+INSERT_DATA_BATCH_SIZE],indent=4)}')
            response_set = responses[i:min(len(responses), i + INSERT_DATA_BATCH_SIZE)]
            is_pass_validation, error_msg = await sync_to_async(validate_docparse)(response_set, doc_type)
            if not is_pass_validation:
                logger.error(f'Term extraction data validation error: \n{json.dumps(error_msg, indent=4)}')
                update_doc_error_status(ids[0], error_msg)
            else:
                logger.info(f'Data validation is successful for data insertion of attributes: {list(response_set[0].keys())}')
            final_response = await POST(url, response_set)
            logger.info(f'Request for Data Insert into DB table of fileId {response_set[0]["fileId"]} with {doc_type} batch {batch_id} and records: [{i}, {min(len(responses), i + INSERT_DATA_BATCH_SIZE)}] has following response:')
            # logger.info(f'{doc_type} batch {batch_id} post with content: ',json.dumps(response_set,indent=4))
            logger.info(f'{doc_type} batch {batch_id} response: {json.dumps(final_response,indent=4)}')
            if final_response and "error" not in final_response:
                r = {'batch_id':batch_id}
                r.update(final_response)
                final_responses.append(r)
            else:
                if final_response:
                    message = 'Encountered ' + str(final_response["status"]) + ' ' + final_response["error"] + ' on ' + final_response["path"]
                    update_doc_error_status(ids[0], message)
                    post_success = False

                file_id_list = list(set([responses[j]["fileId"] for j in range(i,min(len(responses),i+INSERT_DATA_BATCH_SIZE))]))
                r = {
                        'batch_id':batch_id, 
                        "fileId": file_id_list[0] if len(file_id_list)==1 else file_id_list,
                        "fileType": responses[i]["fileType"],
                        "record": [i, min(len(responses),i+INSERT_DATA_BATCH_SIZE)],
                        "success": False
                     }
                final_responses.append(r)
            
            if min(len(responses), i + INSERT_DATA_BATCH_SIZE) == len(responses):
                break
            else:
                i += INSERT_DATA_BATCH_SIZE
                batch_id += 1
        
        if post_success:
            for fileid in ids:
                end_time = time.perf_counter()
                total_time = end_time - start_time
                update_status = {
                    "id": fileid,
                    "fileName": all_response2[int(fileid)]['fileName'],
                    "filePath": all_response2[int(fileid)]['filePath'],
                    "fileType": all_response2[int(fileid)]['fileType'],
                    "fileStatus": "PROCESSED",
                    "processingTime": total_time,
                    "message": None
                }
                await POST(UPDATE_DOC_STATUS_URL, update_status)
    
    response = Response(final_responses, status=status.HTTP_200_OK)
    response.accepted_renderer = JSONRenderer()
    response.accepted_media_type = "application/json"
    response.renderer_context = {}
    response.render()
    return response