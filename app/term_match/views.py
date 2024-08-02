import pandas as pd
import json
import traceback
import sys
import time

from rest_framework import status
from rest_framework.response import Response
from rest_framework.renderers import JSONRenderer
from rest_framework.decorators import authentication_classes, permission_classes
from asgiref.sync import sync_to_async
from adrf.views import APIView
from django.core.exceptions import BadRequest
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi

from config import *
from app.utils import *
from app.term_match.term_match import main
from app.term_match.models import TermMatch
from app.term_match.serializers import TermMatchSerializer

import logging

logger = logging.getLogger(__name__)

sample_dict = {
    'createUser': 'alan',
    'updateUser': 'lu',
    'id': 1
}

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
        description='*REQUIRED*. Task ID',
        default=sample_dict['id']
    )
}

response_schema_dict = {
    "200": openapi.Response(
        description="Success",
        examples={
            "application/json": [
                {
                    "id": 0,
                    "taskId": 1,
                    "indexId": 1,
                    "textBlockId": 1,
                    "pageId": "1",
                    "phraseId": 1,
                    "tsTerm": "string",
                    "tsText": "string",
                    "textElement": "string",
                    "listId": "string",
                    "matchTermList": [
                        "string",
                        "string",
                        "string",
                        "string",
                        "string"
                    ],
                    "identifierList": [
                        "string",
                        "string",
                        "string",
                        "string",
                        "string"
                    ],
                    "similarityList": [
                        0,
                        0,
                        0,
                        0,
                        0
                    ],
                    "matchTypeList": [
                        "string",
                        "string",
                        "string",
                        "string",
                        "string"
                    ],
                    "result": [
                        False,
                        False,
                        False,
                        False,
                        False
                    ]
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
            "errorMessage": f"<ERROR_MESSAGE_FROM_TERM_EXTRACTION>"
            }
        }
    )
    }

def wrap_response(content, success=True):
    if success:
        response = Response(content, status=status.HTTP_200_OK)
    else:
        response = Response(content, status=status.HTTP_200_OK)
        
    response.accepted_renderer = JSONRenderer()
    response.accepted_media_type = "application/json"
    response.renderer_context = {}
    response.render()
    return response


async def termmatch_table(request):
    '''
    Request Body:
    {
        "ts_data": JSON,
        "fa_data": JSON,
        "task_id": integer,
    }

    Response Body:
    [
        {
            "id": 0,
            "taskId": 1,
            "indexId": 1,
            "textBlockId": 1,
            "pageId": "1",
            "phraseId": 1,
            "tsTerm": "string",
            "tsText": "string",
            "textElement": "string",
            "listId": "string",
            "matchTermList": [
                "string",
                "string",
                "string",
                "string",
                "string"
            ],
            "identifierList": [
                "string",
                "string",
                "string",
                "string",
                "string"
            ],
            "similarityList": [
                0,
                0,
                0,
                0,
                0
            ],
            "matchTypeList": [
                "string",
                "string",
                "string",
                "string",
                "string"
            ],
            "result": [
                true,
                true,
                true,
                true,
                true
            ]
        }, ...
    ]

    '''
    
    async def request_term_match(request):        
        ts_data = request.get('ts_data')
        fa_data = request.get('fa_data')
        task_id = request.get('task_id')
        response = {
            "success": True
        }
        try:
            colname_map = {
                'indexId': 'index',
                'textBlockId': 'text_block_id',
                'pageId': 'page_id',
                'phraseId': 'phrase_id',
                'sectionId': 'section_id',
                'sectionContent': 'section',
                'subSectionId': 'sub_section_id',
                'subSection': 'sub_section',
                'scheduleId': 'schedule_id',
                'partId': 'part_id',
                'textElement': 'text_element',
                'listId': 'list_id',
                'textContent': 'text'
            }
            df_ts = pd.DataFrame(data=ts_data)
            df_ts = df_ts.rename(columns=colname_map)
            df_fa = pd.DataFrame(data=fa_data)
            df_fa = df_fa.rename(columns=colname_map)
            logger.info("Start to match terms:")
            df = await sync_to_async(main)(df_ts, df_fa, topN=TOP_N_RESULTS, task_id=task_id)
            term_match_col_map = {
                'TS_term': 'tsTerm',
                'TS_text': 'tsText',
                'identifier_list': 'identifierList',
                'similarity_list': 'similarityList',
                'match_type_list': 'matchTypeList',
                'match_term_list': 'matchTermList',
                'index': 'indexId',
                'text_block_id':'textBlockId',
                'page_id':'pageId',
                'phrase_id': 'phraseId',
                'text_element': 'textElement',
                'list_id': 'listId'              
            }
            df = df.rename(columns=term_match_col_map)
            result = df.to_dict(orient='records')
            i = 0
            responses = []
            while i <= len(result)-1:
                r = result[i]
                r.update(response.copy())
                r['id'] = i
                r['taskId'] = task_id
                r['result'] = [False] * TOP_N_RESULTS if r['identifierList'] else None # by default unselected all matched results in GUI
                responses.append(r)
                i += 1
            return responses
        except Exception as e:
            exc_type, exc_value, exc_tb = sys.exc_info()
            tb = traceback.TracebackException(exc_type, exc_value, exc_tb)
            message = 'In task name: "term match"\n'+''.join(tb.format())
            response['success'] = False
            response['errorMessage'] = message
            return response
        
    try:
        request = json.loads(request.body.decode("utf-8"))
    except:
        pass
    
    flag = True
    content = []
    if isinstance(request, list):
        # TODO: add multi-process in term match
        pass
    elif isinstance(request, dict):
        content = await request_term_match(request)
    
    if isinstance(content, list):
        if any([not i.get('success') for i in content]):
            flag = False
    elif isinstance(content, dict):
        flag = False

    if flag:
        response = Response(content, status=status.HTTP_200_OK)
    else:
        response = Response(content, status=status.HTTP_200_OK) # HTTP_400_BAD_REQUEST
        
    response.accepted_renderer = JSONRenderer()
    response.accepted_media_type = "application/json"
    response.renderer_context = {}
    response.render()
    return response

def validate_term_match(responses):
    
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
    for fieldname in ["taskId", "indexId", "textBlockId", "phraseId"]:
        validated[fieldname], validated[fieldname + '_msg'] = validate(responses, fieldname, int, length_limit=None, can_null=False)
    
    # case "str", length limit 512, cannot be null
    for fieldname in ["pageId"]:
        validated[fieldname], validated[fieldname + '_msg'] = validate(responses, fieldname, str, length_limit=512, can_null=False)
    
    # case "str", length limit 512, can be null
    for fieldname in ["tsTerm"]:
        validated[fieldname], validated[fieldname + '_msg'] = validate(responses, fieldname, str, length_limit=512, can_null=True)
    
    # case "str", no length limit, can be null
    for fieldname in ["tsText"]:
        validated[fieldname], validated[fieldname + '_msg'] = validate(responses, fieldname, str, length_limit=None, can_null=True)
    
    # case "list", length limit = TopN, can be null
    for fieldname in ["matchTermList", "identifierList", "similarityList", "matchTypeList", "result"]:
        validated[fieldname], validated[fieldname + '_msg'] = validate(responses, fieldname, list, length_limit=TOP_N_RESULTS, can_null=True)
        
    for k, v in validated.items():
        if '_msg' in k and v is not None:
            return False, v
    
    return True, None

class TermMatchView(APIView):
    queryset = TermMatch.objects.all()
    serializer_class = TermMatchSerializer
    @authentication_classes([])
    @permission_classes([])
    @swagger_auto_schema(
     operation_summary='term match for parsed TS file and FA file',
     operation_description='perform term matching for TS and FA pair.',
     request_body=openapi.Schema(
        type=openapi.TYPE_OBJECT,
        properties=request_schema_dict
        ),
     responses=response_schema_dict
    )
    async def post(self, request, *args, **krgs):
        start_time = time.perf_counter()
        request_body = request.data
        
        try:
            taskId = request_body.get("id")
        except:
            raise BadRequest(f'id is compulsory field name. Please check if task id is provided in the request')
        try:
            createUser = request_body.get("createUser")
        except:
            createUser = None
        try:
            updateUser = request_body.get("updateUser")
        except:
            updateUser = None
        
        task_data = await POST(QUERY_TASK_URL, {"id": taskId})
        if 'success' in task_data and task_data['success']:
            logger.info(f'Successfully POST task query with id {taskId}')
        task_data = task_data["result"]["records"][0]        
        faFileId = task_data["faFileId"]
        tsFileId = task_data["tsFileId"]
        makerGroup = task_data["makerGroup"]
        taskName = task_data["name"]
        
        logger.info(f'It is going to query TS & FA with record batch size = {QUERY_DATA_BATCH_SIZE} records per batch')
        
        all_ts_data = []
        ts_page_count = None
        ts_current_page = 1
        while True:
            if ts_page_count == 0 or (ts_page_count is not None and ts_current_page > ts_page_count):
                break
            ts_data = await POST(QUERY_TS_URL, {"fileId": tsFileId, "pageSize": QUERY_DATA_BATCH_SIZE, "pageNum": ts_current_page})
            if 'success' in ts_data and ts_data['success'] and 'result' in ts_data and ts_data["result"]["records"]:
                logger.info(f'Successfully POST TS Data query with TS fileid {tsFileId} with batch ID {ts_current_page} and record ID from {(ts_current_page-1)*(QUERY_DATA_BATCH_SIZE)} to {(ts_current_page)*(QUERY_DATA_BATCH_SIZE)}')
                if ts_page_count is None:
                    ts_page_count = ts_data["result"]["pages"]
                all_ts_data += ts_data["result"]["records"]
                ts_current_page += 1
            else:
                logger.error(f'Fail to POST TS Data query with TS fileid {tsFileId} with batch ID {ts_current_page} and record ID from {(ts_current_page-1)*(QUERY_DATA_BATCH_SIZE)} to {(ts_current_page)*(QUERY_DATA_BATCH_SIZE)}')
                if 'result' in ts_data and ts_data["result"]["pages"] == 0:
                    break

        all_fa_data = []
        fa_page_count = None
        fa_current_page = 1
        while True:
            if fa_page_count == 0 or (fa_page_count is not None and fa_current_page > fa_page_count):
                break
            fa_data = await POST(QUERY_FA_URL, {"fileId": faFileId, "pageSize": QUERY_DATA_BATCH_SIZE, "pageNum": fa_current_page})
            if 'success' in fa_data and fa_data['success'] and 'result' in fa_data and fa_data["result"]["records"]:
                logger.info(f'Successfully POST FA Data query with FA fileid {faFileId} with batch ID {fa_current_page} and record ID from {(fa_current_page-1)*(QUERY_DATA_BATCH_SIZE)} to {(fa_current_page)*(QUERY_DATA_BATCH_SIZE)}')
                if fa_page_count is None:
                    fa_page_count = fa_data["result"]["pages"]
                all_fa_data += fa_data["result"]["records"]
                fa_current_page += 1
            else:
                logger.error(f'Fail to POST FA Data query with FA fileid {faFileId} with batch ID {fa_current_page} and record ID from {(fa_current_page-1)*(QUERY_DATA_BATCH_SIZE)} to {(fa_current_page)*(QUERY_DATA_BATCH_SIZE)}')
                if 'result' in fa_data and fa_data["result"]["pages"] == 0:
                    break

        query_fa = await POST(QUERY_DOC_URL, {"id": faFileId})
        if 'success' in query_fa and query_fa['success']:
            logger.info(f'Successfully POST FA document query with FA fileid {faFileId}')

        query_ts = await POST(QUERY_DOC_URL, {"id": tsFileId})
        if 'success' in query_ts and query_ts['success']:
            logger.info(f'Successfully POST TS document query with TS fileid {tsFileId}')
        
        faFileName = query_fa["result"]["records"][0]["fileName"]
        tsFileName = query_ts["result"]["records"][0]["fileName"]
        
        data = {
            "ts_data": all_ts_data,
            "fa_data": all_fa_data,
            "task_id": taskId
        }
        logger.info(f'Term matching on {len(all_ts_data)} sentences in TS filename: {tsFileName} with {len(all_fa_data)} clauses in FA filename: {faFileName} for taskId {taskId}.')
        response = await termmatch_table(data)
        term_match_json_data = json.loads(response.content)
        if isinstance(term_match_json_data, dict) and not term_match_json_data["success"]:
            await update_task_error_status(taskId, makerGroup, taskName, term_match_json_data["errorMessage"])
            response = wrap_response(term_match_json_data, success=False)
            return response
        else:
            logger.info('*** Term matching data is generated successfully ***')
            # logger.info(f'Term matching data for taskId {taskId} with TS filename {tsFileName} & FA filename {faFileName}: \n{json.dumps(term_match_json_data, indent=4)}')
            is_pass_validation, error_msg = await sync_to_async(validate_term_match)(term_match_json_data)
            if not is_pass_validation:
                insert_response = {"success": False}
                await update_task_error_status(taskId, makerGroup, taskName, error_msg)
                insert_response['errorMessage'] = error_msg
                response = wrap_response(insert_response, success=False)
                logger.error(f'Term matching data validation error: \n{json.dumps(insert_response, indent=4)}')
                return response
            insert_response = await POST(INSERT_TERM_MATCH_URL, term_match_json_data)
            if insert_response and "error" in insert_response:
                logger.error(f'Term matching data insertion error: \n{json.dumps(insert_response, indent=4)}')
                invalid_msg = f'Term matching result doesn\'t be inserted succesfully for taskId {taskId}. Please check if the data is valid.'
                await update_task_error_status(taskId, makerGroup, taskName, invalid_msg)
                insert_response['errorMessage'] = invalid_msg
                response = wrap_response(insert_response, success=False)
                return response
            else:
                logger.info(f'Term matching data insertion response: \n{json.dumps(insert_response, indent=4)}')
                logger.info('*** Term matching data is saved to table "tbl_term_matching_data" ***')
                end_time = time.perf_counter()
                total_time = end_time - start_time
                update_status = {
                    "id": taskId,
                    "makerGroup": makerGroup,
                    "name": taskName,
                    "status": "CREATING",
                    "systemStatus": "PROCESSED",
                    "processingTime": total_time,
                    "message": None
                }
                await POST(UPDATE_TASK_URL, update_status)
                logger.info('*** Status for task ID {} is updated to DB***'.format(taskId))
                
        return response
        