import os
import pandas as pd
import shutil

from rest_framework import status
from rest_framework.response import Response
from rest_framework.renderers import JSONRenderer
from rest_framework.decorators import authentication_classes, permission_classes
from asgiref.sync import sync_to_async
from adrf.views import APIView
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi

from config import (
    OUTPUT_ANNOT_DOC_DIR,
    OUTPUT_ANNOT_DOC_TYPE,
    QUERY_DOC_URL,
    QUERY_FA_URL,
    QUERY_TASK_URL,
    QUERY_TERM_MATCH_URL,
    QUERY_DATA_BATCH_SIZE)
from app.utils import *
from app.create_antd_doc.generate_antd_doc import RegenerateWords
from app.create_antd_doc.generate_antd_xlsx import RegenerateXlsx
from app.create_antd_doc.models import CreateAntdDoc
from app.create_antd_doc.serializers import CreateAntdDocSerializer 

import socket
import logging

logger = logging.getLogger(__name__)
host_ip = socket.gethostbyname(socket.gethostname())

sample_dict = {
                'id': 1
                }

request_schema_dict = {
                        'id': openapi.Schema(
                            type=openapi.TYPE_STRING,
                            description='task ID for query term matching results',
                            default=int(sample_dict['id'])
                        ),
                    }

response_schema_dict = {
    "200": openapi.Response(
        description="Success",
        examples={
            "application/json": {
                "success": True,
                "taskId": sample_dict['id'],
                "filename": "<FILENAME>",
                "filepath": "<OUTPUT_PATH>",
                "message": f"match results has been zipped and created to the path <OUTPUT_PATH> at host server IP {host_ip}"
            }
        }
    ),
    "400": openapi.Response(
        description="Error: Bad Request",
        examples={
            "application/json": {
                "success": False,
                "taskId": sample_dict['id'],
                "errorMessage": f"Fail to create term matching annotated document. Please check if the taskId is valid."
            }
        }
    )
    }

class CreateAntdDocView(APIView):
    queryset = CreateAntdDoc.objects.all()
    serializer_class = CreateAntdDocSerializer

    @authentication_classes([])
    @permission_classes([])
    @swagger_auto_schema(
     operation_summary='create MS Words & MS Excel files that annotated term match result on TS content',
     operation_description='query term matching data and FA docparse data to generate annotated TS content with matching result in .docx/.xlsx/.zip format.',
     request_body=openapi.Schema(
        type=openapi.TYPE_OBJECT,
        properties=request_schema_dict
        ),
    responses=response_schema_dict
    )
    async def post(self, request, *args, **krgs):
        'request = {"id": <taskId>}'
        taskId = request.data['id']
        task_data = await POST(QUERY_TASK_URL, request.data)
        faFileId = task_data["result"]["records"][0]["faFileId"]
        tsFileId = task_data["result"]["records"][0]["tsFileId"]
        
        query_fa = await POST(QUERY_DOC_URL, {"id": faFileId})
        query_ts = await POST(QUERY_DOC_URL, {"id": tsFileId})
        
        faFileName = query_fa["result"]["records"][0]["fileName"]
        tsFileName = query_ts["result"]["records"][0]["fileName"]
        fname = tsFileName.split('.')[0]
        dir_name = os.path.abspath(os.path.join(OUTPUT_ANNOT_DOC_DIR, fname))
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        
        output_name1 = fname + '.docx'
        output_path1 = os.path.join(dir_name, output_name1)
        output_name2 = fname + '.xlsx'
        output_path2 = os.path.join(dir_name, output_name2)
                
        all_term_match_data = []
        term_match_page_count = None
        term_match_current_page = 1
        while True:
            if term_match_page_count is not None and term_match_current_page > term_match_page_count:
                break
            term_match_data = await POST(QUERY_TERM_MATCH_URL+ f'?pageNum={str(term_match_current_page)}&pageSize={str(QUERY_DATA_BATCH_SIZE)}&taskId={request.data["id"]}', '')
            if 'success' in term_match_data and term_match_data['success']:
                logger.info(f'Successfully POST Term Match Data query with taskId {request.data["id"]} with batch ID {term_match_current_page} and record ID from {(term_match_current_page-1)*(QUERY_DATA_BATCH_SIZE)} to {(term_match_current_page)*(QUERY_DATA_BATCH_SIZE)}')
                if term_match_page_count is None:
                    term_match_page_count = term_match_data["result"]["pages"]
                all_term_match_data += term_match_data["result"]["records"]
                term_match_current_page += 1
            else:
                logger.error(f'Fail to POST Term Match Data query with taskId {request.data["id"]} with batch ID {term_match_current_page} and record ID from {(term_match_current_page-1)*(QUERY_DATA_BATCH_SIZE)} to {(term_match_current_page)*(QUERY_DATA_BATCH_SIZE)}')

        if OUTPUT_ANNOT_DOC_TYPE in [".zip", ".docx"]:
            await sync_to_async(RegenerateWords(all_term_match_data, tsFileName, faFileName, output_path1).generate)()
            output_path = output_path1
            if not os.path.exists(output_path1):
                logger.error(f"Annotated .docx path {output_path1} doesn't exist. Fail to generate annotated term match result in form of .docx.")
                response = {
                    "success": False,
                    "taskId": request.data['id'],
                    "filename": os.path.basename(output_path),
                    "filepath": output_path,
                    "errorMessage": f"Fail to create annotated docs to the path {output_path} at host server IP {host_ip}"
                    }
            
        if OUTPUT_ANNOT_DOC_TYPE in [".zip", ".xlsx"]:
            writer = pd.ExcelWriter(output_path2, engine='xlsxwriter')
            await sync_to_async(RegenerateXlsx(writer, all_term_match_data, taskId, tsFileName, faFileName, output_path2).generate)()
            # Close the Pandas Excel writer and output the Excel file.
            writer.close()
            output_path = output_path2
            if not os.path.exists(output_path2):
                logger.error(f"Annotated .xlsx path {output_path2} doesn't exist. Fail to generate annotated term match result in form of .xlsx.")
                response = {
                    "success": False,
                    "taskId": request.data['id'],
                    "filename": os.path.basename(output_path),
                    "filepath": output_path,
                    "errorMessage": f"Fail to create annotated docs to the path {output_path} at host server IP {host_ip}"
                    }
        if OUTPUT_ANNOT_DOC_TYPE == ".zip":
            # zip the output docs directory as .zip
            shutil.make_archive(dir_name, 'zip', dir_name)
            output_path = dir_name + '.zip'
            
            if not os.path.exists(output_path):
                logger.error(f"Annotated doc zip filepath {output_path} doesn't exist. Fail to zip all annotated term match docs in form of .zip.")
                response = {
                    "success": False,
                    "taskId": request.data['id'],
                    "filename": os.path.basename(output_path),
                    "filepath": output_path,
                    "errorMessage": f"Fail to zip and create annotated docs to the path {output_path} at host server IP {host_ip}"
                    }
            else:
                # remove the output docs directory
                if os.path.exists(dir_name):
                    shutil.rmtree(dir_name)
        response = {
            "success": True,
            "taskId": request.data['id'],
            "filename": os.path.basename(output_path),
            "filepath": output_path,
            "message": f"match results has been zipped and created to the path {output_path} at host server IP {host_ip}"
            }
        
        if response and "errorMessage" not in response:
            response = Response(response, status=status.HTTP_200_OK)
        else:
            response = Response(response, status=status.HTTP_400_BAD_REQUEST)
            
        response.accepted_renderer = JSONRenderer()
        response.accepted_media_type = "application/json"
        response.renderer_context = {}
        response.render()
        return response
