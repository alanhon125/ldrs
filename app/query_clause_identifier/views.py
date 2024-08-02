from rest_framework import status
from rest_framework.response import Response
from rest_framework.renderers import JSONRenderer
from rest_framework.decorators import authentication_classes, permission_classes
from asgiref.sync import sync_to_async
from adrf.views import APIView

from config import QUERY_FA_URL, QUERY_TASK_URL
from app.utils import *
from app.query_clause_identifier.query_clause_identifier import get_text_identifier_by_bbox
from app.query_clause_identifier.models import QueryClauseId
from app.query_clause_identifier.serializers import QueryClauseIdSerializer 

from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi

sample_dict = {
                'id': 6290, # Term matching record ID
                'taskId': 2, # task ID
                'pageId': 30, # FA document page ID starting from 1
                'bbox': [140.14999389648438, 230.8298797607422, 522.761962890625, 257.3768310546875], # bounding box of text in [x0,y0,x1,y1]
                'width': 595, # width of page of FA document
                'height': 841, # height of page of FA document
                'manualContent': 'the difference between 100% (one hundred per cent.)' # highlighted text
                }

request_schema_dict = {
                        'id': openapi.Schema(
                            type=openapi.TYPE_INTEGER,
                            description='Term matching record ID',
                            default=int(sample_dict['id'])
                        ),
                        'taskId': openapi.Schema(
                            type=openapi.TYPE_INTEGER,
                            description='Task ID for query FA',
                            default=sample_dict['taskId']
                        ),
                        'pageId': openapi.Schema(
                            type=openapi.TYPE_INTEGER,
                            description='Page ID of selected content in FA',
                            default=sample_dict['pageId']
                        ),
                        'bbox': openapi.Schema(
                            type=openapi.TYPE_ARRAY,
                            items=openapi.Items(type=openapi.TYPE_NUMBER),
                            description='Bounding box of the selected FA content',
                            default=sample_dict['bbox']
                        ),
                        'width': openapi.Schema(
                            type=openapi.TYPE_INTEGER,
                            description='Width of page of FA document',
                            default=sample_dict['width']
                        ),
                        'height': openapi.Schema(
                            type=openapi.TYPE_INTEGER,
                            description='Height of page of FA document',
                            default=sample_dict['height']
                        ),
                        'manualContent': openapi.Schema(
                            type=openapi.TYPE_STRING,
                            description='Selected FA content by user on GUI',
                            default=sample_dict['manualContent']
                        ),
                    }

response_schema_dict = {
    "200": openapi.Response(
        description="Success",
        examples={
            "application/json": {
                "success": True,
                "id": sample_dict['id'],
                "identifier": "Cl_8.1(a)(ii)",
                'manualContent': 'the difference between 100% (one hundred per cent.)', # highlighted text
                'textContent': 'the difference between 100% (one hundred per cent.) and the Interest Rate Adjustment', # complete text content of FA 
            }
        }
    ),
    "400": openapi.Response(
        description="Error: Bad Request",
        examples={
            "application/json": {
                "success": False,
                "id": sample_dict['id'],
                "identifier": None,
                'manualContent': 'the difference between 100% (one hundred per cent.)', # highlighted text
                "textContent": None,
                "errorMessage": f"Fail to query clause and identifier from FA document. Please check if the manual highlighted text is correct."
            }
        }
    )
    }

class QueryClauseIdView(APIView):
    queryset = QueryClauseId.objects.all()
    serializer_class = QueryClauseIdSerializer

    @authentication_classes([])
    @permission_classes([])
    @swagger_auto_schema(
     operation_summary='query clause and its FA identifier',
     operation_description='query clause and FA identifier by text bounding box and page ID of selected FA content.',
     request_body=openapi.Schema(
        type=openapi.TYPE_OBJECT,
        properties=request_schema_dict
        ),
    responses=response_schema_dict
    )
    async def post(self, request, *args, **krgs):
        
        tmp_request = {}
        tmp_request['pageNum'] = 1
        tmp_request['pageSize'] = 99999
        tmp_request['id'] = request.data['taskId']
        
        task_data = await POST(QUERY_TASK_URL, tmp_request)
        faFileId = task_data["result"]["records"][0]["faFileId"]
        
        request2 = {
                    "fileId": faFileId,
                    "pageSize":99999
                    }
        fa_data = await POST(QUERY_FA_URL, request2)
        fa_data = fa_data["result"]["records"]
        
        data = []
        for d in fa_data:
            dic = {}
            for k, v in d.items():
                if isinstance(v, list):
                    dic[k] = str(v)
                else:
                    dic[k] = v
            data.append(dic)
 
        tmp_response = await sync_to_async(get_text_identifier_by_bbox)(fa_data, request.data['manualContent'], request.data['pageId'], request.data['bbox'], request.data['width'], request.data['height'])
        
        response = {
            "success": True,
            "id": request.data['id'],
            "identifier": tmp_response['identifier'],
            "manualContent": request.data['manualContent'],
            "textContent": tmp_response['textContent']
            }
        
        if response["success"]:
            response = Response(response, status=status.HTTP_200_OK)
        else:
            response = Response(response, status=status.HTTP_400_BAD_REQUEST)
        
        response.accepted_renderer = JSONRenderer()
        response.accepted_media_type = "application/json"
        response.renderer_context = {}
        response.render()

        return response
