import pandas as pd
import re
import asyncio
from datetime import datetime
from rest_framework import status
from rest_framework.response import Response
from rest_framework.renderers import JSONRenderer
from rest_framework.decorators import authentication_classes, permission_classes
from adrf.views import APIView
from django.core.exceptions import BadRequest
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi
import os

from app.get_sent_pairs.models import GetSentPairs
from app.get_sent_pairs.serializers import GetSentPairsSerializer
from config import QUERY_TERM_MATCH_BY_DATE_URL, QUERY_DATA_BATCH_SIZE
from app.utils import POST

import logging

logger = logging.getLogger(__name__)

sample_dict = {
    'updateStartDate': '2024-03-26',
    'updateEndDate': '2024-03-28'
}

request_schema_dict = {
    'updateStartDate': openapi.Schema(
        type=openapi.TYPE_STRING,
        description='*REQUIRED*. updateStartDate, format: yyyy-mm-dd',
        default=sample_dict['updateStartDate']
    ),
    'updateEndDate': openapi.Schema(
        type=openapi.TYPE_STRING,
        description='*REQUIRED*. updateEndDate, format: yyyy-mm-dd',
        default=sample_dict['updateEndDate']
    )
}

response_schema_dict = {
    "200": openapi.Response(
        description="Success",
        examples={
            "application/json": [
                {'updateTime': '2024-03-27T19:26:53.000',
                 'id': 17469,
                 'taskId': 1,
                 'indexId': 3,
                 'textBlockId': 11,
                 'pageId': '2',
                 'phraseId': 0,
                 'tsTerm': 'Borrower',
                 'tsText': 'Borrower:',
                 'identifier': 'Sched_12【BANKING (EXPOSURE LIMITS) RULES】',
                 'faSection': 'BANKING (EXPOSURE LIMITS) RULES',
                 'faSubSection': None,
                 'faText': '[BORROWER]',
                 'isManualContent': False,
                 'label': 'entailment'},
                {'updateTime': '2024-03-27T19:26:53.000',
                 'id': 17469,
                 'taskId': 1,
                 'indexId': 3,
                 'textBlockId': 11,
                 'pageId': '2',
                 'phraseId': 0,
                 'tsTerm': 'Borrower',
                 'tsText': 'Borrower:',
                 'identifier': 'Cl_1.1-Borrowings_(a)【DEFINITIONS AND INTERPRETATION - Definitions】',
                 'faSection': 'DEFINITIONS AND INTERPRETATION',
                 'faSubSection': 'Definitions',
                 'faText': 'moneys borrowed and debit balances at banks or other financial institutions;',
                 'isManualContent': False,
                 'label': 'contradiction'}
            ]
        }
    ),
    "400": openapi.Response(
        description="Error: Bad Request",
        examples={
            "application/json": {
                "success": False,
                "id": 1,
                "errorMessage": f"<ERROR_MESSAGE_FROM_ANALYTICS>"
            }
        }
    )
}


class GetSentPairsView(APIView):
    queryset = GetSentPairs.objects.all()
    serializer_class = GetSentPairsSerializer

    @authentication_classes([])
    @permission_classes([])
    @swagger_auto_schema(
        operation_summary='generate sentence-pairs (tsText-faText) from user feedback on term matching results',
        operation_description='After user made their feedback on term matching results, it used to pairs up tsText, faText based on result judgement and manually-added content, query by updated start date and updated end date.',
        request_body=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties=request_schema_dict
        ),
        responses=response_schema_dict
    )
    async def post(self, request, *args, **krgs):
        data = request.data
        response = await generate_term_match_sent_pairs(data)
        return response


def flatten_extend(matrix):
    flat_list = []
    for row in matrix:
        flat_list.extend(row)
    return flat_list


async def generate_term_match_sent_pairs(request):
    # try:
    #     taskId = request.get("taskId")
    # except:
    #     raise BadRequest(f'taskId is compulsory field name. Please check if task id is provided in the request')
    try:
        updateStartDate = request.get("updateStartDate")
    except:
        raise BadRequest(
            f'updateStartDate is compulsory field name. Please check if updateStartDate is provided in the request')
    try:
        updateEndDate = request.get("updateEndDate")
    except:
        raise BadRequest(
            f'updateEndDate is compulsory field name. Please check if updateEndDate is provided in the request')
    # updateStartDate = datetime.strptime(updateStartDate, '%Y-%m-%d')
    # updateEndDate = datetime.strptime(updateEndDate, '%Y-%m-%d')

    all_term_match_data = []
    tm_page_count = None
    tm_current_page = 1
    while True:
        if tm_page_count == 0 or (tm_page_count is not None and tm_current_page > tm_page_count):
            break
        term_match_data = POST(
            QUERY_TERM_MATCH_BY_DATE_URL + f'?pageNum={str(tm_current_page)}&pageSize={str(QUERY_DATA_BATCH_SIZE)}&startDate={str(updateStartDate)}&endDate={str(updateEndDate)}',
            '')
        if 'success' in term_match_data and term_match_data['success'] and 'result' in term_match_data and \
                term_match_data["result"]["records"]:
            logger.info(
                f'Successfully POST Term Matching Data query between start date {str(updateStartDate)} and end date {str(updateEndDate)} with batch ID {tm_current_page} and record ID from {(tm_current_page - 1) * (QUERY_DATA_BATCH_SIZE)} to {(tm_current_page) * (QUERY_DATA_BATCH_SIZE)}')
            if tm_page_count is None:
                tm_page_count = term_match_data["result"]["pages"]
            all_term_match_data.append(term_match_data["result"]["records"])
            tm_current_page += 1
        else:
            logger.error(
                f'Fail to POST Term Matching Data query between start date {str(updateStartDate)} and end date {str(updateEndDate)} with batch ID {tm_current_page} and record ID from {(tm_current_page - 1) * (QUERY_DATA_BATCH_SIZE)} to {(tm_current_page) * (QUERY_DATA_BATCH_SIZE)}')
            if 'result' in term_match_data and term_match_data["result"]["pages"] == 0:
                break
    all_term_match_data = flatten_extend(all_term_match_data)

    results = []
    target_fields = ["updateTime",
                     "id",
                     "taskId",
                     "indexId",
                     "textBlockId",
                     "pageId",
                     "phraseId",
                     "tsTerm",
                     "tsText"]
    for record in all_term_match_data:
        updateTime = datetime.strptime(record['updateTime'].split('.')[0], '%Y-%m-%dT%H:%M:%S')
        # if updateStartDate < updateTime and updateEndDate > updateTime:
        del record['createTime'], record['createUser'], record['updateUser'], record['textElement'], record[
            'similarityList'], record['matchTypeList'], record['textContent']
        # when term match result is provided
        if record["matchTermList"]:
            # if result is None, then indicate the top1 result is selected by default
            if record['result'] is None:
                tmp = {key: record[key] for key in record.keys() if key in target_fields}
                try:
                    identifier = re.findall(r'【(.*)】', record['identifierList'][0])[0]
                except:
                    identifier = None
                tmp['identifier'] = record['identifierList'][0]
                tmp['faSection'] = identifier.split(' - ')[0] if identifier else None
                tmp['faSubSection'] = identifier.split(' - ')[1] if identifier and len(
                    identifier.split(' - ')) > 1 else None
                tmp['faText'] = record['matchTermList'][0]
                tmp['isManualContent'] = False
                tmp['label'] = 'entailment'
                results.append(tmp)
            # else iterate the list of result judgement and locate the true result
            else:
                for index, judge in enumerate(record['result']):
                    tmp = {key: record[key] for key in record.keys() if key in target_fields}
                    try:
                        identifier = re.findall(r'【(.*)】', record['identifierList'][index])[0]
                    except:
                        identifier = None
                    tmp['identifier'] = record['identifierList'][index]
                    tmp['faSection'] = identifier.split(' - ')[0] if identifier else None
                    tmp['faSubSection'] = identifier.split(' - ')[1] if identifier and len(
                        identifier.split(' - ')) > 1 else None
                    tmp['faText'] = record['matchTermList'][index]
                    tmp['isManualContent'] = False

                    if judge:
                        tmp['label'] = 'entailment'
                    else:
                        tmp['label'] = 'contradiction'

                    results.append(tmp)
        # when manually-added result is provided
        if record["manualContent"]:
            for index, manualContent in enumerate(record['manualContent']):
                tmp = {key: record[key] for key in record.keys() if key in target_fields}
                try:
                    identifier = re.findall(r'【(.*)】', record['manualIdentifier'][index])[0]
                except:
                    identifier = None
                tmp['identifier'] = record['identifierList'][index]
                tmp['faSection'] = identifier.split(' - ')[0] if identifier else None
                tmp['faSubSection'] = identifier.split(' - ')[1] if identifier and len(
                    identifier.split(' - ')) > 1 else None
                tmp['faText'] = manualContent
                tmp['isManualContent'] = True
                tmp['label'] = 'entailment'
                results.append(tmp)

    OUTPUT_SENTPAIR_CSV = 'data/sentence_pairs/'
    if not os.path.exists(OUTPUT_SENTPAIR_CSV):
        os.mkdir(OUTPUT_SENTPAIR_CSV)
    filename = f'sentence_pairs_{str(updateStartDate)}_{str(updateEndDate)}.csv'
    df_sent_pair = pd.DataFrame(data=results)
    df_sent_pair.to_csv(os.path.join(OUTPUT_SENTPAIR_CSV, filename), index=False, encoding='utf-8-sig')

    response = Response(results, status=status.HTTP_200_OK)
    response.accepted_renderer = JSONRenderer()
    response.accepted_media_type = "application/json"
    response.renderer_context = {}
    response.render()
    return response


if __name__ == "__main__":
    BACKEND_ADDR = '10.6.55.12:8080'
    QUERY_TERM_MATCH_BY_DATE_URL = f"http://{BACKEND_ADDR}/ldrs/dataStorage/queryTermMatchDataByUpdateDateRange"
    QUERY_DATA_BATCH_SIZE = 1500
    updateStartDate = '2024-03-26'
    updateEndDate = '2024-03-28'

    results = asyncio.run(
        generate_term_match_sent_pairs({'updateStartDate': updateStartDate, 'updateEndDate': updateEndDate}))
    df = pd.DataFrame(results.data)