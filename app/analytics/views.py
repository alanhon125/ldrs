from rest_framework import status
from rest_framework.response import Response
from rest_framework.renderers import JSONRenderer
from rest_framework.decorators import authentication_classes, permission_classes
from adrf.views import APIView
from django.core.exceptions import BadRequest
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi

from app.analytics.models import Analytics
from app.analytics.serializers import AnalyticsSerializer
from config import QUERY_TERM_MATCH_URL, INSERT_ANALYTICS_URL, QUERY_DATA_BATCH_SIZE, TOP_N_RESULTS
from app.utils import POST

import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

sample_dict = {
                'taskId': 1,
                }

request_schema_dict = {
                        'taskId': openapi.Schema(
                            type=openapi.TYPE_INTEGER,
                            description='*REQUIRED*. Task ID',
                            default=sample_dict['taskId']
                        )
                    }

response_schema_dict = {
    "200": openapi.Response(
        description="Success",
        examples={
            "application/json": {
                    "taskId": sample_dict['taskId'],
                    "avgAccuracy": 64.71,
                    "termCount": 2,
                    "termContentCount": 10,
                    "irrelavantContentCount": 3,
                    "relavantInstanceRate": 70,
                    "hitCount": 11,
                    "missCount": 6,
                    "topMissTsTerm": "Interest Rate",
                    "hitInstanceCount": 7,
                    "avgHitperHitInstance": 1.57,
                    "manualCount": 15,
                    "manualInstanceCount": 6,
                    "manualInstanceRate": 85.71,
                    "missRate": 88.24,
                    "definitionClauseHitCount": 2,
                    "partiesClauseHitCount": 3,
                    "clauseHitCount": 5,
                    "scheduleHitCount": 1,
                    "definitionClauseMissCount": 7,
                    "partiesClauseMissCount": 2,
                    "clauseMissCount": 5,
                    "scheduleMissCount": 1,
                    "top1HitCount": 3,
                    "top1HitRate": 42.86,
                    "upToTop1HitCount": 11,
                    "upToTop1HitRate": 100,
                    "top2HitCount": 4,
                    "top2HitRate": 57.14,
                    "upToTop2HitCount": 8,
                    "upToTop2HitRate": 72.73,
                    "top3HitCount": 1,
                    "top3HitRate": 14.29,
                    "upToTop3HitCount": 4,
                    "upToTop3HitRate": 36.36,
                    "top4HitCount": 2,
                    "top4HitRate": 28.57,
                    "upToTop4HitCount": 3,
                    "upToTop4HitRate": 27.27,
                    "top5HitCount": 1,
                    "top5HitRate": 14.29,
                    "upToTop5HitCount": 1,
                    "upToTop5HitRate": 9.09
                }
        }
    ),
    "400": openapi.Response(
        description="Error: Bad Request",
        examples={
            "application/json": {
            "success": False,
            "id": sample_dict['taskId'],
            "errorMessage": f"<ERROR_MESSAGE_FROM_ANALYTICS>"
            }
        }
    )
    }

class AnalyticsView(APIView):
    queryset = Analytics.objects.all()
    serializer_class = AnalyticsSerializer
    @authentication_classes([])
    @permission_classes([])
    @swagger_auto_schema(
     operation_summary='analyzing the term matching performance after user feedback',
     operation_description='After user made their feedback on term matching results, it performs hit rate analysis by providing task ID to query user feedback.',
     request_body=openapi.Schema(
        type=openapi.TYPE_OBJECT,
        properties=request_schema_dict
        ),
     responses=response_schema_dict
     )
    async def post(self, request, *args, **krgs):
        data = request.data
        response = await analyze(data)
        return response

def id2idtype(identifier):
    import re
    if identifier is None:
        return None
    elif re.match('^Cl_.*-.*', identifier):
        return 'definitionClause'
    elif re.match('^Parties.*', identifier):
        return 'partiesClause'
    elif re.match('^Cl_.*', identifier):
        return 'clause'
    elif re.match('^Sched_.*', identifier):
        return 'schedule'

async def analyze(request):
    '''
    request = 
    {
        "taskId": 1
    }
    response = 
    {
        "taskId": 0,
        "avgAccuracy": 64.71,
        "termCount": 2,
        "termContentCount": 10,
        "irrelavantContentCount": 3,
        "relavantInstanceRate": 70,
        "hitCount": 11,
        "missCount": 6,
        "topMissTsTerm": "Interest Rate",
        "hitInstanceCount": 7,
        "avgHitperHitInstance": 1.57,
        "manualCount": 15,
        "manualInstanceCount": 6,
        "manualInstanceRate": 85.71,
        "missRate": 88.24,
        "definitionClauseHitCount": 2,
        "partiesClauseHitCount": 3,
        "clauseHitCount": 5,
        "scheduleHitCount": 1,
        "definitionClauseMissCount": 7,
        "partiesClauseMissCount": 2,
        "clauseMissCount": 5,
        "scheduleMissCount": 1,
        "top1HitCount": 3,
        "top1HitRate": 42.86,
        "upToTop1HitCount": 11,
        "upToTop1HitRate": 100,
        "top2HitCount": 4,
        "top2HitRate": 57.14,
        "upToTop2HitCount": 8,
        "upToTop2HitRate": 72.73,
        "top3HitCount": 1,
        "top3HitRate": 14.29,
        "upToTop3HitCount": 4,
        "upToTop3HitRate": 36.36,
        "top4HitCount": 2,
        "top4HitRate": 28.57,
        "upToTop4HitCount": 3,
        "upToTop4HitRate": 27.27,
        "top5HitCount": 1,
        "top5HitRate": 14.29,
        "upToTop5HitCount": 1,
        "upToTop5HitRate": 9.09
    }
    '''
        
    try:
        taskId = request.get("taskId")
    except:
        raise BadRequest(f'taskId is compulsory field name. Please check if task id is provided in the request')
    
    all_term_match_data = []
    tm_page_count = None
    tm_current_page = 1
    while True:
        if tm_page_count == 0 or (tm_page_count is not None and tm_current_page > tm_page_count):
            break
        term_match_data = await POST(QUERY_TERM_MATCH_URL+ f'?pageNum={str(tm_current_page)}&pageSize={str(QUERY_DATA_BATCH_SIZE)}&taskId={str(taskId)}', '')
        if 'success' in term_match_data and term_match_data['success'] and 'result' in term_match_data and term_match_data["result"]["records"]:
            logger.info(f'Successfully POST Term Matching Data query with taskId {taskId} with batch ID {tm_current_page} and record ID from {(tm_current_page-1)*(QUERY_DATA_BATCH_SIZE)} to {(tm_current_page)*(QUERY_DATA_BATCH_SIZE)}')
            if tm_page_count is None:
                tm_page_count = term_match_data["result"]["pages"]
            all_term_match_data.append(term_match_data["result"]["records"])
            tm_current_page += 1
        else:
            logger.error(f'Fail to POST Term Matching Data query with taskId {taskId} with batch ID {tm_current_page} and record ID from {(tm_current_page-1)*(QUERY_DATA_BATCH_SIZE)} to {(tm_current_page)*(QUERY_DATA_BATCH_SIZE)}')
            if 'result' in term_match_data and term_match_data["result"]["pages"] == 0:
                break
                
    if not all_term_match_data:
        error_msg = f"Term matching data for taskId {taskId} doesn't exist. Please check."
        logger.error(error_msg)
        raise BadRequest(error_msg)
        
    '''
    sample feedback response:
    [
        "success": true,
        "message": "",
        "result": {
            "total": 349,
            "size": 99999,
            "current": 1,
            "pages": 1,
            "records": [
                {
                    "id": 1,
                    "taskId": 1,
                    "tsTerm": "Borrower",
                    "tsText": "Borrower",
                    "result": [
                        true,
                        true,
                        true,
                        true,
                        true
                    ],
                    "manualIdentifiers": [
                        "Cl_8.1(a)(ii)",
                        "Cl_8.1(a)(iii)"
                    ]

                }, ...
    ]
    '''
    hit = 0
    miss = 0
    irrelavent_instance_count = 0
    hit_instance = 0
    manual_select_instance = 0
    total_manual_select = 0
    hit_topN = defaultdict(int)
    hit_count_by_identifier_type = defaultdict(int)
    miss_count_by_identifier_type = defaultdict(int)
    miss_count_by_section = defaultdict(int)
    term_list = []
    content_list = []
    for record in term_match_data['result']['records']:
        if record["tsTerm"] and record["tsTerm"] not in term_list:
            term_list.append(record["tsTerm"])
        content_list.append(record["tsText"])
        selected = 0
        if record["result"]:
            selected = sum(record["result"])
            indice_selected = [i for i, e in enumerate(record["result"]) if e == True]
            selected_identifiers = [e for i, e in enumerate(record["identifierList"]) if i in indice_selected]
            selected_identifier_types = [id2idtype(identifier) for identifier in selected_identifiers]
            for typ in selected_identifier_types:
                hit_count_by_identifier_type[typ] += 1
            # unselected choices implies the results are either irrelevant or don't care
            unselected = len(record["result"]) - selected
            hit += selected
            # count hit by rank
            for i, r in enumerate(record["result"]):
                hit_topN[i+1] += int(r)
            # if all results are unselected and no manual-added result, imply the result is irrelevant
            if all(i==False for i in record["result"]) and "manualIdentifier" in record and record["manualIdentifier"] is None:
                irrelavent_instance_count += 1
        # by default result = null and there exists term match result means user accept 1st result as ground-truth such that they didn't edit on the record
        elif record["result"] is None and record["identifierList"] is not None:
            selected = 1
            unselected = 0
            hit += selected
            hit_topN[1] += 1
        # when result = false, the split is irrelavent and ignore the case
        else:
            continue
            

        # if any selected, it means it is a hit instance
        if selected > 0:
            hit_instance += 1
        
        manual_selected = 0
        if "manualIdentifier" in record and record["manualIdentifier"]:
            manualIdentifier = [i for i in record["manualIdentifier"] if i and i != '']
            if manualIdentifier:
                manual_select_instance += 1
                manual_selected = len(manualIdentifier)
                manual_selected_identifier_types = [id2idtype(identifier) for identifier in manualIdentifier]
                for typ in manual_selected_identifier_types:
                    miss_count_by_identifier_type[typ] += 1
                    miss_count_by_section[record["tsTerm"]] += 1
                total_manual_select += manual_selected
            
        # only consider miss count if unselected event occurs
        miss += min(unselected, manual_selected)
    
    assert (hit + miss) > 0, logger.error('There is neither hit result nor manually-added result. There should be at least either one hit result or one manually-added result. Please check.')
    avg_accuracy = round((hit / (hit + miss)) * 100, 2)
    topN_hit_rate = {k: round((hit_topN[k]/(hit + miss)) * 100, 2) if hit>0 else None for k in range(1,TOP_N_RESULTS+1)}
    cumulative_upto_topN = {k: sum([hit_topN[i] for i in range(1,k+1)]) for k in range(1,TOP_N_RESULTS+1)}
    if miss_count_by_section:
        section_top_miss = sorted(miss_count_by_section.items(), key=lambda x:x[1])[0][0]
    else:
        section_top_miss = None
    
    relavent_instance_count = len(content_list) - irrelavent_instance_count
    result = {
        "taskId": taskId,
        "avgAccuracy": avg_accuracy,
        "termCount": len(term_list),
        "termContentCount": len(content_list),
        "irrelavantContentCount": irrelavent_instance_count,
        "relavantInstanceRate": round(relavent_instance_count/ len(content_list) *100, 2),
        "hitCount": hit,
        "missCount": miss,
        "topMissTsTerm": section_top_miss,
        "hitInstanceCount": hit_instance,
        "avgHitperHitInstance": hit / hit_instance if hit_instance>0 else None,
        "manualCount": total_manual_select,
        "manualInstanceCount": manual_select_instance,
        "missRate": round(miss/ (hit + miss) *100, 2),
        "manualInstanceRate": round(manual_select_instance/ relavent_instance_count *100, 2) if relavent_instance_count >0 else None
    }
    
    clause_type = ['definitionClause', 'partiesClause', 'clause', 'schedule']
    for k, v in dict(hit_count_by_identifier_type).items():
        clause_type.remove(k)
        result.update(
            {
                f"{k}HitCount": v
            }
        )
    for k in clause_type:
        result.update(
            {
                f"{k}HitCount": 0
            }
        )
    
    clause_type = ['definitionClause', 'partiesClause', 'clause', 'schedule']
    for k, v in dict(miss_count_by_identifier_type).items():
        clause_type.remove(k)
        result.update(
            {
                f"{k}MissCount": v
            }
        )
    for k in clause_type:
        result.update(
            {
                f"{k}MissCount": 0
            }
        )
    
    for i in range(1,TOP_N_RESULTS+1):
        result.update(
            {
                f"top{str(i)}HitCount": hit_topN[i],
                f"top{str(i)}HitRate": topN_hit_rate[i],
                f"upToTop{str(i)}HitCount": cumulative_upto_topN[i],
                f"upToTop{str(i)}HitRate": round(cumulative_upto_topN[i] / (hit + miss) * 100,2) if hit >0 else None
            }
        )

    response = await POST(INSERT_ANALYTICS_URL, result)
    
    if response and "error" not in response:
        response = Response(response, status=status.HTTP_200_OK)
    else:
        response = Response(response, status=status.HTTP_400_BAD_REQUEST)
        
    response.accepted_renderer = JSONRenderer()
    response.accepted_media_type = "application/json"
    response.renderer_context = {}
    response.render()
    return response

if __name__=='__main__':
    import sys
    import logging
    # Change root logger level from WARNING (default) to NOTSET in order for all messages to be delegated.
    logging.getLogger().setLevel(logging.NOTSET)