import os
import re
import sys
import requests
import json
from collections import defaultdict
from copy import deepcopy


def POST(url,data):
    headers = {'Content-type': 'application/json', 'Accept': 'application/json'}
    r = requests.post(url, data=json.dumps(data), headers=headers)
    return r

doc_list = []
upload_folderPath = '/home/data/frontend_data/upload'
createUser = 'alan'
key1 = ['TS', 'term sheet']
key2 = ['FA', 'facility agreement', 'facilities agreement']
url = 'http://10.6.55.12:8080/ldrs/dataStorage/insertDocument'
url2 = 'http://10.6.55.12:8080/ldrs/dataStorage/insertTask'
target_files = [
        "125_VF_SYN_FA_mkd_20231013.pdf",
        "126_VF_SYN_FA_mkd_20230601.pdf",
        "127_NBFI_SYN_FA_mkd_20230330.pdf",
        "128_NBFI_SYN_FA_mkd_20230224.pdf",
        "129_AW_SYN_FA_mkd_20230626.pdf",
        "130_BL_SYN_FA_mkd_20230710.pdf",
        "131_AF_SYN_FA_mkd_20230718.pdf",
        "132_PF_SYN_FA_mkd_20230106.pdf",
        "133_BL_PRJ_FA_mkd_20230926.pdf",
        "134_GF_PRJ_FA_mkd_20230829.pdf",
        "135_GL_PRJ_FA_mkd_20231004.pdf",
        "136_BL_SYN_FA_mkd_20230418.pdf",
        "137_GL_SYN_FA_mkd_20231121.pdf",
        "138_GF_SYN_FA_mkd_20230130.pdf",
        "139_NBFI_PRJ_FA_mkd_20231122.pdf",
        "140_NBFI_SYN_FA_mkd_20230512.pdf",
        "141_GL_SYN_FA_mkd_20231221.pdf",
        "142_PF_SYN_FA_mkd_20230810.pdf",
        "143_GF_SYN_FA_mkd_undated.pdf",
        "144_NBFI_SYN_FA_mkd_20231031.pdf",
        "145_GF_SYN_FA_mkd_20231031.pdf",
        "146_BL_PRJ_FA_mkd_20230629.pdf",
        "147_GL_PRJ_FA_mkd_20230817.pdf",
        "148_GF_PRJ_FA_mkd_20230919.pdf",
        "149_BL_PRJ_FA_mkd_20231102.pdf"
    ]
if target_files:
    fileList = [f for f in os.listdir(upload_folderPath) if f in target_files]
else:
    fileList = os.listdir(upload_folderPath)
fileList.sort()

task_schema = {
  "createUser": "alan",
  "id": None,
  "name": None,
  "customerNo": "string",
  "borrowerName": "string",
  "cawProposalNo": "string",
  "reviewDate": "2023-09-20",
  "makerGroup": "string",
  "checkerGroup": "string",
  "faFileId": None,
  "tsFileId": None,
  "status": "CREATED",
  "systemStatus": "WAITING_FOR_PROCESSING"
}

TS_dict = defaultdict()
FA_dict = defaultdict()
name_dict = defaultdict()

for file in fileList:
    if file.endswith('.pdf'):
        print(f'file: {file}')
        filePath = os.path.join(upload_folderPath,file)
        if re.match(r'.*' + r'.*|.*'.join(key1), file, flags=re.IGNORECASE):
            fileType = 'TS'
        elif re.match(r'.*' + r'.*|.*'.join(key2), file, flags=re.IGNORECASE):
            fileType = 'FA'
        docId = file.split('_')[0]
        data = {
                "id": 0,
                "createUser": createUser,
                "fileName": file,
                "filePath": filePath,
                "fileType": fileType
                }
        print(f'data: \n{json.dumps(data,indent=4)}')
        response = POST(url,data)
        response = response.json()
        print(f'response: \n{json.dumps(response,indent=4)}')
        fileId = response['result']['id']
        name_dict[docId] = str(file.split('_')[0]) + '_' + file.split('_')[1] + '_'+ file.split('_')[2]
        if fileType == 'TS':
            TS_dict[docId] = fileId
        elif fileType == 'FA':
            FA_dict[docId] = fileId
    
for docid, TSfileId in TS_dict.items():
    data2 = deepcopy(task_schema)
    FAfileId = FA_dict[docid]
    taskname = name_dict[docid]
    data2['id'] = int(docid)
    data2['name'] = taskname
    data2['tsFileId'] = TSfileId
    data2['faFileId'] = FAfileId
    print(f'data2: \n{json.dumps(data2,indent=4)}')
    response2 = POST(url2,data2)
    response2 = response2.json()
    print(f'response2: \n{json.dumps(response2,indent=4)}')
    