
import json
import requests

def POST(url,data):
    headers = {'Content-type': 'application/json', 'Accept': 'application/json'}
    r = requests.post(url, data=json.dumps(data), headers=headers)
    return r

insert_term_match_url = 'http://10.6.55.12:8000/api/termMatch2table'
for taskId in range(1,81):
    request = {"id": taskId}
    response = POST(insert_term_match_url,request).json()
    print(f'Response for taskId {taskId}: {json.dumps(response, indent=4)}')