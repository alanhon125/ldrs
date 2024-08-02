import os
import json
import requests

os.chdir(os.path.dirname(os.path.abspath(__file__)))

'''extract pdf paths'''

pdf_folder = '/home/data/ldrs_analytics/data/pdf/unannotated_fa_ts/'
document_type = 'TS'
target_folder = pdf_folder + document_type
files = sorted([os.path.join(target_folder,i) for i in os.listdir(target_folder) if i.endswith('.pdf')])
data = []

for i, f in enumerate(files):
    data.append(
        {
            "id": i,
            "fileName" : os.path.basename(f),
            "filePath" : f,
            "fileType": document_type
        }
    )
with open(f'{document_type}_list.json', 'w') as output:
    json.dump(data, output, ensure_ascii=False)
    
# headers = {'Content-type': 'application/json', 'Accept': 'application/json'}
# r = requests.post('http://10.6.55.12:8000/api/docparse2json', data=json.dumps(data), headers=headers)

'''extract docparse json paths'''

docparse_folder = '/home/data/ldrs_analytics/data/docparse_json/'
document_type = 'TS'
target_folder = docparse_folder + document_type
files = sorted([os.path.join(target_folder,i) for i in os.listdir(target_folder) if i.endswith('.json')])
data = []

for i, f in enumerate(files):
    data.append(
        {
            "id": i,
            "fileName" : os.path.basename(f),
            "filePath" : f,
            "fileType": document_type
        }
    )
with open(f'{document_type}_docparse_list.json', 'w') as output:
    json.dump(data, output, ensure_ascii=False)
    
# headers = {'Content-type': 'application/json', 'Accept': 'application/json'}
# r = requests.post('http://10.6.55.12:8000/api/docparse2table', data=json.dumps(data), headers=headers)