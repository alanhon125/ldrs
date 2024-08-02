import json
import os

folder1 = [i.replace('_docparse.csv','') for i in os.listdir('/home/data2/ldrs_analytics/data/antd_ts_merge_ts/antd_v3.2_merge_ts_v4.3') if i.endswith('.csv')]
folder2 = [i.replace('_docparse_results.csv','') for i in os.listdir('/home/data2/ldrs_analytics/data/term_matching_csv/0000_s/0123_fintune_raw_ts_v4.3_fa_v4.4') if i.endswith('.csv')]

k1 = []
k2 = []

for f1 in folder1:
    if f1 not in folder2:
        k1.append(f1)

for f2 in folder2:
    if f2 not in folder1:
        k2.append(f2)

print('Exists in TS parse but not in term matching',json.dumps(k1))
print('Exists in term matching but not in TS parse',json.dumps(k2))