import json
import os
import pandas as pd

# check unfinished review filename list
# reviewed_folder = '/home/data/ldrs_analytics/data/reviewed_antd_ts'
# all_folder = '/home/data/ldrs_analytics/data/updated_annotated_ts_v3'
# file_facility = '/home/data/ldrs_analytics/data/doc/filename_facility_type.csv'

# reviewed_files = [f.replace('_docparse','') for f in os.listdir(reviewed_folder) if f.endswith('.csv')]
# all_files = [f.replace('_docparse','') for f in os.listdir(all_folder) if f.endswith('.csv')]

# not_finished = [f.replace('.csv','') for f in all_files if f not in reviewed_files]
# df = pd.read_csv(file_facility)
# df_new = df[df.filename.isin(not_finished)]
# df_new.to_csv('/home/data/ldrs_analytics/data/reviewed_antd_ts/not_finished.csv',index=False)


# with open("/home/data/ldrs_analytics/data/reviewed_antd_ts/t.json", 'r') as f:
#     old = json.load(f)
    
# with open("/home/data/ldrs_analytics/data/reviewed_antd_ts/checked_list.json", 'r') as f:
#     new = json.load(f)

# d = []
# for i in new:
#     if i not in old:
#         d.append(i)
        
# # Serializing json
# json_object = json.dumps(d, indent=4)
 
# # Writing to sample.json
# with open("/home/data/ldrs_analytics/data/reviewed_antd_ts/new.json", "w") as outfile:
#     outfile.write(json_object)

# check unfinished cleaning of batch #1 filename list
TS_clean_doc_folder = '/home/data/ldrs_analytics/data/doc/TS/clean'
TS_doc_folder = '/home/data/ldrs_analytics/data/doc/TS/clean'
file_facility_batch = '/home/data/ldrs_analytics/data/doc/filename_facility_type_batch.csv'

cleaned_files = os.listdir(TS_clean_doc_folder)
df = pd.read_csv(file_facility_batch)
df['batch_id'] = df['batch_id'].astype(int)
df['annotated'] = df['annotated'].astype(bool)
original_fnames = df.loc[df.batch_id.eq(1) & df.annotated.eq(True)]['filename'].values.tolist()

for file in cleaned_files:
    fname = file.split('.')[0].replace('-clean','')
    if fname in original_fnames:
        original_fnames.remove(fname)

# Serializing json
json_object = json.dumps(original_fnames, indent=4)
 
# Writing to sample.json
with open("/home/data/ldrs_analytics/data/doc/TS/clean/not_finished.json", "w") as outfile:
    outfile.write(json_object)