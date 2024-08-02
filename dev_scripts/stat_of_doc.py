import pandas as pd
import os
import re

FA_ver = '_v3.2'
TS_ver = '_v3'
src_fa_folder = f'/home/data/ldrs_analytics/data/docparse_csv/FA{FA_ver}'
src_ts_folder = f'/home/data/ldrs_analytics/data/docparse_csv/TS{TS_ver}'

df = {
    "fa_filename": [],
    "ts_filename": [],
    "facility_type": [],
    "batch_id": [],
    "fa_length": [],
    "ts_length": [],
    "definition_length": [],
    "fa x ts length": [],
    "definition x ts length": []
}

# map filename to facility type
file_facility_type_df = pd.read_csv('/home/data/ldrs_analytics/data/doc/filename_facility_type_batch.csv')
file2facility = dict(zip(file_facility_type_df['filename'].values.tolist(),file_facility_type_df['facility_type'].values.tolist()))
file2batch = dict(zip(file_facility_type_df['filename'].values.tolist(),file_facility_type_df['batch_id'].values.tolist()))

for file in os.listdir(src_ts_folder):
    fa_fname = re.sub(r"(.*)TS_(.*)", r"\1FA_\2", file)
    df_ts = pd.read_csv(os.path.join(src_ts_folder, file))
    df_ts = df_ts[~df_ts.text_element.str.contains('section')]
    facility_type = file2facility.get(re.sub('_docparse.csv','',file),None)
    batch_id = file2batch.get(re.sub('_docparse.csv','',file),None)
    df_fa = pd.read_csv(os.path.join(src_fa_folder, fa_fname))
    df_def = df_fa[~df_fa.definition.isna()]
    df_def = df_def.drop_duplicates(subset=['section_id','sub_section_id','definition'])
    
    ts_length = len(df_ts.index)
    fa_length = len(df_fa.index)
    def_length = len(df_def.index)
    
    df['fa_filename'].append(fa_fname)
    df['ts_filename'].append(file)
    df['facility_type'].append(facility_type)
    df['batch_id'].append(batch_id)
    df['fa_length'].append(fa_length)
    df['ts_length'].append(ts_length)
    df['definition_length'].append(def_length)
    df['fa x ts length'].append(ts_length*fa_length)
    df['definition x ts length'].append(ts_length*def_length)
    
df = pd.DataFrame(data=df)
df['total_combination'] = df['fa x ts length'] + df['definition x ts length']
df['fa_length'] = df['fa_length'].astype(float).map('{:,.0f}'.format)
df['ts_length'] = df['ts_length'].astype(float).map('{:,.0f}'.format)
df['definition_length'] = df['definition_length'].astype(float).map('{:,.0f}'.format)
df['fa x ts length'] = df['fa x ts length'].astype(float).map('{:,.0f}'.format)
df['definition x ts length'] = df['definition x ts length'].astype(float).map('{:,.0f}'.format)
df['total_combination'] = df['total_combination'].astype(float).map('{:,.0f}'.format)
pd.options.display.float_format = '{:,}'.format
df.to_csv(f'/home/data/ldrs_analytics/data/docparse_csv/all_ts_fa_length{FA_ver}.csv', index=False, encoding='utf-8-sig')