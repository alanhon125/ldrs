import pandas as pd
import numpy as np
import re
import os

FA_CSV_DIR = '/home/data/ldrs_analytics/data/docparse_csv/FA/'
TRAILING_DOT_ZERO = r'\.0$' # 1.0, 2.5.0 etc.

fa_fnames = sorted(os.listdir(FA_CSV_DIR))
data = {
    "filename": [],
    "definition_clause_id": [],
    # "identifiers": []
}

for fa_fname in fa_fnames:
    filename = fa_fname.replace('.csv', '')
    df_fa = pd.read_csv(FA_CSV_DIR+fa_fname)
    df_fa = df_fa.add_prefix('fa_')
    df_fa['fa_sub_section'] = df_fa['fa_sub_section'].astype(str)
    find_def_cond2 = df_fa['fa_sub_section'].str.contains('Definitions|Interpretations', na=False, case=False)
    df_fa = df_fa.replace({'nan': None, np.nan: None, 'None': None})
    def_sec_id = df_fa[(find_def_cond2) & df_fa.fa_schedule_id.isna()]['fa_section_id']
    def_sub_sec_id = df_fa[(find_def_cond2) & df_fa.fa_schedule_id.isna()]['fa_sub_section_id']
    def_clause_id = list(set([re.sub(TRAILING_DOT_ZERO,'',str(a)) + '.' + re.sub(TRAILING_DOT_ZERO,'',str(b)) for a, b in list(zip(def_sec_id, def_sub_sec_id)) if str(a)!='nan' and str(b)!='nan']))
    def_identifier = df_fa[(find_def_cond2) & df_fa.fa_schedule_id.isna()]['fa_identifier'].values.tolist()
    data['filename'].append(filename)
    data['definition_clause_id'].append(def_clause_id)
    # data['identifiers'].append(def_identifier)
    
df = pd.DataFrame(data)
df.to_csv('definition_clause_id.csv',index=False)