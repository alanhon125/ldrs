import pandas as pd
import os
import numpy as np
import argparse

PROJ_ROOT_DIR = '/home/data2/ldrs_analytics'


def judge_relationship(row):
    if row['phrase_id'].nunique() == 1:
        ts_split = '1'
    else:
        ts_split = 'n'

    if row['section'].values[0] in ["Documentation", "Documentatio n", "Amendments and Waivers", "Miscellaneous Provisions", "Other Terms"]:
        ts_split = '1'

    clause_ids = row['clause_id'].values[0]
    schedule_ids = row['schedule_id'].values[0]

    if clause_ids:
        isMultipleClause = len(str(clause_ids).split(',')) > 1 or len(
            [i for i in str(clause_ids).split('\n') if i.strip()]) > 1
    if schedule_ids:
        isMultipleSched = len(str(schedule_ids).split(',')) > 1 or len(
            [i for i in str(schedule_ids).split('\n') if i.strip()]) > 1

    if not clause_ids and not schedule_ids:
        fa_split = None
    elif (clause_ids and isMultipleClause) or (schedule_ids and isMultipleSched) or (clause_ids and schedule_ids):
        fa_split = 'm'
    else:
        fa_split = '1'

    if fa_split:
        return ts_split + ' x ' + fa_split
    else:
        return None


parser = argparse.ArgumentParser()
# Required parameters
parser.add_argument(
    "--annotated_csv_dir",
    # f'{PROJ_ROOT_DIR}/data/updated_annotated_ts_v2/'
    default=f'{PROJ_ROOT_DIR}/data/antd_ts_merge_ts/antd_v3.0_merge_ts_v4/',
    type=str,
    required=False,
    help="Path to folder of annotated TS in CSV",
)
args = parser.parse_args()
output_bfill_csv_dir = os.path.join(args.annotated_csv_dir, 'bfilled/')

if not os.path.exists(output_bfill_csv_dir):
    os.mkdir(output_bfill_csv_dir)

files = sorted(os.listdir(args.annotated_csv_dir))
print(f'Backward filling labels for files in {args.annotated_csv_dir} ...')
for file in files:
    if file.endswith('.csv'):
        print(f'filename: {file}')
        try:
            df = pd.read_csv(os.path.join(args.annotated_csv_dir, file), encoding='utf-8-sig')
        except:
            df = pd.read_csv(os.path.join(args.annotated_csv_dir, file), encoding='ISO-8859-1')
        df = df.replace({np.nan: None})
        backward_fill_columns = ['clause_id', 'definition', 'schedule_id', 'annotation']
        # index means text component index, all phrases with same index are the text in same text component
        composite_key = ['index', 'text_block_id', 'page_id']
        for c in backward_fill_columns:
            df[c] = df.groupby(composite_key, as_index=False, group_keys=True)[c].apply(lambda x: x.bfill()).reset_index(level=0, drop=True)
        df2 = df.groupby(['index', 'clause_id', 'definition', 'schedule_id'], dropna=False).apply(lambda x: judge_relationship(x)).reset_index()
        df2 = df2.rename(columns={0: "relationship"})
        df2 = df2.replace({'nan': None, np.nan: None, 'None': None})
        df = df.merge(df2, how='left', on=['index', 'clause_id', 'definition', 'schedule_id'])
        df.to_csv(output_bfill_csv_dir+file, index=False, encoding='utf-8-sig')
print(f'Completed backward filling labels for files in {args.annotated_csv_dir}\n')
