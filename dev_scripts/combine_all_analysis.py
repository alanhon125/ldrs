import pandas as pd
result_tags = ['finetune0130_raw_ts_v5.3_fa_v4.4',
               'finetune0130_raw_ts_v5.4_fa_v4.4',
               'finetune0130_raw_ts_v5.5_fa_v4.4',
               'finetune0130_raw_ts_v5.5_fa_v4.4_concated_without_section',
               'finetune0130_raw_ts_v5.5_fa_v5.1_concated_without_section'
               ]

for i, result_tag in enumerate(result_tags):
    print(result_tag)
    file_name = f'/home/data/ldrs_analytics/data/antd_ts_merge_fa/{result_tag}_top5_filter_human_errors/analysis/(0)/{result_tag}_test_analysis.xlsx'
    df = pd.read_excel(file_name, sheet_name='analy_grpby_doc on (3)').sort_values(['filename'])
    df = df.drop(columns=['Unnamed: 0'])
    df = df.rename(columns={"recall":result_tag})
    if i==0:
        dfs = df
    else:
        dfs = dfs.merge(df,on=['filename','facility_type'])

dfs.to_csv('all_grpby_doc_on_sect.csv')