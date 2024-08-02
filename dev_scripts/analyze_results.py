import os
import pandas as pd
import numpy as np
import argparse
import re

def write_append_excel(df, outpath, sheet_name, cols_show_as_percentage=None):
    import pandas as pd
    if os.path.exists(outpath):
        mode = 'a'
    else:
        mode = 'w'

    if cols_show_as_percentage:
        for column in cols_show_as_percentage:
            df.loc[:, column] = df[column].map(lambda x: '{:.2%}'.format(x) if x is not None and x != np.nan else x)

    if mode == 'a':
        with pd.ExcelWriter(outpath, engine="openpyxl", mode=mode,if_sheet_exists='replace') as writer:
            df.to_excel(writer, sheet_name=sheet_name)
    else:
        with pd.ExcelWriter(outpath, mode=mode) as writer:
            df.to_excel(writer, sheet_name=sheet_name)

def judge_txtgrp(judge_all_grp):
    '''
    Given a 'judge_all' pd.Series from pd.groupby object
    check if there is any 'TP' in a group of judges, return judge_grp as 'TP'
    elif all 'TN' were found in a group of judges, return judge_grp as 'TN'
    elif all 'FN' were found in a group of judges, return judge_grp as 'FN'
    elif all 'FP' were found in a group of judges, return judge_grp as 'FP'
    else return ['FP','FN'] as there should be some 'FP' and some 'FN' in the group
    e.g. judge_all_grp = pd.Series(['FP','FP','TP','FP','FP']), return 'TP'
    e.g. judge_all_grp = pd.Series(['FP','FP','FN']), return ['FP','FN']
    '''
    if judge_all_grp.eq('TP').any():
        return 'TP'
    elif judge_all_grp.eq('TN').all():
        return 'TN'
    elif judge_all_grp.eq('FN').all():
        return 'FN'
    elif judge_all_grp.eq('FP').all():
        return 'FP'
    elif judge_all_grp.eq('FP').any() and judge_all_grp.eq('FN').any():
        return str(['FP', 'FN'])
    # elif judge_all_grp.eq(None).any() or judge_all_grp.eq('None').any() or judge_all_grp.eq(np.nan).any():
    #     return None, 1/judge_all_grp.size
    else:
        return None

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--result_dir",
        default='/home/data2/ldrs_analytics/data/antd_ts_merge_fa/reviewed_ts_v4_fa_v4.4_finetune0130_tm_modified_top5_filter_human_errors/(0)',
        type=str,
        required=False,
        help="Directory of TS merge term matching & FA text results.",
    )
    parser.add_argument(
        "--output_dir",
        default='/home/data2/ldrs_analytics/data/antd_ts_merge_fa/reviewed_ts_v4_fa_v4.4_finetune0130_tm_modified_top5_filter_human_errors/analysis/(0)',
        type=str,
        required=False,
        help="Directory of TS merge term matching & FA text results.",
    )
    parser.add_argument(
        "--result_flag",
        default='reviewed_ts_v4_fa_v4.4_finetune0130_tm_modified',
        type=str,
        required=False,
        help="Directory of TS merge term matching & FA text results.",
    )
    parser.add_argument(
        "--target_input_result_filename",
        default='all_merge&match_results.csv',
        type=str,
        required=False,
        help="Source of input CSV that concatenated all results.",
    )
    parser.add_argument(
        "--analyze_set",
        default=None,
        type=str,
        required=False,
        help="Target split to be analyzed.",
    )
    parser.add_argument(
        "--file_dataset_doc",
        default="/home/data2/ldrs_analytics/data/doc/filename_dataset_split.csv",
        type=str,
        required=False,
        help="Target split to be analyzed.",
    )
    parser.add_argument(
        "--filename_facility_type_doc",
        default="/home/data2/ldrs_analytics/data/doc/filename_facility_type.csv",
        type=str,
        required=False,
        help="Target split to be analyzed.",
    )

    args = parser.parse_args()
    df = pd.read_csv(os.path.join(args.result_dir ,args.target_input_result_filename))
    df = df.replace({'nan': None, np.nan: None, 'None': None, '': None})
    if 'split' not in df.columns:
        # map filename to dataset split
        file_dataset_df = pd.read_csv(args.file_dataset_doc)
        file2dataset = dict(zip(file_dataset_df['filename'].values.tolist(), file_dataset_df['split'].values.tolist()))
        df['split'] = df['filename'].map(lambda x: file2dataset.get(re.sub('_docparse', '', x), None))
    
    file_facility_type_df = pd.read_csv(args.filename_facility_type_doc)
    file2facility = dict(zip(file_facility_type_df['filename'].values.tolist(), file_facility_type_df['facility_type'].values.tolist()))
    if 'facility_type' not in df.columns:
        # map filename to facility type
        df['facility_type'] = df['filename'].map(lambda x: file2facility.get(re.sub('_docparse', '', x), None))
    if args.analyze_set:
        df = df[df['split']==args.analyze_set]
    facility_type = dict(zip(df.filename.values.tolist(),df.facility_type.values.tolist()))
    file_facility_type_df = pd.read_csv(args.filename_facility_type_doc)
    file2facility = dict(zip(file_facility_type_df['filename'].values.tolist(), file_facility_type_df['facility_type'].values.tolist()))
    
    df_gp_doc = df.groupby(['filename'], dropna=False)
    
    df_individ_analysis = pd.DataFrame(data={
                    'support': df_gp_doc.judge_all.size(),
                    'tp': df_gp_doc.judge_all.apply(lambda x: (x == 'TP').sum()),
                    'fp': df_gp_doc.judge_all.apply(lambda x: (x == 'FP').sum()),
                    'tn': df_gp_doc.judge_all.apply(lambda x: (x == 'TN').sum()),
                    'fn': df_gp_doc.judge_all.apply(lambda x: (x == 'FN').sum())
                }).reset_index()
    df_individ_analysis['precision'] = df_individ_analysis.tp / (df_individ_analysis.tp + df_individ_analysis.fp)
    df_individ_analysis['recall'] = df_individ_analysis.tp / (df_individ_analysis.tp + df_individ_analysis.fn)
    df_individ_analysis['f1_score'] = (2 * df_individ_analysis.precision * df_individ_analysis.recall) / (df_individ_analysis.precision + df_individ_analysis.recall)
    df_individ_analysis['accuracy'] = (df_individ_analysis.tp + df_individ_analysis.tn) / (df_individ_analysis.tp + df_individ_analysis.tn + df_individ_analysis.fn + df_individ_analysis.fp)
    df_individ_analysis['facility_type'] = df_individ_analysis['filename'].map(lambda x: file2facility.get(re.sub('_docparse', '', x), None))
    df_individ_analysis = df_individ_analysis.sort_values(['recall'])
    df_individ_analysis.loc[-1] = ['Macro-Average', None, None, None, None, None, df_individ_analysis.precision.mean(), df_individ_analysis.recall.mean(), df_individ_analysis.f1_score.mean(), df_individ_analysis.accuracy.mean(), None]

    df_gp_idx_identifier = df.groupby(['filename', 'index', 'fa_identifier'], dropna=False)

    df_gp_detail = pd.DataFrame(data={
                    'section': df_gp_idx_identifier.section.first(),
                    'text': df_gp_idx_identifier.apply(lambda x: x.text.values.tolist()[0]),
                    'split_count': df_gp_idx_identifier.apply(lambda x: len(x.text.values.tolist())),
                    'match_term_list': df_gp_idx_identifier.apply(lambda x: x.match_term_list.values.tolist()[0]),
                    'identifier_list': df_gp_idx_identifier.apply(lambda x: x.identifier_list.values.tolist()[0]),
                    'judge': df_gp_idx_identifier.apply(lambda x: x.judge.values.tolist()[0]),
                    'judge_all': df_gp_idx_identifier.apply(lambda x: x.judge_all.tolist()[0]),
                    'judge_grp': df_gp_idx_identifier.apply(lambda x: judge_txtgrp(x.judge_all))
                }).reset_index(drop=False)

    df_gp_detail2 = df_gp_detail.groupby('filename')

    df_gp_detail_analysis = pd.DataFrame(data={
                    'support': df_gp_detail2.judge_all.size(),
                    'tp': df_gp_detail2.judge_grp.apply(lambda x: (x == 'TP').sum()),
                    'fp': df_gp_detail2.judge_grp.apply(lambda x: (x == 'FP').sum()),
                    'tn': df_gp_detail2.judge_grp.apply(lambda x: (x == 'TN').sum()),
                    'fn': df_gp_detail2.judge_grp.apply(lambda x: (x == 'FN').sum())
                }).reset_index()
    df_gp_detail_analysis['precision'] = df_gp_detail_analysis.tp / (df_gp_detail_analysis.tp + df_gp_detail_analysis.fp)
    df_gp_detail_analysis['recall'] = df_gp_detail_analysis.tp / (df_gp_detail_analysis.tp + df_gp_detail_analysis.fn)
    df_gp_detail_analysis['f1_score'] = (2 * df_gp_detail_analysis.precision * df_gp_detail_analysis.recall) / (df_gp_detail_analysis.precision + df_gp_detail_analysis.recall)
    df_gp_detail_analysis['accuracy'] = (df_gp_detail_analysis.tp + df_gp_detail_analysis.tn) / (df_gp_detail_analysis.tp + df_gp_detail_analysis.tn + df_gp_detail_analysis.fn + df_gp_detail_analysis.fp)
    df_gp_detail_analysis['facility_type'] = df_gp_detail_analysis['filename'].map(lambda x: file2facility.get(re.sub('_docparse', '', x), None))
    df_gp_detail_analysis = df_gp_detail_analysis.sort_values(['recall'])
    df_gp_detail_analysis.loc[-1] = ['Macro-Average', None, None, None, None, None, df_gp_detail_analysis.precision.mean(), df_gp_detail_analysis.recall.mean(), df_gp_detail_analysis.f1_score.mean(), df_gp_detail_analysis.accuracy.mean(), None]
    
    df_facility_type2 = df_gp_detail_analysis[~df_gp_detail_analysis.recall.isna()].groupby(['facility_type'])[['precision','recall','f1_score','accuracy']].mean().reset_index().sort_values(['recall'])
    df_gp = df.groupby(['filename','section','fa_identifier'])

    df_file_sect_antd_judge = pd.DataFrame(data={
        'judge_list': df_gp['judge_all'].apply(lambda x: x.values.tolist()),
        'judge_all': df_gp['judge_all'].apply(judge_txtgrp),
        'count_splits': df_gp['judge_all'].apply(len),
    }).reset_index()
    
    df_sec = df_file_sect_antd_judge.groupby(['section'])
    df_sect_judge = pd.DataFrame(data={
        'identifier_list': df_sec['fa_identifier'].apply(lambda x: x.values.tolist()),
        'judge_list': df_sec['judge_all'].apply(lambda x: x.values.tolist()),
        'support': df_sec['judge_all'].apply(len),
        'tp': df_sec['judge_all'].apply(lambda x: sum(x=='TP')),
        'fn': df_sec['judge_all'].apply(lambda x: sum(x=='FN')),
        'recall': df_sec['judge_all'].apply(lambda x: sum(x=='TP')/(sum(x=='TP')+sum(x=='FN')) if (sum(x=='TP')+sum(x=='FN'))>0 else None),
    }).reset_index().sort_values(['recall','support'], ascending=[True, False])

    df_gp2 = df_file_sect_antd_judge.groupby(['filename','section'])

    df_file_sect_judge = pd.DataFrame(data={
        'identifier_list': df_gp2['fa_identifier'].apply(lambda x: x.values.tolist()),
        'judge_list': df_gp2['judge_all'].apply(lambda x: x.values.tolist()),
        'support': df_gp2['fa_identifier'].apply(len),
        'tp': df_gp2['judge_all'].apply(lambda x: sum(x=='TP')),
        'fn': df_gp2['judge_all'].apply(lambda x: sum(x=='FN')),
        'recall': df_gp2['judge_all'].apply(lambda x: sum(x=='TP')/(sum(x=='TP')+sum(x=='FN')) if (sum(x=='TP')+sum(x=='FN'))>0 else None),
    }).reset_index().sort_values(['recall','support'], ascending=[True, False])

    df_file_sect_judge['facility_type'] = df_file_sect_judge['filename'].map(facility_type)
    df_file_sect_judge = df_file_sect_judge.replace({'nan': None, np.nan: None, 'None': None, '': None})

    df_sec = df_file_sect_judge[~df_file_sect_judge.recall.isna()].groupby(['section'])['recall'].mean().reset_index().sort_values(['recall'])
    df_sec.loc[-1] = ['Macro-Average', df_sec.recall.mean()]
    df_facility_type = df_file_sect_judge[~df_file_sect_judge.recall.isna()].groupby(['facility_type'])['recall'].mean().reset_index().sort_values(['recall'])
    df_facility_type.loc[-1] = ['Macro-Average', df_facility_type.recall.mean()]
    df_file = pd.DataFrame(data={
        'support': df_file_sect_judge[~df_file_sect_judge.recall.isna()].groupby(['filename'])['support'].sum(),
        'tp': df_file_sect_judge[~df_file_sect_judge.recall.isna()].groupby(['filename'])['tp'].sum(),
        'fn': df_file_sect_judge[~df_file_sect_judge.recall.isna()].groupby(['filename'])['fn'].sum()
        }).reset_index()
    df_file['recall'] = df_file.tp/(df_file.tp+df_file.fn)
    df_file = df_file.sort_values(['recall'])
    df_file['facility_type'] = df_file['filename'].map(facility_type)
    df_file.loc[-1] = ['Macro-Average', df_file.support.sum(), df_file.tp.sum(), df_file.fn.sum(), df_file.recall.mean(), None]
    percentage_columns = ['recall']

    output_filepath = os.path.join(args.output_dir, f'{args.result_flag}_{args.analyze_set}_analysis.xlsx')

    write_append_excel(df_file, output_filepath, 'analy_grpby_doc on (3)', cols_show_as_percentage=percentage_columns)
    write_append_excel(df_facility_type, output_filepath, 'analy_grpby_type on (3)', cols_show_as_percentage=percentage_columns)
    write_append_excel(df_sec, output_filepath, 'analy_grpby_sec on (3)', cols_show_as_percentage=percentage_columns)
    write_append_excel(df_gp_detail_analysis, output_filepath, 'analy_grpby_doc on (1)',cols_show_as_percentage=['precision','recall','f1_score','accuracy'])
    write_append_excel(df_facility_type2, output_filepath, 'analy_grpby_type on (1)',cols_show_as_percentage=['precision','recall','f1_score','accuracy'])
    write_append_excel(df_individ_analysis, output_filepath, 'analy on (0)',cols_show_as_percentage=['precision','recall','f1_score','accuracy'])
    write_append_excel(df_sect_judge, output_filepath, '(3.1)grpby_sec on (2)', cols_show_as_percentage=percentage_columns)
    write_append_excel(df_file_sect_judge, output_filepath, '(3)grpby_doc+sec on (2)', cols_show_as_percentage=percentage_columns)
    write_append_excel(df_file_sect_antd_judge, output_filepath, '(2)grpby_doc+sec+antd on (0)')
    write_append_excel(df_gp_detail, output_filepath, '(1)grpby_doc+index+antd on (0)')
    write_append_excel(df, output_filepath, '(0)source')