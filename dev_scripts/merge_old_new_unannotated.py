import pandas as pd
import os
import nltk
import numpy as np
import re
import nltk
from typing import Union, Dict, Tuple, Callable
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import string
import chardet
import codecs
import ast
from collections import defaultdict

from dev_scripts.merge_left_right_on_text import (
    tryfunction,
    remove_non_alphabet,
    calculate_BoW,
    calculate_two_string_BoW,
    compare_texts_difference,
    merge_old_annot2new
)

nltk.download('stopwords')
SYMBOLS = ['●', '•', '·', '∙', '◉', '○', '⦿', '。', '■', '□', '☐',
        '⁃', '◆', '◇', '◈', '✦', '➢', '➣', '➤', '‣', '▶', '▷', '❖', '_']
punct = [i for i in string.punctuation]
stop_words = stopwords.words('english') + ['etc.'] + punct + SYMBOLS
punct2 = ''.join(punct+SYMBOLS) + ' '

def num_of_identical(lst_1, lst_2):
    if isinstance(lst_1, str):
        try:
            lst_1 = ast.literal_eval(lst_1)
        except:
            pass
    if isinstance(lst_2, str):
        try:
            lst_2 = ast.literal_eval(lst_2)
        except:
            pass
    if not isinstance(lst_1, (list, tuple)) and not isinstance(lst_2, (list, tuple)) or not lst_1 or not lst_2:
        return None
    count = 0
    for i in lst_1:
        if i in lst_2:
            count += 1
    return count


def revert_to_list(s):
    import ast
    if not s or '\n' not in s:
        return s
    s = s.replace('\n', '')
    s = "["+s+']'
    if s.startswith('["'):
        s = s
        try:
            s = ast.literal_eval(s)
        except:
            s = ast.literal_eval("['" + s.lstrip('['))
    elif not s.startswith("['"):
        s = "['" + s.lstrip('[')
        try:
            s = ast.literal_eval(s)
        except:
            s = s.replace('),', ")',")
            s = ast.literal_eval(s)
    return s

def merge_old_new_unannot(args):
    import pandas as pd
    import os

    if args.target_filenames:
        file_list = sorted([file for file in os.listdir(args.src_new_result_csv_dir) if file.endswith('.csv') and file in args.target_filenames])
    else:
        file_list = sorted([file for file in os.listdir(args.src_new_result_csv_dir) if file.endswith('.csv')])
    # file_list = args.target_filenames
    old_all = pd.read_csv(args.src_old_result_csv)
    old_all = old_all.replace({'nan': None, np.nan: None, 'None': None, '': None})
    old_all['TS_term'] = old_all['TS_term'].map(lambda x: x.replace(r':', '').strip() if x else x)
    old_all.loc[old_all.TS_term.notnull(), 'TS_term'] = old_all[old_all.TS_term.notnull()]['TS_term'].map(remove_non_alphabet)
    # with open(args.src_new_result_csv, 'rb') as f:
    #     result = chardet.detect(f.read())
    with codecs.open(args.src_new_result_csv, 'r', encoding='ISO-8859-1') as f:
        new_all = pd.read_csv(f)
    new_all = new_all.replace({'nan': None, np.nan: None, 'None': None, '': None})
    d = defaultdict(list)

    for file in file_list:
        print(f'filename: {file}')
        filename = file.replace('_docparse_results.csv','')
        df_old = old_all[old_all.filename==filename].astype(str)
        df_old['processed_TS_text'] = df_old['TS_text'].map(lambda x: x.strip(punct2) if x else x)
        # df_old[['match_term_list', 'identifier_list', 'similarity_list', 'match_type_list']] = df_old[
        #     ['match_term_list', 'identifier_list', 'similarity_list', 'match_type_list']].apply(
        #     lambda x: tryfunction(x, ast.literal_eval), axis=1)
        # df_old[['match_term_list', 'identifier_list', 'similarity_list', 'match_type_list']] = df_old[
        #     ['match_term_list', 'identifier_list', 'similarity_list', 'match_type_list']].apply(
        #     lambda x: list(x) if isinstance(x, tuple) else x, axis=1)
        # df_new = pd.read_csv(os.path.join(args.src_new_result_csv_dir, file)).astype(str)

        df_new = new_all[new_all['filename']==filename]
        df_new = df_new.replace({'nan': None, np.nan: None, 'None': None, '': None})
        df_new['TS_term'] = df_new['TS_term'].map(lambda x: x.replace(r':', '').strip() if x else x)
        df_new['processed_TS_text'] = df_new['TS_text'].map(lambda x: x.strip(punct2) if x else x)
        df_new.loc[df_new.TS_term.notnull(), 'TS_term'] = df_new[df_new.TS_term.notnull()]['TS_term'].map(remove_non_alphabet)
        df_new[['match_term_list', 'identifier_list', 'similarity_list']] = df_new[['match_term_list', 'identifier_list', 'similarity_list']].map(revert_to_list)
        df_new = df_new.rename(columns={'Judgement':'New Judgement',
                                                      'Matched Count':'New Matched Count',
                                                      'Expected Count':'New Expected Count'})

        df_merge = pd.merge(df_new, df_old, how="left", on=['page_id'])
        df_merge = df_merge.rename(columns={name: name.replace('_x', '').replace('_y', '_old') for name in df_merge.columns.tolist()})
        df_merge = df_merge.replace({'nan': None, np.nan: None, 'None': None, '': None})

        # check if updated_text is in original_text or original_text is in updated_text
        df_merge['isEqualText'] = df_merge.apply(lambda x: remove_non_alphabet(str(x['TS_text'])).lower() == remove_non_alphabet(str(x['TS_text_old'])).lower(), axis=1)
        df_merge['isSubstring'] = df_merge.apply(lambda x: (str(x['processed_TS_text_old']) in str(x['processed_TS_text'])) and (str(x['processed_TS_text_old']).lower().strip() not in [None,'']+stop_words), axis=1) # str(x['processed_TS_text']).lower() in str(x['processed_TS_text_old']).lower() or
        count_vectorizer = CountVectorizer()
        update_sec_map_old_sec = defaultdict(str)
        for updated_sec in list(set(df_new['TS_term'].tolist())):
            ratio = 1
            for sec in list(set(df_old['TS_term'].tolist())):
                tmp_ratio = calculate_two_string_BoW(sec, updated_sec, count_vectorizer)[1]
                if tmp_ratio < ratio:
                    ratio = tmp_ratio
                    update_sec_map_old_sec[updated_sec] = sec
        
        df_merge['isMatchedSection'] = df_merge[["TS_term", "TS_term_old"]].apply(lambda i: update_sec_map_old_sec.get(i.TS_term, None) == i.TS_term_old, axis=1)
        # Calculate cosine similarity by Bag-Of-Words
        df_merge[["difference_BoW_more", "difference_BoW_less"]] = df_merge[["processed_TS_text", "processed_TS_text_old"]].apply(
            lambda i: calculate_BoW(i, i.processed_TS_text_old, i.processed_TS_text, count_vectorizer), axis=1)
        df_merge["isBoWRatioLessThanThreshold"] = (df_merge["difference_BoW_less"] <= args.BoW_ratio_threshold) & (df_merge["difference_BoW_more"] <= args.BoW_ratio_threshold)
        # Calculate ratios of differences between to texts
        # df_merge[["difference_distance_more", "difference_distance_less"]] = df_merge[["processed_TS_text_old", "processed_TS_text"]].apply(
        #     lambda i: compare_texts_difference(i.processed_TS_text_old, i.processed_TS_text), axis=1)
        # calculate the string length difference as abs(len(text)-len(original_text))
        df_merge['StringlengthDiff'] = (df_merge['processed_TS_text'].astype(str).map(len) - df_merge['processed_TS_text_old'].astype(str).map(len)).abs()
        
        
        # merge tables on processed_section
        df_merge2 = pd.merge(df_new, df_old, how="left", on=['processed_section'])
        df_merge2 = df_merge2.rename(columns={name: name.replace('_x', '').replace('_y', '_old') for name in df_merge2.columns.tolist()})
        
        # check if updated_text is in text_old or text_old is in updated_text
        df_merge2['isEqualText'] = df_merge2.apply(lambda x: str(x['TS_text']).lower().strip(" :") == str(x['TS_text_old']).lower().strip(" :") , axis=1)
        df_merge2['isSubstring'] = df_merge2.apply(lambda x: (str(x['processed_TS_text_old']) in str(x['processed_TS_text'])) and (str(x['processed_TS_text_old']).lower().strip() not in [None,'']+stop_words), axis=1) # str(x['processed_TS_text']).lower() in str(x['processed_TS_text_old']).lower() or 
        count_vectorizer = CountVectorizer()

        # Calculate cosine similarity by Bag-Of-Words
        df_merge2[["difference_BoW_more", "difference_BoW_less"]] = df_merge2[["processed_TS_text", "processed_TS_text_old"]].apply(lambda i: calculate_BoW(i, i.processed_TS_text_old, i.processed_TS_text, count_vectorizer), axis=1)
        df_merge2["isBoWRatioLessThanThreshold"] = (df_merge2["difference_BoW_less"] <= args.BoW_ratio_threshold) & (df_merge2["difference_BoW_more"] <= args.BoW_ratio_threshold)
        # Calculate ratios of differences between to texts
        df_merge2[["difference_distance_more", "difference_distance_less"]] = df_merge2[["processed_TS_text", "processed_TS_text_old"]].apply(lambda i: compare_texts_difference(i.processed_TS_text_old, i.processed_TS_text),axis=1)
        
        # calculate the string length difference as abs(len(text)-len(text_old))
        df_merge2['StringlengthDiff'] = (df_merge2['processed_TS_text'].astype(str).map(len) - df_merge2['processed_TS_text_old'].astype(str).map(len)).abs()

        # df_merge.to_csv(os.path.join(os.path.dirname(args.output_merge_dir), file.split('.')[0]+'_merge.csv'), index=False, encoding='utf-8-sig')
        match_annot_df = df_merge.groupby(['index', 'page_id', 'phrase_id', 'TS_term', 'TS_text']).apply(lambda group: merge_old_annot2new(group)) # .reset_index()
        match_annot_df2 = df_merge2.groupby(['index', 'page_id', 'phrase_id', 'TS_term', 'TS_text']).apply(lambda group: merge_old_annot2new(group, check_section=False)) # .reset_index()
        tmp = match_annot_df[match_annot_df['merge_reason']=='not_found'].drop(['match_term_list','identifier_list','similarity_list','match_type_list', 'text_old','merge_reason'],axis=1)
        match_annot_df.loc[match_annot_df['merge_reason']=='not_found'] = pd.merge(tmp, match_annot_df2, how="inner", on=['index','page_id','phrase_id','section','text'])
        
        df_merge_true = pd.merge(df_new, match_annot_df, how="left", on=['index', 'page_id', 'phrase_id', 'TS_term', 'TS_text'])  # df_merge.iloc[true_instance_indices]
        df_merge_true = df_merge_true.drop_duplicates(['index', 'page_id', 'phrase_id', 'TS_term', 'TS_text'], keep="first")

        df_merge_true[['index', 'text_block_id', 'phrase_id']] = df_merge_true[['index', 'text_block_id', 'phrase_id']].astype(int)
        df_new[['index', 'text_block_id', 'phrase_id']] = df_new[['index', 'text_block_id', 'phrase_id']].astype(int)
        df_merge_true = df_merge_true.replace({'': None, 'nan': None, np.nan: None, 'None': None})
        df_merge_true = df_merge_true.sort_values(by=['index', 'text_block_id', 'page_id', 'phrase_id'])
        df_merge_true = df_merge_true.rename(columns={'Judgement':'Old Judgement',
                                                      'Matched Count':'Old Matched Count',
                                                      'Expected Count':'Old Expected Count'})
        # df_merge_true[['Judgement', 'Matched Count', 'Expected Count']] = None

        df_new2 = df_new.merge(df_merge_true, how='left', on=['index', 'text_block_id', 'page_id', 'phrase_id'])['TS_text_y']

        assert len(df_merge_true) == len(df_new), f'In filename: {file}, the updated TS left join old TS is not as same length as the updated TS. Please check.'
        df_merge_true[['match_term_list_old', 'identifier_list_old', 'similarity_list_old']] = df_merge_true[['match_term_list_old', 'identifier_list_old', 'similarity_list_old']].apply(lambda x: x.replace("\n","") if isinstance(x, str) else x, axis=1)
        df_merge_true[['match_term_list_old', 'identifier_list_old', 'similarity_list_old']] = df_merge_true[['match_term_list_old', 'identifier_list_old', 'similarity_list_old']].apply(lambda x: x.replace(")nan",')').replace("nan(",'(').replace("]nan",']').replace("nan[",'[').replace("][",', ').replace(")('",", '").replace(')("',', "').replace('](',', ').replace(')[',', ') if isinstance(x,str) else x, axis=1)
        df_merge_true[['match_term_list', 'identifier_list', 'similarity_list', 'match_term_list_old', 'identifier_list_old', 'similarity_list_old']] = df_merge_true[['match_term_list', 'identifier_list', 'similarity_list', 'match_term_list_old', 'identifier_list_old', 'similarity_list_old']].apply(lambda x: tryfunction(x, ast.literal_eval), axis=1)
        df_merge_true['identical_identifier_count'] = df_merge_true[['identifier_list', 'identifier_list_old']].apply(lambda x: num_of_identical(x.identifier_list, x.identifier_list_old), axis=1)

        df_merge_true = df_merge_true[['index', 'text_block_id', 'page_id', 'phrase_id', 'list_id', 'text_element', 'TS_term', 'TS_text', 'TS_text_old', 'merge_reason', 'match_term_list', 'identifier_list', 'similarity_list', 'New Judgement', 'New Matched Count', 'New Expected Count', 'match_term_list_old', 'identifier_list_old', 'similarity_list_old', 'identical_identifier_count', 'Old Judgement', 'Old Matched Count', 'Old Expected Count']]
        df_merge_true.to_csv(os.path.join(args.output_merge_dir, file), index=False, encoding='utf-8-sig')
        re_judge_count = len(df_merge_true.loc[(df_merge_true['New Matched Count'].isna()) & (df_merge_true['identical_identifier_count']<5)])
        re_judge_count0 = len(df_merge_true.loc[(df_merge_true['New Matched Count'].isna()) & (
                    df_merge_true['identical_identifier_count'] == 0)])
        re_judge_count1 = len(df_merge_true.loc[(df_merge_true['New Matched Count'].isna()) & (
                    df_merge_true['identical_identifier_count'] == 1)])
        re_judge_count2 = len(df_merge_true.loc[(df_merge_true['New Matched Count'].isna()) & (
                    df_merge_true['identical_identifier_count'] == 2)])
        re_judge_count3 = len(df_merge_true.loc[(df_merge_true['New Matched Count'].isna()) & (
                    df_merge_true['identical_identifier_count'] == 3)])
        re_judge_count4 = len(df_merge_true.loc[(df_merge_true['New Matched Count'].isna()) & (
                    df_merge_true['identical_identifier_count'] == 4)])
        d['filenames'].append(filename)
        d['num_of_identical_match'].append(len(df_merge_true.loc[(df_merge_true['merge_reason']=='identical_text')]))
        d['num_of_substring_match'].append(len(df_merge_true.loc[(df_merge_true['merge_reason'] == 'substring')]))
        d['num_of_BoW_match'].append(len(df_merge_true.loc[(df_merge_true['merge_reason'] == 'min_BoW_less_diff_ratio')]))
        d['num_of_split'].append(len(df_merge_true))
        d['num_of_judge'].append(len(df_merge_true.loc[(~df_merge_true['New Matched Count'].isna())]))
        d['num_of_re-judge'].append(re_judge_count)
        d['num_of_re-judge_with_0_identifier_identical'].append(re_judge_count0)
        d['num_of_re-judge_with_1_identifier_identical'].append(re_judge_count1)
        d['num_of_re-judge_with_2_identifier_identical'].append(re_judge_count2)
        d['num_of_re-judge_with_3_identifier_identical'].append(re_judge_count3)
        d['num_of_re-judge_with_4_identifier_identical'].append(re_judge_count4)


    DF = pd.DataFrame(d)
    DF['ratio_of_identical_match'] = DF['num_of_identical_match'] / DF['num_of_split']
    DF['ratio_of_substring_match'] = DF['num_of_substring_match'] / DF['num_of_split']
    DF['ratio_of_BoW_match'] = DF['num_of_BoW_match'] / DF['num_of_split']
    DF['ratio_of_judge'] = DF['num_of_judge'] / DF['num_of_split']
    DF['ratio_of_re-judge'] = DF['num_of_re-judge'] / DF['num_of_split']
    DF['ratio_of_re-judge_with_0_identifier_identical'] = DF['num_of_re-judge_with_0_identifier_identical'] / DF['num_of_split']
    DF['ratio_of_re-judge_with_1_identifier_identical'] = DF['num_of_re-judge_with_1_identifier_identical'] / DF['num_of_split']
    DF['ratio_of_re-judge_with_2_identifier_identical'] = DF['num_of_re-judge_with_2_identifier_identical'] / DF['num_of_split']
    DF['ratio_of_re-judge_with_3_identifier_identical'] = DF['num_of_re-judge_with_3_identifier_identical'] / DF['num_of_split']
    DF['ratio_of_re-judge_with_4_identifier_identical'] = DF['num_of_re-judge_with_4_identifier_identical'] / DF['num_of_split']
    DF['ratio_of_all_non-judge_and_5_identifier_identical'] = 1 - DF['ratio_of_re-judge']
    DF.to_csv(os.path.join(args.output_merge_dir, 'analysis.csv'), index=False, encoding='utf-8-sig')


if __name__ == "__main__":
    import warnings
    import argparse
    warnings.filterwarnings("ignore")

    PROJ_ROOT_DIR = '.'

    parser = argparse.ArgumentParser()
    # Required parameters

    parser.add_argument(
        "--src_old_result_csv",
        default=f'{PROJ_ROOT_DIR}/unannotated_results.csv',
        type=str,
        required=False,
        help="path to old result .csv",
    )
    parser.add_argument(
        "--src_new_result_csv_dir",
        default=f'{PROJ_ROOT_DIR}/finetune0130_raw_ts_v5.5_fa_v4.4_section/',
        type=str,
        required=False,
        help="Folder path to new term matching result .csv",
    )
    parser.add_argument(
        "--src_new_result_csv",
        default=f'{PROJ_ROOT_DIR}/all_unannotated_data_0326.csv',
        type=str,
        required=False,
        help="Folder path to new term matching result .csv",
    )
    parser.add_argument(
        "--output_merge_dir",
        default=f'{PROJ_ROOT_DIR}/new_merge_old_unannotated_results/',
        type=str,
        required=False,
        help="Output folder for storing merged TS .csv",
    )
    parser.add_argument(
        "--BoW_ratio_threshold",
        default=0.6,
        type=float,
        required=False,
        help="threshold (minimum value) to accept string as similar when ratio of Bag-of-Word count vector difference between two strings less than or equal to that",
    )
    parser.add_argument(
        "--target_filenames",
        default=[
        "125_VF_SYN_TS_mkd_20231013_docparse_results.csv",
        "126_VF_SYN_TS_mkd_20230601_docparse_results.csv",
        "127_NBFI_SYN_TS_mkd_20230330_docparse_results.csv",
        "128_NBFI_SYN_TS_mkd_20230224_docparse_results.csv",
        "129_AW_SYN_TS_mkd_20230626_docparse_results.csv",
        "130_BL_SYN_TS_mkd_20230710_docparse_results.csv",
        "131_AF_SYN_TS_mkd_20230718_docparse_results.csv",
        "132_PF_SYN_TS_mkd_20230106_docparse_results.csv",
        "133_BL_PRJ_TS_mkd_20230926_docparse_results.csv",
        "134_GF_PRJ_TS_mkd_20230829_docparse_results.csv",
        "135_GL_PRJ_TS_mkd_20231004_docparse_results.csv",
        "136_BL_SYN_TS_mkd_20230418_docparse_results.csv",
        "137_GL_SYN_TS_mkd_20231121_docparse_results.csv",
        "138_GF_SYN_TS_mkd_20230130_docparse_results.csv",
        "139_NBFI_PRJ_TS_mkd_20231122_docparse_results.csv",
        "140_NBFI_SYN_TS_mkd_20230512_docparse_results.csv",
        "141_GL_SYN_TS_mkd_20231221_docparse_results.csv",
        "142_PF_SYN_TS_mkd_20230810_docparse_results.csv",
        "143_GF_SYN_TS_mkd_undated_docparse_results.csv",
        "144_NBFI_SYN_TS_mkd_20231031_docparse_results.csv",
        "145_GF_SYN_TS_mkd_20231031_docparse_results.csv",
        "146_BL_PRJ_TS_mkd_20230629_docparse_results.csv",
        "147_GL_PRJ_TS_mkd_20230817_docparse_results.csv",
        "148_GF_PRJ_TS_mkd_20230919_docparse_results.csv",
        "149_BL_PRJ_TS_mkd_20231102_docparse_results.csv"
    ],
        type=list,
        required=False,
        help="Output folder for storing merged TS .csv",
    )

    args = parser.parse_args()
    if not os.path.exists(args.output_merge_dir):
        os.mkdir(args.output_merge_dir)

    merge_old_new_unannot(args)
