import pandas as pd
import os
import re
import numpy as np
import ast

def remove_non_alphabet(s):
    '''
    Remove non-alphabet character from a string, e.g. 10. Documentations: -> Documentation
    '''
    import re
    import string

    SYMBOLS = ['●', '•', '·', '∙', '◉', '○', '⦿', '。', '■', '□', '☐', '⁃', '◆', '◇', '◈', '✦', '➢', '➣', '➤', '‣', '▶', '▷', '❖']
    bullet_symbols = ''.join(SYMBOLS)
    punct = re.escape(re.sub(r'\[|\(', '', string.punctuation))

    # remove leading non-alphabet character, e.g. 1. Borrower -> Borrower
    if re.match(r'^([0-9' + punct + bullet_symbols + r'\s]+)(.+)', s):
        s = re.match(r'^([0-9' + punct + bullet_symbols + r'\s]+)(.+)', s).groups()[1]

    # remove trailing non-alphabet character, e.g. Borrower1: -> Borrower
    if re.match(r'(.+[^0-9'+ punct + bullet_symbols +'])([0-9' + punct + bullet_symbols + r'\s]+)$', s):
        s = re.match(r'(.+[^0-9'+ punct + bullet_symbols +'])([0-9' + punct + bullet_symbols + r'\s]+)$', s).groups()[0]

    # replace multiple whitespaces into one whitespace
    s = re.sub(' +',' ',s)
    s = re.sub('"','',s)
    s = re.sub("'",'',s)
    
    return replace_ordinal_numbers(s)

def replace_ordinal_numbers(text):
    '''
    e.g. 1st -> First; 2nd -> Second
    '''
    import re
    from num2words import num2words
    
    re_results = re.findall('(\d+\s*(st|nd|rd|th))', text)
    for enitre_result, suffix in re_results:
        num = int(enitre_result[:-len(suffix)])
        text = text.replace(enitre_result, num2words(num, ordinal=True).capitalize())
    return text

if __name__ == "__main__":
    
    reviewed_ts_version = 'v4'
    parsed_ts_versions = ['v4', 'v4.1', 'v4.2', 'v4.3', 'v4.4', 'v5','v5.1', 'v5.2', 'v5.3', 'v5.4', 'v5.4.1']
    reviewed_ts_folder = f'/home/data2/ldrs_analytics/data/antd_ts_merge_fa/merge_antdTS_{reviewed_ts_version}_FA_v5/'
    output_analysis_folder = '/home/data2/ldrs_analytics/data/docparse_csv'
    files_with_TS_layout_issues = [
        '62_PF_SYN_TS_mkd_20220331_docparse.csv',
        '65_PF_SYN_TS_mkd_docparse.csv',
        'Acquisition and share pledge_(1) TS_docparse.csv',
        '(11) sample TS_docparse.csv'
    ]
    UAT = [
        "1_GL_SYN_TS_mkd_20221215_docparse.csv",
        "13_BL_SYN_TS_mkd_20220713_docparse.csv",
        "19_GL_SYN_TS_mkd_20220718_docparse.csv",
        "23_BL_SYN_TS_mkd_20220715_docparse.csv",
        "24_GF_PRJ_TS_mkd_20220225_docparse.csv",
        "25_NBFI_PRJ_TS_mkd_20220613_docparse.csv",
        "28_GF_PRJ_TS_mkd_20221111_docparse.csv",
        "29_PF_PRJ_TS_mkd_20200624_docparse.csv",
        "3_GF_SYN_TS_mkd_20221018_docparse.csv",
        "31_GL_PRJ_TS_mkd_20220630_docparse.csv",
        "33_BL_PRJ_TS_mkd_20200727_docparse.csv",
        "34_AWSP_PRJ_TS_mkd_20211230_docparse.csv",
        "41_AF_SYN_TS_mkd_20190307_docparse.csv",
        "43_AF_SYN_TS_mkd_20151101_docparse.csv",
        "45_AF_PRJ_TS_mkd_20170331_docparse.csv",
        "49_VF_PRJ_TS_mkd_20151130_docparse.csv",
        "54_VF_PRJ_TS_mkd_20191018_docparse.csv",
        "58_VF_SYN_TS_mkd_20111201_docparse.csv",
        "59_AWSP_SYN_TS_mkd_20210814_docparse.csv",
        "63_AW_SYN_TS_mkd_20221025_docparse.csv",
        "66_PF_SYN_TS_mkd_20230106_docparse.csv",
        "68_PF_SYN_TS_mkd_20221104_docparse.csv",
        "72_NBFI_SYN_TS_mkd_20221215_docparse.csv",
        "74_NBFI_SYN_TS_mkd_20220401_docparse.csv",
        "8_GF_SYN_TS_mkd_20230215_docparse.csv"
    ]
    LAYOUT_TYPE = {
        "TS_layout_table_with_header": [
            "(2) sample TS_1_docparse.csv",
            "(1) sample TS_docparse.csv",
            "(8) sample TS_docparse.csv",
            "64_AWSP_SYN_TS_mkd_20220622_docparse.csv",
            "65_PF_SYN_TS_mkd_docparse.csv",
            "66_PF_SYN_TS_mkd_20230106_docparse.csv"
        ],
        "TS_layout_list_form": [
            "5_AF_SYN_TS_mkd_20221213_docparse.csv",
            "30_GL_PRJ_TS_mkd_20220915_docparse.csv",
            "40_AF_SYN_TS_mkd_20190130_docparse.csv",
            "41_AF_SYN_TS_mkd_20190307_docparse.csv",
            "42_AF_SYN_TS_mkd_20180511_docparse.csv",
            "76_NBFI_SYN_TS_mkd_20211231_docparse.csv",
            "78_GF_PRJ_TS_mkd_20210722_docparse.csv"
        ]
    }
    
    df_all_section_check = pd.DataFrame(columns=[
        'TS_version',
        'macro-avg_missed_sec_rate',
        'micro-avg_missed_sec_rate',
        'UAT_macro-avg_missed_sec_rate',
        'UAT_micro-avg_missed_sec_rate',
        'total_missed_sec',
        'total_extra_sec',
        'total_reviewed_sec',
        'total_reviewed_sec_with_ann',
        'macro-avg_missed_label_rate',
        'micro-avg_missed_label_rate',
        'UAT_macro-avg_missed_label_rate',
        'UAT_micro-avg_missed_label_rate',
        'total_missed_label',
        'total_reviewed_label',
        'min_missed_sec_rate',
        'max_missed_sec_rate',
        'sd_missed_sec_rate',
        'range_missed_sec_rate',
        'miss_secs_count>=10_filenames'
        
    ], index=[i for i in range(len(parsed_ts_versions))])
    
    for i, parsed_ts_version in enumerate(parsed_ts_versions):
    
        parsed_ts_folder = f'/home/data2/ldrs_analytics/data/docparse_csv/TS_{parsed_ts_version}/'
        
        reviewed_ts_filenames = [f for f in os.listdir(reviewed_ts_folder) if f.endswith('.csv') and f not in files_with_TS_layout_issues and not (f.startswith('all') or f.startswith('sentence_pairs'))]
        parsed_ts_filenames = [f for f in os.listdir(parsed_ts_folder) if f.endswith('.csv') and not f.startswith('all') and f not in files_with_TS_layout_issues]

        check_section = []
        missed_sec_count_greater_ten = []
        
        for f in reviewed_ts_filenames:
            df_reviewed = pd.read_csv(os.path.join(reviewed_ts_folder, f))
            df_reviewed = df_reviewed.replace({np.nan: None, 'nan': None, 'None': None})
            section2label_count = df_reviewed.groupby('section')['fa_identifier'].nunique().to_dict()
            total_unique_label_count = sum(list(section2label_count.values()))
            
            reviewed_secs = sorted([k for k,v in section2label_count.items()])
            reviewed_secs_with_ann = sorted([k for k,v in section2label_count.items() if v>0])
            
            # df_reviewed['combined_ann'] = df_reviewed.apply(lambda i: str(i['clause_id']) + str(i['definition']) + str(i['schedule_id']),axis=1)

            # reviewed_secs = sorted(list(set([s.strip() for s in df_reviewed[df_reviewed.text_element=='section'].section.values.tolist() if s])))
            # df_reviewed_with_ann = df_reviewed[df_reviewed.combined_ann!='NoneNoneNone']
            # reviewed_secs_with_ann = sorted(list(set([s.strip() for s in df_reviewed_with_ann.section.values.tolist() if s])))
            
            df_parsed = pd.read_csv(os.path.join(parsed_ts_folder, f.replace('docparse_merge_FA','docparse')))
            df_parsed = df_parsed.replace({np.nan: None, 'nan': None, 'None': None})
            # parsed_secs = list(set([s for s in df_parsed[df_parsed.text_element=='section'].section if s]))
            parsed_secs = sorted(list(set([s for s in df_parsed.section.values.tolist() if s])))
            
            missed_sec = []
            missed_sec_with_antd = []
            missed_sec_label_count = 0
            partial_matched = []
            ### compare gt and match
            for reviewed_sec in reviewed_secs:
                processed_reviewed_sec = remove_non_alphabet(reviewed_sec)
                reviewed_sec_lower = ''.join(processed_reviewed_sec.split()).lower() # lowercase & remove multiple in-between whitespaces
                reviewed_sec_lower = re.sub('\"|\'|\d+', '', reviewed_sec_lower)
                match_flag = False
                for parsed_sec in parsed_secs:
                    processed_parsed_sec = remove_non_alphabet(parsed_sec)
                    parsed_sec_lower = ''.join(processed_parsed_sec.split()).lower() # lowercase & remove multiple in-between whitespaces
                    parsed_sec_lower = re.sub('\"+|\'+|\d+', '', parsed_sec_lower)
                    if parsed_sec_lower in reviewed_sec_lower and parsed_sec_lower != reviewed_sec_lower and not parsed_sec.lower() in [i.lower() for i in reviewed_secs]:
                        partial_matched.append(parsed_sec)
                    if reviewed_sec_lower == parsed_sec_lower:
                        match_flag = True
                        break
                    
                if not match_flag:
                    missed_sec.append(reviewed_sec)
                    if section2label_count[reviewed_sec]>0:
                        missed_sec_with_antd.append(reviewed_sec)
                    missed_sec_label_count += section2label_count[reviewed_sec]
            
            extra_parse_sec = [s for s in parsed_secs if s not in reviewed_secs]
            
            typ = None
            for t, lst in LAYOUT_TYPE.items():
                if f.replace('docparse_merge_FA','docparse') in lst:
                    typ = t
                    
            check_section.append(
                {
                    'filename': f.replace('docparse_merge_FA','docparse'),
                    'is_UAT': f.replace('docparse_merge_FA','docparse') in UAT,
                    'TS_layout_type': typ,
                    'reviewed_sec_count': len(reviewed_secs),
                    'reviewed_sec_with_ann_count': len(reviewed_secs_with_ann),
                    'parsed_sec_count': len(parsed_secs),
                    'reviewed_secs': reviewed_secs,
                    'reviewed_secs_with_ann': reviewed_secs_with_ann,
                    'parsed_secs': parsed_secs,
                    'missed_sec_count': len(missed_sec),
                    'missed_sec_with_annotation_count': len(missed_sec_with_antd),
                    'missed_sec_rate': len(missed_sec) / len(reviewed_secs),
                    'missed_sec_with_annotation_rate': len(missed_sec_with_antd) / len(reviewed_secs_with_ann),
                    'missed_label_by_sec_count': missed_sec_label_count,
                    'reviewed_label_by_sec_count': total_unique_label_count,
                    'missed_label_by_sec_rate': missed_sec_label_count / total_unique_label_count,
                    'missed_sec': missed_sec,
                    'partial_matched_sec': list(set(partial_matched)),
                    'extra_parse_sec': extra_parse_sec,
                    'extra_parse_sec_count': len(extra_parse_sec)
                }
            )
            if len(missed_sec)>=10:
                missed_sec_count_greater_ten.append(f[:9])
        
        df_check_sec = pd.DataFrame(data=check_section)
        
        df_all_section_check.loc[i,'TS_version'] = parsed_ts_version
        df_all_section_check.loc[i,'macro-avg_missed_sec_rate'] = df_check_sec.missed_sec_rate.mean()
        df_all_section_check.loc[i,'micro-avg_missed_sec_rate'] = df_check_sec.missed_sec_count.sum() / df_check_sec.reviewed_sec_count.sum()
        df_all_section_check.loc[i,'UAT_macro-avg_missed_sec_rate'] = df_check_sec[df_check_sec['is_UAT']==True].missed_sec_rate.mean()
        df_all_section_check.loc[i,'UAT_micro-avg_missed_sec_rate'] = df_check_sec[df_check_sec['is_UAT']==True].missed_sec_count.sum() / df_check_sec[df_check_sec['is_UAT']==True].reviewed_sec_count.sum()
        df_all_section_check.loc[i,'total_missed_sec'] = df_check_sec.missed_sec_count.sum()
        df_all_section_check.loc[i,'total_missed_sec_with_annotation'] = df_check_sec.missed_sec_with_annotation_count.sum()
        df_all_section_check.loc[i,'total_extra_sec'] = df_check_sec.extra_parse_sec_count.sum()
        df_all_section_check.loc[i,'total_reviewed_sec'] = df_check_sec.reviewed_sec_count.sum()
        df_all_section_check.loc[i,'total_reviewed_sec_with_ann'] = df_check_sec.reviewed_sec_with_ann_count.sum()
        df_all_section_check.loc[i,'macro-avg_missed_label_rate'] = df_check_sec.missed_label_by_sec_rate.mean()
        df_all_section_check.loc[i,'micro-avg_missed_label_rate'] = df_check_sec.missed_label_by_sec_count.sum() / df_check_sec.reviewed_label_by_sec_count.sum()
        df_all_section_check.loc[i,'UAT_macro-avg_missed_label_rate'] = df_check_sec[df_check_sec['is_UAT']==True].missed_label_by_sec_rate.mean()
        df_all_section_check.loc[i,'UAT_micro-avg_missed_label_rate'] = df_check_sec[df_check_sec['is_UAT']==True].missed_label_by_sec_count.sum() / df_check_sec[df_check_sec['is_UAT']==True].reviewed_label_by_sec_count.sum()
        df_all_section_check.loc[i,'total_missed_label'] = df_check_sec.missed_label_by_sec_count.sum()
        df_all_section_check.loc[i,'total_reviewed_label'] = df_check_sec.reviewed_label_by_sec_count.sum()
        df_all_section_check.loc[i,'min_missed_sec_rate'] = df_check_sec.missed_sec_rate.min()
        df_all_section_check.loc[i,'max_missed_sec_rate'] = df_check_sec.missed_sec_rate.max()
        df_all_section_check.loc[i,'sd_missed_sec_rate'] = df_check_sec.missed_sec_rate.std() 
        df_all_section_check.loc[i,'range_missed_sec_rate'] = df_check_sec.missed_sec_rate.max() - df_check_sec.missed_sec_rate.min()
        df_all_section_check.loc[i,'miss_secs_count>=10_filenames'] = str(missed_sec_count_greater_ten)
        
        df_check_sec = df_check_sec.sort_values(by='missed_sec_count', ascending=False)
        df_check_sec.loc[:, ['missed_sec_rate','missed_label_by_sec_rate']] = df_check_sec[['missed_sec_rate','missed_label_by_sec_rate']].map(lambda x: '{:.2%}'.format(x) if x or x != np.nan else x)
        df_check_sec.to_csv(os.path.join(output_analysis_folder,f'compare_sec_reviewedTS_{reviewed_ts_version}_TS_{parsed_ts_version}.csv'))
    
    df_all_section_check.loc[:, [
        'macro-avg_missed_sec_rate',
        'micro-avg_missed_sec_rate',
        'UAT_macro-avg_missed_sec_rate',
        'UAT_micro-avg_missed_sec_rate',
        'macro-avg_missed_label_rate',
        'micro-avg_missed_label_rate',
        'UAT_macro-avg_missed_label_rate',
        'UAT_micro-avg_missed_label_rate',
        'min_missed_sec_rate',
        'max_missed_sec_rate',
        'sd_missed_sec_rate',
        'range_missed_sec_rate']] = df_all_section_check[[
                                    'macro-avg_missed_sec_rate',
                                    'micro-avg_missed_sec_rate',
                                    'UAT_macro-avg_missed_sec_rate',
                                    'UAT_micro-avg_missed_sec_rate',
                                    'macro-avg_missed_label_rate',
                                    'micro-avg_missed_label_rate',
                                    'UAT_macro-avg_missed_label_rate',
                                    'UAT_micro-avg_missed_label_rate',
                                    'min_missed_sec_rate',
                                    'max_missed_sec_rate',
                                    'sd_missed_sec_rate',
                                    'range_missed_sec_rate']].map(lambda x: '{:.2%}'.format(x) if x or x != np.nan else x)
    df_all_section_check.to_csv(os.path.join(output_analysis_folder,f'all_compare_sec_reviewedTS_{reviewed_ts_version}_TS.csv'))