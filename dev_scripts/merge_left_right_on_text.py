import pandas as pd
import os
import nltk
import numpy as np
import re
import ast
from typing import Callable
from utils import multiprocess
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import string
from collections import defaultdict
import traceback

nltk.download('stopwords')
SYMBOLS = ['●', '•', '·', '∙', '◉', '○', '⦿', '。', '■', '□', '☐',
        '⁃', '◆', '◇', '◈', '✦', '➢', '➣', '➤', '‣', '▶', '▷', '❖', '_']
punct = [i for i in string.punctuation]
punct2 = ''.join(punct+SYMBOLS) + ' '
stop_words = stopwords.words('english') + ['etc.','',None] + punct + SYMBOLS

def tryfunction(value, func: Callable):
    '''
    try a function, if encounter exception or error, return a given value
    '''
    try:
        return func(value)
    except:
        return value

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
    if re.match(r'^([0-9' + punct + bullet_symbols + r'\s]+(?![ st| nd| rd| th]))(.+)', s):
        s = re.match(r'^([0-9' + punct + bullet_symbols + r'\s]+(?![ st| nd| rd| th]))(.+)', s).groups()[1]

    # remove trailing non-alphabet character, e.g. Borrower1: -> Borrower
    if re.match(r'([a-zA-Z ]+)([0-9' + punct + bullet_symbols + r'\s]+)$', s):
        s = re.match(r'([a-zA-Z ]+)([0-9' + punct + bullet_symbols + r'\s]+)$', s).groups()[0]

    # replace multiple whitespaces into one whitespace
    s = re.sub(' +',' ',s)
    return s

def calculate_BoW(texts: pd.Series, ori_text, new_text, count_vectorizer: CountVectorizer, threshold=0.6) -> pd.Series:
    
    if not ori_text and not new_text:
        return pd.Series([0, 0])
    elif not ori_text:
        return pd.Series([1, 0])
    elif not new_text:
        return pd.Series([0, 1])
    
    # if single word in both side of text
    if len(str(new_text).split()) == 1 and len(str(ori_text).split()) == 1:
        if new_text == ori_text:
            # identical word means no less no more word
            return pd.Series([0, 0])
        else:
            # different words means 1 word less and 1 word more
            return pd.Series([1, 1])

    try:
        matrix = count_vectorizer.fit_transform(texts)
    except ValueError:
        new_text = re.sub(r' +','',str(new_text))
        ori_text = re.sub(r' +','',str(ori_text))
        if new_text == ori_text:
            # identical word means no less no more word
            return pd.Series([0, 0])
        else:
            # different words means 1 word less and 1 word more
            return pd.Series([1, 1])
        
    diff_array = matrix[0] - matrix[1]

    if matrix[0].sum() == 0:
        more_ratio = 1
        less_ratio = 0
    else:
        more_ratio = -diff_array[diff_array < 0].sum() / matrix[0].sum()
        less_ratio = diff_array[diff_array > 0].sum() / matrix[0].sum()
        
    if more_ratio >= threshold:
        more_ratio = 1
    if less_ratio >= threshold:
        less_ratio = 1
    return pd.Series([round(more_ratio, 4), round(less_ratio, 4)])

def calculate_two_string_BoW(ori_text, new_text, count_vectorizer: CountVectorizer) -> pd.Series:
    import re

    if not ori_text and not new_text:
        return [0, 0]
    elif not ori_text:
        return [1, 0]
    elif not new_text:
        return [0, 1]

    # if single word in both side of text
    if len(str(new_text).split()) == 1 and len(str(ori_text).split()) == 1:
        if new_text == ori_text:
            # identical word means no less no more word
            return [0, 0]
        else:
            # different words means 1 word less and 1 word more
            return [1, 1]

    try:
        matrix = count_vectorizer.fit_transform([new_text, ori_text])
    except ValueError:
        new_text = re.sub(r' +', '', str(new_text))
        ori_text = re.sub(r' +', '', str(ori_text))
        if new_text == ori_text:
            # identical word means no less no more word
            return [0, 0]
        else:
            # different words means 1 word less and 1 word more
            return [1, 1]

    diff_array = matrix[0] - matrix[1]

    if matrix[0].sum() == 0:
        more_ratio = 1
        less_ratio = 0
    else:
        more_ratio = -diff_array[diff_array < 0].sum() / matrix[0].sum()
        less_ratio = diff_array[diff_array > 0].sum() / matrix[0].sum()

    return [round(more_ratio, 4), round(less_ratio, 4)]

def compare_texts_difference(ori_text: str, new_text: str) -> pd.Series:
    import difflib
    
    more_count_no_space = 0
    less_count_no_space = 0

    for s in difflib.ndiff(str(ori_text), str(new_text)):
        if s[0] == "+" and s[-1] != " ":
            more_count_no_space += 1
        elif s[0] == "-" and s[-1] != " ":
            less_count_no_space += 1

    length_of_string_no_space = sum(map(len, str(ori_text).split()))

    if length_of_string_no_space:
        more_ratio = more_count_no_space / length_of_string_no_space
        less_ratio = less_count_no_space / length_of_string_no_space
        return pd.Series([round(more_ratio, 4), round(less_ratio, 4)])
    else:
        return pd.Series([1, 0])

def merge_old_annot2new(group, check_section=True):

    # if there is any equal text / substring / ngram substring between original and df_left text
    existEqualText = group['isEqualText'].any()
    existSubstring = group['isSubstring'].any()
    left_df_prev_next_section = group['left_df_prev_next_section'].values[0]
    right_df_prev_next_section = group['right_df_prev_next_section'].values[0]
    if not check_section:
        group["isMatchedSection"] = True
        
    text = group['text'].values[0]
    
    if text.strip() in stop_words or re.match('^\d+\.*\d*\.*\d*$', text.strip()):
        g = pd.Series([None,None,None,None,None],index=['clause_id','definition', 'schedule_id', 'annotation', 'text_old'])
        g['merge_reason'] = 'ignore_stopwords_punctuations_empty'
        return g

    if existEqualText:
        indices = group[group['isEqualText'] == True].index.values[0]
        g = group.loc[indices, ['clause_id', 'definition', 'schedule_id', 'annotation', 'text_old']]
        g['merge_reason'] = 'identical_text'
        # g['difference_BoW_less'] = None
    elif existSubstring:
        # return the index of first occurrence of minimum in string length difference between original and updated text
        indices = group[((group['isSubstring'] == True)|(group["isBoWRatioLessThanThreshold"] == True))&(group["isMatchedSection"] == True)]['StringlengthDiff'].index
        g = group.loc[indices, ['clause_id', 'definition', 'schedule_id', 'annotation', 'text_old']].drop_duplicates()
        if len(g)>1:
            # consider group of isString text which are consecutive
            g = g[(g.index.diff()==1)|(g.index.diff(-1)==-1)]
        g = pd.Series(data={
            'clause_id': g['clause_id'].str.cat(sep='\n'),
            'definition': g['definition'].str.cat(sep='\n'),
            'schedule_id': g['schedule_id'].str.cat(sep='\n'),
            'annotation': g['annotation'].str.cat(sep='\n'),
            'text_old': g['text_old'].str.cat(sep='\n'),
        })
        g = g.map(lambda x: re.sub('(\n*nan)+$|^(\n*nan)+', '', x))
        g = g.map(lambda x: re.sub('(\n*None)+$|^(\n*None)+', '', x))
        g['merge_reason'] = 'substring'

        if not g['text_old'].strip():
            if group[(group["isMatchedSection"] == True)]['difference_BoW_less'].empty:
                g = pd.Series([None,None,None,None,None],index=['clause_id','definition', 'schedule_id', 'annotation', 'text_old'])
                g['merge_reason'] = 'not_found'
                return g

            indices = group[(group["isMatchedSection"] == True)]['difference_BoW_less'].idxmin()
            g = group.loc[indices, ['clause_id', 'definition', 'schedule_id',
                                    'annotation', 'text_old']]
            g['merge_reason'] = 'min_BoW_less_diff_ratio'
    else:
        if group[(group["isMatchedSection"] == True)]['difference_BoW_less'].empty:
            g = pd.Series([None,None,None,None,None],index=['clause_id','definition', 'schedule_id', 'annotation', 'text_old'])
            g['merge_reason'] = 'not_found'
            return g
        indices = group[(group["isMatchedSection"] == True)]['difference_BoW_less'].idxmin()
        g = group.loc[indices, ['clause_id','definition', 'schedule_id', 'annotation', 'text_old']]
        g['merge_reason'] = 'min_BoW_less_diff_ratio'

    return g

def merge_annot2tm(group, check_section=True):
    # if there is any equal text / substring / ngram substring between original and df_left text
    existEqualText = group['isEqualText'].any()
    existSubstring = group['isSubstring'].any()
    left_df_prev_next_section = group['left_df_prev_next_section'].values[0]
    right_df_prev_next_section = group['right_df_prev_next_section'].values[0]
    text = group['text'].values[0]
    if not check_section:
        group["isMatchedSection"] = True
    
    if text.strip() in stop_words or re.match('^\d+\.*\d*\.*\d*$', text.strip()):
        g = pd.Series([None,None,None,None,None],index=['match_term_list','identifier_list','similarity_list','match_type_list', 'text_old'])
        g['merge_reason'] = 'ignore_stopwords_punctuations_empty'

        return g

    if existEqualText:
        indices = group[group['isEqualText'] == True].index.values[0]
        g = group.loc[indices, ['match_term_list','identifier_list','similarity_list','match_type_list', 'text_old']]
        g['merge_reason'] = 'identical_text'
        # g['difference_BoW_less'] = None
    elif existSubstring:
        # return the index of first occurrence of minimum in string length difference between original and updated text
        indices = group[((group['isSubstring'] == True)|(group["isBoWRatioLessThanThreshold"] == True))&(group["isMatchedSection"] == True)]['StringlengthDiff'].index
        g = group.loc[indices, ['match_term_list','identifier_list','similarity_list','match_type_list', 'text_old']].drop_duplicates()
        if len(g)>1:
            # consider group of isString text which are consecutive
            g = g[(g.index.diff()==1)|(g.index.diff(-1)==-1)]
        # g = g.astype(str)
        # g = g.agg(sum)
        g = pd.Series(data={
            'match_term_list': g['match_term_list'].str.cat(),
            'identifier_list': g['identifier_list'].str.cat(),
            'similarity_list': g['similarity_list'].str.cat(),
            'match_type_list': g['match_type_list'].str.cat(),
            'text_old': g['text_old'].str.cat(),
        })
        g = g.map(lambda x: re.sub('(None)+$|^(None)+', '', x))
        g = g.apply(lambda x: None if isinstance(x,(int, float)) else x)
        g = g.apply(lambda x: x.replace(")None",')').replace("None(",'(').replace("]None",']').replace("None[",'['))
        g = g.apply(lambda x: x.replace(")nan",')').replace("nan(",'(').replace("]nan",']').replace("nan[",'[').replace("][",', ').replace(")('",", '").replace(')("',', "').replace('](',', ').replace(')[',', ') if isinstance(x,str) else x)
        g = g.apply(lambda x: re.sub(r'\)\(0.',', 0.',x) if isinstance(x,str) else x)
        g = g.apply(lambda x: re.sub(r'^\(','[',x) if isinstance(x,str) else x)
        g = g.apply(lambda x: re.sub(r'\)$',']',x) if isinstance(x,str) else x)
        g['merge_reason'] = 'substring'

        if not g['text_old'] or not g['text_old'].strip():
            if group[(group["isMatchedSection"] == True)]['difference_BoW_less'].empty:
                g = pd.Series([None,None,None,None,None],index=['match_term_list','identifier_list','similarity_list','match_type_list', 'text_old'])
                g['merge_reason'] = 'not_found'
                return g
            indices = group[(group["isMatchedSection"] == True)]['difference_BoW_less'].idxmin()
            g = group.loc[indices, ['match_term_list', 'identifier_list', 'similarity_list',
                                    'match_type_list', 'text_old']]
            g['merge_reason'] = 'min_BoW_less_diff_ratio'
    else:
        if group[(group["isMatchedSection"] == True)]['difference_BoW_less'].empty:
            g = pd.Series([None,None,None,None,None],index=['match_term_list','identifier_list','similarity_list','match_type_list', 'text_old'])
            g['merge_reason'] = 'not_found'
            return g
        indices = group[(group["isMatchedSection"] == True)]['difference_BoW_less'].idxmin()
        g = group.loc[indices, ['match_term_list','identifier_list','similarity_list','match_type_list', 'text_old']]
        g['merge_reason'] = 'min_BoW_less_diff_ratio'

    return g

def split_newline(row):
    '''
    To split each string in a list at newline character
    @param row: list of string that may contain newline character
    @type row: list
    @return : list of string that splitting at newline character and insert to the original list
    @rtype : list
    '''
    if not row:
        return [row]
    for index, s_id in enumerate(row):
        if '\n' in s_id:
            row.remove(s_id)
            s_ids = s_id.splitlines()
            for i in s_ids:
                row.insert(index, i)
    return row

def batch_merge_left_right(input_file_list, args):
    for file in input_file_list:
        try:
            merge_left_right(file, args)
        except Exception as e:
            print(f'Error occurs in: {file} with error:\n{e}')
            traceback.print_exc()

def merge_left_right(file, args):
    import pandas as pd
    import os

    print(f'filename: {file}')
    if args.is_merge_tm:
        file = file.replace('docparse_results','docparse').replace('docparse','docparse_results')
    if not os.path.exists(os.path.join(args.src_right_table_csv_dir, file)):
        print(f'Skipped. No such file or directory: {os.path.join(args.src_right_table_csv_dir, file)}')
        return
    df_right = pd.read_csv(os.path.join(args.src_right_table_csv_dir, file)).astype(str)
    if args.is_merge_tm:
        # if is_merge_tm, df_right = term matching table, df_left = annotated ts table
        df_right = df_right.rename(columns={"TS_term": "section", "TS_text": "text"})
        df_right = df_right.replace({np.nan: None, 'None': None, 'nan': None})
        df_right[['match_term_list', 'identifier_list', 'similarity_list', 'match_type_list']] = df_right[['match_term_list', 'identifier_list', 'similarity_list', 'match_type_list']].apply(lambda x: tryfunction(x, ast.literal_eval), axis=1)
        df_right[['match_term_list', 'identifier_list', 'similarity_list', 'match_type_list']] = df_right[['match_term_list', 'identifier_list', 'similarity_list', 'match_type_list']].apply(lambda x: list(x) if isinstance(x,tuple) else x, axis=1)
        
    # df_right = df_right[~df_right.text_element.str.contains('section')]
    # extract section and text only from df_right TS
    df_right['processed_text'] = df_right['text'].map(lambda x: x.strip(punct2) if x else x)
    df_right = df_right.loc[:, ~df_right.columns.isin(['index', 'text_block_id', 'phrase_id', 'text_element', 'list_id', 'keyphrase', 'text_granularity', 'docparse_datetime'])]
    df_right['processed_section'] = df_right['section'].map(lambda x: x.replace(r':', '').replace(' ', '').strip() if x else x)
    df_right.loc[df_right.section.notnull(), 'processed_section'] = df_right[df_right.section.notnull()]['processed_section'].map(remove_non_alphabet)
    df_right['prev_processed_section'] = df_right['processed_section'].shift().where(df_right['processed_section'].ne(df_right['processed_section'].shift())).ffill()
    df_right['next_processed_section'] = df_right['processed_section'].where(df_right['processed_section'].ne(df_right['processed_section'].shift())).shift(-1).bfill()
    df_right['left_df_prev_next_section'] = df_right['prev_processed_section'] + df_right['next_processed_section']
    df_right = df_right.drop(['section','prev_processed_section','next_processed_section'], axis=1)
    df_right = df_right.replace({np.nan: None, 'None': None, 'nan': None})
    
    # df_right.to_csv(os.path.join(os.path.dirname(args.output_merge_dir), file.split('.')[0]+'_right_df.csv'), index=False, encoding='utf-8-sig')

    df_left = pd.read_csv(os.path.join(args.src_left_table_csv_dir, file.replace('_results',''))).astype(str)
    df_left['processed_text'] = df_left['text'].map(lambda x: x.strip(punct2) if x else x)
    df_left['processed_section'] = df_left['section'].map(lambda x: x.replace(r':', '').replace(' ', '').strip() if x else x)
    df_left.loc[df_left.section.notnull(), 'processed_section'] = df_left[df_left.section.notnull()]['processed_section'].map(remove_non_alphabet)
    # exclude all annotation columns in a updated, clean version of docparsed TS
    df_left = df_left.loc[:, ~df_left.columns.isin(['clause_id', 'definition', 'schedule_id', 'annotation'])]
    df_left['prev_processed_section'] = df_left['processed_section'].shift().where(df_left['processed_section'].ne(df_left['processed_section'].shift())).ffill()
    df_left['next_processed_section'] = df_left['processed_section'].where(df_left['processed_section'].ne(df_left['processed_section'].shift())).shift(-1).bfill()
    df_left['right_df_prev_next_section'] = df_left['prev_processed_section'] + df_left['next_processed_section']
    df_left = df_left.drop(['prev_processed_section','next_processed_section'], axis=1)
    df_left = df_left.replace({np.nan: None, 'None': None, 'nan': None})

    # df_left.to_csv(os.path.join(os.path.dirname(args.output_merge_dir), file.split('.')[0]+'_left_df.csv'), index=False, encoding='utf-8-sig')
    
    # merge tables on page_id
    df_merge = pd.merge(df_left, df_right, how="left", on=['page_id'])
    df_merge = df_merge.rename(columns={name: name.replace('_x', '').replace('_y', '_old') for name in df_merge.columns.tolist()})

    # check if updated_text is in text_old or text_old is in updated_text
    df_merge['isEqualText'] = df_merge.apply(lambda x: str(x['text']).lower().strip(" :") == str(x['text_old']).lower().strip(" :") , axis=1)
    df_merge['isSubstring'] = df_merge.apply(lambda x: (str(x['processed_text_old']) in str(x['processed_text'])) and (str(x['processed_text_old']).lower().strip() not in [None,'']+stop_words), axis=1) # str(x['processed_text']).lower() in str(x['processed_text_old']).lower() or 
    count_vectorizer = CountVectorizer()

    update_sec_map_old_sec = defaultdict(str)
    for updated_sec in list(set(df_left['processed_section'].tolist())):
        ratio = 1
        for sec in list(set(df_right['processed_section'].tolist())):
            tmp_ratio = calculate_two_string_BoW(sec, updated_sec, count_vectorizer)[1]
            if tmp_ratio < ratio:
                ratio = tmp_ratio
                update_sec_map_old_sec[updated_sec] = sec
                
    df_merge['isMatchedSection'] = df_merge[["processed_section", "processed_section_old"]].apply(lambda i: update_sec_map_old_sec.get(i.processed_section, None)==i.processed_section_old, axis=1)
    
    # Calculate cosine similarity by Bag-Of-Words
    df_merge[["difference_BoW_more", "difference_BoW_less"]] = df_merge[["processed_text", "processed_text_old"]].apply(lambda i: calculate_BoW(i, i.processed_text_old, i.processed_text, count_vectorizer), axis=1)
    df_merge["isBoWRatioLessThanThreshold"] = (df_merge["difference_BoW_less"] <= args.BoW_ratio_threshold) & (df_merge["difference_BoW_more"] <= args.BoW_ratio_threshold)
    # Calculate ratios of differences between to texts
    df_merge[["difference_distance_more", "difference_distance_less"]] = df_merge[["processed_text", "processed_text_old"]].apply(lambda i: compare_texts_difference(i.processed_text_old, i.processed_text),axis=1)
    
    # calculate the string length difference as abs(len(text)-len(text_old))
    df_merge['StringlengthDiff'] = (df_merge['processed_text'].astype(str).map(len) - df_merge['processed_text_old'].astype(str).map(len)).abs()
    df_merge = df_merge.replace({'': None, np.nan: None, 'None': None, 'nan': None})
    # df_merge.to_csv(os.path.join(os.path.dirname(args.output_merge_dir), file.split('.')[0]+'_merge.csv'), index=False, encoding='utf-8-sig')
    
    # merge tables on processed_section
    df_merge2 = pd.merge(df_left, df_right, how="left", on=['processed_section'])
    df_merge2 = df_merge2.rename(columns={name: name.replace('_x', '').replace('_y', '_old') for name in df_merge2.columns.tolist()})
    
    # check if updated_text is in text_old or text_old is in updated_text
    df_merge2['isEqualText'] = df_merge2.apply(lambda x: str(x['text']).lower().strip(" :") == str(x['text_old']).lower().strip(" :") , axis=1)
    df_merge2['isSubstring'] = df_merge2.apply(lambda x: (str(x['processed_text_old']) in str(x['processed_text'])) and (str(x['processed_text_old']).lower().strip() not in [None,'']+stop_words), axis=1) # str(x['processed_text']).lower() in str(x['processed_text_old']).lower() or 
    count_vectorizer = CountVectorizer()

    # Calculate cosine similarity by Bag-Of-Words
    df_merge2[["difference_BoW_more", "difference_BoW_less"]] = df_merge2[["processed_text", "processed_text_old"]].apply(lambda i: calculate_BoW(i, i.processed_text_old, i.processed_text, count_vectorizer), axis=1)
    df_merge2["isBoWRatioLessThanThreshold"] = (df_merge2["difference_BoW_less"] <= args.BoW_ratio_threshold) & (df_merge2["difference_BoW_more"] <= args.BoW_ratio_threshold)
    # Calculate ratios of differences between to texts
    df_merge2[["difference_distance_more", "difference_distance_less"]] = df_merge2[["processed_text", "processed_text_old"]].apply(lambda i: compare_texts_difference(i.processed_text_old, i.processed_text),axis=1)
    
    # calculate the string length difference as abs(len(text)-len(text_old))
    df_merge2['StringlengthDiff'] = (df_merge2['processed_text'].astype(str).map(len) - df_merge2['processed_text_old'].astype(str).map(len)).abs()
    df_merge2 = df_merge2.replace({'': None, np.nan: None, 'None': None, 'nan': None})
    # df_merge2.to_csv(os.path.join(os.path.dirname(args.output_merge_dir), file.split('.')[0]+'_merge2.csv'), index=False, encoding='utf-8-sig')
    
    if args.is_merge_tm:
        match_annot_df = df_merge.groupby(['index', 'page_id', 'phrase_id', 'section', 'text'],dropna=False).apply(lambda group: merge_annot2tm(group))
        match_annot_df = match_annot_df.replace({'': None, np.nan: None, 'None': None, 'nan': None})
        # match_annot_df.to_csv(os.path.join(os.path.dirname(args.output_merge_dir), file.split('.')[0]+'_match_annot_df.csv'), index=False, encoding='utf-8-sig')
        match_annot_df2 = df_merge2.groupby(['index', 'page_id', 'phrase_id', 'section', 'text'],dropna=False).apply(lambda group: merge_annot2tm(group, check_section=False)) # .reset_index()
        match_annot_df2 = match_annot_df2.replace({'': None, np.nan: None, 'None': None, 'nan': None})
        # match_annot_df2.to_csv(os.path.join(os.path.dirname(args.output_merge_dir), file.split('.')[0]+'_match_annot_df2.csv'), index=False, encoding='utf-8-sig')
        tmp = match_annot_df[match_annot_df['merge_reason']=='not_found'].drop(['match_term_list','identifier_list','similarity_list','match_type_list', 'text_old','merge_reason'],axis=1)
        match_annot_df.loc[match_annot_df['merge_reason']=='not_found'] = pd.merge(tmp, match_annot_df2, how="inner", on=['index','page_id','phrase_id','section','text'])
    else:
        match_annot_df = df_merge.groupby(['index', 'page_id', 'phrase_id', 'section', 'text'],dropna=False).apply(lambda group: merge_old_annot2new(group))
        match_annot_df = match_annot_df.replace({'': None, np.nan: None, 'None': None, 'nan': None})
        match_annot_df2 = df_merge2.groupby(['index', 'page_id', 'phrase_id', 'section', 'text'],dropna=False).apply(lambda group: merge_old_annot2new(group, check_section=False))
        match_annot_df2 = match_annot_df2.replace({'': None, np.nan: None, 'None': None, 'nan': None})
        tmp = match_annot_df[match_annot_df['merge_reason']=='not_found'].drop(['clause_id','definition','schedule_id','annotation','text_old','merge_reason'],axis=1)
        match_annot_df.loc[match_annot_df['merge_reason']=='not_found'] = pd.merge(tmp, match_annot_df2, how="inner", on=['index','page_id','phrase_id','section','text'])
    
    df_merge_true = pd.merge(df_left, match_annot_df, how="left", on=[
                             'page_id', 'text'])  # df_merge.iloc[true_instance_indices]
    df_merge_true = df_merge_true.drop_duplicates(['index', 'page_id', 'phrase_id', 'section', 'text'])
    df_merge_true = df_merge_true.drop(['processed_section','right_df_prev_next_section'], axis=1)

    df_merge_true[['index', 'text_block_id', 'phrase_id']] = df_merge_true[['index', 'text_block_id', 'phrase_id']].astype(int)
    df_left[['index', 'text_block_id', 'phrase_id']] = df_left[['index', 'text_block_id', 'phrase_id']].astype(int)
    df_merge_true = df_merge_true.replace({'': None, 'nan': None, np.nan: None, 'None': None})
    df_merge_true = df_merge_true.reindex(columns=(list([a for a in df_merge_true.columns if a != 'docparse_datetime']+['docparse_datetime'])))
    df_merge_true = df_merge_true.sort_values(by=['index', 'text_block_id', 'page_id', 'phrase_id'])
    
    if args.is_merge_tm:
        df_merge_true = df_merge_true.rename(columns={"text_old": "original_text_from_tm"})
    else:
        df_merge_true = df_merge_true.rename(columns={"text_old": "original_text_from_antdTS"})
    
    # if not args.is_merge_tm:
    #     df_merge_true.loc[(df_merge_true.text.str.len() <= 1) | (df_merge_true.text_element.str.contains('section')), ['clause_id', 'definition', 'schedule_id', 'annotation']] = None  # erase all annotation for text with length less than or equal 1
    # else:
    #     df_merge_true[['match_term_list', 'identifier_list', 'similarity_list', 'match_type_list']] = df_merge_true[['match_term_list', 'identifier_list', 'similarity_list', 'match_type_list']].apply(lambda x: None if not isinstance(x, (list,tuple)) else x, axis=1)
    right_df2 = df_left.merge(df_merge_true, how='left', on=['index', 'text_block_id', 'page_id', 'phrase_id'])['text_y']

    assert len(df_merge_true) == len(df_left), f'In filename: {file}, the updated TS left join annotated TS is not as same length as the updated TS. Please check.'

    df_merge_true.to_csv(os.path.join(args.output_merge_dir, file), index=False, encoding='utf-8-sig')

if __name__ == "__main__":
    import warnings
    import argparse
    warnings.filterwarnings("ignore")

    reviewed_version = '_v4'
    ts_version = '_v5.5'
    PROJ_ROOT_DIR = '/home/data2/ldrs_analytics'

    parser = argparse.ArgumentParser()
    # Required parameters

    parser.add_argument(
        "--src_left_table_csv_dir",
        default=f'{PROJ_ROOT_DIR}/data/docparse_csv/TS{ts_version}/',
        type=str,
        required=False,
        help="Folder path to parsed TS .csv",
    )
    parser.add_argument(
        "--src_right_table_csv_dir",
        default=f'{PROJ_ROOT_DIR}/data/reviewed_antd_ts{reviewed_version}/',
        type=str,
        required=False,
        help="Folder path to reviewed annotated TS .csv",
    )
    parser.add_argument(
        "--output_merge_dir",
        default=f'{PROJ_ROOT_DIR}/data/antd_ts_merge_ts/antd{reviewed_version}_merge_ts{ts_version}',
        type=str,
        required=False,
        help="Output folder for storing merged TS .csv",
    )
    parser.add_argument(
        "--is_merge_tm",
        default=False,
        type=bool,
        required=False,
        help="is merging attempt to annotated ts (left) merge term match result (right)",
    )
    parser.add_argument(
        "--BoW_ratio_threshold",
        default=0.6,
        type=float,
        required=False,
        help="threshold (minimum value) to accept string as similar when ratio of Bag-of-Word count vector difference between two strings less than or equal to that",
    )

    args = parser.parse_args()

    if not os.path.exists(args.output_merge_dir):
        os.mkdir(args.output_merge_dir)

    file_list = sorted([file for file in os.listdir(
        args.src_left_table_csv_dir) if file.endswith('.csv')])
    # for file in file_list:
    #     merge_left_right(file ,args)
    
    # file_list = ['43_AF_SYN_TS_mkd_20151101_docparse.csv']
    if args.is_merge_tm:
        print('Processing annotated TS left join term matching of parsed TS & parsed FA on TS page_id ...\n')
    else:
        print('Processing parsed TS left join annotated TS on page_id, bringing annotation to parsed TS ...\n')
    multiprocess(batch_merge_left_right, file_list, args=args)
    if args.is_merge_tm:
        print('Completed merging term matching results to annotated TS\n')
    else:
        print('Completed merging annotated TS to updated TS\n')
