import pandas as pd
import os
import nltk
import numpy as np
import re
import ast
from typing import Union, Dict, Tuple, Callable
from utils import multiprocess
import nltk
from nltk.corpus import stopwords
 
nltk.download('stopwords')
stop_words = stopwords.words('english') + ['etc.']

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

def match_word_count(a_str, b_str):
    from nltk import word_tokenize
    import re
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    # removing punctuations in string using regex
    a_str = re.sub(r'[^\w\s]', '', str(a_str))
    b_str = re.sub(r'[^\w\s]', '', str(b_str))
    # tokenize sentence into list of words
    a_str_words = word_tokenize(a_str.lower())
    b_str_words = word_tokenize(b_str.lower())
    a_in_b_count = sum([a in b_str_words for a in a_str_words])
    b_in_a_count = sum([b in a_str_words for b in b_str_words])
    return max(a_in_b_count, b_in_a_count)

def is_ngram_substring(a_str, b_str):
    '''
    N-grams are continuous sequences of words or symbols, or tokens in a document.
    e.g. Trigram of “I reside in Bengaluru” = [“I reside in”, “reside in Bengaluru”]
    Parameters
    ----------
    a_str: string to be split into (n-1)grams tuples and compare with b_str
    b_str: to be tested as a whole string

    Returns: boolean that indicate if (n-1)grams of a_str is in b_str
    -------
    '''

    from nltk import ngrams
    n = len(str(a_str).split())
    m = len(str(b_str).split())
    if n <= 1 or m <= 1:
        return False, 0

    for k in reversed(range(2, n - 1)):
        n_minus_1_grams = ngrams(str(a_str).lower().split(), k)
        for grams in n_minus_1_grams:
            gram_str = ' '.join(grams)
            if gram_str in str(b_str).lower():
                word_count = len(gram_str.split(' '))
                return True, word_count

    for k in reversed(range(2, m - 1)):
        m_minus_1_grams = ngrams(str(b_str).lower().split(), m - 1)
        for grams in m_minus_1_grams:
            gram_str = ' '.join(grams)
            if gram_str in str(a_str).lower():
                word_count = len(gram_str.split(' '))
                return True, word_count

    return False, 0

def merge_old_annot2new(group):
    # if there is any equal text / substring / ngram substring between original and updated text
    existEqualText = group['isEqualText'].any()
    existSubstring = group['isSubstring'].any()
    existNgramsSubstring = group['isNgramsSubstring'].any()
    annotated_prev_next_section = group['annotated_prev_next_section'].values[0]
    updated_prev_next_section = group['updated_prev_next_section'].values[0]
    text = group['text'].values[0]
    
    if text.strip() in stop_words or re.match('^\d+\.*\d*\.*\d*$', text.strip()):
        return pd.Series([None,None,None,None],index=['clause_id','definition', 'schedule_id', 'annotation'])

    if existEqualText:
        indices = group[group['isEqualText'] == True].index.values[0]
        g = group.loc[indices, ['clause_id', 'definition', 'schedule_id', 'annotation']]
    elif existSubstring:
        # return the index of first occurrence of minimum in string length difference between original and updated text
        indices = group[group['isSubstring'] == True]['StringlengthDiff'].index
        g = group.loc[indices, ['clause_id', 'definition', 'schedule_id', 'annotation']].drop_duplicates()
        g = pd.Series(data={
            'clause_id': g['clause_id'].str.cat(sep='\n'),
            'definition': g['definition'].str.cat(sep='\n'),
            'schedule_id': g['schedule_id'].str.cat(sep='\n'),
            'annotation': g['annotation'].str.cat(sep='\n')
        })
        g = g.map(lambda x: re.sub('(\n*nan)+$|^(\n*nan)+', '', x))
        # return group[group['isSubstring'] == True]['StringlengthDiff'].index #.idxmin()
    elif existNgramsSubstring:
        # return the index of first occurrence of maximum in match word count between original and updated text
        indices = group[group['isNgramsSubstring']== True]['matchWordCount'].idxmax()
        g = group.loc[indices, ['clause_id','definition', 'schedule_id', 'annotation']]
    else:
        # if annotated_prev_next_section != updated_prev_next_section:
        #     return pd.Series([None,None,None,None],index=['clause_id','definition', 'schedule_id', 'annotation'])
        # if all string matching fails, choose the updated text with the largest match word count
        indices = group['matchWordCount'].idxmax()
        g = group.loc[indices, ['clause_id','definition', 'schedule_id', 'annotation']]

    return g

def merge_annot2tm(group):
    # if there is any equal text / substring / ngram substring between original and updated text
    existEqualText = group['isEqualText'].any()
    existSubstring = group['isSubstring'].any()
    existNgramsSubstring = group['isNgramsSubstring'].any()
    annotated_prev_next_section = group['annotated_prev_next_section'].values[0]
    updated_prev_next_section = group['updated_prev_next_section'].values[0]
    text = group['text'].values[0]
    
    if text.strip() in stop_words or re.match('^\d+\.*\d*\.*\d*$', text.strip()):
        return pd.Series([None,None,None,None],index=['match_term_list','identifier_list','similarity_list','match_type_list'])

    if existEqualText:
        indices = group[group['isEqualText'] == True].index.values[0]
        g = group.loc[indices, ['match_term_list','identifier_list','similarity_list','match_type_list']]
    elif existSubstring:
        # return the index of first occurrence of minimum in string length difference between original and updated text
        indices = group[group['isSubstring'] == True]['StringlengthDiff'].index
        g = group.loc[indices, ['match_term_list','identifier_list','similarity_list','match_type_list']].drop_duplicates()
        g = g.agg(sum)
        g = g.apply(lambda x: None if isinstance(x,(int, float)) else x)
        g = g.apply(lambda x: x.replace(")nan",')').replace("nan(",'(').replace("]nan",']').replace("nan[",'[').replace("][",', ').replace(")('",", '").replace(')("',', "').replace('](',', ').replace(')[',', ') if isinstance(x,str) else x)
        g = g.apply(lambda x: re.sub(r'\)\(0.',', 0.',x) if isinstance(x,str) else x)
        g = g.apply(lambda x: re.sub(r'^\(','[',x) if isinstance(x,str) else x)
        g = g.apply(lambda x: re.sub(r'\)$',']',x) if isinstance(x,str) else x)
        # return group[group['isSubstring'] == True]['StringlengthDiff'].index #.idxmin()
    elif existNgramsSubstring:
        # return the index of first occurrence of maximum in match word count between original and updated text
        indices = group[group['isNgramsSubstring']== True]['matchWordCount'].idxmax()
        g = group.loc[indices, ['match_term_list','identifier_list','similarity_list','match_type_list']]
    else:
        # if annotated_prev_next_section != updated_prev_next_section:
        #     return pd.Series([None,None,None,None],index=['match_term_list','identifier_list','similarity_list','match_type_list'])
        # if all string matching fails, choose the updated text with the largest match word count
        indices = group['matchWordCount'].idxmax()
        g = group.loc[indices, ['match_term_list','identifier_list','similarity_list','match_type_list']]

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

def batch_merge_old_new(input_file_list, args):
    for file in input_file_list:
        try:
            merge_old_new(file, args)
        except Exception as e:
            print(f'Error occurs in: {file} with error:\n{e}')

def merge_old_new(file, args):
    import pandas as pd
    import os

    print(f'filename: {file}')
    annotated = pd.read_csv(os.path.join(args.src_antd_ts_dir, file)).astype(str)
    if args.is_merge_tm:
        # if is_merge_tm, annotated = term matching table, update = annotated ts table
        annotated = annotated.rename(columns={"TS_term": "section", "TS_text": "text"})
        annotated = annotated.replace({np.nan: None, 'None': None})
        annotated[['match_term_list', 'identifier_list', 'similarity_list', 'match_type_list']] = annotated[['match_term_list', 'identifier_list', 'similarity_list', 'match_type_list']].apply(lambda x: tryfunction(x, ast.literal_eval), axis=1)
    # annotated = annotated[~annotated.text_element.str.contains('section')]
    # extract section and text only from annotated TS
    annotated = annotated.loc[:, ~annotated.columns.isin(['index', 'text_block_id', 'phrase_id', 'text_element', 'list_id', 'keyphrase', 'text_granularity', 'docparse_datetime'])]
    annotated['processed_section'] = annotated['section'].map(lambda x: x.replace(r':', '').replace(' ', '').strip())
    annotated.loc[annotated.section.notnull(), 'processed_section'] = annotated[annotated.section.notnull()]['processed_section'].map(remove_non_alphabet)
    annotated['prev_processed_section'] = annotated['processed_section'].shift().where(annotated['processed_section'].ne(annotated['processed_section'].shift())).ffill()
    annotated['next_processed_section'] = annotated['processed_section'].where(annotated['processed_section'].ne(annotated['processed_section'].shift())).shift(-1).bfill()
    annotated['annotated_prev_next_section'] = annotated['prev_processed_section'] + annotated['next_processed_section']
    annotated = annotated.drop(['section','prev_processed_section','next_processed_section'], axis=1)

    updated = pd.read_csv(os.path.join(args.src_ts_csv_dir, file.replace('_results',''))).astype(str)
    updated['processed_section'] = updated['section'].map(lambda x: x.replace(r':', '').replace(' ', '').strip())
    updated.loc[updated.section.notnull(), 'processed_section'] = updated[updated.section.notnull()]['processed_section'].map(remove_non_alphabet)
    # exclude all annotation columns in a updated, clean version of docparsed TS
    updated = updated.loc[:, ~updated.columns.isin(['clause_id', 'definition', 'schedule_id', 'annotation'])]
    updated['prev_processed_section'] = updated['processed_section'].shift().where(updated['processed_section'].ne(updated['processed_section'].shift())).ffill()
    updated['next_processed_section'] = updated['processed_section'].where(updated['processed_section'].ne(updated['processed_section'].shift())).shift(-1).bfill()
    updated['updated_prev_next_section'] = updated['prev_processed_section'] + updated['next_processed_section']
    updated = updated.drop(['prev_processed_section','next_processed_section'], axis=1)

    # updated.to_csv(os.path.join(os.path.dirname(args.output_merge_dir), file.split('.')[0]+'_updated.csv'), index=False, encoding='utf-8-sig')
    # annotated.to_csv(os.path.join(os.path.dirname(args.output_merge_dir), file.split('.')[0]+'_annotated.csv'), index=False, encoding='utf-8-sig')
    df_merge = pd.merge(updated, annotated, how="left", on=['page_id'])
    df_merge = df_merge.rename(columns={'text_x': 'text', 'text_y': 'original_text'})

    # check if updated_text is in original_text or original_text is in updated_text
    df_merge['isEqualText'] = df_merge.apply(lambda x: str(x['text']).lower() == str(x['original_text']).lower(), axis=1)
    df_merge['isSubstring'] = df_merge.apply(lambda x: str(x['text']).lower() in str(x['original_text']).lower() or str(x['original_text']).lower() in str(x['text']).lower(), axis=1)
    df_merge['isNgramsSubstring'], df_merge['matchWordCount'] = zip(*df_merge.apply(lambda x: is_ngram_substring(x['text'], x['original_text']), axis=1))
    # calculate the string length difference as abs(len(text)-len(original_text))
    df_merge['StringlengthDiff'] = (df_merge['text'].astype(str).map(len) - df_merge['original_text'].astype(str).map(len)).abs()

    # df_merge.to_csv(os.path.join(os.path.dirname(args.output_merge_dir), file.split('.')[0]+'_merge.csv'), index=False, encoding='utf-8-sig')
    if args.is_merge_tm:
        match_annot_df = df_merge.groupby(['index', 'page_id', 'phrase_id', 'section', 'text']).apply(lambda group: merge_annot2tm(group))  # .reset_index()
    else:
        match_annot_df = df_merge.groupby(['index', 'page_id', 'phrase_id', 'section', 'text']).apply(lambda group: merge_old_annot2new(group))  # .reset_index()
    df_merge_true = pd.merge(updated, match_annot_df, how="left", on=[
                             'page_id', 'text'])  # df_merge.iloc[true_instance_indices]
    df_merge_true = df_merge_true.drop_duplicates(['index', 'page_id', 'phrase_id', 'section', 'text'])
    df_merge_true = df_merge_true.drop(['processed_section','updated_prev_next_section'], axis=1)

    df_merge_true[['index', 'text_block_id', 'phrase_id']] = df_merge_true[['index', 'text_block_id', 'phrase_id']].astype(int)
    updated[['index', 'text_block_id', 'phrase_id']] = updated[['index', 'text_block_id', 'phrase_id']].astype(int)
    df_merge_true = df_merge_true.replace({'': None, 'nan': None, np.nan: None, 'None': None})
    df_merge_true = df_merge_true.reindex(columns=(list([a for a in df_merge_true.columns if a != 'docparse_datetime']+['docparse_datetime'])))
    df_merge_true = df_merge_true.sort_values(by=['index', 'text_block_id', 'page_id', 'phrase_id'])
    
    # if not args.is_merge_tm:
    #     df_merge_true.loc[(df_merge_true.text.str.len() <= 1) | (df_merge_true.text_element.str.contains('section')), ['clause_id', 'definition', 'schedule_id', 'annotation']] = None  # erase all annotation for text with length less than or equal 1
    # else:
    #     df_merge_true[['match_term_list', 'identifier_list', 'similarity_list', 'match_type_list']] = df_merge_true[['match_term_list', 'identifier_list', 'similarity_list', 'match_type_list']].apply(lambda x: None if not isinstance(x, (list,tuple)) else x, axis=1)
    updated2 = updated.merge(df_merge_true, how='left', on=['index', 'text_block_id', 'page_id', 'phrase_id'])['text_y']

    assert len(df_merge_true) == len(updated), f'In filename: {file}, the updated TS left join annotated TS is not as same length as the updated TS. Please check.'

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
        "--src_antd_ts_dir",
        default=f'{PROJ_ROOT_DIR}/data/reviewed_antd_ts{reviewed_version}/',
        type=str,
        required=False,
        help="Folder path to annotated TS .csv",
    )
    parser.add_argument(
        "--src_ts_csv_dir",
        default=f'{PROJ_ROOT_DIR}/data/docparse_csv/TS{ts_version}/',
        type=str,
        required=False,
        help="Folder path to docparse TS .csv",
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

    args = parser.parse_args()

    if not os.path.exists(args.output_merge_dir):
        os.mkdir(args.output_merge_dir)

    file_list = sorted([file for file in os.listdir(
        args.src_antd_ts_dir) if file.endswith('.csv')])
    # for file in file_list:
    #     merge_old_new(file ,args)
    
    # file_list = ['28_GF_PRJ_TS_mkd_20221111_docparse.csv']
    
    print('Processing merging existing annotated TS to updated TS ...')
    multiprocess(batch_merge_old_new, file_list, args=args)
    print('Completed merging existing annotated TS to updated TS\n')
