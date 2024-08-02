import os
import re
import pandas as pd
import numpy as np
from copy import deepcopy
from random import randint
from config import *


def timeit(func):
    from functools import wraps
    import time

    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        row = {'task': func.__name__,
               'sentence': args[0],
               'result': result,
               'runtime': total_time}
        log2csv(LOG_DIR + '/log_text_processing.csv', row)
        # print(f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
        return result
    return timeit_wrapper


def log2csv(csv_name, row):
    '''
    log the record 'row' into path 'csv_name'
    '''
    import csv
    import os.path

    file_exists = os.path.isfile(csv_name)
    # Open the CSV file in "append" mode
    with open(csv_name, 'a', newline='') as f:
        # Create a dictionary writer with the dict keys as column fieldnames
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()  # file doesn't exist yet, write a header
            # Append single row to CSV
        writer.writerow(row)


def concat_features_to_text(row, doc_type, apply_parent_text=False):
    '''
    @param row: pandas.Series object with indices ['text', 'section', 'fa_text', 'fa_section', 'fa_sub_section', 'parent_caption', 'parent_list']
    @param doc_type: either 'TS' or 'FA'
    @return: return a concatenated string of useful features (section, sub-section, caption, list item and main text) as a complete content
    @rtype: str
    '''
    parent_caption = parent_list = None
    
    text = row['text']
    text_element = row['text_element']
    
    # if column 'parent_caption' or 'parent_list' exists, then get the text
    if 'parent_caption' in row.index.tolist():
        parent_caption = row['parent_caption']
    if 'parent_list' in row.index.tolist():
        parent_list = row['parent_list']
    
    # if not apply parent text to concatenated processed text, erase all parent caption and parent list
    if not apply_parent_text:
        parent_caption = parent_list = None
        
    if not text:
        return None
    # else:
    #     text = convert_arabic_num2word(str(text))
    
    if doc_type == 'FA':
        fa_section = row['fa_section']
        fa_sub_section = row['fa_sub_section']
        
        # erase section / sub-section when text_element is section / sub-section because text standalone is already a section / sub-section
        if text_element and 'sub_section' in text_element:
            fa_sub_section = parent_caption = parent_list = None
            # remove leading digit in FA section text, eg. 21.2 Financial Covanents -> Financial Covanents
            text = re.sub('^\d+\.*\d*\.*\d*','',text)
        elif text_element and 'section' in text_element:
            fa_section = fa_sub_section = parent_caption = parent_list = None
            # remove leading digit in FA section text, eg. 21.2 Financial Covanents -> Financial Covanents
            text = re.sub('^\d+\.*\d*\.*\d*','',text)
        
        # erase section / sub-section that contains word "definition" or "interpretation" because it is meaningless
        if fa_section and re.search('definition|interpretation',fa_section,re.IGNORECASE):
            fa_section = None
        if fa_sub_section and re.search('definition|interpretation',fa_sub_section,re.IGNORECASE):
            fa_sub_section = None
        
        if fa_sub_section and fa_section and text:
            if parent_caption and parent_list:
                return fa_section + ' - ' + fa_sub_section + ': ' + parent_caption + ' ' + parent_list + ': ' + text
            elif parent_caption:
                return fa_section + ' - ' + fa_sub_section + ': ' + parent_caption + ' ' +  text
            elif parent_list:
                return fa_section + ' - ' + fa_sub_section + ': ' + parent_list + ': ' + text
            else:
                return fa_section + ' - ' + fa_sub_section + ': ' + text
        elif fa_sub_section and not fa_section and text:
            if parent_caption and parent_list:
                return fa_sub_section + ': ' + parent_caption + ' ' + parent_list + ': ' + text
            elif parent_caption:
                return fa_sub_section + ': ' + parent_caption + ' ' + text
            elif parent_list:
                return fa_sub_section + ': ' + parent_list + ': ' + text
            else:
                return fa_sub_section + ': ' + text
        elif not fa_sub_section and fa_section and text:
            if parent_caption and parent_list:
                return fa_section + ': ' + parent_caption + ' ' + parent_list + ': ' + text
            elif parent_caption:
                return fa_section + ': ' + parent_caption + ' ' + text
            elif parent_list:
                return fa_section + ': ' + parent_list + ': ' + text
            else:
                return fa_section + ': ' + text
        else:
            if parent_caption and parent_list:
                return parent_caption + ' ' + parent_list + ': ' + text
            elif parent_caption:
                return parent_caption + ' ' + text
            elif parent_list:
                return parent_list + ': ' + text
            else:
                return text

    elif doc_type == 'TS':

        section = row['section']
        if text_element and 'section' in text_element:
            parent_caption = parent_list = section = None
            
        # erase section that contains word "documentation" because it is meaningless
        if section and re.search('documentation',section,re.IGNORECASE):
            section = None
        
        # if both section and text not null
        if section and text:
            if parent_caption and parent_list:
                return section + ': ' + parent_caption + ' ' + parent_list + ': ' + text
            elif parent_caption:
                return section + ': ' + parent_caption + ' ' + text
            elif parent_list:
                return section + ': ' + parent_list + ': ' + text
            else:
                return section + ': ' + text
        else:
            if parent_caption and parent_list:
                return parent_caption + ' ' + parent_list + ': ' + text
            elif parent_caption:
                return parent_caption + ' ' + text
            elif parent_list:
                return parent_list + ': ' + text
            else:
                return text

class TextProcessing(object):

    def __init__(self, df):
        from transformers import AutoTokenizer
        import spacy
        import os
        self.df = df
        self.sent2processedSent = dict()
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            os.system('python3 -m spacy download en_core_web_sm')
            self.nlp = spacy.load("en_core_web_md")
        self.nlp.Defaults.stop_words -= {"no", "nor", "not", "none", "nobody", "nothing", "nowhere", "never", "cannot", "n't",
                                         "empty", "few", "neither", "rather", "nevertheless", "unless", "except", "however", "though", "although", "otherwise"}
        # 'sentence-transformers/all-MiniLM-L6-v2'
        self.model_name = 'TaylorAI/gte-tiny'
        self.tokenizer = AutoTokenizer.from_pretrained(f'{self.model_name}')

    def text_processing(self, sentence):
        """
        Lemmatize, lowercase, remove numbers and stop words
        Args:
        sentence: The sentence we want to process.   
        Returns:
        A list of processed words
        """
        if sentence in self.sent2processedSent.keys():
            return self.sent2processedSent[sentence]
        else:
            processed_sentence = [token.lemma_.lower()
                                  for token in self.nlp(sentence)
                                  if token.is_alpha and not token.is_stop]
            processed_sentence = ' '.join(processed_sentence).strip()
            self.sent2processedSent.update({sentence: processed_sentence})
            return processed_sentence

    def tokenize_sent_length(self, sentence):
        tokens = self.tokenizer.tokenize(sentence)
        return len(tokens)

    def process_df_text(self):
        self.df['processed_ts_text'] = self.df['ts_text'].map(lambda x: self.text_processing(x))
        self.df['processed_fa_text'] = self.df['fa_text'].map(lambda x: self.text_processing(x))
        self.df['processed_ts_text_length'] = self.df['processed_ts_text'].map(lambda x: self.tokenize_sent_length(str(x)) if x else 0)
        self.df['processed_fa_text_length'] = self.df['processed_fa_text'].map(lambda x: self.tokenize_sent_length(str(x)) if x else 0)
        return self.df

    def calculate_ts_text_length(self):
        return self.df['ts_text'].map(lambda x: self.tokenize_sent_length(str(x)) if x else 0)

    def calculate_fa_text_length(self):
        return self.df['fa_text'].map(lambda x: self.tokenize_sent_length(str(x)) if x else 0)


def shuffle(lst):
    temp_lst = deepcopy(lst)
    m = len(temp_lst)
    while (m):
        m -= 1
        i = randint(0, m)
        temp_lst[m], temp_lst[i] = temp_lst[i], temp_lst[m]
    return temp_lst

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


if __name__ == "__main__":
    import os
    import argparse
    import string

    test_set = [
        '1_GL_SYN_TS_mkd_20221215_docparse',
        '13_BL_SYN_TS_mkd_20220713_docparse',
        '19_GL_SYN_TS_mkd_20220718_docparse',
        '23_BL_SYN_TS_mkd_20220715_docparse',
        '24_GF_PRJ_TS_mkd_20220225_docparse',
        '25_NBFI_PRJ_TS_mkd_20220613_docparse',
        '28_GF_PRJ_TS_mkd_20221111_docparse',
        '29_PF_PRJ_TS_mkd_20200624_docparse',
        '3_GF_SYN_TS_mkd_20221018_docparse',
        '31_GL_PRJ_TS_mkd_20220630_docparse',
        '33_BL_PRJ_TS_mkd_20200727_docparse',
        '34_AWSP_PRJ_TS_mkd_20211230_docparse',
        '41_AF_SYN_TS_mkd_20190307_docparse',
        '43_AF_SYN_TS_mkd_20151101_docparse',
        '45_AF_PRJ_TS_mkd_20170331_docparse',
        '49_VF_PRJ_TS_mkd_20151130_docparse',
        '54_VF_PRJ_TS_mkd_20191018_docparse',
        '58_VF_SYN_TS_mkd_20111201_docparse',
        '59_AWSP_SYN_TS_mkd_20210814_docparse',
        '63_AW_SYN_TS_mkd_20221025_docparse',
        '66_PF_SYN_TS_mkd_20230106_docparse',
        '68_PF_SYN_TS_mkd_20221104_docparse',
        '72_NBFI_SYN_TS_mkd_20221215_docparse',
        '74_NBFI_SYN_TS_mkd_20220401_docparse',
        '8_GF_SYN_TS_mkd_20230215_docparse',
    ]

    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--input_csv_path",
        default=f'{PROJ_ROOT_DIR}/data/antd_ts_merge_fa/merge/all_merge_results.csv',
        type=str,
        required=False,
        help="Path to input CSV",
    )
    parser.add_argument(
        "--output_csv_dir",
        default=f'{PROJ_ROOT_DIR}/data/antd_ts_merge_fa/merge/',
        type=str,
        required=False,
        help="Path to output all sentence pairs (with labels entailment, contradiction) and train-dev-test splits in a CSV",
    )
    parser.add_argument(
        "--version",
        default='_v6.0',
        type=str,
        required=False,
        help="Path to output all sentence pairs (with labels entailment, contradiction) and train-dev-test splits in a CSV",
    )
    args = parser.parse_args()
    df = pd.read_csv(args.input_csv_path)
    if 'relationship' not in df.columns:
        df2 = df.groupby(['index', 'clause_id', 'definition', 'schedule_id'], dropna=False).apply(lambda x: judge_relationship(x)).reset_index()
        df2 = df2.rename(columns={0: "relationship"})
        df2 = df2.replace({'nan': None, np.nan: None, 'None': None})
        df = df.merge(df2, how='left', on=['index', 'clause_id', 'definition', 'schedule_id'])
    
    assert all(i in df.columns for i in ['index', 'page_id', 'phrase_id', 'section', 'text', 'fa_text', 'fa_section', 'fa_sub_section', 'fa_identifier', 'fa_text_length', 'relationship', 'filename']), "please make sure input csv contains columns ['index','page_id','phrase_id','section','text', 'fa_text','fa_section','fa_sub_section','fa_identifier','fa_text_length','relationship','filename']"
    df = df.drop_duplicates()
    df = df.T.drop_duplicates().T
    df['text'] = df['text'].astype(str)

    # text preprocessing
    df = df.replace({np.nan: None, 'nan': None, 'None': None})
    df['text'] = df['text'].map(lambda x: re.sub(' ('+r'|'.join(re.escape(i) for i in string.punctuation if i !='(')+')', r'\1', x) if x else x)  # remove space in space plus punctuation
    # remove consecutive punctuation in ts text
    df['text'] = df['text'].map(lambda x: re.sub(r'(:|\(|\)|,| )\1+', r'\1', x) if x else x)
    df['fa_text'] = df['fa_text'].map(lambda x: re.sub(' ('+r'|'.join(re.escape(i) for i in string.punctuation if i != '(')+')', r'\1', x) if x else x)  # remove space in space plus punctuation
    df['fa_text'] = df['fa_text'].map(lambda x: re.sub(r'(:|\(|\)|,| )\1+', r'\1', x) if x else x)  # remove consecutive punctuation in ts text
    # remove leading non-alphabet characters in section
    df['section'] = df['section'].map(lambda x: re.sub(r'^[^A-Za-z]+(?!$)', '', x).strip() if x else x)
    # drop pairs that either ts_text or fa_text is None
    df = df.dropna(subset=['text', 'fa_text'])
    # drop rows from a dataframe if fa_text is purely numeric, e.g. fa_text = "17.1"
    df = df[pd.to_numeric(df['fa_text'], errors='coerce').isna()]
    df['fa_text'] = df.apply(lambda x: concat_features_to_text(x, 'FA'), axis=1)
    df['fa_text'] = df['fa_text'].map(lambda x: x.replace('\r\n', ''))

    ts_text_group = df.groupby(by=['index', 'page_id', 'fa_identifier', 'filename'], group_keys=True).apply(lambda x: ' '.join([i for i in x['text'] if i])).reset_index()
    ts_text_group[['index']] = ts_text_group[['index']].astype(float).astype('Int64')
    ts_text_group[['page_id']] = ts_text_group[['page_id']].astype(str)
    df[['index']] = df[['index']].astype(float).astype('Int64')
    df[['page_id']] = df[['page_id']].astype(str)
    df.drop('text', axis=1, inplace=True)
    df = pd.merge(df, ts_text_group, on=['index', 'page_id', 'fa_identifier', 'filename'], how='inner')
    df.rename(columns={df.columns[-1]: 'text'}, inplace=True)

    df['text'] = df.apply(lambda x: concat_features_to_text(x, 'TS'), axis=1)
    df.rename(columns={'text': 'ts_text', 'section': 'ts_section','relationship': 'original_relationship'}, inplace=True)
    df = df.explode('ts_text')  # explode ts_text if it is a list
    df = df.drop_duplicates(subset=['ts_text', 'fa_identifier', 'filename'])
    processor = TextProcessing(df)
    df['ts_text_length'] = processor.calculate_ts_text_length()
    df['fa_text_length'] = processor.calculate_fa_text_length()
    df['split'] = df['filename'].map(lambda x: 'test' if x in test_set else 'train')
    df = df[['split', 'index', 'page_id', 'ts_section', 'ts_text', 'fa_text', 'fa_section', 'fa_sub_section', 'fa_identifier', 'ts_text_length', 'fa_text_length', 'original_relationship', 'filename']]

    # df = df.drop_duplicates()
    df = df[df['fa_text'].notna()]
    df['fa_text'] = df['fa_text'].map(lambda x: x.replace('\r\n', ''))
    assert not df['fa_text'].isnull().values.any() and not df['fa_text'].eq('').values.any() and not df['ts_text'].isnull().values.any() and not df['ts_text'].eq('').values.any(), 'There exists empty string or null content. Please check the ts_text or fa_text'
    df.to_csv(args.output_csv_dir +f'sentence_pairs{args.version}.csv', index=False, encoding='utf-8-sig')

    # df = TextProcessing(df).process_df_text()
    # print(((df['processed_ts_text_length'] > 512) | (df['processed_fa_text_length'] > 512)).sum())
    # df.to_csv(args.output_csv_dir + 'all_sentence_pairs.csv', index=False, encoding='utf-8-sig')
    # df = df[(df['processed_ts_text_length'] <= 512) & (df['processed_fa_text_length'] <= 512)]
    # df.insert(loc = 0, column = 'label', value = 'entailment')
    # # df = df[df.label == 'entailment']

    # df_2 = df[:]
    # df_2['label'] = 'contradiction'
    # for col in df_2.columns.tolist():
    #   df_2[col] = shuffle(df_2[col].tolist())
    # # df_2 = df_2[df_2.label == 'contradiction']

    # df1 = df.sample(frac = 0.8, random_state = 200)
    # df2 = df_2.sample(frac = 0.8, random_state = 200)
    # train = pd.concat([df1, df2])
    # train.insert(loc = 0, column = 'split', value = 'train')
    # df3 = df.drop(df1.index).sample(frac = 0.5, random_state = 200)
    # df4 = df_2.drop(df2.index).sample(frac = 0.5, random_state = 200)
    # dev = pd.concat([df3, df4])
    # dev.insert(loc = 0, column = 'split', value = 'dev')
    # test = pd.concat([df.drop(df1.index).drop(df3.index), df_2.drop(df2.index).drop(df4.index)])
    # test.insert(loc = 0, column = 'split', value = 'test')
    # df = pd.concat([train, dev, test])
    # #df = df[df.label.isin(['entailment', 'contradiction'])]
    # #df = df[df.split.isin(['train', 'dev', 'test'])]
    # df.to_csv(args.output_csv_dir+'sentence_pairs_v4.1.csv', index=False, encoding='utf-8-sig')

    # df = pd.read_csv('/home/data/ldrs_analytics/data/antd_ts_merge_fa/20230719/1_GL_SYN_TS_mkd_20221215_docparse_merge_FA.csv')

    '''
    for file in os.listdir(args.output_csv_dir):
        if file.endswith('.csv'):
            #print(f'filename: {file}')
            new_fname = file.split('.')[0] + '.csv'
            #f_id = re.sub('\D', '', new_fname[0:3])
            df = pd.read_csv(args.output_csv_dir+file)
            df = df[['ts_text_group', 'fa_text']]
            #df = df.replace({np.nan: None})
            df['split'] = pd.NA
            split = ['train', 'test']
            df['split'] = df.apply(lambda x: np.random.choice(split, p = [0.8, 0.2]))
            col = df.pop('split')
            df.insert(loc = 0, column = 'split', value = col)
            df.to_csv(clean_csv_dir+new_fname, index=False, encoding='utf-8-sig')
    '''
