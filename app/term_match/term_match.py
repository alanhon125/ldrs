'''
Term Match for TS file and FA file
'''
import re
import pandas as pd
import numpy as np
import requests
import json
import os
import torch
from ast import literal_eval
from sentence_transformers import SentenceTransformer, util
import logging

try:
    from app.utils import *
    from config import (COS_SIM_URL, 
                        LOG_DIR, 
                        OUTPUT_TERM_MATCH_CSV, 
                        OUTPUT_DOCPARSE_CSV, 
                        SENT_BERT_MODEL_PATH, 
                        SENT_EMBEDDINGS_BATCH_SIZE, 
                        DEV_MODE, 
                        TOP_N_RESULTS, 
                        LOG_FILEPATH)
    from regexp import DEF_RULES, SCEHD_RULES, TS_SECTIONS_WITH_TERMS
except ModuleNotFoundError:
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    currentdir = os.path.dirname(os.path.abspath(__file__))
    parentdir = os.path.dirname(currentdir)
    parentparentdir = os.path.dirname(parentdir)
    import sys
    sys.path.insert(0, parentdir)
    from utils import *
    sys.path.insert(0, parentparentdir)
    from config import (COS_SIM_URL, 
                        LOG_DIR, 
                        OUTPUT_TERM_MATCH_CSV, 
                        OUTPUT_DOCPARSE_CSV, 
                        SENT_BERT_MODEL_PATH, 
                        SENT_EMBEDDINGS_BATCH_SIZE, 
                        DEV_MODE, 
                        TOP_N_RESULTS, 
                        LOG_FILEPATH)
    from regexp import DEF_RULES, SCEHD_RULES, TS_SECTIONS_WITH_TERMS

logger = logging.getLogger(__name__)

def is_date(string, fuzzy=False):
    """
    Return whether the string can be interpreted as a date.
    e.g. December -> True, 1 January 2020 -> True

    :param string: str, string to check for date
    :param fuzzy: bool, ignore unknown tokens in string if True
    """
    from dateutil.parser import parse
    import re
    if re.match('\d{5,}', string):
        return False
    try:
        parse(string, fuzzy=fuzzy)
        return True

    except:
        return False

def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False

def convert_arabic_num2word(string):
    '''
    @param string: any non-empty string
    @return: return a converted string that has attempted to capture arabic numerals (if exists but except date) and convert into number in words literally (e.g. US$5,000,000 -> US$ five million)
    @rtype: str
    '''
    from num2words import num2words
    import re

    string = re.sub(r',(\d+)',r'\1',string)
    string = re.sub(r'(\d+)m', r'\1 million', string)
    string = string.replace(',',' ,')
    string = string.replace('(','( ')
    string = string.replace(')', ' )')
    string = string.replace('[','[ ')
    string = string.replace(']', ' ]')
    string = string.replace('$', '$ ')
    string = string.replace('€', '€ ')
    string = string.replace('¥', '¥ ')
    for i in ['HKD','USD','EUR','RMB','CNY','AUD','JPY']:
        string = string.replace(i, f'{i} ')
    string = string.replace('%', ' per cent')
    string = re.sub(' +', ' ', string)
    split_txts = string.split()
    new_string = ''
    for i, split_txt in enumerate(split_txts):
        if i+1< len(split_txts) and \
                not is_date(split_txts[i+1]) and \
                not re.match(r'(20|19)\d{2}',split_txt) and \
                (split_txt.isdigit() or isfloat(split_txt) or re.match("(\d+(st|nd|rd|th))",split_txt)):
            try:
                split_txt = num2words(split_txt)
            except:
                pass
        new_string += ' ' + split_txt
    new_string = new_string.replace(', ',',')
    new_string = new_string.replace('( ','(')
    new_string = new_string.replace(' )', ')')
    new_string = new_string.replace('[ ','[')
    new_string = new_string.replace(' ]', ']')

    return new_string.strip()

def concat_features_to_text(row, text, doc_type, apply_parent_text=False):
    '''
    @param row: pandas.Series object with indices ['text', 'section', 'fa_text', 'fa_section', 'fa_sub_section', 'parent_caption', 'parent_list']
    @param doc_type: either 'TS' or 'FA'
    @return: return a concatenated string of useful features (section, sub-section, caption, list item and main text) as a complete content
    @rtype: str
    '''
    parent_caption = parent_list = None

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
    
    text_element = row['text_element']
    
    if doc_type == 'FA':
        fa_section = row['section']
        fa_sub_section = row['sub_section']
        
        # erase section / sub-section when text_element is section / sub-section because text standalone is already a section / sub-section
        if 'sub_section' in text_element:
            fa_sub_section = parent_caption = parent_list = None
            # remove leading digit in FA section text, eg. 21.2 Financial Covanents -> Financial Covanents
            text = re.sub('^\d+\.*\d*\.*\d*','',text)
        elif 'section' in text_element:
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
        if 'section' in text_element:
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


def concat_section_to_text(row, doc_type):
    '''
    @param row: pandas.Series object with columns ['text', 'section', 'fa_text', 'fa_section', 'fa_sub_section']
    @param doc_type: either 'TS' or 'FA'
    '''
    text_element = row['text_element']
    text = row['text']
    if doc_type == 'FA':
        # if row['concatenated_child_text']:
        #     text = row['concatenated_child_text']
        # else:
        #     text = row['text']
        fa_section = row['section']
        fa_sub_section = row['sub_section']
        # erase section / sub-section when text_element is section / sub-section because text standalone is already a section / sub-section
        if 'sub_section' in text_element:
            fa_sub_section = None
            # remove leading digit in FA section text, eg. 21.2 Financial Covanents -> Financial Covanents
            text = re.sub('^\d+\.*\d*\.*\d*','',text)
        elif 'section' in text_element:
            fa_section = fa_sub_section = None
            # remove leading digit in FA section text, eg. 21.2 Financial Covanents -> Financial Covanents
            text = re.sub('^\d+\.*\d*\.*\d*','',text)
        
        # erase section / sub-section that contains word "definition" or "interpretation" because it is meaningless
        if fa_section and re.search('definition|interpretation',fa_section,re.IGNORECASE):
            fa_section = None
        if fa_sub_section and re.search('definition|interpretation',fa_sub_section,re.IGNORECASE):
            fa_sub_section = None
        
        if fa_sub_section and fa_section and text:
            return fa_section + ' - ' + fa_sub_section + ': ' + text
        elif fa_sub_section and not fa_section and text:
            return fa_sub_section + ': ' + text
        elif not fa_sub_section and fa_section and text:
            return fa_section + ': ' + text
        else:
            return text
        
    elif doc_type == 'TS':
        section = row['section']
        
        if 'section' in text_element:
            section = None
            
        # erase section that contains word "documentation" because it is meaningless
        if section and re.search('documentation',section,re.IGNORECASE):
            section = None
        
        if section and text:
            return section + ': ' + text
        else:
            return text
        

def merge_idf_list(row):
    idf_list = row['identifier_list']
    term_list = row['match_term_list']
    sim_list = row['similarity_list']
    type_list = row['match_type_list']
    rgx = '\(|\)|_|\[|\]|\.|%|\"'
    if not idf_list or all(i is None for i in idf_list):
        return idf_list, term_list, sim_list, type_list
    else:
        merged = [None]
        merged_terms = [None]
        merged_sims = [None]
        merged_types = [None]

        # idf_list = ast.literal_eval(idf_list)
        # term_list = ast.literal_eval(term_list)
        # sim_list = ast.literal_eval(sim_list)
        # type_list = ast.literal_eval(type_list)

        idf_list = [i for i in idf_list if i]
        term_list = [t for t in term_list if t]
        sim_list = [i for i in sim_list if i]
        type_list = [i for i in type_list if i]

        if idf_list[0]:
            merged = [idf_list[0]]
            merged_terms = [term_list[0]]
            merged_sims = [sim_list[0]]
            merged_types = [type_list[0]]
            for i in range(1, len(idf_list)):
                flag = 0
                idf = idf_list[i]
                idf_r = re.sub(rgx, 'TEST', idf)
                if re.search('\_', idf):
                    idf_def_text = re.split('\_', re.split('-', idf)[-1])[0]
                    idf_def_suffix = '_'.join(re.split('\_', re.split('-', idf)[-1])[1:])
                else:
                    idf_def_text = re.split('-', idf)[-1]
                    idf_def_suffix = ''
                if re.search('-', idf):  # 'Cl_1.1-Security Documents', 'Cl_1.1-Security Documents_(b)'
                    for j in range(0, len(merged)):
                        if re.search('-', merged[j]):
                            if re.search('\_', merged[j]):
                                def_text = re.split('\_', re.split('-', merged[j])[-1])[0]
                                def_suffix = '_'.join(re.split('\_', re.split('-', merged[j])[-1])[1:])
                            else:
                                def_text = re.split('-', merged[j])[-1]
                                def_suffix = ''

                            if def_text == idf_def_text:
                                if not def_suffix:
                                    flag = 1
                                    merged[j] = idf
                                    merged_terms[j] = term_list[i]
                                    # merged_sims[j] = sim_list[i]
                                    merged_types[j] = type_list[i]
                                else:
                                    if idf_def_suffix:
                                        if re.search(re.sub(rgx, 'TEST', def_suffix),
                                                     re.sub(rgx, 'TEST', idf_def_suffix)):
                                            flag = 1
                                            merged[j] = idf
                                            merged_terms[j] = term_list[i]
                                            # merged_sims[j] = sim_list[i]
                                            merged_types[j] = type_list[i]

                    if flag == 0:
                        # 'Cl_1.1-Security Documents_(b)', 'Cl_1.1-Security Documents'
                        # 'Cl_1.1-Security Documents_(b)', 'Cl_1.1-Security Documents_(a)'
                        for m in merged:
                            if re.search('-', m):
                                text_part = re.split('-', m)[-1]
                                def_text = re.split('\_', re.split('-', merged[j])[-1])[0]
                                def_suffix = ''
                                if re.search('\_', m):
                                    def_suffix = '_'.join(re.split('\_', re.split('-', merged[j])[-1])[1:])

                                if idf_def_text == def_text:
                                    if re.search(re.sub(rgx, 'TEST', idf_def_suffix), re.sub(rgx, 'TEST', def_suffix)):
                                        flag = 1
                                        break
                        if flag == 0:
                            merged.append(idf)
                            merged_terms.append(term_list[i])
                            merged_sims.append(sim_list[i])
                            merged_types.append(type_list[i])
                else:
                    for j in range(0, len(merged)):
                        # 20.1, 20.1(a)
                        if re.search(re.sub(rgx, 'TEST', merged[j]), idf_r):
                            flag = 1
                            merged[j] = idf
                            merged_terms[j] = term_list[i]
                            # merged_sims[j] = sim_list[i]
                            merged_types[j] = type_list[i]
                    if flag == 0:
                        # 20.1(a), 20.1
                        check_idf = [1 if re.search(idf_r, re.sub(rgx, 'TEST', m)) else 0 for m in merged]
                        if sum(check_idf) == 0:
                            merged.append(idf)
                            merged_terms.append(term_list[i])
                            merged_sims.append(sim_list[i])
                            merged_types.append(type_list[i])
        
        # after all operations, remove duplicates
        lst = list(zip(merged, merged_terms, merged_sims, merged_types))
        mylist = list(dict.fromkeys(lst))
        merged, merged_terms, merged_sims, merged_types = list(zip(*mylist))

        return merged, merged_terms, merged_sims, merged_types

def modify_term_match(df_match, df_fa, topN=TOP_N_RESULTS):
    
    # dict map definition term to identifier , e.g. {"Borrower": "Cl_1.1-Borrower【 Definitions & Interpretations 】"}
    def2idf = dict(zip(
        df_fa.definition, df_fa.identifier
    ))
    # dict map definition term to definition clause text, e.g. {"Borrower": "\"Borrower\" means xxxx"}
    def2text = def2text_dict(df_fa)
    schedules = [s for s in set(df_fa.schedule) if s]
    definitions = [d for d in set(df_fa.definition) if d]
    
    results = []
    for idx, row in df_match.iterrows():
        row_n = row.copy()
        ts_term = row['TS_term']
        ts_text = row['TS_text']
        for col in ['match_term_list', 'identifier_list', 'similarity_list']:
            if isinstance(row[col], tuple):
                row[col] = list(row[col])
        if row['text_element'] != 'section' and ts_term and ts_text:
            add_def = []
            add_idf = []
            for rule in DEF_RULES:
                find_sec, find_text, find_def = rule
                
                if re.search(find_sec, ts_term, re.IGNORECASE) and re.search(find_text, ts_text, re.IGNORECASE):
                    for definition in definitions:
                        if re.search(find_def, definition, re.IGNORECASE):
                            idf = def2idf[definition]
                            identifier_list = row['identifier_list']
                            
                            # all section, text and definition match with the rule tuple but identifier not exists in identifier_list, add defintion and identifer to the list
                            if idf not in identifier_list[:topN]:
                                if idf not in add_idf:
                                    add_idf.append(idf)
                                    add_def.append(def2text.get(definition))
        
            row_n['match_term_list'] = add_def + list(row['match_term_list'])
            row_n['identifier_list'] = add_idf + list(row['identifier_list'])
            row_n['similarity_list'] = [1]*len(add_def) + list(row['similarity_list'])
            row_n['match_type_list'] = ['rule']*len(add_def) + list(row['match_type_list'])
            
            check_idfs = []
            # if "security" appears in TS section, look for identifiers from schedule with "security document"
            for item in SCEHD_RULES:
                search_sec = item[0]
                search_txt = item[2]
                check_idfs = []
                if re.search(search_sec, ts_term, re.I):
                    check_scheds = list(set([s for s in schedules if re.search(search_txt, s, re.I)]))
                    if check_scheds:
                        check_idfs = list(set(df_fa[df_fa.schedule.isin(check_scheds)].identifier))
                    else:
                        check_defs = list(set([s for s in definitions if re.search(search_txt, s, re.I)]))
                        if check_defs:
                            check_idfs = list(set(df_fa[df_fa.definition.isin(check_defs)].identifier))
                    if check_idfs:
                        break
            # if re.search('security', ts_term, re.I):
            #     check_scheds = list(set([s for s in schedules if re.search('^security document', s, re.I)]))
            #     if check_scheds:
            #         check_idfs = list(set(df_fa[df_fa.schedule.isin(check_scheds)].identifier))
            #     else:
            #         check_defs = list(set([s for s in definitions if re.search('^security document', s, re.I)]))
            #         if check_defs:
            #             check_idfs = list(set(df_fa[df_fa.definition.isin(check_defs)].identifier))
            
            # if "security" appears in TS section and any matched identifier is in identifiers from schedule with "security document",
            # re-rank the identifier list such that the first 3 identifiers appears is belonging to schedule with "security document"
            if check_idfs:
                priority_idx_list = []
                not_priority_idx_list = []
                for idx, idf in enumerate(row_n['identifier_list']):
                    if idf in check_idfs:
                        priority_idx_list.append(idx)
                    else:
                        not_priority_idx_list.append(idx)
                rerank_n = 4
                new_priority_idx = priority_idx_list[:rerank_n] + not_priority_idx_list + priority_idx_list[rerank_n:]
                
                row_n['identifier_list'] = [row_n['identifier_list'][i] for i in new_priority_idx]
                row_n['match_term_list'] = [row_n['match_term_list'][i] for i in new_priority_idx]
                # row_n['similarity_list'] = [row_n['similarity_list'][i] for i in new_priority_idx]
                row_n['match_type_list'] = [row_n['match_type_list'][i] for i in new_priority_idx]
            
        results.append(row_n)
    df_match_updated = pd.DataFrame(data=results)
    return df_match_updated


def query_section_by_id(df_fa, identifier):
    '''
    @param df_fa: pd.DataFrame of FA docparse, containing column names ['section', 'sub_section', 'definition', 'schedule', 'identifier']
    @param identifier: FA clause identifier which express in FA docparse result, e.g. "Cl_7.5(e)(ii)"
    @return: the clause headings, say "PREPAYMENT AND CANCELLATION - Right of prepayment and cancellation in relation to a single Lender"
    '''
    import numpy as np
    import re

    if identifier is None:
        return None

    df_fa = df_fa.replace({np.nan: None, 'nan': None, 'None': None})
    matched_df = df_fa[df_fa['identifier'].str.contains(re.escape(identifier),na=False)]
    if matched_df.shape[0] > 0:  # exist at least one matched record
        section = matched_df.section.values[0]
        sub_section = matched_df.sub_section.values[0]
        schedule = matched_df.schedule.values[0]

        if schedule and not schedule is np.nan:
            return schedule
        elif section and not section is np.nan and sub_section and not sub_section is np.nan:
            return section + ' - ' + sub_section
        elif not section and section is np.nan and sub_section and not sub_section is np.nan:
            return sub_section
        elif section and not section is np.nan:
            return section
        else:
            return None
    else:
        return None


def def2text_dict(df_fa):
    '''
    To generate dictionary of definition mapping to FA content with DataFrame of FA provided
    Sample usage:
    df_fa:

        definition          identifier                  text
        Authorisation       Cl_1.1-Authorisation	    "Authorisation" means:
        Authorisation       Cl_1.1-Authorisation_(a)	an authorisation, consent, approval, resolution, licence, exemption, filing, notarisation, lodgement or registration
        Authorisation       Cl_1.1-Authorisation_(b)	in relation to anything which will be fully or partly prohibited or restricted by law if a Governmental Agency intervenes or acts in any way within a specified period after lodgement, filing, registration or notification, the expiry of that period without intervention or action.

    return
        {
            "Authorisation": "\"Authorisation\" means:\n
                            an authorisation, consent, approval, resolution, licence, exemption, filing, notarisation, lodgement or registration\n
                            in relation to anything which will be fully or partly prohibited or restricted by law if a Governmental Agency intervenes or acts in any way within a specified period after lodgement, filing, registration or notification, the expiry of that period without intervention or action."
        }

    @param df_fa: pandas DataFrame of FA contains 'section_id','sub_section_id','definition' and 'text'
    @return: a dictionary of definition map to FA content
    '''
    df_fa = df_fa.replace({'nan': None, np.nan: None, 'None': None})
    # groupby section_id, sub_section_id and definition
    df_def_gp = df_fa.groupby(['section_id', 'sub_section_id', 'definition'], dropna=False)
    # concatenate the text associate with the same definition within the group with newline character, drop records where definition == NaN
    df_def = df_def_gp.text.apply(lambda x: '\n'.join(
        [i if i else '' for i in x.values.tolist()])).reset_index().dropna(subset=['definition'])
    # create a dictionary {"Definition": "FA_content"}
    def2text = pd.Series(df_def.text.values, index=df_def.definition).to_dict()
    return def2text

def remove_particular_punct(s):
    '''
    Remove particular punctuations in a string
    '''

    import string
    import re

    keep_punct = ['(', ')', ':', ',', '.','\\','$']
    punct_list = string.punctuation
    remove_punct = [re.escape(i) for i in punct_list.translate({ord(c): None for c in keep_punct})]
    s = s.replace('\r\n', ' ')
    cleaned_text = re.sub('|'.join(remove_punct), '', s)
    cleaned_text = re.sub(r'(:|\(|\)|,| )\1+', r'\1', cleaned_text)

    return cleaned_text

def concatenated_fa_text_by_identifier(df, idx, txt_block_id, definition, fa_identifier):
    import string
    import re

    if definition is not None:
        df = df.loc[(df.identifier.str.contains('^'+re.escape(fa_identifier), na=False)) &
                    (df.definition.eq(definition)) &
                    (df.index>=idx) &
                    (df.text_block_id==txt_block_id)]
    else:
        df = df.loc[(df.identifier.str.contains('^' + re.escape(fa_identifier), na=False)) &
                    (df.index >= idx) &
                    (df.text_block_id == txt_block_id)]
    # remove all punctuations such as [, ", {, } , ] in table JSON text
    df.loc[df.text_element.str.contains('table'), 'text'] = df[df.text_element.str.contains('table')]['text'].map(remove_particular_punct)
    fa_text = df[~df.text.isna()]['text'].values.tolist()

    # if query list of text with length more than 1, concatenate a text in the list else return None
    if len(fa_text)>1:
        concatenated_child_text = ''.join([c + (' ' if c.endswith(tuple([i for i in string.punctuation])) else '; ') for c in fa_text]).strip()
    else:
        concatenated_child_text = None

    return concatenated_child_text

def compute_text_embedding_cos_sim(model, text_list1, text_list2):
    import requests
    import numpy as np
    import torch
    from sentence_transformers import util
    
    # if GPU exists in this environment, perform cosine similarity computation in this environment
    if torch.cuda.is_available():
        # Start the multi-process pool on all available CUDA devices
        pool = model.start_multi_process_pool()
        embeddings1 = model.encode_multi_process(text_list1, pool, batch_size=SENT_EMBEDDINGS_BATCH_SIZE)
        embeddings2 = model.encode_multi_process(text_list2, pool, batch_size=SENT_EMBEDDINGS_BATCH_SIZE)
        # Optional: Stop the processes in the pool
        model.stop_multi_process_pool(pool)
        cos_scores = util.cos_sim(embeddings1, embeddings2).cpu().numpy()
    else:
        embeddings1 = model.encode(text_list1, convert_to_tensor=True, batch_size=SENT_EMBEDDINGS_BATCH_SIZE)
        embeddings2 = model.encode(text_list2, convert_to_tensor=True, batch_size=SENT_EMBEDDINGS_BATCH_SIZE)
        cos_scores = util.cos_sim(embeddings1, embeddings2).cpu().numpy()
    
    # if no GPU, request cosine similarity computation from remote server at 10.6.55.3
    # else:
    #     request = {
    #         "sentences1": text_list1,
    #         "sentences2": text_list2
    #     }
    #     logger.info('Request sentences text embeddings and compute cosine similarity, waiting for response ...')
    #     headers = {'Content-type': 'application/json',
    #             'Accept': 'application/json'}
    #     try:
    #         response = requests.post(COS_SIM_URL, data=json.dumps(request), headers=headers)
    #         response = json.loads(response.content.decode('utf-8'))
    #         cos_scores = response["cosine_similarities"]
    #         cos_scores = np.array(json.loads(cos_scores))
    #     except requests.exceptions.ConnectionError:
    #         # Start the multi-process pool on all available CUDA devices
    #         pool = model.start_multi_process_pool()
    #         embeddings1 = model.encode_multi_process(text_list1, pool, batch_size=SENT_EMBEDDINGS_BATCH_SIZE)
    #         embeddings2 = model.encode_multi_process(text_list2, pool, batch_size=SENT_EMBEDDINGS_BATCH_SIZE)
    #         # Optional: Stop the processes in the pool
    #         model.stop_multi_process_pool(pool)
    #         cos_scores = util.cos_sim(embeddings1, embeddings2).cpu().numpy()
    logger.info('Complete computation of cosine similarity.')
        
    return cos_scores

def main(df_ts, df_fa, topN=TOP_N_RESULTS, task_id=None, model_path=SENT_BERT_MODEL_PATH):
    # read term matching results from csv file if it is in DEV_mode and cache file exists
    if DEV_MODE and task_id and os.path.exists(os.path.join(OUTPUT_TERM_MATCH_CSV, f'{task_id}.csv')):
        df_final = pd.read_csv(os.path.join(OUTPUT_TERM_MATCH_CSV, f'{task_id}.csv'))
        df_final = df_final.replace({np.nan: None, 'nan': None, 'None': None})
        df_final[['page_id', 'TS_text']] = df_final[['page_id', 'TS_text']].astype(str)
        df_final[['match_term_list', 'identifier_list', 'similarity_list', 'match_type_list']] = df_final[['match_term_list', 'identifier_list', 'similarity_list', 'match_type_list']].map(lambda x: literal_eval(str(x)) if x else None)
        return df_final

    # Processing TS text
    # df_ts = pd.DataFrame(data=ts_data)
    df_ts = df_ts.replace({np.nan: None, 'nan': None, 'None': None})
    df_ts['processed_text'] = df_ts.apply(lambda i: concat_features_to_text(i, i.text, 'TS'), axis=1)
    # df_ts['processed_text'] = df_ts.apply(lambda i: concat_section_to_text(i, 'TS'), axis=1)
    df_ts = df_ts[~df_ts.processed_text.isna()]

    # Processing FA text
    # df_fa = pd.DataFrame(data=fa_data).astype(str)
    df_fa = df_fa.astype(str)
    df_fa['index'] = df_fa['index'].astype(int)
    df_fa = df_fa.replace({np.nan: None, 'nan': None, 'None': None})
    
    # remove leading digit in FA section text, eg. 21.2 Financial Covanents -> Financial Covanents
    df_fa.loc[df_fa.text_element.str.contains('section', na=False), 'text'] = df_fa[df_fa.text_element.str.contains('section', na=False)]['text'].map(lambda x: re.sub('^\d+\.*\d*\.*\d*','',x) if x else x)
    df_fa['processed_fa_text'] = df_fa.apply(lambda i: concat_features_to_text(i, i.text, 'FA'), axis=1)
    # df_fa['concatenated_child_text'] = df_fa[['text_block_id', 'definition', 'identifier']].apply(lambda x: concatenated_fa_text_by_identifier(df_fa, x.name, x[0], x[1], x[2]) if isinstance(x[2], str) else None, axis=1)
    # df_fa['processed_concatenated_child_text'] = df_fa.apply(lambda i: concat_features_to_text(i, i.concatenated_child_text, 'FA'), axis=1)
    # extract section + sub-section string by fa identifier
    df_fa['section_subsection'] = df_fa['identifier'].map(lambda x: query_section_by_id(df_fa, x))
    df_fa = df_fa.replace({np.nan: None, 'nan': None, 'None': None})

    # df_parties = df_fa[(df_fa.section_id == "0") | (df_fa.section_id == 0)] # cols: definition + text
    # line 149 to string, some section ids are like 0.0, not int.
    df_parties = df_fa[df_fa.section == 'PARTIES']
    df_parties = df_parties[~df_parties.definition.isna()]
    # extract definition clause section
    df_def = df_fa[~df_fa.definition.isna()]
    df_def = df_def.drop_duplicates(subset=['section_id', 'sub_section_id', 'definition'])

    # extract sesction excluding parties & definition clause
    df_others = df_fa[~(df_fa.section.str.contains("DEFINITION|INTERPRETATION", na=False, case=False)) & (df_fa.section_id != "0") & (df_fa.section != 'PARTIES') & (df_fa.section != 'TABLE OF CONTENTS')]
    # extract schedule section
    df_sched = df_others.loc[df_others.schedule.notnull()]
    # main clause
    df_clause = df_others.loc[~df_others.schedule.notnull()]
    df_clause = df_clause[~df_clause.section.isna()]

    df_def_idf = df_def[['definition', 'identifier']].drop_duplicates().rename(columns={'definition': 'processed_fa_text'})
    df_parties_idf = df_parties[['definition', 'identifier']].drop_duplicates().rename(columns={'definition': 'processed_fa_text'})

    df_content_idf = df_fa[['processed_fa_text', 'identifier']].drop_duplicates()
    # df_concat_content_idf = df_fa[~df_fa.processed_concatenated_child_text.isna()]
    # df_concat_content_idf = df_concat_content_idf[['processed_concatenated_child_text', 'identifier']].drop_duplicates().rename(columns={'processed_concatenated_child_text': 'processed_fa_text'})

    # create dict to map fa_identifier to section + sub-section string, e.g. {"Cl_1.1-Borrower": "Definitions - Definitions & Interpretations"}
    id2sec_subsec = dict(zip(df_fa.identifier, df_fa.section_subsection))

    # dict map definition term to definition clause text, e.g. {"Borrower": "\"Borrower\" means xxxx"}
    def2text = def2text_dict(df_fa)

    # dict mapping fa_processed_text : fa_text, including fa_processed_concatenated_child_text : fa_concatenated_child_text
    FA_TEXT_DICT = dict(zip(df_fa.processed_fa_text, df_fa.text))
    # FA_SECTEXT2CONCAT_TEXT = dict(zip(df_fa[df_fa.text_element.str.contains('section')].processed_fa_text, df_fa.concatenated_child_text))
    # FA_CONCAT_TEXT_DICT = dict(zip(df_fa.processed_concatenated_child_text, df_fa.concatenated_child_text))
    # FA_TEXT_DICT.update(FA_CONCAT_TEXT_DICT)
    # FA_TEXT_DICT.update(FA_SECTEXT2CONCAT_TEXT)
    
    # create complete df of FA term and df of all FA content
    df_term_idf = pd.concat([df_def_idf, df_parties_idf])
    # df_all_idf = df_content_idf
    df_all_idf = pd.concat([df_content_idf, df_term_idf])
    # df_all_idf = pd.concat([df_content_idf, df_concat_content_idf])

    # create list of texts for cosine similarity computation
    ts_text_list = list(set(df_ts.processed_text))
    fa_text_list = list(set(df_all_idf.processed_fa_text))
    fa_term_list = list(set(df_term_idf.processed_fa_text))

    ts_text_list = [t for t in ts_text_list if t]
    fa_text_list = [t for t in fa_text_list if t]
    fa_term_list = [t for t in fa_term_list if t]

    fa_all_text_list = fa_text_list + fa_term_list
    
    # compute cosine similarity scores
    model = SentenceTransformer(model_path)

    ts_section_list = [sec.strip() for sec in set(df_ts.section) if sec]
    split_sections = [sec for sec in ts_section_list if re.search(TS_SECTIONS_WITH_TERMS, sec, re.I)]
    
    df_ts_split = df_ts[df_ts.section.isin(split_sections)&(df_ts.text_element!='section')].reset_index(drop=True)
    df_ts_split_updated = pd.DataFrame()
    if not df_ts_split.empty:
        l4 = list(df_ts_split.processed_text)
        fa_section_df = df_clause[~df_clause.section.isna()][['section', 'identifier']].drop_duplicates(subset=['section'])
        fa_section_df = fa_section_df.rename(columns={'section': 'text'})
        fa_sub_section_df = df_clause[~df_clause.sub_section.isna()][['sub_section', 'identifier']].drop_duplicates(subset=['sub_section'])
        fa_sub_section_df = fa_sub_section_df.rename(columns={'sub_section': 'text'})

        df_fa_section_all = pd.concat([fa_section_df, fa_sub_section_df])
        if not df_fa_section_all.empty:
            SECTION_IDF_DICT = dict(zip(
                df_fa_section_all.text,
                df_fa_section_all.identifier
            ))
            logger.info('Start text embeddings on ts_processed_text of TS sections with terms, fa_section, fa_sub_section of FA main clause ...')
            cosine_scores_sec = compute_text_embedding_cos_sim(model, l4, list(df_fa_section_all.text))

            ts_split_data = []
            top_n_section = 10
            for i, row in df_ts_split.iterrows():
                sims = cosine_scores_sec[i]
                top_sims_idx = np.array(sims).argsort()[-1:-top_n_section-1:-1]
                row['similarity_list'] = [sims[idx].item() for idx in top_sims_idx]
                row['match_term_list'] = [list(df_fa_section_all.text)[idx] for idx in top_sims_idx]
                row['identifier_list'] = [SECTION_IDF_DICT[t] for t in row['match_term_list']]
                row['match_type_list'] = ['section']*top_n_section
                ts_split_data.append(row)
            df_ts_split_updated = pd.DataFrame(data=ts_split_data)
    
    
    # Compute the embeddings using the multi-process pool
    logger.info('Start text embeddings on all ts_processed_text and all fa_definitions, all fa_processed_text ...')
    cos_scores = compute_text_embedding_cos_sim(model, ts_text_list, fa_all_text_list)
    
    # list of cosine similarity of TS Text to FA Term (i.e. definitions, parties)
    ts_text2fa_term_cos_scores = cos_scores[:, len(fa_text_list):]
    # list of cosine similarity of TS Text to FA content (i.e. parties clauses, definition clauses, clauses, schedule content)
    ts_text2fa_text_cos_scores = cos_scores[:, :len(fa_text_list)]

    ts_cols = [
        'index',
        'text_block_id',
        'page_id',
        'phrase_id',
        'list_id',
        'text_element',
    ]

    added_cols = [
        'section',
        'processed_text',
        'fa_text',
        'identifier',
        'similarity',
        'match_type'
    ]

    data = {
        'term': {k: [] for k in ts_cols + added_cols},  # term includes definition and parties
        'content': {k: [] for k in ts_cols + added_cols}
        # content includes definition clauses, main clauses and schedule clauses
    }

    topN_type = {
        'term': 40,  # extract top 20 from results list of ts_text match fa_definition & fa_parties
        'content': 40  # extract top 20 from results list of ts_text match fa_text
    }

    for _, row in df_ts.iterrows():
        if row['processed_text'] is None:
            continue
        i = ts_text_list.index(row['processed_text'])
        for match_type in data.keys():
            if match_type == 'term':
                sims = ts_text2fa_term_cos_scores[i]
                ref_df = df_term_idf
            else:
                sims = ts_text2fa_text_cos_scores[i]
                ref_df = df_all_idf

            top_sims_idx = np.array(sims).argsort()[-1:-topN_type[match_type] - 1:-1]
            top_scores = [sims[idx].item() for idx in top_sims_idx]
            if match_type == 'term':
                top_fa_text = [fa_term_list[idx] for idx in top_sims_idx]
            else:
                top_fa_text = [fa_text_list[idx] for idx in top_sims_idx]

            all_top_idf = []
            all_top_scores = []
            all_top_fa_text = []
            all_match_types = []
            for j in range(len(top_fa_text)):
                fa_text = top_fa_text[j]
                idf = ref_df[ref_df.processed_fa_text == fa_text].identifier
                if match_type == 'term':
                    fa_text = def2text.get(fa_text)
                if not idf.empty:
                    no_of_idf = len(list(idf))
                    all_top_idf.extend(list(idf))
                    all_top_fa_text.extend([fa_text] * no_of_idf)
                    all_top_scores.extend([top_scores[j]] * no_of_idf)
                    all_match_types.extend([match_type] * no_of_idf)
                else:
                    all_top_idf.append(None)
                    all_top_fa_text.append(fa_text)
                    all_top_scores.append(top_scores[j])
                    all_match_types.append(match_type)
            
            if match_type == 'content':
                if not df_ts_split_updated.empty:
                    df_section_s = df_ts_split_updated[(df_ts_split_updated.section==row['section'])&(df_ts_split_updated.processed_text==row['processed_text'])]
                    if not df_section_s.empty:
                        all_top_idf.extend(list(df_section_s.identifier_list.tolist()[0]))
                        all_top_fa_text.extend(list(df_section_s.match_term_list.tolist()[0]))
                        all_top_scores.extend(list(df_section_s.similarity_list.tolist()[0]))
                        all_match_types.extend(list(df_section_s.match_type_list.tolist()[0]))
            
            length = len(all_top_idf)
            for col in ts_cols:
                data[match_type][col].extend([row[col]] * length)
            data[match_type]['section'].extend([row['section']] * length)
            data[match_type]['processed_text'].extend([row['processed_text']] * length)
            data[match_type]['fa_text'].extend(all_top_fa_text)
            data[match_type]['identifier'].extend(all_top_idf)
            data[match_type]['similarity'].extend(all_top_scores)
            data[match_type]['match_type'].extend(all_match_types)

    df_match_term = pd.DataFrame(data=data['term'])
    df_match_term = df_match_term.sort_values(by=['section', 'processed_text', 'similarity'], ascending=False)
    df_match_term = df_match_term.drop_duplicates()
    df_match_term = df_match_term[~df_match_term.identifier.isna()].drop_duplicates(subset=['section', 'processed_text', 'identifier'])
    df_match_term['fa_text'] = df_match_term['fa_text'].map(lambda i: FA_TEXT_DICT[i] if FA_TEXT_DICT.get(i) else i)

    df_match_content = pd.DataFrame(data=data['content'])
    df_match_content = df_match_content.sort_values(by=['section', 'processed_text', 'similarity'], ascending=False)
    df_match_content = df_match_content.drop_duplicates()
    df_match_content = df_match_content[~df_match_content.identifier.isna()].drop_duplicates(subset=['section', 'processed_text', 'identifier'])
    df_match_content['fa_text'] = df_match_content['fa_text'].map(lambda i: FA_TEXT_DICT[i] if FA_TEXT_DICT.get(i) else i)

    logger.info('Term Match results is generated ..')

    # adding set of matched results attribute to match section and definition into list of content matching results
    final_data = []
    for i, row in df_ts.iterrows():
        section = row['section']
        processed_text = row['processed_text']
        text_element = row['text_element']
        # if TS text is not a section-text, consider matched FA terms and matched FA content search by TS content
        if text_element == 'section':
            row['result'] = False
        else:
            row['result'] = None
        df_s1 = df_match_content[(df_match_content.section == section) & (df_match_content.processed_text == processed_text)]
        # df_s2 = df_match_term[(df_match_term.section == section) & (df_match_term.processed_text == processed_text)]  # e.g. found ts_section "Borrower" in matched with results of definition/parties
        # df_s = pd.concat([df_s1, df_s2])
        df_s = df_s1
        df_s = df_s.sort_values(by=['similarity'], ascending=False)
        df_s = df_s.drop_duplicates(subset=['identifier'])

        row['TS_term'] = section
        row['TS_text'] = row['text']
        row['match_term_list'] = None
        row['identifier_list'] = None
        row['similarity_list'] = None
        row['match_type_list'] = None

        max_topN = sum(topN_type.values())

        # if not df_s.empty and row['text_element'] != 'section':
        if not df_s.empty:
            result_length = len([item for item in list(df_s['fa_text']) if item])
            padding_null = [None] * max(max_topN - result_length, 0)
            row['match_term_list'] = [item for item in list(df_s['fa_text']) if item][:max_topN] + padding_null
            row['identifier_list'] = [item for item in list(df_s['identifier']) if item][:max_topN] + padding_null
            row['similarity_list'] = [item for item in list(df_s['similarity']) if item][:max_topN] + padding_null
            row['match_type_list'] = [item for item in list(df_s['match_type']) if item][:max_topN] + padding_null
        final_data.append(row)

    cols = [
        'index',
        'text_block_id',
        'page_id',
        'phrase_id',
        'list_id',
        'text_element',
        'TS_term',
        'TS_text',
        'match_term_list',
        'identifier_list',
        'similarity_list',
        'match_type_list'
    ]

    df_final = pd.DataFrame(data=final_data)[cols]
    df_final = df_final.replace({np.nan: None, 'nan': None, 'None': None})
    df_final = df_final.sort_values(by=['index', 'phrase_id'])
    
    # merge results
    not_in_split_sec = ~df_final.TS_term.isin(split_sections)
    list_columns = ['identifier_list', 'match_term_list', 'similarity_list', 'match_type_list']
    df_final.loc[not_in_split_sec, list_columns] = df_final[not_in_split_sec].apply(lambda x: pd.Series(merge_idf_list(x), index=list_columns), axis=1, result_type='expand')
    
    # add wordbank
    df_final = modify_term_match(df_final, df_fa)
    df_final[['match_term_list', 'identifier_list', 'similarity_list', 'match_type_list']] = df_final[
        ['match_term_list', 'identifier_list', 'similarity_list', 'match_type_list']].map(lambda x: list(x) if isinstance(x, tuple) else x)

    # filter top N results on list of matched results
    df_final[['match_term_list', 'identifier_list', 'similarity_list', 'match_type_list']] = df_final[
        ['match_term_list', 'identifier_list', 'similarity_list', 'match_type_list']].map(lambda x: x[:topN] if isinstance(x, (list, tuple)) else x)
    
    # add additional information 【Section -Sub_section】after each identifier
    df_final['identifier_list'] = df_final['identifier_list'].map(lambda x: [i + f'【{id2sec_subsec.get(i)}】' if id2sec_subsec.get(i) else i for i in x] if x else x)

    if DEV_MODE and task_id:
        try:
            output_csv = os.path.join(OUTPUT_TERM_MATCH_CSV, f'{task_id}.csv')
            df_final.to_csv(output_csv, index=False, encoding='utf-8-sig')
            logger.debug(f'Term Match results for taskId {task_id} is generated and successfully save as {output_csv}.')
        except:
            pass

    return df_final


if __name__ == "__main__":
    import os
    import re
    import argparse
    import warnings
    import sys
    import logging
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=UserWarning)
    
    # Change root logger level from WARNING (default) to NOTSET in order for all messages to be delegated.
    logging.getLogger().setLevel(logging.NOTSET)

    # Add stdout handler, with level INFO
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    formater = logging.Formatter('【%(asctime)s】【%(filename)s:%(lineno)d】【%(levelname)-8s】%(message)s')
    console.setFormatter(formater)
    logging.getLogger().addHandler(console)

    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--src_fa_folder",
        default=f'{OUTPUT_DOCPARSE_CSV}FA_v4.4',
        type=str,
        required=False,
        help="The source FA docparse csv to be used for matching with TS.",
    )
    parser.add_argument(
        "--src_ts_folder",
        default=f'{OUTPUT_DOCPARSE_CSV}TS_v5.5',
        type=str,
        required=False,
        help="The source TS docparse csv to be used for matching with FA.",
    )
    parser.add_argument(
        "--model_path",
        default=SENT_BERT_MODEL_PATH,
        type=str,
        required=False,
        help="The path to sentence-transformer model apply for cosine similarity computation.",
    )
    parser.add_argument(
        "--output_term_match_folder",
        default=f'{OUTPUT_TERM_MATCH_CSV}tm_ts_v5.5_fa_v4.4_0307',
        type=str,
        required=False,
        help="The output folder of term matching results csv.",
    )
    parser.add_argument(
        "--target_files_list",
        default=None,
        type=list,
        required=False,
        help="The target TS filenames list that used to be test for development purpose.",
    )
    args = parser.parse_args()
    if not os.path.exists(args.output_term_match_folder):
        os.makedirs(args.output_term_match_folder)
    
    if not args.target_files_list:
        ts_file_list = sorted(os.listdir(args.src_ts_folder))
    else:
        ts_file_list = [f for f in sorted(os.listdir(args.src_ts_folder)) if f in args.target_files_list]

    # unannotated ts files
    # ts_file_list = [
    #     '125_VF_SYN_TS_mkd_20231013_docparse.csv',
    #     '126_VF_SYN_TS_mkd_20230601_docparse.csv',
    #     '127_NBFI_SYN_TS_mkd_20230330_docparse.csv',
    #     '128_NBFI_SYN_TS_mkd_20230224_docparse.csv',
    #     '129_AW_SYN_TS_mkd_20230626_docparse.csv',
    #     '130_BL_SYN_TS_mkd_20230710_docparse.csv',
    #     '131_AF_SYN_TS_mkd_20230718_docparse.csv',
    #     '132_PF_SYN_TS_mkd_20230106_docparse.csv',
    #     '133_BL_PRJ_TS_mkd_20230926_docparse.csv',
    #     '134_GF_PRJ_TS_mkd_20230829_docparse.csv',
    #     '136_BL_SYN_TS_mkd_20230418_docparse.csv',
    #     '137_GL_SYN_TS_mkd_20231121_docparse.csv',
    #     '138_GF_SYN_TS_mkd_20230130_docparse.csv',
    #     '139_NBFI_PRJ_TS_mkd_20231122_docparse.csv',
    #     '140_NBFI_SYN_TS_mkd_20230512_docparse.csv',
    #     '141_GL_SYN_TS_mkd_20231221_docparse.csv',
    #     '142_PF_SYN_TS_mkd_20230810_docparse.csv',
    #     '143_GF_SYN_TS_mkd_undated_docparse.csv',
    #     '144_NBFI_SYN_TS_mkd_20231031_docparse.csv',
    #     '145_GF_SYN_TS_mkd_20231031_docparse.csv',
    #     '146_BL_PRJ_TS_mkd_20230629_docparse.csv',
    #     '147_GL_PRJ_TS_mkd_20230817_docparse.csv',
    #     '135_GL_PRJ_TS_mkd_20231004_docparse.csv',
    #     '148_GF_PRJ_TS_mkd_20230919_docparse.csv',
    #     '149_BL_PRJ_TS_mkd_20231102_docparse.csv',
    # ]
    
    # UAT files
    ts_file_list = [
        '1_GL_SYN_TS_mkd_20221215_docparse.csv',
        '13_BL_SYN_TS_mkd_20220713_docparse.csv',
        '19_GL_SYN_TS_mkd_20220718_docparse.csv',
        '23_BL_SYN_TS_mkd_20220715_docparse.csv',
        '24_GF_PRJ_TS_mkd_20220225_docparse.csv',
        '25_NBFI_PRJ_TS_mkd_20220613_docparse.csv',
        '28_GF_PRJ_TS_mkd_20221111_docparse.csv',
        '29_PF_PRJ_TS_mkd_20200624_docparse.csv',
        '3_GF_SYN_TS_mkd_20221018_docparse.csv',
        '31_GL_PRJ_TS_mkd_20220630_docparse.csv',
        '33_BL_PRJ_TS_mkd_20200727_docparse.csv',
        '34_AWSP_PRJ_TS_mkd_20211230_docparse.csv',
        '41_AF_SYN_TS_mkd_20190307_docparse.csv',
        '43_AF_SYN_TS_mkd_20151101_docparse.csv',
        '45_AF_PRJ_TS_mkd_20170331_docparse.csv',
        '49_VF_PRJ_TS_mkd_20151130_docparse.csv',
        '54_VF_PRJ_TS_mkd_20191018_docparse.csv',
        '58_VF_SYN_TS_mkd_20111201_docparse.csv',
        '59_AWSP_SYN_TS_mkd_20210814_docparse.csv',
        '63_AW_SYN_TS_mkd_20221025_docparse.csv',
        '66_PF_SYN_TS_mkd_20230106_docparse.csv',
        '68_PF_SYN_TS_mkd_20221104_docparse.csv',
        '72_NBFI_SYN_TS_mkd_20221215_docparse.csv',
        '74_NBFI_SYN_TS_mkd_20220401_docparse.csv',
        '8_GF_SYN_TS_mkd_20230215_docparse.csv',
    ]
    
    for ts_fname in ts_file_list:
        logger.info(f'Term matching on filename: {ts_fname}')
        df_ts = pd.read_csv(os.path.join(args.src_ts_folder, ts_fname))
        fa_fname = re.sub('TS_', 'FA_', ts_fname)
        df_fa = pd.read_csv(os.path.join(args.src_fa_folder, fa_fname))
        new_name = re.sub('_docparse', '_docparse_results', ts_fname)
        output_term_match_path = os.path.join(args.output_term_match_folder, new_name)
        if not os.path.exists(output_term_match_path):
            df_term_match = main(df_ts, df_fa, topN=TOP_N_RESULTS, model_path=args.model_path)
            df_term_match.to_csv(output_term_match_path, index=False, encoding='utf-8-sig')
            logger.info(f'Term-matching results export as csv and save into: {output_term_match_path} \n')
        else:
            logger.info(f'{output_term_match_path} file exists. Skipped. \n')
