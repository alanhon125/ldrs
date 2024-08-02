'''
Function of ts_merge_fa.py:
1. Merge annotated docparse TS CSV (Doc 1) with docparse FA results in CSV (Doc 2) by clause id
2. Semantic Textual Similarity Matching between TS Section/Content and FA Keys/Content
Hierarchy of documents must be as following:

TS (Doc 1):
└── Title
└── Declaration
    └── Section keys (Left Column)
        └── Term description (Right Column)
        (Content Components: Caption, List, Paragraph, Phrase)

FA (Doc 2):
└── Title (with Date, Borrower, Amount, Agreement Type)
└── Table of Content
└── Parties Clauses (with party role)
└── Clause Section (with ID, description)
└── Clause Sub-Section (with ID , description)
     └── 1.1 Definition (Optional, with term)
     └── Paragraph/Caption + List items (Optional ,with ID, content)
└── Schedule (with ID , description)
└── Part(Optional, with ID , description)
     └── Section (Optional, with ID , description)
          └── Paragraph/Caption + List items (Optional, with ID)
└── Execution Pages
'''
import argparse
import pandas as pd
import numpy as np
import os
import json
import re
from utils import discard_str_with_unwant_char, log_task2csv, remove_punctuation, remove_particular_punct, stem_identifier, stem_string
from config import *
from ast import literal_eval
from operator import itemgetter
import time
from copy import deepcopy
import unicodedata
import traceback
import sys

def timeit(func):
    from functools import wraps
    import time

    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        row = {'test_list': args[0],
               'test_list_size': len(args[0]),
               'target_list': args[1],
               'target_list_size': len(args[1]),
               'result': result,
               'runtime': total_time}
        log_task2csv(LOG_DIR + 'log_semantic_textual_similarity.csv', row)
        # print(f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
        return result
    return timeit_wrapper


def is_roman_number(num):
    '''
    Check if a string is a roman numerals or not
    @param num: a string that may contains roman numeral
    @type num: str
    @return: True if it contains roman numerals else False
    @rtype: boolean
    '''
    pattern = re.compile(ROMAN_NUMERALS, re.VERBOSE | re.IGNORECASE)
    if re.match(pattern, num):
        return True

    return False


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
                row.insert(index, i.strip() if i else i)
    return row


def pairing_clauses_def(row, clause_id_def, file):
    '''
    To pair up the clause id and definition in a row of record
    Input a reference list of clause id that possess definitions, with variable name "def_clause_id"
    Example input:
        row["clause_id"] = ["Parties", "1.1"]
        row["definition"] = ["Original Lenders", "Lender"]
        clause_id_def = [('1.1','Borrower'), ('1.2','Interest'), ('1.3','Lender'), ('1.4','Facility'), ('12.1','Gurantor')]
    Example Output:
        pairs = [("Parties","Original Lenders"),("1.1","Lender")]
    @param row: row of record from TS DataFrame that contains column fields "clause_id", "definition"
    @type row: pd.Series
    @param def_clause_id: list of clause id that possess definitions in FA
    @type def_clause_id: list
    @return : list of pairs in (<CLAUSE_ID>, <DEFINITION>)
    @rtype : list of tuples
    '''
    clause_id = row['clause_id']
    definitions = row['definition']
    if clause_id_def:
        def_clause_id = list(set(list(zip(*clause_id_def))[0]))
        defs = list(zip(*clause_id_def))[1]
    else:
        def_clause_id = []
        defs = []

    pairs = []
    if not clause_id and not definitions:
        return [(None, None)]
    elif not clause_id and isinstance(definitions, list):
        clause_id = ['1.1'] * len(definitions)
    elif isinstance(clause_id, list) and not definitions:
        definitions = ['nan'] * len(clause_id)

    # split clause id string with expression '1.1\n2.1' into list ['1.1', '2.1']
    for index, c_id in enumerate(clause_id):
        if '\n' in c_id:
            clause_id.remove(c_id)
            c_ids = c_id.splitlines()
            for i in c_ids:
                clause_id.insert(index, i)
    # split definitions string with expression 'Borrower\nBusiness Day' into list ['Borrower', 'Business Day']
    for index, d in enumerate(definitions):
        if '\n' in d:
            definitions.remove(d)
            ds = d.splitlines()
            for i in ds:
                definitions.insert(index, i)
    clause_id = [i.strip() for i in clause_id if i.strip() and i.strip() != 'nan']
    definitions = [i.strip() for i in definitions if i.strip() and i.strip() != 'nan']
    last_c_id = None

    for c_id in clause_id:
        if c_id in ['Parties', 'parties', 'Party', 'party'] + def_clause_id:
            # if c_id not in ['Parties', 'parties', 'Party', 'party']:
            #     candidate_idx = [i for i, Id in enumerate(def_clause_id) if c_id == Id]
            #     candidate_def = [defs[i] for i in candidate_idx]
            try:
                pairs.append((c_id, definitions[0]))
                definitions.pop(0)
                last_c_id = c_id
            except IndexError:
                print(f"In file: {file}\nIndex: {row['index']},\n def_clause_id: {def_clause_id},\n clause_id: {clause_id},\n definitions: {definitions}.\n Definition or party name has to be provided in TS annotation csv. \nPlease review Index {row['index']}")

        elif c_id and c_id != 'nan':
            pairs.append((c_id, None))

    while definitions:
        pairs.append((last_c_id, definitions[0]))
        definitions.pop(0)

    return pairs if pairs else [(None, None)]


def extract_id(id_string, ref_id_type, id_type):
    '''
    extract specific id_type from string of ID with pattern like 1.1.2(a)
    @param id_string: id_string with pattern <SECTION_ID>.<SUBSECTION_ID>.<LIST_ID>
    @type id_string: str
    @param ref_id_type: reference_id_type in ['schedule_id', 'clause_id', 'definition']
    @type ref_id_type: str
    @param id_type: id_type in ['fa_section_id', 'fa_schedule_id', 'fa_sub_section_id', 'fa_part_id', 'fa_list_id']
    @type id_type: str
    @return: string of specific id or None if not match the pattern
    @rtype : str or None
    '''
    import re
    match = re.match(CLAUSE_SEC_SUBSEC_LIST, str(id_string))
    match2 = re.match(CLAUSE_SEC_LIST, str(id_string))
    match_sched = re.match(SCHED_SEC_LIST, str(id_string))
    match_def = re.match('.*_(.*)', str(id_string))  # Lender_(a)
    if match_sched and ref_id_type == 'schedule_id':
        group = match_sched.groups()
        if id_type == 'fa_schedule_id':
            return str(group[0])
        elif id_type == 'fa_part_id':
            if group[1] == '0':
                return None
            return str(group[1])
        # elif id_type == 'fa_section_id':
        #     return str(group[2])
        elif id_type == 'fa_list_id':
            list_id = group[-1]
            return str(list_id)
    elif match and ref_id_type == 'clause_id':
        group = match.groups()
        if id_type == 'fa_section_id':
            return str(group[0])
        elif id_type == 'fa_sub_section_id':
            if group[1] == '0':
                return None
            return str(group[1])
        else:
            list_id = group[-1]
            return str(list_id)
    elif match2 and ref_id_type == 'clause_id':
        group = match2.groups()
        if id_type == 'fa_section_id':
            return str(group[0])
        elif id_type == 'fa_sub_section_id':
            return None
        else:
            list_id = group[-1]
            return str(list_id)
    elif match_def and ref_id_type == 'definition':
        group = match_def.groups()
        return str(group[0])
    else:
        return None

def decompose_fa_identifier(fa_identifier):
    import re
    fa_section_id = fa_sub_section_id = fa_schedule_id = fa_part_id = fa_definition = fa_clause_list_id = fa_schedule_list_id = None
    if not fa_identifier:
        return fa_section_id, fa_sub_section_id, fa_schedule_id, fa_part_id, fa_definition, fa_clause_list_id, fa_schedule_list_id

    if fa_identifier.startswith('Cl'):
        fa_identifier = fa_identifier.lstrip('Cl_')
        splits = fa_identifier.split('.')
        if len(splits) == 1:
            fa_section_id = fa_identifier
        elif len(splits) == 2:
            fa_section_id, ids2 = splits
            if '-' in ids2:
                splits2 = ids2.split('-')
                if len(splits2) == 3:
                    id3, fa_definition1, fa_definition2 = splits2
                    fa_definition = fa_definition1 + '-' + fa_definition2
                else:
                    id3, fa_definition = splits2
                if ')' in id3:
                    fa_sub_section_id , fa_clause_list_id = re.match(r'(\w+\.*\w*\.*\w*)(\(*\w+\))',id3).groups()
                else:
                    fa_sub_section_id = id3
            else:
                if ')' in ids2:
                    fa_sub_section_id , fa_clause_list_id = re.match(r'(\w+\.*\w*\.*\w*)(\(*\w+\))',ids2).groups()
                else:
                    fa_sub_section_id = ids2
        elif len(splits) == 3:
            fa_section_id, fa_sub_section_id, fa_clause_list_id = splits

    elif fa_identifier.startswith('Sched'):
        fa_identifier = fa_identifier.lstrip('Sched_')
        splits = fa_identifier.split('_')
        if len(splits) == 1:
            if '.' in fa_identifier and len(fa_identifier.split('.'))==2:
                fa_schedule_id, fa_part_id = fa_identifier.split('.')
            else:
                fa_schedule_id = fa_identifier
        elif len(splits) == 2:
            ids, fa_schedule_list_id = splits
            if '.' in ids and len(ids.split('.'))==2:
                fa_schedule_id, fa_part_id = ids.split('.')
            else:
                fa_schedule_id = ids
        elif len(splits) == 3:
            ids, fa_section_id, fa_schedule_list_id = splits
            if '.' in ids and len(ids.split('.'))==2:
                fa_schedule_id, fa_part_id = ids.split('.')
            else:
                fa_schedule_id = ids

    elif fa_identifier.startswith('Parties'):
        splits = fa_identifier.split('-')
        if len(splits) == 2:
            _, parties = splits
        elif len(splits) == 3:
            _, parties1, parties2 = splits
            parties = parties1 + '-' + parties2
        fa_definition = parties
        fa_section_id = '0'
    else:
        return fa_section_id, fa_sub_section_id, fa_schedule_id, fa_part_id, fa_definition, fa_clause_list_id, fa_schedule_list_id

    return fa_section_id, fa_sub_section_id, fa_schedule_id, fa_part_id, fa_definition, fa_clause_list_id, fa_schedule_list_id

def remove_articles(text):
    import re
    return re.sub('(a|an|and|the)(\s+)', '\2', text, flags=re.I).strip('\x02')


def stem_word(sentence, ps):
    # from nltk.stem import PorterStemmer
    from nltk.tokenize import word_tokenize

    if sentence is None:
        return sentence

    # ps = PorterStemmer()
    sentence = remove_articles(sentence)
    words = word_tokenize(sentence)
    new_sent = ' '.join([ps.stem(w) for w in words])
    return new_sent


def extract_fa_text(df_ts, df_fa, ps):
    '''
    To extract FA content by TS annotation keys (Section ID, Subsection ID, Definitions, Schedule ID, Part ID, List ID)
    @param df_ts: Pandas Dataframe of TS
    @type: pd.DataFrame
    @param df_fa: Pandas Dataframe of FA
    @type: pd.DataFrame
    '''
    import string

    isParties = re.match('Parties|Party', df_ts['clause_id'], re.IGNORECASE) if df_ts['clause_id'] else df_ts['clause_id']
    sec_id = df_ts['fa_section_id'] if df_ts['fa_section_id'] else df_ts['fa_section_id']
    sub_sec_id = df_ts['fa_sub_section_id'] if df_ts['fa_sub_section_id'] else df_ts['fa_sub_section_id']
    sched_id = df_ts['fa_schedule_id'] if df_ts['fa_schedule_id'] else df_ts['fa_schedule_id']
    part_id = df_ts['fa_part_id'] if df_ts['fa_part_id'] else df_ts['fa_part_id']
    clause_list_id = df_ts['fa_clause_list_id'] if df_ts['fa_clause_list_id'] else df_ts['fa_clause_list_id']
    sched_list_id = df_ts['fa_schedule_list_id'] if df_ts['fa_schedule_list_id'] else df_ts['fa_schedule_list_id']
    if sched_list_id and re.match(r'(\d+)(\(*.*\))*', sched_list_id):
        sched_sec_id, sched_list_id2 = re.match(r'(\d+)(\(*.*\))*', sched_list_id).groups()
    else:
        sched_sec_id = sched_list_id2 = None

    definition = re.sub(r'_.*', '', df_ts['definition']) if df_ts['definition'] else df_ts['definition']
    definition2 = stem_word(definition, ps)
    # definition2 = df_ts['definition'].removesuffix('s')+'|'+df_ts['definition'] if df_ts['definition'] else df_ts['definition']

    # extract parties clauses df
    # isPartiesStart = df_fa.fa_text.str.contains(PARTIES_BEGIN_PATTERN, na=False, case=False)
    # isPartiesEnd = df_fa.fa_text.str.contains(PARTIES_END_PATTERN, na=False, case=False)
    # partiesBeginID = df_fa[isPartiesStart]['fa_index'].values[0] + 1
    # partiesEndID = df_fa[isPartiesEnd]['fa_index'].values[0] - 1
    # parties = df_fa[df_fa['fa_index'].between(partiesBeginID, partiesEndID)]

    parties = df_fa[df_fa.fa_section.eq('PARTIES')]
    df_fa = df_fa.replace({'nan': None, np.nan: None, 'None': None})
    df_fa['fa_list_id'] = df_fa['fa_list_id'].astype(str).map(lambda x: x.replace('.', '') if x else x)
    # if 'clause_id' == 'Parties' or 'Party'
    if isParties:
        parties['fa_definition_stem'] = parties['fa_definition'].map(lambda x: stem_word(x, ps))
        parties['fa_definition_in_ts_definition'] = parties['fa_definition_stem'].map(lambda x: bool(re.match(rf'.*{re.escape(str(x))}$', str(definition2), re.IGNORECASE)) if x else str(x) == str(definition2))
        parties_cond = (parties['fa_definition_stem'].str.contains(rf'^{re.escape(str(definition2))}|{re.escape(str(definition2))}$', na=False)
                        if definition2 else parties['fa_definition_stem'].astype(str) == str(definition2)) | (parties['fa_definition_in_ts_definition'])
        # [parties['fa_text'].str.contains(re.escape(str(definition)), na=False, case=False)]
        df = parties.loc[parties_cond]
        if len(df) == 0:
            return None, None
        elif len(df) == 1:
            return df[['fa_text', 'fa_identifier']].values[0].tolist()

    # if 'clause_id' not in string ['Parties','Party']
    if all(i is None for i in [sec_id, sub_sec_id, sched_id, part_id, clause_list_id, sched_list_id]):
        return None, None
    else:
        # if list_id exists, either clause_list_id or sched_list_id exists
        if not clause_list_id and sched_list_id and sched_sec_id and sched_list_id2:
            list_id_cond = (df_fa['fa_list_id'].str.contains('^'+re.escape(str(sched_list_id))+'$', regex=True, na=False, case=False) | ((df_fa['fa_section_id'].astype(str) == str(sched_sec_id)) & df_fa['fa_list_id'].str.contains('^'+re.escape(str(sched_list_id2))+'$', regex=True, na=False, case=False)))
        elif not clause_list_id and sched_list_id and sched_sec_id and not sched_list_id2:
            list_id_cond = (df_fa['fa_list_id'].str.contains('^'+re.escape(str(sched_list_id))+'$', regex=True, na=False, case=False) | (df_fa['fa_section_id'].astype(str) == str(sched_sec_id)))
        elif not clause_list_id and sched_list_id and not sched_sec_id and not sched_list_id2:
            list_id_cond = (df_fa['fa_list_id'].str.contains('^'+re.escape(str(sched_list_id))+'$', regex=True, na=False, case=False))
        elif clause_list_id and not sched_list_id:
            list_id_cond = (df_fa['fa_list_id'].str.contains('^'+re.escape(str(clause_list_id))+'$', regex=True, na=False, case=False))
        else:
            list_id_cond = pd.Series(data=[True] * len(df_fa))

        if sec_id:
            sec_id_cond = (df_fa['fa_section_id'].astype(str) == str(sec_id))
        else:
            sec_id_cond = pd.Series(data=[True] * len(df_fa))

        if sub_sec_id:
            sub_section_id_cond = ((df_fa['fa_sub_section_id'].astype(str) == str(sub_sec_id)) | (df_fa['fa_sub_section_id'].str.lstrip('0') == str(sub_sec_id).lstrip('0')))
        else:
            sub_section_id_cond = pd.Series(data=[True] * len(df_fa))

        if sched_id:
            sched_id_cond = (df_fa['fa_schedule_id'].astype(str) == str(sched_id))
        else:
            sched_id_cond = pd.Series(data=[True] * len(df_fa))

        if part_id:
            part_id_cond = (df_fa['fa_part_id'].astype(str) == str(part_id))
        else:
            part_id_cond = pd.Series(data=[True] * len(df_fa))

        df_fa['fa_definition_stem'] = df_fa['fa_definition'].map(lambda x: stem_word(x, ps))
        # defintion_cond = df_fa['fa_definition_stem'].str.contains(rf'^{re.escape(str(definition2))}$',na=False) if definition2 else df_fa['fa_definition_stem'] == str(definition2)
        # defintion_cond = df_fa['fa_definition_stem'].str.contains(rf'^{re.escape(str(definition2))}|{re.escape(str(definition2))}$',na=False) if definition2 else df_fa['fa_definition_stem'].astype(str) == str(definition2)
        df_fa['fa_definition_in_ts_definition'] = df_fa['fa_definition_stem'].map(lambda x: bool(re.match(rf'.*{re.escape(str(x))}$', str(definition2), re.IGNORECASE)) if x else str(x) == str(definition2))
        defintion_cond = (df_fa['fa_definition_stem'].str.contains(rf'^{re.escape(str(definition2))}$|^{re.escape(str(definition2))} or .*$|.* or {re.escape(str(definition2))}.*$',
                          na=False) if definition2 else df_fa['fa_definition_stem'].astype(str) == str(definition2)) | (df_fa['fa_definition_in_ts_definition'])
        df = df_fa.loc[sec_id_cond &
                       sub_section_id_cond &
                       sched_id_cond &
                       part_id_cond &
                       defintion_cond &
                       list_id_cond, :]
        df['fa_identifier'] = df.apply(generate_identifier, axis=1)
        if len(df) == 0:
            return None, None
        elif len(df) == 1:
            return df[['fa_text', 'fa_identifier']].values[0].tolist()
        else:
            # reconstruct the string which definition exists
            fa_text_all_ids = df['fa_identifier'].values.tolist()
            df.loc[df.fa_text_element.str.contains('table'), 'fa_text'] = df[df.fa_text_element.str.contains('table')]['fa_text'].map(remove_particular_punct)
            fa_text = [str(i) for i in df['fa_text'].values.tolist() if i]
            fa_text = ''.join([c + (' ' if c.endswith(tuple([i for i in string.punctuation])) else '; ') for c in fa_text]).strip()

            return fa_text, fa_text_all_ids


@timeit
def get_similarity_sentbert(test_list, target_list, sim_threshold=0.6,
                            model_path=SENTBERT_MODEL_PATH, pretrained=None,
                            return_single_response=True, topN=None):
    '''
    calculate string similarity between test string and reference list of strings using SentenceTransformer,
    return (the most similar string from reference list, similarity score) which greater than given similarity threshold.

    @param test_list: a string or list of string that to be examinate its similarity to reference list of string
    @type test_list: list or str
    @param target_list: a reference list of list of strings OR string
    @type target_list: list of list or str
    @param sim_threshold: a similarity score threshold to accept the max similarity score and return string (by default = 0.6)
    @type sim_threshold: float
    @param model_path: the local model path of Sentence-BERT
    @type model_path: str
    @param pretrained: if prediction leverage pre-trained model from huggingface, provide the name of model, otherwise set to None
    @type pretrained: str
    @param return_single_response: set True if single response of (string, similarity score) as a result, otherwise response (list of string, list of similarity score)
    @type return_single_response: boolean
    @rtype: tuple
    @return: tuple of (the most similar string from first item of reference list, similarity score)

    If return_single_response==False, it generates the dictionary of test candidate to sim_score and target candidate by input list of result candidates and list of list of target candidate keywords
    Example output of test_target_similar_candidate_dict:
    {
        "CO2 emissions": {                           # test string
            "score": 0.7978004217147827,             # similarity score
            "similar_candidate": "Carbon Dioxides"   # target candidate string
        }, ...
    }
    '''
    from sentence_transformers import SentenceTransformer, util
    import torch

    use_cuda = torch.cuda.is_available()
    device = torch.device(f'cuda:0' if use_cuda else 'cpu')

    if pretrained is not None:
        model = SentenceTransformer(pretrained, device=device)
    else:
        model = SentenceTransformer(model_path, device=device)

    if isinstance(test_list, str):
        test_words_list = test_list
        try:
            embeddings1 = model.encode(test_words_list, convert2tensor=True)
        except:
            embeddings1 = model.encode(test_words_list)

        all_sims = []
        relevant = False

        if isinstance(target_list, list) or isinstance(target_list, tuple):
            for data_field in target_list:
                sims = []
                # for data_field in data_field_list:
                #     if data_field or data_field != '':
                # data_field = data_field.lower()
                try:
                    embeddings2 = model.encode(data_field, convert2tensor=True)
                except:
                    embeddings2 = model.encode(data_field)
                similarity = util.pytorch_cos_sim(embeddings1, embeddings2).item()
                sims.append(similarity)
                max_sim = max(sims)
                all_sims.append(max_sim)

            if return_single_response:
                max_all_sims = max(all_sims)
                max_index = all_sims.index(max_all_sims)

                if max_all_sims >= sim_threshold:
                    relevant = True
                    sim_score = round(max_all_sims, 2)
                    ref_string = target_list[max_index][0]
            else:
                ref_string = []
                sim_score = []
                for sim in all_sims:
                    if sim >= sim_threshold:
                        above_threshold_idx = all_sims.index(sim)
                        relevant = True
                        sim_score.append(round(sim, 2))
                        ref_string.append(target_list[above_threshold_idx][0])

        else:
            try:
                embeddings2 = model.encode(target_list, convert2tensor=True)
            except:
                embeddings2 = model.encode(target_list)
            sim_score = round(util.pytorch_cos_sim(embeddings1, embeddings2).item(), 2)
            if sim_score >= sim_threshold:
                relevant = True
                ref_string = target_list

        if not relevant:
            return None, 0
        else:
            return ref_string, sim_score

    elif isinstance(test_list, list) or isinstance(test_list, tuple):
        # test_words_list = [s.lower() for s in test_string]
        test_words_list = test_list
        target_words_list = []
        schema_map = {}
        similar_pairs = {}
        for data_field in target_list:
            # for data_field in data_field_list:
            #     if data_field or data_field != '':
            # data_field = data_field.lower()
            target_words_list.append(data_field)
            # schema_map.update({data_field: data_field_list[0]})
        try:
            embeddings1 = model.encode(test_words_list, convert2tensor=True)
            embeddings2 = model.encode(target_words_list, convert2tensor=True)
        except:
            embeddings1 = model.encode(test_words_list)
            embeddings2 = model.encode(target_words_list)
        cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)

        for i in range(0, len(test_words_list)):
            if return_single_response:
                max_index = list(cosine_scores[i]).index(max(cosine_scores[i]))
                max_all_sims = max(cosine_scores[i]).item()
                if max_all_sims >= sim_threshold:
                    sim_score = round(max_all_sims, 2)
                    # schema_map.get(target_words_list[max_index])
                    ref_string = target_words_list[max_index]
                    # else:
                    #     sim_score = 0
                    #     ref_string = None

                    similar_pairs.update({
                        test_words_list[i]: {
                            'similar_candidates': ref_string,
                            'score': sim_score
                        }
                    })
            else:
                all_score = list(cosine_scores[i])
                above_threshold_idx = [all_score.index(k) for k in [j for j in all_score if j >= sim_threshold]]
                above_threshold_sims = [j.item() for j in all_score if j >= sim_threshold]
                idx_sims = list(zip(above_threshold_idx, above_threshold_sims))
                idx_sims = sorted(idx_sims, key=lambda x: x[1], reverse=True)
                ref_string = []
                sim_score = []
                for idx, sims in idx_sims:
                    # schema_map.get(target_words_list[idx])
                    string = target_words_list[idx]
                    if string not in ref_string:
                        ref_string.append(string)
                        sim_score.append(round(sims, 2))

                if ref_string:
                    similar_pairs.update({
                        test_words_list[i]: {
                            'similar_candidates': ref_string[:topN],
                            'score': sim_score[:topN]
                        }
                    })

        return similar_pairs


def term_matching(df_ts, df_fa, id_mapper, sim_threshold=0, topN=5):
    '''
    perform semantic similarity match between ts and fa

    @param df_ts: a string or list of string that to be examinate its similarity to reference list of string
    @type df_ts: list or str

    @param df_ts: a string or list of string that to be examinate its similarity to reference list of string
    @type df_ts: list or str
    '''
    # extract schedule df
    schedBeginID = df_fa[df_fa.fa_text.str.contains(SCHED_TITLE, na=False, case=False) & (~df_fa.fa_text.str.contains(EXCLUD_STR_IN_SCHED_TITLE, na=False, case=False))]['fa_index'].values[0]
    schedEndID = df_fa["fa_index"].iloc[-1]
    sched_index = df_fa['fa_index'].between(schedBeginID, schedEndID)
    df_sched = df_fa[sched_index]
    df_fa_sched = df_sched[~df_sched.fa_schedule.isna()].drop_duplicates(["fa_schedule"])
    df_fa_part = df_sched[~df_sched.fa_part.isna()].drop_duplicates(["fa_part"])
    part_list = [i for i in list(df_fa_part.fa_part) if i]

    # extract clauses df
    clause_index = df_fa['fa_index'].between(0, schedBeginID - 1)
    df_clauses = df_fa[clause_index]
    df_fa_sec = df_clauses[~df_clauses.fa_section.isna()].drop_duplicates(["fa_section"])
    df_fa_sub_sec = df_clauses[~df_clauses.fa_sub_section.isna()].drop_duplicates(["fa_sub_section"])
    df_fa_def = df_clauses[~df_clauses.fa_definition.isna()].drop_duplicates(["fa_definition"])

    # extract parties df
    df_parties = df_fa[df_fa.fa_section.eq('PARTIES')]

    ts_section = df_ts[~df_ts['section'].isin([''])]['section']
    ts_sec_list = list(ts_section.unique())
    sec_list = [i for i in list(df_fa_sec.fa_section) if i]
    sub_sec_list = [i for i in list(df_fa_sub_sec.fa_sub_section) if i]
    def_list = [i for i in list(df_fa_def.fa_definition) if i]
    sched_list = [i for i in list(df_fa_sched.fa_schedule) if i]
    parties_list = [i for i in df_parties.fa_text.values.tolist() if i]

    # extract a list of content text that without punctuation, non-numeric, non-roman-numeric for similarity matching
    ts_text = df_ts[~df_ts.text_element.str.contains('section', na=False)]['text']  # .map(remove_punctuation)
    # contain alphabet, numeral, whitespaces
    ts_text = ts_text[ts_text.str.contains('[a-zA-Z\s\d]+$', na=False)]
    ts_text = ts_text[~ts_text.map(lambda x: is_roman_number(x.replace(" ", "")))]  # not roman numerals
    ts_text = ts_text[~ts_text.map(lambda x: x.replace(" ", "").isnumeric())]
    ts_text_list = list(ts_text.unique())

    fa_text = df_fa[~df_fa.fa_text_element.str.contains('section', na=False)]['fa_text']  # .map(remove_punctuation)
    # contain alphabet, numeral, whitespaces
    fa_text = fa_text[fa_text.str.contains('[a-zA-Z\s\d]+$')]
    fa_text = fa_text[~fa_text.map(lambda x: is_roman_number(x.replace(" ", "")))]  # not roman numerals
    fa_text = fa_text[~fa_text.map(lambda x: x.replace(" ", "").isnumeric())]
    fa_text_list = list(fa_text.unique())

    sec2type = {i: 'sec' for i in sec_list}
    def2type = {i: 'def' for i in def_list}
    parties2type = {i: 'parties' for i in parties_list}
    sub_sec2type = {i: 'sub_sec' for i in sub_sec_list}
    sched2type = {i: 'sched' for i in sched_list}
    part2type = {i: 'part' for i in part_list}
    fa_text2type = {i: 'clause_text' for i in fa_text_list}
    L = [sec2type, def2type, parties2type, sub_sec2type, sched2type, part2type, fa_text2type]
    all_fa_string2type = {k: v for d in L for k, v in d.items()}

    all_ts_text = ts_sec_list + ts_text_list

    all_fa_text = []
    targetFAList = [sec_list, def_list, parties_list, sec_list, sub_sec_list, def_list, sched_list, part_list, fa_text_list]
    for l in targetFAList:
        if l:
            all_fa_text += l
    simdict = get_similarity_sentbert(all_ts_text, all_fa_text, sim_threshold=sim_threshold, return_single_response=False, topN=topN)
    for ts_type in ['text', 'section']:
        if 'section' == ts_type:
            ts_type2 = 'sec'
            df_ts_content = df_ts.text_element.str.contains('section', na=False)
        else:
            df_ts_content = ~df_ts.text_element.str.contains('section', na=False)
        # add semantic textual similarity by different matching keys to df_ts, splitting into matched_result, sim_score, id and judgement
        df_ts.loc[df_ts_content, 'match_term_list'] = df_ts[df_ts_content][ts_type].map(lambda x: map_sim(x, simdict)[0])
        df_ts.loc[df_ts_content, 'similarity_list'] = df_ts[df_ts_content][ts_type].map(lambda x: map_sim(x, simdict)[1])
        df_ts.loc[df_ts_content, 'match_term_type'] = df_ts[df_ts_content]['match_term_list'].map(lambda x: (all_fa_string2type.get(i) for i in x))
        df_ts.loc[df_ts_content, 'zip_type_term'] = df_ts[df_ts_content].apply(lambda x: list(zip(*x[['match_term_type', 'match_term_list']])), axis=1)  # e.g. [('sec','Definitions'),...]
        df_ts.loc[df_ts_content, 'identifier_list'] = df_ts[df_ts_content]['zip_type_term'].map(lambda x: (id_mapper[i[0]][i[1]] for i in x))
        df_ts.loc[df_ts_content, 'match_type_list'] = df_ts[df_ts_content]['match_term_type'].map(lambda x: (ts_type2 + '2' + i for i in x))  # sec2sec, sec2def etc.

    # # create list pair dictionary that key: pairing strategy, value: (testStringType, testList, targetStringType, targetList)
    # listPairDict = {
    #     'sec2sec': ('section', ts_sec_list, 'section', sec_list),
    #     'sec2def': ('section', ts_sec_list, 'definition', def_list),
    #     'sec2parties': ('section', ts_sec_list, 'parties', parties_list),
    #     'text2sec': ('text', ts_text_list, 'section', sec_list),
    #     'text2sub_sec': ('text', ts_text_list, 'sub_section', sub_sec_list),
    #     'text2def': ('text', ts_text_list, 'definition', def_list),
    #     'text2sched': ('text', ts_text_list, 'schedule', sched_list),
    #     'text2part': ('text', ts_text_list, 'part', part_list),
    #     'text2clause_text': ('text', ts_text_list, 'text', fa_text_list)
    # }
    # sim_list_cols = []
    # for k, (testStringType, testList, targetStringType, targetList) in listPairDict.items():
    #     if targetList:
    #         if 'section' in testStringType:
    #             df_ts_content = df_ts.text_element.str.contains('section',na=False)
    #         else:
    #             df_ts_content = ~df_ts.text_element.str.contains('section',na=False)
    #         # create dictionaries of semantic similarity mapping
    #         simdict = get_similarity_sentbert(testList, targetList, sim_threshold=sim_threshold, return_single_response=False, topN=topN)
    #         # add semantic textual similarity by different matching keys to df_ts, splitting into matched_result, sim_score, id and judgement
    #         df_ts.loc[df_ts_content, k] = df_ts[df_ts_content][testStringType].map(lambda x: map_sim(x,simdict, targetStringType)[0])
    #         df_ts.loc[df_ts_content, k+'_sim'] = df_ts[df_ts_content][testStringType].map(lambda x: map_sim(x, simdict, targetStringType)[1])
    #         df_ts.loc[df_ts_content, k+'_id'] = df_ts[df_ts_content][testStringType].map(lambda x: map_sim(x, simdict, targetStringType)[2])
    #         sim_list_cols += [k, k+'_sim', k+'_id']
    # combine_list_cols = ['match_term_list','identifier_list','similarity_list','match_type_list']
    # df_ts[combine_list_cols] = pd.DataFrame(data={k:[()]*(len(df_ts)+1) for k in combine_list_cols})
    # for k, (testStringType, testList, targetStringType, targetList) in listPairDict.items():
    #     if targetList:
    #         df_ts.loc[~df_ts[k].isnull(),'match_term_list'] += df_ts[~df_ts[k].isnull()][k]
    #         df_ts.loc[~df_ts[k].isnull(),'identifier_list'] += df_ts[~df_ts[k].isnull()][k+'_id']
    #         df_ts.loc[~df_ts[k].isnull(),'similarity_list'] += df_ts[~df_ts[k].isnull()][k+'_sim']
    #         df_ts.loc[~df_ts[k].isnull(),'match_type_list'] += df_ts[~df_ts[k].isnull()][k].map(lambda x: tuple([k])*len(x))

    # df_ts = df_ts.replace({'nan': None, np.nan: None, 'None': None, '': None})
    # emptyTuple = df_ts.match_type_list.str.len() == 0
    # df_ts.loc[emptyTuple,['match_term_list','identifier_list','similarity_list','match_type_list']] = df_ts[emptyTuple][['match_term_list','identifier_list','similarity_list','match_type_list']].applymap(lambda x: None)# tuple([None])*len(listPairDict.keys()))
    # # df_ts.loc[emptyTuple, 'similarity_list'] = df_ts[emptyTuple]['similarity_list'].map(lambda x: tuple([0]) * len(listPairDict.keys()))
    # # df_ts.loc[emptyTuple, 'match_type_list'] = df_ts[emptyTuple]['match_type_list'].map(lambda x: tuple(list(listPairDict.keys())))
    # emptyTuple = df_ts.match_term_list.isna() | df_ts.identifier_list.isna() | df_ts.similarity_list.isna() | df_ts.match_type_list.isna()
    # df_ts.loc[~emptyTuple,'zip_result'] = df_ts[~emptyTuple].apply(lambda x: list(zip(*x[combine_list_cols])),axis=1)
    # df_ts.loc[~emptyTuple,'zip_result'] = df_ts[~emptyTuple]['zip_result'].map(lambda x: sorted(x,key=itemgetter(2),reverse=True))
    # df_ts.loc[~emptyTuple,'zip_result'] = df_ts[~emptyTuple]['zip_result'].map(lambda x: unique_tuples(x,1))
    # for i, col in enumerate(combine_list_cols):
    #     df_ts.loc[~emptyTuple,col] = df_ts[~emptyTuple]['zip_result'].map(lambda x: list(zip(*x))[i][:topN])
    # df_ts.drop(columns=['zip_result'], inplace=True)
    # if not args.keep_individual_strategy_result:
    #     df_ts.drop(columns=sim_list_cols, inplace=True)

    return df_ts


def map_sim(text, sim_dict):
    '''
    @param text: string
    @type text: str
    @param sim_dict: similarity result dictionary
    @type sim_dict: dict in form:
    {
        <STRING>: {
            "score": <SIMILARITY_SCORE>,
            "similar_candidate": <CANDIDATE_STRING>
        }, ...
    }
    @param key_type: key type which is the keys of id_mapper
    Return (similar_candidates, scores, key_ids)
    '''
    text = str(text)
    text = re.sub(r'<s>|^nan$|^None$', '', text).strip()
    # text = remove_punctuation(text)
    # text = text.lower() if text else text
    sim = sim_dict.get(text, None)
    if sim:
        similar_candidates = tuple(sim['similar_candidates'])
        scores = tuple(sim['score'])
        return similar_candidates, scores
    else:
        return None, None


def map_text_granularity(row: pd.DataFrame, df: pd.DataFrame):
    '''
    categorize the content text into sentence, phrase, term based on sentence pattern, word count and text frequency within the same text component group
    '''
    from nltk.tokenize import word_tokenize
    import re
    index = row['index']
    previous_text = str(row['previous_text']) if row['previous_text'] else None
    text = str(row['text']) if row['text'] else None
    text_element = row['text_element']
    text_grp_count = len(df[df['index'] == index])
    if not text:
        return None
    if text_grp_count > 1:
        if str(previous_text).endswith(('.', ':')) and text.endswith(('.')) or len(word_tokenize(text)) > 10 and text.endswith(':'):
            return 'sentence'
        elif len(word_tokenize(text)) < 10 and not text.endswith(('.', 'and', ';')) and re.match(r'paragraph.*|list.*|caption.*', text_element):
            return 'term'
        elif len(word_tokenize(text)) < 10 and not text.endswith(('and', ';')) and re.match(r'list.*', text_element):
            return 'term'
        else:
            return 'phrase'
    elif text_grp_count == 1:
        if len(word_tokenize(text)) < 10 and not text.endswith(('.', 'and', ';')) and re.match(r'paragraph.*|list.*|caption.*', text_element):
            return 'term'
        elif len(word_tokenize(text)) < 10 and not text.endswith(('and', ';')) and re.match(r'list.*', text_element):
            return 'term'
        else:
            return 'sentence'


def annotation_granularity(df, label_col='fa_identifier'):
    '''
    To examine the annotation granularity, the examination logic as follow:
    If all identifier are the same in a group and text_granularity is phrase, implies it is an annotation at paragraph-level;
    If all identifier are the same in a group and text_granularity is sentence, implies it is an annotation at sentence-level;
    If NOT all identifier are the same in a group and text_granularity is phrase, implies it is an annotation at phrase-level;
    If identifier is None in a group, implies there is no annotation;
    @param df: df groupby object that contains column field "text_granularity" and label_col
    @type df: pandas.DataFrame.groupby object
    @param label_col: a label column name in df that contain identifier
    @type label_col: str
    @return : the label of annotation granularity
    @rtype : str or None
    '''
    text_gran = df['text_granularity'].values[0]
    annotation = df[label_col].values[0]
    if annotation and (df[label_col] == annotation).all() and text_gran == 'phrase':
        return 'paragraph'
    elif annotation and text_gran == 'sentence':
        return 'sentence'
    elif annotation and text_gran == 'phrase':
        return 'phrase'
    elif annotation and text_gran == 'term':
        return 'term'
    else:
        return None


def unique_tuples(input_list, distinct_key_index):
    unique_dict = {}
    for t in input_list:
        key = t[distinct_key_index]
        if key not in unique_dict:
            unique_dict[key] = t
    output_list = list(unique_dict.values())
    return output_list


def generate_identifier(row):
    '''
    @param row: a row of pd.Series that contains column fields 'fa_section_id', 'fa_sub_section_id', 'fa_schedule_id', 'fa_part_id', 'fa_clause_list_id', 'fa_schedule_list_id'
    @type row: pd.Series
    @return : the identifier that concatenate by all IDs
    @rtype : str
    '''
    try:
        clause_id = row['clause_id']
    except:
        clause_id = row['fa_section']
    isParties = re.match('^Parties$|^Party$', str(clause_id), re.IGNORECASE)
    sec_id = row['fa_section_id']
    sub_sec_id = row['fa_sub_section_id']
    schedule_id = row['fa_schedule_id']
    part_id = row['fa_part_id']
    try:
        clause_list_id = row['fa_clause_list_id']
        schedule_list_id = row['fa_schedule_list_id']
        if clause_list_id and not schedule_list_id:
            list_id = clause_list_id
        elif not clause_list_id and schedule_list_id:
            list_id = schedule_list_id
        else:
            list_id = None
    except:
        list_id = row['fa_list_id']
    try:
        definition = re.sub(r'_.*', '', row['definition']) if isinstance(
            row['definition'], str) else row['definition']
    except:
        definition = row['fa_definition']
    if (sec_id or isParties) and schedule_id in [None, np.nan, 'None', 'nan']:
        if isParties and str(definition) not in ['nan', '', 'None']:
            definition = re.sub(ARTICLES, '', definition).strip()
            identifier = 'Parties-' + definition
        elif isParties and str(definition) in ['nan', '', 'None']:
            identifier = 'Parties'
        elif sec_id:
            target_ids = [sec_id, sub_sec_id] if sub_sec_id else [sec_id]
            identifier = 'Cl_' + '.'.join([re.sub(TRAILING_DOT_ZERO, '', str(i)) for i in target_ids if i])
            if str(definition) not in ['nan', '', 'None']:
                definition = re.sub(ARTICLES, '', definition).strip()
                identifier += '-' + definition
    elif schedule_id and schedule_id not in [np.nan, 'None', 'nan']:
        target_ids = [schedule_id, part_id] if str(
            part_id) not in ['nan', '', 'None'] else [schedule_id]
        id1 = '.'.join([re.sub(TRAILING_DOT_ZERO, '', str(i)) for i in target_ids if i])
        if sec_id and str(sec_id) not in ['nan', '', 'None']:
            id2 = re.sub(TRAILING_DOT_ZERO, '', str(sec_id))
            identifier = 'Sched_' + id1 + '_' + id2
        else:
            identifier = 'Sched_' + id1
    else:
        identifier = ''
    if list_id and str(list_id) !='None' and (schedule_id or (sec_id != '0' and str(definition) not in ['nan', '', 'None']) or  # if there exists definition
                    (identifier.strip() and identifier[-1].isdigit() and list_id[0].isdigit())):  # or identifier endwith digit and list id startwith digit
        identifier += '_' + str(list_id)
    elif list_id and sec_id != '0' and str(definition) in ['nan', '', 'None']:
        identifier += str(list_id)
    return identifier


def id2idtype(identifier):
    import re
    if identifier is None:
        return None
    elif re.match(CLAUSE_SECTION_PATTERN, identifier):
        return 'clause_section'
    elif re.match(DEF_CLAUSE_PATTERN, identifier):
        return 'definition_clause'
    elif re.match(PARTIES_CLAUSE_PATTERN, identifier):
        return 'parties_clause'
    elif re.match(CLAUSE_PATTERN, identifier):
        return 'clause'
    elif re.match(SCHED_PATTERN, identifier):
        return 'schedule'
    elif re.match(SCHED_PART_PATTERN, identifier):
        return 'schedule_part'
    elif re.match(PARAG_SCHED_PATTERN, identifier):
        return 'paragraph_of_schedule'


def id2stringid(x, typ):
    if isinstance(x, tuple) or isinstance(x, list):
        target_ids = list(x)
    else:
        target_ids = [x]
    if '0.0' in target_ids:
        identifier = 'Parties'
    else:
        identifier = typ + '.'.join([str(i).replace('.0', '') for i in target_ids if i])
    return identifier


def judge_match(row, match_col, match_key_type, from_sectype=None):
    '''
    Make judgement (TP, TN, FP, FN)
    on result by comparing result clause/schedule id with the ground truth clause/schedule id
    @param row:
    @type row: pd.Series
    @param match_col:
    @type match_col: str
    @param match_key_type: match key type in ['section','sub_section','definition','schedule','part','text','party']
    @type match_key_type: str
    @return :
    @rtype :
    '''
    clause_id = row['clause_id']
    clause_sec_id = row['fa_section_id']
    clause_sec_id = 'Cl_' + str(clause_sec_id).split('.0')[0] if clause_sec_id else 'None'
    clause_subsec_id = row['fa_sub_section_id']
    clause_subsec_id = str(clause_subsec_id).split('.0')[0] if clause_subsec_id else 'None'
    clause_list_id = row['fa_clause_list_id']
    sched_id = row['fa_schedule_id']
    sched_id = 'Sched_' + str(sched_id).split('.0')[0] if sched_id else 'None'
    sched_part_id = row['fa_part_id']
    sched_sec_id = row['fa_section_id']
    sched_sec_id = str(sched_sec_id).split('.0')[0] if sched_sec_id else 'None'
    sched_list_id = row['fa_schedule_list_id']
    definition = row['definition']
    text_element = row['text_element']
    matches = row[match_col]

    clauseisNone = clause_id in ['nan', 'None', '', None]
    scheduleisNone = sched_id in ['nan', 'None', '', None]
    clauseisNotParties = clause_id not in ['Parties', 'parties', 'party', 'Party']

    if from_sectype and 'section' not in text_element:
        return None
    elif not from_sectype and 'section' in text_element:
        return None

    if match_key_type in ['section', 'sub_section', 'definition']:
        if not clauseisNone and matches:
            if match_key_type == 'section':
                if any(clause_sec_id == str(i) for i in matches):
                    return 'TP'
            elif match_key_type == 'sub_section':
                if any(clause_sec_id+'.'+clause_subsec_id == str(i) for i in matches) or any(clause_sec_id+'.0'+clause_subsec_id == str(i) for i in matches):
                    return 'TP'
            elif match_key_type == 'definition':
                if any(definition.lower() in str(i).lower() for i in matches if definition):
                    return 'TP'
            return 'FP'
        elif clauseisNone and matches:
            return 'FP'
        elif not clauseisNone and clauseisNotParties and not matches:
            return 'FN'
        elif clauseisNone and not matches:
            return 'TN'

    elif match_key_type in ['schedule', 'part']:
        if not scheduleisNone and matches:
            if match_key_type == 'schedule':
                if any(sched_id == str(i) for i in matches if i):
                    return 'TP'
            elif match_key_type == 'part':
                if any(sched_id + '.' + sched_part_id == str(i) for i in matches if sched_part_id):
                    return 'TP'
            return 'FP'
        elif scheduleisNone and matches:
            return 'FP'
        elif not scheduleisNone and not matches:
            return 'FN'
        elif scheduleisNone and not matches:
            return 'TN'
    elif match_key_type == 'text':
        if not clauseisNone and matches:
            if clause_id in ['Parties', 'parties', 'party', 'Party']:
                identifier = 'Parties'
                if definition:
                    identifier += '-' + definition
            else:
                target_ids = [clause_sec_id, clause_subsec_id]
                identifier = '.'.join([i for i in target_ids if i != 'None'])
                if definition:
                    identifier += '-' + definition
                if clause_list_id:
                    identifier += str(clause_list_id)
            if any(identifier == i for i in matches):
                return 'TP'
            return 'FP'
        elif not scheduleisNone and matches:
            target_ids = [sched_id, sched_part_id] if sched_part_id else [sched_id]
            id1 = '.'.join([i for i in target_ids if i and i != 'None'])
            if sched_sec_id and sched_sec_id != 'None':
                identifier = id1 + '_' + sched_sec_id
            else:
                identifier = id1
            if sched_list_id and sched_list_id != 'None':
                identifier += '_' + str(sched_list_id)
            if any(identifier == i for i in matches):
                return 'TP'
        elif clauseisNone and scheduleisNone and matches:
            return 'FP'
        elif (not clauseisNone or not scheduleisNone) and not matches:
            return 'FN'
        elif clauseisNone and scheduleisNone and not matches:
            return 'TN'
    elif match_key_type == 'parties':
        if not clauseisNotParties and matches:
            identifier = 'Parties'
            if definition:
                identifier += '-' + definition
            if any(identifier == i for i in matches):
                return 'TP'
        elif clauseisNotParties and matches:
            return 'FP'
        elif not clauseisNotParties and not matches:
            return 'FN'
        elif clauseisNotParties and not matches:
            return 'TN'


def judge_match_list(row, def_clause_id):
    '''
    Check if the Ground-truth ID is equal or is a substring of matched result ID
    @param row: pd.Series that contains fields 'identifier_list' and 'fa_identifier'
    @param def_clause_id: list of definition clause id
    @return judges: list of judge
    @return flag: flag to illustrate if gt_id in lower-level but matched id in higher-level, or vice versa
    '''
    from nltk.stem import PorterStemmer
    ps = PorterStemmer()
    section = row['section']
    id_list = row['identifier_list']
    match_term_list = row['match_term_list']
    gt_id = row['fa_identifier']
    flag = None
    if not gt_id and (not id_list or id_list==(None,)):  # if gt_id is None and id_list is None
        return 'TN', flag
    elif not gt_id and id_list:  # if gt_id is None and id_list is not None -> False Alarm
        return ['FP']*len(id_list), flag  # [None]*len(id_list)
    elif gt_id and not id_list:  # if gt_id is not None and id_list is None -> Miss
        return 'FN', flag
    elif gt_id and id_list:  # if gt_id is not None and id_list is not None
        if '-' in gt_id:
            if len(gt_id.split('-')) == 2:
                gt_id, gt_definition = gt_id.split('-')
                gt_definition = stem_identifier(gt_definition, ps)
       # In case misintepretate ground truth, list all possible variants that ground truth may represent, e.g. Sched_2.2(a)
        if 'Sched' in gt_id and '_' in gt_id:
            if '.' in gt_id:
                splits = gt_id.replace('.', '_').split('_')
            else:
                splits = gt_id.split('_')
            if len(splits) == 5:
                new = ['Sched_' + splits[1] + '_' + splits[2] + '_' + splits[3] + '_' +  splits[4],
                    'Sched_' + splits[1] + '_' + splits[2] + '_' + splits[3] + '_' +  splits[4].replace('(', '.('),
                    'Sched_' + splits[1] + '_' + splits[2] + '_' + splits[3] + '_' +  splits[4].replace('(', '_('),
                    'Sched_' + splits[1] + '_' + splits[2] + '_' + splits[3] + '.' + splits[4],
                    'Sched_' + splits[1] + '_' + splits[2] + '_' + splits[3] + '.' + splits[4].replace('(', '.('),
                    'Sched_' + splits[1] + '_' + splits[2] + '_' + splits[3] + '.' + splits[4].replace('(', '_('),
                    'Sched_' + splits[1] + '_' + splits[2] + '_' + splits[3] + '_' + splits[4],
                    'Sched_' + splits[1] + '_' + splits[2] + '_' + splits[3] + '_' + splits[4].replace('(', '.('),
                    'Sched_' + splits[1] + '_' + splits[2] + '_' + splits[3] + '_' + splits[4].replace('(', '_('),
                    'Sched_' + splits[1] + '.' + splits[2] + '_' + splits[3] + '_' + splits[4],
                    'Sched_' + splits[1] + '.' + splits[2] + '_' + splits[3] + '_' + splits[4].replace('(', '.('),
                    'Sched_' + splits[1] + '.' + splits[2] + '_' + splits[3] + '_' + splits[4].replace('(', '_('),
                    'Sched_' + splits[1] + '.' + splits[2] + '_' + splits[3] + '.' + splits[4],
                    'Sched_' + splits[1] + '.' + splits[2] + '_' + splits[3] + '.' + splits[4].replace('(', '.('),
                    'Sched_' + splits[1] + '.' + splits[2] + '_' + splits[3] + '.' + splits[4].replace('(', '_('),
                    'Sched_' + splits[1] + '.' + splits[2] + '_' + splits[3] + '_' + splits[4],
                    'Sched_' + splits[1] + '.' + splits[2] + '_' + splits[3] + '_' + splits[4].replace('(', '.('),
                    'Sched_' + splits[1] + '.' + splits[2] + '_' + splits[3] + '_' + splits[4].replace('(', '_('),
                    ]
            elif len(splits) == 4:
                new = ['Sched_' + splits[1] + '_' + splits[2] + '_' + splits[3],
                    'Sched_' + splits[1] + '_' + splits[2] + '_' + splits[3].replace('(', '.('),
                    'Sched_' + splits[1] + '_' + splits[2] + '_' + splits[3].replace('(', '_('),
                    'Sched_' + splits[1] + '.' + splits[2] + '_' + splits[3],
                    'Sched_' + splits[1] + '.' + splits[2] + '_' + splits[3],
                    'Sched_' + splits[1] + '.' + splits[2] + '_' + splits[3].replace('(', '.('),
                    'Sched_' + splits[1] + '.' + splits[2] + '_' + splits[3].replace('(', '_(')
                    ]
            elif len(splits) == 3:

                new = ['Sched_' + splits[1] + '_' + splits[2],
                    'Sched_' + splits[1] + '_' + splits[2].replace('(', '.('),
                    'Sched_' + splits[1] + '_' + splits[2].replace('(', '_('),
                    'Sched_' + splits[1] + '.' + splits[2],
                    'Sched_' + splits[1] + '.' + splits[2].replace('(', '.('),
                    'Sched_' + splits[1] + '.' + splits[2].replace('(', '_(')
                    ]
            elif len(splits) == 2:
                new = ['Sched_' + splits[1],
                    'Sched_' + splits[1].replace('(', '_('),
                    'Sched_' + splits[1].replace('(', '.(')
                    ]
            else:
                new = []
            tmp = new + [gt_id]
            tmp = [i.replace('_(', '(') for i in tmp] + [i.replace('__', '_') for i in tmp]
            gt_sched_id_variants = list(set(tmp))
        else:
            gt_sched_id_variants = []
        
        judges = []
        flag = []
        for i, Id in enumerate(id_list):
            match_term = match_term_list[i]
            if Id:
                if '【' in Id and '】' in Id :
                    Id = Id.split('【')[0]
                    # Id = re.match(r'(.*)(【.*】)', Id).groups()[0]
            term = stem_string(str(match_term_list[i]), ps)
            section = stem_string(str(section), ps)
            if Id and '-' in Id:
                if len(Id.split('-')) == 2:
                    Id, Id_definition = Id.split('-')
                    Id_definition = stem_identifier(Id_definition, ps)
            def_suffix = ['Parties.*', re.escape(gt_id)] + [re.escape('Cl_' + i) for i in def_clause_id]
            if "gt_definition" in locals() and "Id_definition" in locals() and re.match('|'.join(def_suffix),Id) and \
            (re.search('means', match_term, re.IGNORECASE) and re.search(re.escape(gt_definition), match_term, re.IGNORECASE) or (re.search(rf'{re.escape(Id_definition)}', gt_definition, re.IGNORECASE) or re.search(rf'{re.escape(gt_definition)}', Id_definition, re.IGNORECASE))):
                judges.append('TP')
                flag.append(None)
            elif "gt_definition" not in locals() and "Id_definition" not in locals():
                if (Id and gt_id == Id) or (Id and 'Sched' in gt_id and gt_sched_id_variants and any(i in Id for i in gt_sched_id_variants)):
                    judges.append('TP')
                    flag.append(None)
                elif Id and (re.search(rf'^{re.escape(gt_id)}\D*$', Id, re.IGNORECASE) or re.search(re.escape(gt_id), Id, re.IGNORECASE)):
                    judges.append('TP')
                    flag.append('Annotation and Matched ID in same level or Matched ID in lower level than Annotation')
                elif Id and (re.search(rf'^{re.escape(Id)}\D*$', gt_id, re.IGNORECASE) or re.search(re.escape(Id), gt_id, re.IGNORECASE)):
                    judges.append('FN')
                    flag.append('Annotation in lower-level but Matched ID in higher-level')
                else:
                    judges.append('FN')
                    if section in term:
                        flag.append('Search result based on ts_section overwhelm the ts_content result')
                    elif Id and '(' in Id and ')' in Id and '(' in gt_id and ')' in gt_id:
                        Id_split = [i for i in re.split('(\(.*\))', Id) if i]
                        gt_id_split = [i for i in re.split('(\(.*\))', gt_id) if i]
                        if Id_split[0] == gt_id_split[0] and Id_split[1] != gt_id_split[1]:
                            flag.append('Annotation and Matched ID are sibling in same level')
                        else:
                            flag.append('Matched ID mismatch with Annotation')
                    else:
                        flag.append('Matched ID mismatch with Annotation')
            else:  # Not Match
                judges.append('FN')
                if section in term:
                    flag.append('Search result based on ts_section overwhelm the ts_content result')
                elif Id and '(' in Id and ')' in Id and '(' in gt_id and ')' in gt_id:
                    Id_split = [i for i in re.split('(\(.*\))', Id) if i]
                    gt_id_split = [i for i in re.split('(\(.*\))', gt_id) if i]
                    if Id_split[0] == gt_id_split[0] and Id_split[1] != gt_id_split[1]:
                        flag.append('Annotation and Matched ID are sibling in same level')
                    else:
                        flag.append('Matched ID mismatch with Annotation')
                else:
                    flag.append('Matched ID mismatch with Annotation')
            if "Id_definition" in locals():
                del Id_definition

        if any([i == 'Annotation and Matched ID in same level or Matched ID in lower level than Annotation' for i in flag]):
            flag = 'Annotation and Matched ID in same level or Matched ID in lower level than Annotation'
        elif any([i == 'TP' for i in judges]):
            flag = None
        elif any([i == 'Annotation in lower-level but Matched ID in higher-level' for i in flag]):
            flag = 'Annotation in lower-level but Matched ID in higher-level'
        elif any([i == 'Annotation and Matched ID are sibling in same level' for i in judges]):
            flag = 'Annotation and Matched ID are sibling in same level'
        elif any([i == 'Search result based on ts_section overwhelm the ts_content result' for i in judges]):
            flag = 'Search result based on ts_section overwhelm the ts_content result'
        else:
            flag = 'Matched ID mismatch with Annotation'

        return judges, flag


def overall_judge(x):
    '''
    Check if the judges is a list
    if it is a list, check if there is any 'TP' in the list of judges, return overall_judge as 'TP'
    else return 'FP'
    if it is a str or None, return itself
    e.g. x = ['FP','FP','TP','FP','FP'], return 'TP'
    '''
    if isinstance(x, list):
        if 'TP' in x:
            return 'TP'
        elif all(i is None for i in x):
            return None
        elif all(i == 'FN' for i in x):
            return 'FN'
        elif all(i == 'FP' for i in x):
            return 'FP'
    else:
        return x


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

def judge_grp(judge_all_grp):
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
        return 'TP', 1/judge_all_grp.size
    elif judge_all_grp.eq('TN').all():
        return 'TN', 1/judge_all_grp.size
    elif judge_all_grp.eq('FN').all():
        return 'FN', 1/judge_all_grp.size
    elif judge_all_grp.eq('FP').all():
        return 'FP', 1/judge_all_grp.size
    elif judge_all_grp.eq('FP').any() and judge_all_grp.eq('FN').any():
        return str(['FP', 'FN']), 1/judge_all_grp.size
    # elif judge_all_grp.eq(None).any() or judge_all_grp.eq('None').any() or judge_all_grp.eq(np.nan).any():
    #     return None, 1/judge_all_grp.size
    else:
        return None, 1/judge_all_grp.size  # str(['FP','FN'])


def judge_paragraph_lv(row):
    judge_all = row['judge_all']
    if any(i == 'TP' for i in judge_all):
        return 'TP'
    elif any(i == 'FP' for i in judge_all):
        return 'FP'
    elif all(i == 'TN' for i in judge_all):
        return 'TN'
    else:
        return 'FN'


def find(lst, a):
    return [i for i, x in enumerate(lst) if x == a]


def calculate_length(obj):
    if isinstance(obj, list) or isinstance(obj, tuple):
        return len(obj)
    elif isinstance(obj, str):
        return 1
    else:
        return 0


def map_tp_fp_sim(row, judgement):
    '''
    @ param judgement: 'TP' or 'FP'
    '''
    judges = row['judge']
    sim_scores = row['similarity_list']
    if isinstance(judges, list):
        match_judge_idx = find(judges, judgement)
        if match_judge_idx:
            sims = [sim_scores[idx] for idx in match_judge_idx]
            return sims[0] if len(sims) == 1 else sims
        else:
            return None
    elif isinstance(judges, str) and judges == judgement:
        return sim_scores
    else:
        return None


def map_tp_fp_rank(judges, judgement):
    if isinstance(judges, list):
        all_judges = find(judges, judgement)
        if all_judges:
            all_judges = [i+1 for i in all_judges]
            return all_judges[0] if len(all_judges) == 1 else all_judges
        else:
            return None
    elif isinstance(judges, str) and judges == judgement:
        return 1
    else:
        return None


def performance_metric(x):
    ''' Performance metric calculation with labels 'TP', 'FP', 'TN', 'FN' in a pd.Series
    @param x: the pd.Series that contains 'TP', 'FP', 'TN', 'FN'
    @type x: pd.Series
    @return : a tuple of (precision, recall, f1)
    @rtype : tuple
    '''
    tp = (x == 'TP').sum()
    fp = (x == 'FP').sum()
    fn = (x == 'FN').sum()
    if tp and fp and tp + fp != 0:
        precision = tp/(tp+fp)
    else:
        precision = None
    if tp and fn and tp + fn != 0:
        recall = tp/(tp+fn)
    else:
        recall = None
    if precision and recall and precision + recall != 0:
        f1 = (2 * precision * recall)/(precision + recall)
    else:
        f1 = None
    return precision, recall, f1


def write_append_excel(df, outpath, sheet_name, cols_show_as_percentage=None):
    import pandas as pd
    if os.path.exists(outpath):
        mode = 'a'
    else:
        mode = 'w'

    if cols_show_as_percentage:
        for column in cols_show_as_percentage:
            df.loc[:, column] = df[column].map(lambda x: '{:.2%}'.format(x) if x or x != np.nan else x)

    if mode == 'a':
        with pd.ExcelWriter(outpath, engine="openpyxl", mode=mode,if_sheet_exists='replace') as writer:
            df.to_excel(writer, sheet_name=sheet_name)
    else:
        with pd.ExcelWriter(outpath, mode=mode) as writer:
            df.to_excel(writer, sheet_name=sheet_name)


def export_ind_strategy_analysis(df_ts, outpath, listPairDict):
    df_ts_sec = df_ts[~df_ts.section.isna()].drop_duplicates(["section"])
    df_ts_content = df_ts[~df_ts.text_element.str.contains('section', na=False)]
    groups = [('text_granularity', 'individual_strategy_tokenlv_eval')]
    for lv_type, sheetname in groups:
        df_gp_sec = df_ts_sec.groupby(lv_type)
        df_gp_content = df_ts_content.groupby(lv_type)
        data = {}
        percent_cols = []
        for k, (testStringType, testList, targetStringType, targetList) in listPairDict.items():
            if testStringType == 'section':
                DF = df_gp_sec
            else:
                DF = df_gp_content
            if targetList:
                data[k + '_precision'] = DF[k + '_judge'].apply(lambda x: performance_metric(x)[0])
                data[k + '_recall'] = DF[k + '_judge'].apply(lambda x: performance_metric(x)[1])
                data[k + '_f1'] = DF[k + '_judge'].apply(lambda x: performance_metric(x)[2])
                percent_cols += [k + '_precision', k + '_recall', k + '_f1']
        df_gp = pd.DataFrame(data=data).reset_index()
        write_append_excel(df_gp, outpath, sheetname, cols_show_as_percentage=percent_cols)
    data2 = {}
    percent_cols = []
    for k, (testStringType, testList, targetStringType, targetList) in listPairDict.items():
        if testStringType == 'section':
            DF = df_gp_sec
        else:
            DF = df_gp_content
        if targetList:
            data2[k + '_precision'] = [performance_metric(DF[k + '_judge'])[0]]
            data2[k + '_recall'] = [performance_metric(DF[k + '_judge'])[1]]
            data2[k + '_f1'] = [performance_metric(DF[k + '_judge'])[2]]
            percent_cols += [k + '_precision', k + '_recall', k + '_f1']
    df_gp2 = pd.DataFrame(data=data2)
    sheetname = 'all_eval'
    write_append_excel(df_gp2, outpath, sheetname, cols_show_as_percentage=percent_cols)


def export_merge_analysis(df_ts, ts_fname, outpath):
    # map filename to list of fa identifier that annotated but not exist in FA
    df_indeterminate_id = pd.read_csv(args.indeterminate_doc)
    df_indeterminate_id = df_indeterminate_id[df_indeterminate_id['not_exist_in_FA'] == True]
    filename2fa_identifier_not_exist = df_indeterminate_id.groupby('filename')['fa_identifier'].apply(list).to_dict()
    if ts_fname in filename2fa_identifier_not_exist.keys():
        not_exist_id_list = filename2fa_identifier_not_exist[ts_fname]
        df_ts = df_ts[~df_ts['fa_identifier'].isin(not_exist_id_list)]

    df_gp = df_ts.groupby(['index', 'fa_identifier'], dropna=False)
    df_gp2 = df_ts.groupby(['index', 'fa_identifier'], dropna=True)
    df_stat = pd.DataFrame({
        "split_without_annotation_count": [df_gp['fa_identifier'].apply(lambda x: (x.isna())).sum()],
        "successful_query_text_count": [df_gp['fa_text'].apply(lambda x: ~(x.isna())).sum()],
        "indeterminate_count": [df_gp2['fa_text'].apply(lambda x: (x.isna())).sum()]
    })
    df_stat['successful_query_text_rate'] = df_stat['successful_query_text_count'] / \
        (df_stat['successful_query_text_count'] +
         df_stat['indeterminate_count'])
    df_stat['indeterminate_rate'] = df_stat['indeterminate_count'] / \
        (df_stat['successful_query_text_count'] +
         df_stat['indeterminate_count'])
    df_stat['successful_query_text_rate'] = df_stat['successful_query_text_rate'].map(lambda x: '{:.2%}'.format(x) if (x or x != np.nan) and not str(x).endswith('%') else x)
    df_stat['indeterminate_rate'] = df_stat['indeterminate_rate'].map(lambda x: '{:.2%}'.format(x) if (x or x != np.nan) and not str(x).endswith('%') else x)

    # df_stat.to_csv(outpath, index=False, encoding='utf-8-sig')

    split_without_annotation_count = df_stat['split_without_annotation_count'].values[0]
    successful_query_text_count = df_stat['successful_query_text_count'].values[0]
    indeterminate_count = df_stat['indeterminate_count'].values[0]
    successful_query_text_rate = df_stat['successful_query_text_rate'].values[0]
    indeterminate_rate = df_stat['indeterminate_rate'].values[0]

    return split_without_annotation_count, successful_query_text_count, indeterminate_count, successful_query_text_rate, indeterminate_rate


def export_analysis(df_ts, outpath, analysis_set):
    df_ts[['match_type_list', 'judge']] = df_ts[['match_type_list', 'judge']].applymap(lambda x: tuple(literal_eval(str(x))) if isinstance(x, str) and x.startswith('(') and x != 'None' else x)
    # df_ts_analysis_expand = df_ts[analysis_set].explode(
    #     ['match_type_list', 'judge'])
    df_ts_analysis = df_ts[analysis_set]
    txt_ele_id_keys = ['index', 'fa_identifier']
    if 'filename' in df_ts.columns:
        txt_ele_id_keys.append('filename')
    groups = [([], 'all_eval', df_ts_analysis),
              (txt_ele_id_keys, 'grp_eval', df_ts),
              (['text_granularity'], 'tokenlv_eval', df_ts_analysis),
              (['fa_identifier_type'], 'clausetype_eval', df_ts_analysis)]
    for groupby, sheetname, df_ts_analy in groups:
        if groupby:
            df_gp = df_ts_analy.groupby(groupby, dropna=False)
            # if groupby == txt_ele_id_keys:
            #     df_ts2 = deepcopy(df_ts)
            #     df_ts2['judge_grp'] = df_gp.judge_all.transform(lambda x: judge_grp(x)[0])
            #     df_ts2['judge_grp_weight'] = df_gp.judge_all.transform(lambda x: judge_grp(x)[1])
            #     df_ts2['judge_grp'] = df_ts2.judge_grp.map(lambda x: literal_eval(x) if isinstance(x, str) and x.startswith('[') else x)
                # col = df_ts2.pop('judge_grp')
                # df_ts2.insert(df_ts2.columns.get_loc("judge_all")+1, col.name, col)
                # df_ts2 = df_ts2.explode('judge_grp')
                # df_ts2 = df_ts2.drop_duplicates(subset=txt_ele_id_keys+['judge_grp'])
        else:
            df_gp = df_ts_analy
    
        if 'match_type_list' in groupby:
            df_gp = pd.DataFrame(data={
                'support': df_gp.judge.size(),
                'tp': df_gp.judge.apply(lambda x: (x == 'TP').sum()),
                'fp': df_gp.judge.apply(lambda x: (x == 'FP').sum()),
                'tn': df_gp.judge.apply(lambda x: (x == 'TN').sum()),
                'fn': df_gp.judge.apply(lambda x: (x == 'FN').sum()),
            }).reset_index()
        elif groupby == txt_ele_id_keys:
            # df_gp = df_ts2.groupby('judge_grp')['judge_grp_weight'].sum().transpose()
            # df_gp = df_gp.rename({'FN': 'fn', 'FP': 'fp', 'TN': 'tn', 'TP': 'tp'})
            # # df_gp = df_gp.round(0).astype(int)
            # df_gp['support'] = df_gp.sum()
            # df_gp = df_gp.to_frame().transpose().reset_index(drop=True)
            # l = ['support']
            # judges = ['tp', 'fp', 'tn', 'fn']
            # not_exists = []
            # for j in judges:
            #     if j in df_gp.columns:
            #         l.append(j)
            #     else:
            #         not_exists.append(j)
            # df_gp = df_gp[l]
            # if not_exists:
            #     for i in not_exists:
            #         df_gp[i] = 0
            # recall = df_gp.tp / (df_gp.tp + df_gp.fn)
            
            df_gp_detail = pd.DataFrame(data={
                'section': df_gp.section.first(),
                'text': df_gp.apply(lambda x: x.text.values.tolist()),
                'split_count': df_gp.apply(lambda x: len(x.text.values.tolist())),
                'match_term_list': df_gp.apply(lambda x: x.match_term_list.values.tolist()),
                'identifier_list': df_gp.apply(lambda x: x.identifier_list.values.tolist()),
                'judge': df_gp.apply(lambda x: x.judge.values.tolist()),
                'judge_all': df_gp.apply(lambda x: x.judge_all.tolist()),
                'judge_grp': df_gp.apply(lambda x: judge_txtgrp(x.judge_all))
            }).reset_index(drop=False)
            df_gp_detail.loc[df_gp_detail.split_count==1, ['text','match_term_list','identifier_list','judge','judge_all']] = df_gp_detail[df_gp_detail.split_count==1][['text','match_term_list','identifier_list','judge','judge_all']].map(lambda x:x[0])
            df_gp_detail = df_gp_detail.replace({np.nan: None, 'nan': None})
            df_gp_detail['fa_identifier_type'] = df_gp_detail.fa_identifier.map(id2idtype)
            df_gp_detail = df_gp_detail[['index', 'section', 'text', 'split_count', 'fa_identifier', 'fa_identifier_type','match_term_list', 'identifier_list', 'judge', 'judge_all', 'judge_grp']]
            
            df_gp = pd.DataFrame(data={
                'support': df_gp_detail.count().values[0],
                'tp': df_gp_detail[(df_gp_detail.judge_grp=='TP')].count().values[0],
                'fp': df_gp_detail[(df_gp_detail.judge_grp=='FP')].count().values[0],
                'tn': df_gp_detail[(df_gp_detail.judge_grp=='TN')].count().values[0],
                'fn': df_gp_detail[(df_gp_detail.judge_grp=='FN')].count().values[0],
                'definition_clause_support': df_gp_detail[(df_gp_detail.fa_identifier_type == 'definition_clause')].count().values[0],
                'parties_clause_support': df_gp_detail[(df_gp_detail.fa_identifier_type == 'parties_clause')].count().values[0],
                'clause_section_support': df_gp_detail[(df_gp_detail.fa_identifier_type == 'clause_section')].count().values[0],
                'clause_support': df_gp_detail[(df_gp_detail.fa_identifier_type == 'clause')].count().values[0],
                'schedule_support': df_gp_detail[(df_gp_detail.fa_identifier_type == 'schedule')].count().values[0],
                'schedule_part_support': df_gp_detail[(df_gp_detail.fa_identifier_type == 'schedule_part')].count().values[0],
                'paragraph_of_schedule_support': df_gp_detail[(df_gp_detail.fa_identifier_type == 'paragraph_of_schedule')].count().values[0],
                'definition_clause_fn': df_gp_detail[(df_gp_detail.judge_grp=='FN') & (df_gp_detail.fa_identifier_type == 'definition_clause')].count().values[0],
                'parties_clause_fn': df_gp_detail[(df_gp_detail.judge_grp=='FN') & (df_gp_detail.fa_identifier_type == 'parties_clause')].count().values[0],
                'clause_section_fn': df_gp_detail[(df_gp_detail.judge_grp=='FN') & (df_gp_detail.fa_identifier_type == 'clause_section')].count().values[0],
                'clause_fn': df_gp_detail[(df_gp_detail.judge_grp=='FN') & (df_gp_detail.fa_identifier_type == 'clause')].count().values[0],
                'schedule_fn': df_gp_detail[(df_gp_detail.judge_grp=='FN') & (df_gp_detail.fa_identifier_type == 'schedule')].count().values[0],
                'schedule_part_fn': df_gp_detail[(df_gp_detail.judge_grp=='FN') & (df_gp_detail.fa_identifier_type == 'schedule_part')].count().values[0],
                'paragraph_of_schedule_fn': df_gp_detail[(df_gp_detail.judge_grp=='FN') & (df_gp_detail.fa_identifier_type == 'paragraph_of_schedule')].count().values[0],
            }, index=[0]).reset_index(drop=True)
        elif not groupby:
            df_gp = pd.DataFrame(data={
                'support': df_gp.loc[~df_gp['judge_all'].isna(), 'judge_all'].size,
                'tp': df_gp.judge_all.apply(lambda x: (x == 'TP')).sum(),
                'fp': df_gp.judge_all.apply(lambda x: (x == 'FP')).sum(),
                'tn': df_gp.judge_all.apply(lambda x: (x == 'TN')).sum(),
                'fn': df_gp.judge_all.apply(lambda x: (x == 'FN')).sum(),
            }, index=[0]).reset_index(drop=True)
        else:
            df_gp = pd.DataFrame(data={
                'support': df_gp.judge_all.size(),
                'tp': df_gp.judge_all.apply(lambda x: (x == 'TP').sum()),
                'fp': df_gp.judge_all.apply(lambda x: (x == 'FP').sum()),
                'tn': df_gp.judge_all.apply(lambda x: (x == 'TN').sum()),
                'fn': df_gp.judge_all.apply(lambda x: (x == 'FN').sum())
            }).reset_index()
        df_gp['precision'] = df_gp.tp / (df_gp.tp + df_gp.fp)
        df_gp['recall'] = df_gp.tp / (df_gp.tp + df_gp.fn)
        df_gp['f1_score'] = (2 * df_gp.precision * df_gp.recall) / (df_gp.precision + df_gp.recall)
        df_gp['accuracy'] = (df_gp.tp + df_gp.tn) / (df_gp.tp + df_gp.tn + df_gp.fn + df_gp.fp)
        write_append_excel(df_gp, outpath, sheetname, cols_show_as_percentage=['precision', 'recall', 'f1_score', 'accuracy'])
        if groupby == txt_ele_id_keys:
            recall = df_gp['recall']
    return recall.values[0]

# def export_text_segment_stat(df_ts, outpath, suppl_subset=None):
#     df_ts2 = df_ts.drop_duplicates(
#         subset=['index'] + [suppl_subset] if suppl_subset else ['index'])
#     df_ts3 = df_ts.drop_duplicates(
#         subset=['index', 'text'] + [suppl_subset] if suppl_subset else ['index', 'text'])
#     term_count = len(df_ts[df_ts['text_granularity'] == 'term'])
#     phrase_count = len(df_ts[df_ts['text_granularity'] == 'phrase'])
#     sentence_count = len(df_ts[~df_ts.text_element.str.contains('section', na=False, case=False) & df_ts[
#         'text_granularity'] == 'sentence'])
#     paragraph_count = len(
#         df_ts2[df_ts2.text_element.str.contains('paragraph', na=False, case=False)])
#     section_count = len(
#         df_ts3[df_ts3.text_element.str.contains('section', na=False, case=False)])
#     antd_on_para_count = len(df_ts3[df_ts3['annotated_on'] == 'paragraph'])
#     antd_on_phrase_count = len(df_ts3[df_ts3['annotated_on'] == 'phrase'])
#     antd_on_sent_count = len(df_ts3[df_ts3['annotated_on'] == 'sentence'])
#     antd_on_term_count = len(df_ts3[df_ts3['annotated_on'] == 'term'])
#     df_txt_gran = pd.DataFrame(data={
#         'count_on': [term_count, phrase_count, paragraph_count],
#         'antd_count_on': [antd_on_term_count, antd_on_phrase_count, antd_on_para_count],
#         '%_of_antd': [antd_on_term_count / term_count if term_count else None,
#                       antd_on_phrase_count / phrase_count if phrase_count else None,
#                       antd_on_para_count / paragraph_count if paragraph_count else None]
#     }, index=['term', 'phrase', 'paragraph'])
#     write_append_excel(df_txt_gran, outpath, 'phraseSplit_antd_stat',
#                        cols_show_as_percentage=['%_of_antd'])


def export_tp_fp_sim_stat(df_ts, outpath):
    df_tp_stats = []
    df_ts[['TP_sim', 'TP_rank', 'FP_sim', 'FP_rank']] = df_ts[['TP_sim', 'TP_rank', 'FP_sim', 'FP_rank']].applymap(lambda x: tuple(literal_eval(str(x))) if isinstance(x, str) and x.startswith('[') and x != 'None' else x)
    for col in ['TP_sim', 'TP_rank', 'FP_sim', 'FP_rank']:
        mean = df_ts[col].explode().astype(float).mean()
        min_ = df_ts[col].explode().astype(float).min()
        max_ = df_ts[col].explode().astype(float).max()
        sd = df_ts[col].explode().astype(float).std()
        median = df_ts[col].explode().astype(float).median()
        df_tp_stat = pd.DataFrame(data={
            f'mean_{col}': [mean],
            f'min_{col}': [min_],
            f'max_{col}': [max_],
            f'sd_{col}': [sd],
            f'median_{col}': [median]
        })
        df_tp_stats.append(df_tp_stat)
    df_tp_stats = pd.concat(df_tp_stats, axis=1)
    write_append_excel(df_tp_stats, outpath, 'tp_sim_stat', cols_show_as_percentage=['mean_TP_sim',
                                                                                     'min_TP_sim',
                                                                                     'max_TP_sim',
                                                                                     'sd_TP_sim',
                                                                                     'median_TP_sim',
                                                                                     'mean_FP_sim',
                                                                                     'min_FP_sim',
                                                                                     'max_FP_sim',
                                                                                     'sd_FP_sim',
                                                                                     'median_FP_sim'])


def length_tokenize_sent(sequence):
    global tokenizer
    tokens = tokenizer.tokenize(sequence)
    return len(tokens)

def multiprocess(function, input_list, args):
    '''
    multiprocessing the function at a time

    @param function: a function
    @type function: def
    @param input_list: a list of input that accept by the function
    @type input_list: list
    @param args: arguments variables
    @type args: any argument type as required by function
    '''
    import multiprocessing

    input_length = len(input_list)
    num_processor = multiprocessing.cpu_count()
    print(f'there are {num_processor} CPU cores')
    batch_size = max(input_length // num_processor, 1)
    num_batch = int(input_length / batch_size) + (input_length % batch_size > 0)
    pool = multiprocessing.Pool(num_processor)
    processes = [pool.apply_async(function, args=(input_list[idx * batch_size:(idx + 1) * batch_size], args)) for idx in range(num_batch)]
    results = [p.get() for p in processes]
    # results = [i for r in results if r for i in r]

    return results

def get_parent_list_item(df_grp):
    '''
    input a DataFrame with "list_id" and "text" to query the nearest parent item among the group and 
    form a new DataFrame with columns "parent_list_ids", "parent_list" to indicate the matched parent items and corresponding list of ids

    @param df_grp: DataFrame that must contain list_id, text
    @type df_grp: pandas.DataFrame
    @return: DataFrame with columns "parent_list_ids" (list of parent list_ids that the associate with child list_id), "parent_list" (concatenated text content of associating parent list_ids)
    @rtype: pandas.DataFrame
    '''
    all_text = []
    all_ref_ids = []
    list_ids = df_grp.list_id
    index_list_ids = list(zip(list_ids.index.tolist(), list_ids.values.tolist()))
    for index, list_id in index_list_ids:
        if list_id == np.nan or list_id is None:
            all_ref_ids.append(None)
            all_text.append(None)
        else:
            ids = re.findall(r'(\(*\w+\)*\.*)', str(list_id))
            texts = []
            ref_ids = []
            while ids:
                ids.pop(-1)
                if ids:
                    ref_id = ''.join(ids)
                    matches = df_grp[df_grp.list_id==ref_id]
                    if not matches.empty:
                        matches['index_diff'] = matches.index.map(lambda x: abs(index-int(x)))
                        ref_text = matches.loc[matches['index_diff'].idxmin(),'text']
                        ref_ids.append(ref_id)
                        texts.append(ref_text)
            if ref_ids:
                ref_ids.reverse()
                texts.reverse()
                texts = ': '.join(texts).replace('::',':')
                all_ref_ids.append(ref_ids)
                all_text.append(texts)
            else:
                all_ref_ids.append(None)
                all_text.append(None)
    return pd.DataFrame(data={"parent_list_ids": all_ref_ids, "parent_list": all_text}, index=(list_ids.index.tolist()))

def ts_merge_match_fa(files, args):
    from nltk.stem import PorterStemmer
    from nltk.tokenize import word_tokenize

    # map filename to facility type
    filename_facility_type_batch_df = pd.read_csv(args.filename_facility_type_batch_doc)
    file2facility = dict(zip(filename_facility_type_batch_df['filename'].values.tolist(), filename_facility_type_batch_df['facility_type'].values.tolist()))

    # map filename to dataset split
    file_dataset_df = pd.read_csv(args.file_dataset_doc)
    file2dataset = dict(zip(file_dataset_df['filename'].values.tolist(), file_dataset_df['split'].values.tolist()))

    ps = PorterStemmer()

    results = []
    all_stat = []

    for file in files:
        try:
            print('filename: ', file)
            ts_fname = file.split('.csv')[0]
            if args.do_merge and args.do_match:
                task_type = 'merge&match'
            elif args.do_match:
                task_type = 'match'
            elif args.do_merge:
                task_type = 'merge'
            new_fname = ts_fname + f'_{task_type}_FA.csv'
            if os.path.exists(os.path.join(args.output_csv_dir, new_fname)):
                df_ts = pd.read_csv(os.path.join(args.output_csv_dir, new_fname))
                df_ts['filename'] = ts_fname
                results.append(df_ts)

                if args.do_merge and args.do_match:
                    df_analy = pd.read_excel(
                        args.output_analy_dir + ts_fname+'_analysis.xlsx', sheet_name='grp_eval')
                    recall = df_analy.recall.values[0]
                    all_stat.append(recall)
                continue
            # import ts from csv file
            df_ts = pd.read_csv(os.path.join(args.src_ts_csv_dir, file)).astype(str)
            ts_columns = df_ts.columns
            if not 'text_granularity' in ts_columns:
                df_ts['previous_text'] = df_ts.text.shift(1)
                df_ts['text_granularity'] = df_ts[['index', 'previous_text', 'text', 'text_element']].apply(lambda x: map_text_granularity(x, df_ts), axis=1)

            if not 'parent_list' in ts_columns:
                # trace back with child list id and provide parent list id and content
                df_ts_parent_list = df_ts.groupby(['text_block_id'], dropna=False).apply(get_parent_list_item).reset_index(drop=True)
                df_ts = pd.concat([df_ts, df_ts_parent_list], axis=1)
                df_ts['parent_list_ids'] = df_ts['parent_list_ids'].astype(str)

            if not 'parent_caption' in ts_columns:
                # trace back to get parent caption
                df_ts['prev_text_element'] = df_ts['text_element'].shift()
                df_ts.loc[((df_ts['prev_text_element'].str.contains('caption'))==True) & (df_ts.section==df_ts.section.shift()), 'parent_caption'] = df_ts.text.shift()
                df_ts['parent_caption'] = df_ts.groupby(['index','text_block_id'])['parent_caption'].transform('first')
            
            # import fa
            fa_fname = re.sub(r"(.*)TS_(.*)", r"\1FA_\2", file)
            df_fa = pd.read_csv(os.path.join(args.src_fa_csv_dir, fa_fname)).astype(str)
            df_fa['index'] = df_fa['index'].astype(int)
            df_fa = df_fa.sort_values(by=['index'])
            df_fa = df_fa.add_prefix('fa_')
            fa_columns = df_fa.columns
            df_fa.loc[df_fa.fa_text_element.str.contains('table'), 'fa_text'] = df_fa[df_fa.fa_text_element.str.contains('table')]['fa_text'].astype(str)
            df_fa['fa_text'] = df_fa['fa_text'].map(lambda x: re.sub(r'\n', ' ', str(x)).strip())
            # find_def_cond = df_fa['fa_section'].str.contains('Definition|Interpretation', na=False, case=False)
            # find_def_cond2 = df_fa['fa_sub_section'].str.contains('Definition|Interpretation', na=False, case=False)
            df_fa = df_fa.replace({'nan': None, np.nan: None, 'None': None})
            # def_sec_id = df_fa[(find_def_cond) & df_fa.fa_schedule_id.isna()]['fa_section_id']
            # def_sub_sec_id = df_fa[(find_def_cond2) & df_fa.fa_schedule_id.isna()]['fa_sub_section_id']
            def_sec_id = df_fa[~df_fa.fa_definition.isna() & ~df_fa.fa_section.eq('PARTIES') & df_fa.fa_schedule_id.isna()]['fa_section_id']
            def_sub_sec_id = df_fa[~df_fa.fa_definition.isna() & ~df_fa.fa_section.eq('PARTIES') & df_fa.fa_schedule_id.isna()]['fa_sub_section_id']
            defs = df_fa[~df_fa.fa_definition.isna() & ~df_fa.fa_section.eq('PARTIES') & df_fa.fa_schedule_id.isna()]['fa_definition']
            clause_id_def = list(set([(re.sub(TRAILING_DOT_ZERO, '', str(a)) + '.' + re.sub(TRAILING_DOT_ZERO, '', str(b)), c) for a, b, c in list(zip(def_sec_id, def_sub_sec_id, defs)) if str(a) != 'None' and str(b) != 'None']))
            if clause_id_def:
                def_clause_id = list(set(list(zip(*clause_id_def))[0]))
            else:
                def_clause_id = []
            df_fa[['fa_section_id', 'fa_sub_section_id', 'fa_schedule_id', 'fa_part_id', 'fa_list_id']] = df_fa[[ 'fa_section_id', 'fa_sub_section_id', 'fa_schedule_id', 'fa_part_id', 'fa_list_id']].astype(str)
            df_fa[['fa_section_id', 'fa_sub_section_id', 'fa_schedule_id', 'fa_part_id', 'fa_list_id']] = df_fa[['fa_section_id', 'fa_sub_section_id', 'fa_schedule_id', 'fa_part_id', 'fa_list_id']].applymap(lambda x: re.sub(TRAILING_DOT_ZERO, '', str(x)))
            df_fa['fa_part_id'] = df_fa['fa_part_id'].map(lambda x: x if x != '0' else None)
            df_fa = df_fa.replace({'nan': None, np.nan: None, 'None': None})

            # ========================== Create FA key2id and id2key dictionary ==========================
            # extract schedule df
            try:
                schedBeginID = df_fa[df_fa.fa_text.str.contains(SCHED_TITLE, na=False, case=False) & (~df_fa.fa_text.str.contains(EXCLUD_STR_IN_SCHED_TITLE, na=False, case=False))]['fa_index'].values[0]
            except IndexError as e:
                print(f"【Error Message】filename: {fa_fname}, Schedule clause could not be recognized. Please review the if schedule opening sentence or heading exists or if the regular expression for schedule clause extraction is correct.")
            schedEndID = df_fa["fa_index"].iloc[-1]
            sched_index = df_fa['fa_index'].between(schedBeginID, schedEndID)
            df_sched = df_fa[sched_index]
            df_fa_sched = df_sched[~df_sched.fa_schedule.isna()].drop_duplicates(["fa_schedule"])
            df_fa_part = df_sched[~df_sched.fa_part.isna()].drop_duplicates(["fa_part"])
            part_list = [i for i in list(df_fa_part.fa_part) if i]

            # extract clauses df
            clause_index = df_fa['fa_index'].between(0, schedBeginID - 1)
            df_clauses = df_fa[clause_index]
            df_fa_sec = df_clauses[~df_clauses.fa_section.isna()].drop_duplicates(["fa_section"])
            df_fa_sub_sec = df_clauses[~df_clauses.fa_sub_section.isna()].drop_duplicates(["fa_sub_section"])
            df_fa_def = df_clauses[~df_clauses.fa_definition.isna()].drop_duplicates(["fa_definition"])

            # extract parties df
            df_parties = df_fa[df_fa.fa_section.eq('PARTIES')]
            parties_clause_list = [i for i in df_parties.fa_text.values.tolist() if i]

            # create FA clause/schedule key ids to keys dictionaries
            # e.g. {1.0 :'definition and interpretation',...}
            id2sec = dict(zip(df_fa_sec.fa_section_id, df_fa_sec.fa_section))
            id2subsec = dict(zip(zip(df_fa_sub_sec.fa_section_id, df_fa_sub_sec.fa_sub_section_id),df_fa_sub_sec.fa_sub_section))  # e.g. {(1.0, 1.0) :'definition',...}
            id2sched = dict(zip(df_fa_sched.fa_schedule_id, df_fa_sched.fa_schedule))  # e.g. {1.0: 'THE ORIGINAL LENDERS',...}
            id2part = dict(zip(zip(df_fa_sched.fa_schedule_id, df_fa_sched.fa_part_id), df_fa_sched.fa_part))  # e.g. {(3.0,'1.0'): 'FORM OF UTILISATION REQUEST',...}
            if 'fa_parent_caption' in fa_columns:
                identifier2parent_caption = dict(zip(df_fa.fa_identifier, df_fa.fa_parent_caption))
            if 'fa_parent_list' in fa_columns:
                identifier2parent_list = dict(zip(df_fa.fa_identifier, df_fa.fa_parent_list))
            if 'fa_parent_list_ids' in fa_columns:
                identifier2parent_list_ids = dict(zip(df_fa.fa_identifier, df_fa.fa_parent_list_ids))
            # id2content = dict(zip(df_fa.fa_identifier,df_fa.fa_text))
            # id2content = dict(zip(zip(df_fa.fa_section_id,
            #                           df_fa.fa_sub_section_id,
            #                           df_fa.fa_schedule_id,
            #                           df_fa.fa_part_id,
            #                           df_fa.fa_list_id), df_fa.fa_text))

            id2sec[None] = None

            # create FA clause/schedule keys to key ids dictionaries
            sec2id = {v: id2stringid(k, 'Cl_') for k, v in id2sec.items()}
            subsec2id = {v: id2stringid(k, 'Cl_') for k, v in id2subsec.items()}
            sched2id = {v: id2stringid(k, 'Sched_') for k, v in id2sched.items()}
            part2id = {v: id2stringid(k, 'Sched_') for k, v in id2part.items()}
            # part2id = {v: '.'.join([re.sub(TRAILING_DOT_ZERO,'',str(i)) for i in k if i]) for k, v in id2part.items()}
            df_fa_content = df_fa[~df_fa.fa_text_element.str.contains('section', na=False)]
            def2id = dict(zip(df_fa_def.fa_definition, df_fa_def.fa_identifier))
            content2id = dict(zip(df_fa_content.fa_text, df_fa_content.fa_identifier))
            df_fa_parties = df_fa_content[df_fa_content['fa_section_id'] == 0]
            parties2id = dict(zip(df_fa_parties.fa_text, df_fa_parties.fa_identifier))

            # key to id mapper dictionary
            id_mapper = {
                'sec': sec2id,
                'sub_sec': subsec2id,
                'sched': sched2id,
                'part': part2id,
                'clause_text': content2id,
                'def': def2id,
                'parties': parties2id
            }

            # ========================== TS Match FA ==========================
            if args.do_match:

                if not args.apply_ext_sim_match:
                    df_ts = term_matching(df_ts, df_fa, id_mapper, sim_threshold=args.sim_threshold, topN=args.topN)
                    match_type_list = list(df_ts.match_type_list.unique())
                else:
                    # extract ts semantic match fa from a folder
                    ts_match_fname = file.split('.csv')[0] + '_results.csv'
                    try:
                        df_ts_match = pd.read_csv(os.path.join(args.matched_result_csv_dir, ts_match_fname)).astype(str)
                        df_ts_match = df_ts_match.replace({'nan': None, np.nan: None, 'None': None, '': None})
                        print(f'source matched filename: {os.path.join(args.matched_result_csv_dir, ts_match_fname)}')
                    except FileNotFoundError:
                        print(f'No such file or directory: \'{os.path.join(args.matched_result_csv_dir, ts_match_fname)}\'. Skip to next doc')
                        continue

                    
                    if 'text_element' in df_ts_match.columns:
                        df_ts_match.drop(columns=['text_element'], inplace=True)
                    if 'list_id' in df_ts_match.columns:
                        df_ts_match.drop(columns=['list_id'], inplace=True)
                    # df_ts_match["TS_term"] = df_ts_match["TS_term"].map(lambda x: x.strip() if x else x)
                    # df_ts_match["TS_text"] = df_ts_match["TS_text"].map(lambda x: x.strip() if x else x)
                    df_ts_match = df_ts_match.rename(columns={"TS_term": "section", "TS_text": "text"})
                    # df_ts_match[['index','text_block_id','page_id','phrase_id']] = df_ts_match[['index','text_block_id','page_id','phrase_id']].astype(int)
                    # df_ts[['index','text_block_id','page_id','phrase_id']] = df_ts[['index','text_block_id','page_id','phrase_id']].astype(int)
                    df_ts = df_ts.merge(df_ts_match, how='left', on=['index', 'text_block_id', 'page_id', 'phrase_id','text']) # , 'section'
                    # df_ts.drop(columns=["TS_term", "TS_text"], inplace=True)
                    df_ts = df_ts.rename(columns={"section_x":"section"}) # "text_x": "text", 
                    df_ts = df_ts.replace({'nan': None, np.nan: None, 'None': None, '': None})
                    df_ts[['match_term_list', 'identifier_list', 'similarity_list', 'match_type_list']] = df_ts[['match_term_list', 'identifier_list',
                                                                                                                'similarity_list', 'match_type_list']].applymap(lambda x: x.replace('nan, ', "None, ").replace('nan]', "None]") if x else x)
                    df_ts[['match_term_list', 'identifier_list', 'similarity_list', 'match_type_list']] = df_ts[['match_term_list', 'identifier_list',
                                                                                                                'similarity_list', 'match_type_list']].applymap(lambda x: tuple(literal_eval(str(x))) if x and x != 'None' else None) # [:args.topN]
                    if args.sim_threshold > 0:
                        df_ts['similarity_list'] = df_ts['similarity_list'].map(lambda x: tuple([i for i in x if i >= args.sim_threshold]) if x else x)
                        df_ts['similarity_list_length'] = df_ts['similarity_list'].map(lambda x: len(x) if x else 0)
                        for col in ['match_term_list', 'identifier_list', 'match_type_list']:
                            df_ts[col] = df_ts[[col, 'similarity_list_length']].apply(lambda x: x[col][:x['similarity_list_length']] if x['similarity_list_length'] else None, axis=1)
                    match_type_list = list(df_ts['match_type_list'].unique())
                    match_type_list = tuple(set([i for l in match_type_list if l for i in l]))

            # ========================== TS Merge FA ==========================
            antd_ts_cols = ['fa_section_id', 'fa_sub_section_id', 'fa_clause_list_id',
                            'fa_schedule_id', 'fa_part_id', 'fa_schedule_list_id', 'fa_identifier']
            if args.do_merge:
                # split string of clause_id or schedule_id annotation at "," into list of ids
                for c in ['clause_id', 'definition', 'schedule_id']:
                    df_ts[c] = df_ts[c].map(lambda x: [i.strip() if i else i for i in re.split(r',(?!\d{3})', str(x))] if x else x)
                df_ts['schedule_id'] = df_ts['schedule_id'].map(split_newline)
                # matching clause id==1.1 with definition term
                df_ts['clause_id_def_pair'] = df_ts.apply(lambda x: pairing_clauses_def(x, clause_id_def, args.src_ts_csv_dir + file), axis=1)
                # expand the clause_id, schedule_id into records
                df_ts2 = pd.melt(df_ts, id_vars=['index', 'text'], value_vars=['clause_id_def_pair', 'schedule_id'], var_name='id_type', value_name='id')
                df_ts2 = df_ts2.explode(['id']).sort_values(by=['index'])
                df_ts = df_ts.merge(df_ts2, how='left', on=['index', 'text'])
                df_ts.drop(columns=['clause_id', 'schedule_id','clause_id_def_pair', 'definition'], inplace=True)
                df_ts = df_ts.replace({'nan': None, np.nan: None, 'None': None})
                df_ts = df_ts.drop_duplicates()
                id_type_is_sched = df_ts['id_type'] == 'schedule_id'
                df_ts = df_ts[~(id_type_is_sched & (df_ts['id'].isna()))]
                id_type_is_sched = df_ts['id_type'] == 'schedule_id'
                id_type_is_clause = df_ts['id_type'] == 'clause_id_def_pair'

                # splitting clause id, schedule id, definition from annotations
                df_ts.loc[id_type_is_clause, 'clause_id'] = df_ts[id_type_is_clause]['id'].map(lambda x: x[0] if isinstance(x, tuple) else None)
                df_ts.loc[id_type_is_clause, 'definition'] = df_ts[id_type_is_clause]['id'].map(lambda x: x[1] if isinstance(x, tuple) else None)
                df_ts.loc[id_type_is_sched, 'schedule_id'] = df_ts[id_type_is_sched]['id'].map(lambda x: x if isinstance(x, (float, int, str)) else None)
                isDef = ~df_ts.definition.isna()

                id_series_mapList = [
                    ('fa_section_id', 'clause_id', 'fa_section_id'),
                    ('fa_sub_section_id', 'clause_id', 'fa_sub_section_id'),
                    ('fa_clause_list_id', 'clause_id', 'fa_list_id'),
                    ('fa_clause_list_id', 'definition', 'fa_list_id'),
                    ('fa_schedule_id', 'schedule_id', 'fa_schedule_id'),
                    ('fa_part_id', 'schedule_id', 'fa_part_id'),
                    ('fa_schedule_list_id', 'schedule_id', 'fa_list_id'),
                    ('fa_section_id', 'schedule_id', 'fa_section_id')
                ]
                for id_col, ref_id_col, id_type in id_series_mapList:
                    if ref_id_col == 'schedule_id': # and id_type == 'fa_section_id':
                        df_ts.loc[id_type_is_sched, id_col] = df_ts[id_type_is_sched][ref_id_col].map(lambda x: extract_id(x, ref_id_col, id_type))
                    elif ref_id_col == 'clause_id' and id_type == 'fa_section_id':
                        df_ts.loc[id_type_is_clause, id_col] = df_ts[id_type_is_clause][ref_id_col].map(lambda x: extract_id(x, ref_id_col, id_type))
                    elif ref_id_col == 'definition' and id_type == 'fa_list_id':
                        df_ts.loc[isDef, id_col] = df_ts[isDef][ref_id_col].map(lambda x: extract_id(x, ref_id_col, id_type))
                    else:
                        df_ts[id_col] = df_ts[ref_id_col].map(lambda x: extract_id(x, ref_id_col, id_type))
                        
                # df_ts[['fa_section_id', 'fa_schedule_id']] = df_ts[
                #     ['fa_section_id', 'fa_schedule_id']].astype(float)
                # df_ts['fa_sub_section_id'] = df_ts['fa_sub_section_id'].astype(str)
                df_ts = df_ts.replace({'nan': None, np.nan: None, 'None': None})
                df_ts['fa_identifier'] = df_ts.apply(generate_identifier, axis=1)
                df_ts['fa_identifier_type'] = df_ts['fa_identifier'].map(id2idtype)

                df_ts = df_ts.replace({'': None, 'nan': None, np.nan: None, 'None': None})

                if args.do_match:
                    # if not args.apply_ext_sim_match and args.keep_individual_strategy_result:
                    #     sim_list_cols = []
                    #     for k, (testStringType, testList, targetStringType, targetList) in listPairDict.items():
                    #         if testStringType == 'section':
                    #             from_sectype = True
                    #         else:
                    #             from_sectype = False
                    #         if targetList:
                    #             df_ts[k+'_judge'] = df_ts.apply(lambda x: judge_match(x, k+'_id', targetStringType,from_sectype=from_sectype),axis=1)
                    #             sim_list_cols += [k+'_judge']
                    df_ts = df_ts.replace({np.nan: None})
                    df_ts_gp_sect = pd.DataFrame(data={
                        'page_id': df_ts.groupby('text_block_id')['page_id'].first(),
                        'section': df_ts.groupby('text_block_id')['section'].first(),
                        'text': df_ts.groupby('text_block_id')['text'].apply(lambda x: ' '.join([i for i in x.tolist() if i])),
                        'fa_identifier': df_ts.groupby('text_block_id')['fa_identifier'].apply(lambda x: list(set([i for i in x.tolist() if i]))),
                        'match_term_list': df_ts.groupby('text_block_id')['match_term_list'].apply(lambda x: [i for i in list(flatten([literal_eval(i) if isinstance(i, str) else i for i in x.tolist()])) if i]),
                        'identifier_list': df_ts.groupby('text_block_id')['identifier_list'].apply(lambda x: [i for i in list(flatten([literal_eval(i) if isinstance(i, str) else i for i in x.tolist()])) if i]),
                        'similarity_list': df_ts.groupby('text_block_id')['similarity_list'].apply(lambda x: [i for i in list(flatten([literal_eval(i) if isinstance(i, str) else i for i in x.tolist()])) if i]),
                    }
                    )
                    df_ts_gp_sect = df_ts_gp_sect.explode('fa_identifier')
                    df_ts_gp_sect = df_ts_gp_sect.replace({np.nan: None})
                    df_ts_gp_sect['judge'] = df_ts_gp_sect.apply(lambda x: judge_match_list(x, def_clause_id)[0], axis=1)
                    df_ts_gp_sect['judge_all'] = df_ts_gp_sect['judge'].map(overall_judge)
                    
                    df_ts_gp_sect['tp'] = df_ts_gp_sect.judge.apply(lambda x: len([i for i in x if i == 'TP']))
                    df_ts_gp_sect['fp'] = df_ts_gp_sect.judge.apply(lambda x: len([i for i in x if i == 'FP']))
                    df_ts_gp_sect['tn'] = df_ts_gp_sect.judge.apply(lambda x: len([i for i in x if i == 'TN']))
                    df_ts_gp_sect['fn'] = df_ts_gp_sect.judge.apply(lambda x: len([i for i in x if i == 'FN']))
                    df_ts_gp_sect['flag'] = df_ts_gp_sect.apply(lambda x: judge_match_list(x, def_clause_id)[1], axis=1)
                    df_ts_gp_sect[['fa_section_id', 'fa_sub_section_id', 'fa_schedule_id', 'fa_part_id', 'definition', 'fa_clause_list_id', 'fa_schedule_list_id' ]] = pd.DataFrame(df_ts_gp_sect.fa_identifier.apply(decompose_fa_identifier).tolist(),index= df_ts_gp_sect.index)
                    df_ts_gp_sect = df_ts_gp_sect.rename(columns={'fa_identifier':'clause_id'})
                    df_ts_gp_sect[['fa_text', 'fa_text_sources']] = df_ts_gp_sect.apply(lambda x: extract_fa_text(x, df_fa, ps), axis=1, result_type='expand')
                    df_ts_gp_sect.loc[df_ts_gp_sect.fa_text.isna() & ~df_ts_gp_sect.clause_id.isna(), 'judge_all'] = None
                    df_ts_gp_sect = df_ts_gp_sect.drop(columns=['fa_section_id', 'fa_sub_section_id', 'fa_schedule_id', 'fa_part_id', 'definition', 'fa_clause_list_id', 'fa_schedule_list_id' ])
                    df_ts_gp_sect = df_ts_gp_sect.rename(columns={'clause_id':'fa_identifier'})
                    if not os.path.exists(os.path.join(args.output_analy_dir, 'grpby_sect/')):
                        os.makedirs(os.path.join(args.output_analy_dir, 'grpby_sect/'))

                    outpath_grpby_sect = args.output_analy_dir + 'grpby_sect/' + ts_fname + '_grpby_sect.csv'
                    df_ts_gp_sect.to_csv(outpath_grpby_sect, index=False, encoding='utf-8-sig')

                    tp = df_ts_gp_sect.judge_all.apply(lambda x: (x == 'TP')).sum()
                    fp = df_ts_gp_sect.judge_all.apply(lambda x: (x == 'FP')).sum()
                    tn = df_ts_gp_sect.judge_all.apply(lambda x: (x == 'TN')).sum()
                    fn = df_ts_gp_sect.judge_all.apply(lambda x: (x == 'FN')).sum()
                    df_ts_gp_sect_analy = pd.DataFrame(data={
                        'precision': [tp / (tp + fp)],
                        'recall': [tp / (tp + fn)],
                        'accuracy': [(tp + tn) / (tp + tn + fn + fp)]
                    })
                    df_ts_gp_sect_analy['f1_score'] = (2 * df_ts_gp_sect_analy.precision * df_ts_gp_sect_analy.recall) / (df_ts_gp_sect_analy.precision + df_ts_gp_sect_analy.recall)

                    outpath_grpby_sect2 = args.output_analy_dir + 'grpby_sect/' + ts_fname + '_grpby_sect_analysis.csv'
                    df_ts_gp_sect_analy.to_csv(outpath_grpby_sect2, index=False, encoding='utf-8-sig')

                    df_ts['judge'] = df_ts.apply(lambda x: judge_match_list(x, def_clause_id)[0], axis=1)
                    df_ts['flag'] = df_ts.apply(lambda x: judge_match_list(x, def_clause_id)[1], axis=1)
                    df_ts['judge_all'] = df_ts['judge'].map(overall_judge)
                    df_ts['TP_sim'] = df_ts.apply(lambda x: map_tp_fp_sim(x, 'TP'), axis=1)
                    df_ts['TP_rank'] = df_ts['judge'].map(lambda x: map_tp_fp_rank(x, 'TP'))
                    df_ts['FP_sim'] = df_ts.apply(lambda x: map_tp_fp_sim(x, 'FP'), axis=1)
                    df_ts['FP_rank'] = df_ts['judge'].map(lambda x: map_tp_fp_rank(x, 'FP'))
                    df_ts['judge'] = df_ts['judge'].map(lambda x: tuple(x) if isinstance(x, list) else x)
                    df_ts.loc[df_ts['judge'].isin(['FN', 'TN']), 'match_type_list'] = df_ts[df_ts['judge'].isin(['FN', 'TN'])]['judge'].map(lambda x: match_type_list)
                    df_ts.loc[df_ts['judge'].isin(['FN', 'TN']), ['match_term_list', 'similarity_list', 'identifier_list']] = df_ts[df_ts['judge'].isin(['FN', 'TN'])][['match_term_list', 'identifier_list']].applymap(lambda x: tuple([None]*len(match_type_list)))
                    df_ts.loc[df_ts['judge'].isin(['FN', 'TN']), 'judge'] = df_ts[df_ts['judge'].isin(['FN', 'TN'])]['judge'].map(lambda x: tuple([x]*len(match_type_list)))

                # get keys (section, sub-section, schedule, part) by ids from df_fa
                df_ts['fa_section'] = df_ts['fa_section_id'].map(lambda x: id2sec.get(x, None))
                df_ts['fa_sub_section'] = df_ts[['fa_section_id', 'fa_sub_section_id']].apply(lambda x: id2subsec.get((x['fa_section_id'], x['fa_sub_section_id']), None), axis=1)
                df_ts['fa_schedule'] = df_ts['fa_schedule_id'].map(lambda x: id2sched.get(x, None))
                df_ts['fa_part'] = df_ts[['fa_schedule_id', 'fa_part_id']].apply(lambda x: id2part.get((x['fa_schedule_id'], x['fa_part_id']), None), axis=1)
                if 'fa_parent_caption' in fa_columns:
                    df_ts['fa_parent_caption'] = df_ts['fa_identifier'].map(lambda x: identifier2parent_caption.get(x, None))
                if 'fa_parent_list_ids' in fa_columns:
                    df_ts['fa_parent_list_ids'] = df_ts['fa_identifier'].map(lambda x: identifier2parent_list_ids.get(x, None))
                if 'fa_parent_list' in fa_columns:
                    df_ts['fa_parent_list'] = df_ts['fa_identifier'].map(lambda x: identifier2parent_list.get(x, None))

                # extract corresponding FA content if TS text has annotation
                df_ts[['fa_text', 'fa_text_sources']] = None
                df_ts[['fa_text', 'fa_text_sources']] = df_ts.apply(lambda x: extract_fa_text(x, df_fa, ps), axis=1, result_type='expand')
                df_ts['fa_text_sources_size'] = df_ts['fa_text_sources'].map(lambda x: calculate_length(x))
                df_ts['fa_text_length'] = df_ts['fa_text'].map(lambda x: length_tokenize_sent(x) if x else x)

                if args.do_match:
                    df_ts[['reason_for_FN', 'FN_error_type_id']] = None
                    df_ts[['reason_for_FN', 'FN_error_type_id']] = df_ts.apply(lambda x: fn_reason_analyzer(x), axis=1, result_type='expand')
                if args.do_match and args.eval_filter_human_errors:
                    '''
                    explanation of FN human errors:
                    FN_error_type_id_1 = "Couldn\'t query FA text by annotation. Unsure if annotation belong to FA that we used Or wrong annotation was made"
                    FN_error_type_id_2 = "Wrong positioning annotation. Should be put at content not section"
                    FN_error_type_id_3 = "Probably mix up definition and parties when making annotation"
                    filter all human errors with error ids 1,2,3
                    '''
                    df_ts['judge_all'] = df_ts[['judge_all', 'FN_error_type_id']].apply(lambda x: None if x[1] in [1, 2, 3] else x[0], axis=1)

            # ========================== Export results ==========================
            # analysis is available only if merging (provide true id labels) and matching (provide predicted id labels) are finished
            if args.do_merge and args.do_match:
                outpath = args.output_analy_dir + ts_fname + '_analysis.xlsx'
                # if not args.apply_ext_sim_match and args.keep_individual_strategy_result:
                #     # performance evaluation by matching strategy, annotation location and text granularity when results of matching strategy are split into different columns
                #     export_ind_strategy_analysis(df_ts, outpath, listPairDict)
                # else:
                ordered_cols = ['index',
                                'text_block_id',
                                'page_id',
                                'phrase_id',
                                'section',
                                'text_element',
                                'text',
                                'list_id',
                                'parent_caption',
                                'parent_list_ids',
                                'parent_list',
                                'text_granularity',
                                'clause_id',
                                'schedule_id',
                                'definition',
                                'fa_identifier',
                                'fa_identifier_type',
                                'annotation',
                                'match_term_list',
                                'identifier_list',
                                'similarity_list',
                                'match_type_list',
                                'judge',
                                'judge_all',
                                'flag',
                                'reason_for_FN',
                                'TP_sim',
                                'TP_rank',
                                'FP_sim',
                                'FP_rank',
                                'fa_section',
                                'fa_sub_section',
                                'fa_schedule',
                                'fa_part',
                                'fa_section_id',
                                'fa_sub_section_id',
                                'fa_schedule_id',
                                'fa_part_id',
                                'fa_clause_list_id',
                                'fa_schedule_list_id',
                                'fa_text',
                                'fa_text_sources',
                                'fa_text_sources_size',
                                'fa_text_length']
                if 'fa_parent_caption' in fa_columns:
                    ordered_cols += ['fa_parent_caption',
                                'fa_parent_list_ids',
                                'fa_parent_list']
                df_ts = df_ts[ordered_cols]
            # ========================== Analyze results by doc ==========================
                # statistics of true positive similarity score
                export_tp_fp_sim_stat(df_ts, outpath)
                # performance evaluation by matching strategy, annotation location and text granularity
                analysis_set = ['index', 'text_granularity',
                                'fa_identifier_type', 'match_type_list', 'judge', 'judge_all']
                # expand the match_type_list, judge into rows
                recall = export_analysis(df_ts, outpath, analysis_set)
                all_stat.append(recall)
            elif args.do_merge:
                ordered_cols = ['index',
                                'text_block_id',
                                'page_id',
                                'phrase_id',
                                'section',
                                'text_element',
                                'list_id',
                                'text',
                                'parent_caption',
                                'parent_list_ids',
                                'parent_list',
                                'text_granularity',
                                'clause_id',
                                'schedule_id',
                                'definition',
                                'annotation',
                                'fa_identifier',
                                'fa_identifier_type',
                                'fa_section',
                                'fa_sub_section',
                                'fa_schedule',
                                'fa_part',
                                'fa_section_id',
                                'fa_sub_section_id',
                                'fa_schedule_id',
                                'fa_part_id',
                                'fa_clause_list_id',
                                'fa_schedule_list_id',
                                'fa_text',
                                'fa_text_sources',
                                'fa_text_sources_size',
                                'fa_text_length']
                if 'fa_parent_caption' in fa_columns:
                    ordered_cols += ['fa_parent_caption',
                                'fa_parent_list_ids',
                                'fa_parent_list']
                df_ts = df_ts[ordered_cols]
                outpath = args.output_analy_dir + ts_fname + '_analysis.csv'
                # expand the match_type_list, judge into rows
                statistics = export_merge_analysis(df_ts, ts_fname, outpath)
                all_stat.append(statistics)
            elif args.do_match:
                ordered_cols = ['index',
                                'text_block_id',
                                'page_id',
                                'phrase_id',
                                'section',
                                'text_element',
                                'list_id',
                                'text',
                                'text_granularity',
                                'match_term_list',
                                'identifier_list',
                                'similarity_list',
                                'match_type_list']
                df_ts = df_ts[ordered_cols]
            df_ts.to_csv(args.output_csv_dir + new_fname,index=False, encoding='utf-8-sig')
            df_ts['filename'] = ts_fname
            df_ts['facility_type'] = df_ts['filename'].map(lambda x: file2facility.get(re.sub('_docparse', '', x), None))
            df_ts['split'] = df_ts['filename'].map(lambda x: file2dataset.get(re.sub('_docparse', '', x), None))

            results.append(df_ts)
        except Exception as e:
            exc_type, exc_value, exc_tb = sys.exc_info()
            tb = traceback.TracebackException(exc_type, exc_value, exc_tb)
            print(f'Error occurs in: {file} with error:\n'+'\n'.join(tb.format()))

    return results, all_stat


def flatten(lst):
    for el in lst:
        if isinstance(el, tuple):
            el = list(el)
        if isinstance(el, list):
            # recurse
            yield from flatten(el)
        else:
            # generate
            yield el


def main(source_ts_files, args):
    all_results = multiprocess(ts_merge_match_fa, source_ts_files, args)
    # all_results = ts_merge_match_fa(source_ts_files, args)
    results = []
    all_statistics = []
    for result, all_stat in all_results:
        results.append(result)
        all_statistics.append(all_stat)

    if args.do_merge and not args.do_match:
        split_without_annotation_count, successful_query_text_count, indeterminate_count, successful_query_text_rate, indeterminate_rate = list(zip(*[x[0] for x in all_statistics]))
    results = list(flatten(results))
    if args.do_merge and args.do_match:
        all_recalls = {'filename': source_ts_files,
                       'recall': list(flatten(all_statistics))}
    elif args.do_merge:
        all_stats = {'filename': source_ts_files,
                     'split_without_annotation_count': split_without_annotation_count,
                     'successful_query_text_count': successful_query_text_count,
                     'indeterminate_count': indeterminate_count,
                     'successful_query_text_rate': successful_query_text_rate,
                     'indeterminate_rate': indeterminate_rate
                     }
    assert len(results) == len(source_ts_files), f'Input {len(source_ts_files)} files but yields {len(results)} results. Please check'

    if len(results) > 0:
        df_result = pd.concat(results, ignore_index=True).sort_values(by=['filename', 'index', 'phrase_id'])
        df_result.to_csv(os.path.join(args.output_csv_dir, f'all_{task_type}_results.csv'), index=False, encoding='utf-8-sig')
        # ========================== Analyze all doc results ==========================
        if args.do_merge and args.do_match:
            outpath = args.output_analy_dir + 'all_analysis.xlsx'
            # if not args.apply_ext_sim_match and args.keep_individual_strategy_result:
            #     # performance evaluation by matching strategy, annotation location and text granularity when results of matching strategy are split into different columns
            #     export_ind_strategy_analysis(df_result, outpath, listPairDict)
            # else:
            num_files = len(df_result['filename'].unique())
            # statistics of true positive similarity score
            export_tp_fp_sim_stat(df_result, outpath)
            analysis_set = ['index', 'text_granularity', 'fa_identifier_type',
                            'match_type_list', 'judge', 'judge_all']
            analysis_set += ['filename']
            export_analysis(df_result, outpath, analysis_set)
    # if all_recalls['recall']:

    # map filename to facility type
    file_facility_type_df = pd.read_csv(args.filename_facility_type_doc)
    file2facility = dict(zip(file_facility_type_df['filename'].values.tolist(), file_facility_type_df['facility_type'].values.tolist()))

    # map filename to dataset split
    file_dataset_df = pd.read_csv(args.file_dataset_doc)
    file2dataset = dict(zip(file_dataset_df['filename'].values.tolist(), file_dataset_df['split'].values.tolist()))

    if args.do_merge and args.do_match:
        df_recall = pd.DataFrame(all_recalls)
        df_recall['recall'] = df_recall['recall'].map(lambda x: '{:.2%}'.format(x) if (x or x != np.nan) and not str(x).endswith('%') else x)
        df_recall['facility_type'] = df_recall['filename'].map(lambda x: file2facility.get(re.sub('_docparse', '', x), None))
        df_recall['split'] = df_recall['filename'].map(lambda x: file2dataset.get(re.sub('_docparse', '', x), None))
        df_recall.to_csv(args.output_analy_dir +f'all_recalls.csv', index=False, encoding='utf-8-sig')
        
        import glob
        all_grpby_sect = []
        for name in glob.glob(args.output_analy_dir + 'grpby_sect/*_grpby_sect_analysis.csv'):
            df_grpby_sect = pd.read_csv(name)
            ts_fname = re.search(re.escape(args.output_analy_dir + 'grpby_sect/') +'(.*)'+ '_grpby_sect_analysis.csv', name).group(1)
            df_grpby_sect['filename'] = ts_fname
            df_grpby_sect['facility_type'] = df_grpby_sect['filename'].map(lambda x: file2facility.get(re.sub('_docparse', '', x), None))
            df_grpby_sect['split'] = df_grpby_sect['filename'].map(lambda x: file2dataset.get(re.sub('_docparse', '', x), None))
            all_grpby_sect.append(df_grpby_sect)
        df_grpby_sect_recall = pd.concat(all_grpby_sect)
        df_grpby_sect_recall = df_grpby_sect_recall[['filename','precision','recall','accuracy','f1_score','facility_type','split']]
        cols_show_as_percentage = ['precision', 'recall', 'accuracy', 'f1_score']
        for column in cols_show_as_percentage:
            df_grpby_sect_recall.loc[:, column] = df_grpby_sect_recall[column].map(lambda x: '{:.2%}'.format(x) if isinstance(x, float) and x != np.nan else x) 
        df_grpby_sect_recall.to_csv(args.output_analy_dir + 'grpby_sect/' + f'all_grpby_sect_analysis.csv',index=False, encoding='utf-8-sig')
            
    elif args.do_merge:
        df_stat = pd.DataFrame(all_stats)
        df_stat['facility_type'] = df_stat['filename'].map(lambda x: file2facility.get(re.sub('_docparse', '', x), None))
        df_stat['split'] = df_stat['filename'].map(lambda x: file2dataset.get(re.sub('_docparse', '', x), None))
        df_stat.to_csv(args.output_analy_dir + f'all_merge_stats.csv',index=False, encoding='utf-8-sig')


def fn_reason_analyzer(row):
    err_msg_1 = "Couldn\'t query FA text by annotation. Unsure if annotation belong to FA that we used Or wrong annotation was made"
    err_msg_2 = "Wrong positioning annotation. Should be put at content not section"
    err_msg_3 = "Couldn\'t query FA text by annotation. Probably mix up definition and parties when making annotation"
    err_msg_4 = "Semantic similarity matching model fail to match anything"
    err_msg_5 = "Semantic similarity matching model yield matching results but not match with annotation provided. Or probably wrong annotation was made"

    judge_all = row['judge_all']
    if judge_all != 'FN':
        return None, None

    fa_text = row['fa_text']
    fa_identifier = row['fa_identifier']
    identifier_list = str(row['identifier_list'])
    text_element = row['text_element']
    if identifier_list == '(None,)' and text_element == 'section':
        return err_msg_2, 2
    if fa_text is None:
        if fa_identifier.startswith('Parties') or fa_identifier.startswith('Cl_1.1'):
            return err_msg_3, 3
        else:
            return err_msg_1, 1
    else:
        if identifier_list == '(None,)':
            return err_msg_4, 4
        else:
            return err_msg_5, 5


if __name__ == "__main__":
    from transformers import AutoTokenizer
    import warnings
    import json
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    warnings.filterwarnings("ignore")
    # configurations (P.S.: some configurations are in config.py)
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--do_merge",
        default=True,
        type=bool,
        required=False,
        help="Set True if merging annotated TS on FA.",
    )
    parser.add_argument(
        "--do_match",
        default=False,
        type=bool,
        required=False,
        help="Set True if perform semantic similarity matching between TS text and FA text.",
    )
    parser.add_argument(
        "--apply_ext_sim_match",
        default=True,
        type=bool,
        required=False,
        help="Set True if leverage external source of term matching result instead of perform matching here.",
    )
    parser.add_argument(
        "--keep_individual_strategy_result",
        default=False,
        type=bool,
        required=False,
        help="Set True if seperate results by each term matching strategy as individual column.",
    )
    parser.add_argument(
        "--topN",
        default=5,
        type=int,
        required=False,
        help="To indicate the top N results of term matching return and keep.",
    )
    parser.add_argument(
        "--sim_threshold",
        default=0,
        type=int,
        required=False,
        help="To indicate the top N results of term matching return and keep.",
    )
    parser.add_argument(
        "--src_ts_csv_dir",
        default=ANTD_TS_CSV,
        type=str,
        required=False,
        help="Folder path to input TS docparse csv.",
    )
    parser.add_argument(
        "--src_fa_csv_dir",
        default=FA_CSV,
        type=str,
        required=False,
        help="Folder path to input FA docparse csv.",
    )
    parser.add_argument(
        "--src_ts_csv_filenames",
        default=None,
        type=list,
        required=False,
        help="List of filename to be used as input. Set value to None to accept all files found in src_ts_csv_dir."
    )
    parser.add_argument(
        "--matched_result_csv_dir",
        default=MATCHED_FA_CSV,
        type=str,
        required=False,
        help="Folder path to input TS matched FA result csv.",
    )
    parser.add_argument(
        "--output_csv_dir",
        default=OUTPUT_MERGE_MATCH_CSV_DIR,
        type=str,
        required=False,
        help="Folder path to output result csv.",
    )
    parser.add_argument(
        "--output_analy_dir",
        default='',
        type=str,
        required=False,
        help="Folder path to output analysis result xlsx.",
    )
    parser.add_argument(
        "--eval_filter_human_errors",
        default=False,
        type=bool,
        required=False,
        help="Set True to filter false negatives contribute by human errors in performing evaluation. Human error may include those ground-truth annotation couldn't query FA clauses or wrong positioning labels",
    )
    parser.add_argument(
        "--filename_facility_type_doc",
        default='/home/data/ldrs_analytics/data/doc/filename_facility_type.csv',
        type=str,
        required=False,
        help="A reference document to map filenames to facility type.",
    )
    parser.add_argument(
        "--filename_facility_type_batch_doc",
        default='/home/data/ldrs_analytics/data/doc/filename_facility_type_batch.csv',
        type=str,
        required=False,
        help="A reference document to map filenames to facility type and batch ID.",
    )
    parser.add_argument(
        "--file_dataset_doc",
        default='/home/data/ldrs_analytics/data/doc/filename_dataset_split.csv',
        type=str,
        required=False,
        help="A reference document to map filename to dataset split (either train or test)",
    )
    parser.add_argument(
        "--indeterminate_doc",
        default='/home/data/ldrs_analytics/data/doc/not_exist_fa_identifier.csv',
        type=str,
        required=False,
        help="A reference document to check if fa_identifier is truely indeterminate.",
    )
    parser.add_argument(
        "--files_with_TS_layout_issues",
        default=[
            '62_PF_SYN_TS_mkd_20220331_docparse.csv',
            '65_PF_SYN_TS_mkd_docparse.csv',
            'Acquisition and share pledge_(1) TS_docparse.csv',
            '(11) sample TS_docparse.csv'
        ],
        type=list,
        required=False,
        help="A list of filenames with TS layout issues.",
    )

    args = parser.parse_args()
    print(f'do-merge: {args.do_merge}, do-match: {args.do_match}')
    assert not (not args.do_merge and not args.do_match), "you must choose either apply annotated TS for merging FA or perform term matching"

    # define output folder path for merge or match result and analysis
    if args.do_match and args.do_merge:
        print('Processing merging term match results with label and ground truth FA content ...')
        task_type = 'merge&match'
        # if args.apply_ext_sim_match:
        #     args.output_csv_dir = OUTPUT_MERGE_MATCH_CSV_DIR
        # else:
        #     args.output_csv_dir = OUTPUT_MERGE_MATCH_CSV_DIR2
        args.output_csv_dir = args.output_csv_dir.rstrip('/') + f'_top{str(args.topN)}/'

        if args.eval_filter_human_errors:
            args.output_csv_dir = args.output_csv_dir.rstrip('/') + '_filter_human_errors/'
        args.output_analy_dir = os.path.join(os.path.join(
            args.output_csv_dir, 'analysis/'), f'({str(args.sim_threshold)})/')
        args.output_csv_dir = os.path.join(
            args.output_csv_dir, f'({str(args.sim_threshold)})/')
    elif args.do_merge:
        print('Processing merging TS with label and ground truth FA content ...')
        task_type = 'merge'
        # args.output_csv_dir = OUTPUT_MERGE_CSV_DIR
        args.output_analy_dir = os.path.join(args.output_csv_dir, 'analysis/')
    else:
        print('Processing term matching ...')
        task_type = 'match'
        # args.src_ts_csv_dir = TS_CSV

    if not os.path.exists(args.output_csv_dir):
        os.makedirs(args.output_csv_dir)
    if args.do_merge and not os.path.exists(args.output_analy_dir):
        os.makedirs(args.output_analy_dir)

    # checks if path is a directory
    isDirectory = os.path.isdir(args.src_ts_csv_dir)
    filename_list = args.src_ts_csv_filenames

    assert isDirectory, 'Please check if you input a correct input folder path.'

    print(f'output csv folder: {args.output_csv_dir}')
    print(f'source annotated csv folder: {args.src_ts_csv_dir}')
    source_ts_files = sorted(os.listdir(args.src_ts_csv_dir))
    if filename_list:
        source_ts_files = [file for file in source_ts_files if file.endswith('.csv') and not file.startswith("~$") and os.path.basename(file) in filename_list]
    else:
        source_ts_files = [file for file in source_ts_files if file.endswith('.csv') and not file.startswith("~$")]
    files_with_TS_layout_issues = [
        # '62_PF_SYN_TS_mkd_20220331_docparse.csv',
        # '65_PF_SYN_TS_mkd_docparse.csv',
        # 'Acquisition and share pledge_(1) TS_docparse.csv',
        # '(11) sample TS_docparse.csv'
    ]
    target_files = [
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
    # if os.path.exists("/home/data/ldrs_analytics/data/reviewed_antd_ts/checked_list.json"):
    #     with open("/home/data/ldrs_analytics/data/reviewed_antd_ts/checked_list.json", 'r') as f:
    #         target_files = json.load(f)

    if target_files:
        source_ts_files = [f for f in source_ts_files if f in target_files]
        not_source_ts_files = [f for f in source_ts_files if f not in target_files]
        print(f'target list files count: {len(target_files)}')
        print(f'Not in target list files: {json.dumps(not_source_ts_files, indent=4)}')

    source_ts_files = sorted([i for i in source_ts_files if i not in files_with_TS_layout_issues])
    main(source_ts_files, args)
    print('Complete merging or matching\n')
