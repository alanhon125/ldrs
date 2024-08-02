'''
Extract and convert docparse results of FA and TS in JSON into CSV, store them in different folder, named 'FA', 'TS'
Move the JSON in current directory into 'FA/' and 'TS/' under current directory
Perform key phrases extraction by kbir-inspec model
Take log of runtime in model-based key phrases extraction on every phrase in CSV
'''
import os
import re
import json
import pandas as pd
import numpy as np
import string
from roman_arabic_numerals import conv
from typing import Union, Dict, Tuple, Callable
from collections.abc import Iterable
import spacy
import warnings
warnings.filterwarnings("ignore")
import logging

try:
    from app.utils import *
    from config import *
    from regexp import *
except:
    currentdir = os.path.dirname(os.path.abspath(__file__))
    parentdir = os.path.dirname(currentdir)
    parentparentdir = os.path.dirname(parentdir)
    import sys
    sys.path.insert(0, parentdir)
    from utils import *
    sys.path.insert(0, parentparentdir)
    from config import *
    from regexp import *

try:
    nlp = spacy.load("en_core_web_md")
except:
    os.system('python3 -m spacy download en_core_web_md')
    nlp = spacy.load("en_core_web_md")

logger = logging.getLogger(__name__)
# logging.basicConfig(
#     filename=LOG_FILEPATH,
#     filemode='a',
#     format='【%(asctime)s】【%(filename)s:%(lineno)d】【%(levelname)-8s】%(message)s',
#     level=os.environ.get("LOGLEVEL", "INFO"),
#     datefmt='%Y-%m-%d %H:%M:%S'
#     )
    
nlp = config_nlp_model(nlp)

def return_on_failure(f):
    '''
    To apply and try a function, if it fails print error message
    '''
    def applicator(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except:
            logger.warning(f'Fail to complete feature extraction on {os.path.basename(args[0][0])}')

    return applicator

def split_at_seperators(x):
    dot_pattern = '(?<!\d)(?<![a-zA-Z]\.)(?<!etc)\.(?!\d)(?![a-zA-Z]\.)'
    seperators = ['(?<!")(?<!s)\)(?!\s/)','(?<!")\](?!\s/)','XXX',dot_pattern] + [i+' :' for i in TS_SECTIONS_LIST] + [':']
    seperators2 = [':', '(?<!")(?<!s)\)(?!\s/)', '(?<!")\](?!\s/)', 'XXX', dot_pattern]
    lst = [i for i in re.split('('+r')|('.join(seperators)+')', x) if i and i.strip()!='']
    new_lst = []
    for i, v in enumerate(lst):
        if re.match('|'.join(seperators2), v.strip()):
            # if new_lst:
            tmp = new_lst.pop(-1)+v+' '
            new_lst.append(tmp)
        elif v and not re.match('|'.join(seperators2), v):
            new_lst.append(v)
    new_lst = [i.strip() for i in new_lst]
    return new_lst

def join_seperated_letter(s: str) -> str:
    import string
    punct = string.punctuation+'–'
    if not s:
        return s
    
    def join_seperated_capital_letter(s: str) -> str:
        '''
        @param s: string of text that words segmentate into sub words and need to be restored
        e.g. 'F ORM OF I NCREASE C ONFIRMATION' -> 'FORM OF INCREASE CONFIRMATION'
        '''
        import re

        s = str(s).split()
        new_s = []
        k = iter(s)
        for Id, i in enumerate(k):
            tmp = i
            if Id+1 <= len(s)-1 and len(i) == 1 and len(s[Id + 1]) == 1:
                new_s.append(tmp)
                continue
            if Id+1 <= len(s)-1 and (len(i) == 1 and i not in punct and i not in ['a', 'A', 'I'] and not re.match(r'\d{1}', i) or
                                    len(i) == 1 and Id - 1 >= 0 and i not in punct and not i in ['a', 'A' ,'I', 'V'] and not re.match(r'part\s*|schedule\s*|tranche\s*|appendix\s*', s[Id - 1], re.IGNORECASE) and not re.match(r'\d{1}', i) or
                                    i in ['A','I'] and Id != 0 and not re.match(r'part\s*|schedule\s*|tranche\s*|appendix\s*', s[Id - 1], re.IGNORECASE) or
                                    len(i) == 2 and any(i.startswith(item) for item in punct) and not any(s[Id+1].startswith(item) for item in punct) or
                                    len(i) > 1 and i.endswith('-')):
                if s[Id+1] not in ['and', 'or', 'only', 'only)']:
                    tmp += s[Id+1]
                    s.remove(s[Id+1])
            elif Id+1 > len(s)-1 and (len(i) == 1 and i not in punct and i not in ['a', 'A', 'I'] and not re.match(r'\d{1}', i) or
                                    len(i) == 1 and Id - 1 >= 0 and i not in punct and not i in ['a', 'A' ,'I', 'V'] and not re.match(r'part\s*|schedule\s*|tranche\s*|appendix\s*', s[Id - 1], re.IGNORECASE) and not re.match(r'\d{1}', i) or
                                    i in ['A','I'] and Id != 0 and not re.match(r'part\s*|schedule\s*|tranche\s*|appendix\s*', s[Id - 1], re.IGNORECASE) or
                                    len(i) > 1 and i.endswith('-')):
                if s[Id-1] not in ['and', 'or', 'only', 'only)','Part','Schedule','Tranche','Appendix']:
                    tmp = s[Id - 1] + tmp
                    if s[Id - 1] in new_s:
                        new_s.remove(s[Id - 1])
            new_s.append(tmp)
        new_s = ' '.join(new_s)
        return new_s
    
    s = str(s).split()
    counter = 0
    while any(len(i) == 1 and i not in punct and i not in ['a', 'A', 'I'] and not re.match(r'\d{1}', i) for i in s):
        join_text = join_seperated_capital_letter(' '.join(s))
        s = join_text.split()
        counter +=1
        if counter>5:
            break
    
    return ' '.join(s)

def split_section_id(obj: Union[pd.DataFrame, str]) -> list[str]:
    '''
    @param obj: an object either be an pandas.DataFrame object with columns 'text', 'next_text', 'text_element' and 'next_text_element' or a string
    split section ID and section description.
    In facility agreement, the hierarchy of doc consists of 'Clause Section', 'Clause Sub-section', 'Schedule' and 'Part'
    Each heading has its corresponding ID, thus it splits the string to extract those info
    '''
    section_id, sub_section_id, schedule_id, part_id, section, sub_section, schedule, part = [None] * 8

    def isfloat(num):
        try:
            float(num)
            return True
        except ValueError:
            return False

    if isinstance(obj, pd.Series) or isinstance(obj, pd.DataFrame):
        text = obj['text']
        next_text = obj['next_text']
        text_element = obj['text_element']
        next_element = obj['next_text_element']
    elif isinstance(obj, str):
        text = obj

    if re.match('|'.join([SCHED_DIGIT_PATTERN, SCHED_ROMAN_PATTERN]), str(text), flags=re.IGNORECASE):
        split = [i for i in re.split('|'.join(
            [SCHED_DIGIT_PATTERN, SCHED_ROMAN_PATTERN]), str(text), flags=re.IGNORECASE) if i]
        if len(split) == 2 and split[-1] not in string.punctuation:
            schedule_id, schedule = split
        else:
            schedule_id = split[0]
            schedule = next_text
        if re.match(CAP_SIMPLE_ROMAN_PATTERN, schedule_id, flags=re.IGNORECASE):
            if schedule_id.lower() in ['i','ii','iii','iv','v','vi','vii','viii','ix','x']:
                schedule_id = conv.rom_arab(schedule_id)
        if schedule and schedule.startswith('to'):
            schedule_id = schedule = None

    elif re.match(SCHED_ORDINAL_PATTERN, str(text), re.IGNORECASE):
        split = [i for i in re.split(SCHED_ORDINAL_PATTERN, str(text), flags=re.IGNORECASE) if i]
        if len(split) == 2 and split[-1] not in string.punctuation:
            ordinal_num, schedule = split
            schedule_id = ORDINAL2NUM[ordinal_num.lower()]
        else:
            schedule_id = split[0]
            try:
                schedule_id = ORDINAL2NUM[schedule_id.lower()]
            except:
                pass
            schedule = next_text
    elif re.match(SCHED_ORDINAL_PATTERN2, str(text), re.IGNORECASE):
        split = [i for i in re.split(
            SCHED_ORDINAL_PATTERN2, str(text), flags=re.IGNORECASE) if i]
        if len(split) == 2 and split[-1] not in string.punctuation:
            ordinal_num, schedule = split
            schedule_id = ORDINAL2NUM[ordinal_num.lower()]
        else:
            schedule_id = split[0]
            try:
                schedule_id = ORDINAL2NUM[schedule_id.lower()]
            except:
                pass
            schedule = next_text
    elif re.match('|'.join([PART_DIGIT_PATTERN, PART_ROMAN_PATTERN, PART_ALPHA_PATTERN]), str(text), re.IGNORECASE):
        split = [i for i in re.split('|'.join(
            [PART_DIGIT_PATTERN, PART_ROMAN_PATTERN, PART_ALPHA_PATTERN]), str(text), flags=re.IGNORECASE) if i]
        if len(split) == 2 and split[-1] not in string.punctuation:
            part_id, part = split
        else:
            part_id = split[0]
            part = next_text
        if re.match(CAP_SIMPLE_ROMAN_PATTERN, part_id):
            if part_id.lower() in ['i','ii','iii','iv','v','vi','vii','viii','ix','x']:
                part_id = conv.rom_arab(part_id)
    elif re.match(SECTION_PATTERN, str(text), re.IGNORECASE):
        split = [i for i in re.split(SECTION_PATTERN, str(text)) if i]
        if len(split) == 4:
            section_id = split[2]
            section = split[-1]
    elif re.match(SEC_SUBSEC_PATTERN, str(text), re.IGNORECASE):
        split = [i for i in re.split(SEC_SUBSEC_PATTERN, str(text)) if i]
        if len(split) == 3:
            section_id, sub_section_id, sub_section = split
            if ':' in sub_section:
                sub_section = sub_section.split(':')[0].strip()
        elif len(split) == 2:
            if not all([isfloat(i) for i in split]):
                section_id, section = split
                if 'Note:' in section or 'note:' in section:
                    section_id = section = None
            else:
                section_id, sub_section_id = split
                if 'next_text' in locals() and 'sub_section' in next_element:
                    sub_section = next_text
                    if ':' in sub_section:
                        sub_section = sub_section.split(':')[0].strip()
        elif len(split) == 1:
            if 'next_text' in locals() and next_element == 'section':
                section = next_text
    elif re.match(SEC_SUBSEC_ID_PATTERN, str(text), re.IGNORECASE):
        split = [i for i in re.split(SEC_SUBSEC_ID_PATTERN, str(text)) if i]
        if len(split) == 2:
            section_id, sub_section_id = split
        elif len(split) == 1:
            section_id = split[0]
    # else:
    #     if text_element == 'section':
    #         section = text
    #     elif 'sub_section' in text_element:
    #         sub_section = text
    if section and section.startswith('[{'):
        section = None
    if sub_section and sub_section.startswith('[{'):
        sub_section = None
    if schedule and schedule.startswith('[{'):
        schedule = None
    if part and part.startswith('[{'):
        part = None

    return section_id, sub_section_id, schedule_id, part_id, section, sub_section, schedule, part

def extract_parties(text: str) -> str:
    '''
    @param text: a string of text in parties clauses
    In 'Parties' section, split the name of party role from the clause content
    '''
    import re

    try:
        party_with_together = re.match(PARTIES_PATTERN_TGT, text)
        party_with_together2 = re.match(PARTIES_PATTERN_TGT_2, text)
        party_in_bracket = re.match(PARTIES_PATTERN_BRACK, text)
        party_with_as = re.match(PARTIES_PATTERN_AS, text)
        party_with_as_stop_by_brack = re.match(PARTIES_PATTERN_AS_OF_BRACK, text)
        party_with_as_stop_by_brack2 = re.match(PARTIES_PATTERN_AS_BRACK, text)
        party_in_quote = re.findall(TERMS_IN_QUOTE, text)
        if party_in_quote:
            party = re.findall(TERMS_IN_QUOTE, text)
            party = [remove_punctuation(p) for p in party if p.strip()]
            party = [re.sub(ARTICLES, '', p, flags=re.IGNORECASE) for p in party]
            party = [p.replace("  ", " ").strip() for p in party]
            party = ', '.join(party)
        elif party_with_as_stop_by_brack:
            party = remove_punctuation(party_with_as_stop_by_brack.groups()[0])
        elif party_with_as_stop_by_brack2:
            party = remove_punctuation(party_with_as_stop_by_brack2.groups()[0])
        elif party_with_together:
            party = remove_punctuation(party_with_together.groups()[0])
        elif party_with_together2:
            party = remove_punctuation(party_with_together2.groups()[0])
        elif party_with_as:
            party = remove_punctuation(party_with_as.groups()[0])
        else:
            party = remove_punctuation(party_in_bracket.groups()[0])

        party = re.sub(ARTICLES, '', party, flags=re.IGNORECASE)
        party = party.replace("  ", " ").strip()
    except:
        party = None

    party = re.sub('\[|\]|\(|\)|"|\'|“|”|:', '', party).strip() if party else party
    party = re.sub('\s+', ' ', party) if party else party

    return party

def extract_definition(text: Union[int, float, str]) -> list[str]:
    '''
    @param text: string of text in definition clauses
    In 'Definitions' section, the representation must contain the following words: ' means ', ' means, ', ' means:', ' has the meaning given to it in '
    It splits the definition term and its description by these wordings
    '''
    if text is None:
        return None, None

    elif isinstance(text, float) or isinstance(text, int):
        text = str(text)

    # if all keys are not be found in text, then text doesn't contains definition terms
    if all([not re.search(k, text) for k in DEF_KEYS]) and all([not re.search(k, text) for k in DEF_KEYS_2]):
        return None, None

    case_basel2 = re.search(BASEL2_PATTERN, str(text))
    if case_basel2:
        definition = case_basel2.group(1)
        definition = re.sub(r'["“”]', '', definition)
        description = case_basel2.group(2)
        return definition, description

    definition_all = re.findall(TERMS_IN_QUOTE, text)
    # if "" find in string pattern
    if definition_all:
        definition_all = [d for d in definition_all if d.strip()]
        definition_all = [re.sub(rf'\b(the|The|An)\b|\.', '', d, flags=re.IGNORECASE) for d in definition_all]
        definition_all = [d.replace("  ", " ").strip() for d in definition_all]
        if len(definition_all) > 1:
            definition = ''
            for i, d in enumerate(definition_all):
                if i == 0:
                    definition += d
                elif i < len(definition_all) - 1:
                    definition += ', ' + d
                else:
                    definition += ' or ' + d
        elif len(definition_all) == 1:
            definition = definition_all[0]
        else:
            definition = None

        # definition = ' or '.join(definition_all)
        return definition, text

    # try to find if keys in list is in string pattern
    try:
        match = re.search(r'|'.join(['(' + i + ')' for i in DEF_KEYS]), str(text))[0]
    except:
        match = None
    split = [i.strip() for i in re.split(r'|'.join(DEF_KEYS), str(text)) if i]
    description = definition = None

    if not re.match(rf'(.*)means:(.*)', str(text)):
        try:
            matched_key = re.search('|'.join(DEF_KEYS), str(text))[0]
        except:
            matched_key = ''
        if len(split) > 2:
            try:
                definition = re.search(rf'.*,(.*)', split[0]).group(1)
            except:
                definition = split[0]
            description = [f' {match} '.join(split[1:])][0]
        elif len(split) == 2:
            definition, description = split
            try:
                definition = re.search(rf'.*,(.*)', definition).group(1)
            except:
                pass
            description = matched_key.strip() + ' ' + description
        elif len(split) == 1:
            description = split[-1]
    else:
        if len(split) == 2:
            definition = split[0]
            try:
                definition = re.search(rf'.*,(.*)', definition).group(1)
            except:
                pass
            description = 'means:'
        elif len(split) == 1:
            definition = split[-1]
            description = 'means:'

    if definition and ':' in definition:
        definition = definition.split(':')[0]
        
    definition = re.sub('\[|\]|\(|\)|"|\'|“|”|:', '',
                        definition).strip() if definition else definition
    definition = re.sub('\s+', ' ', definition) if definition else definition

    if definition and len(definition.split()) > 20:
        definition = None

    return definition, description

def get_identifier(row: pd.DataFrame) -> str:
    '''
    Given the list of keys, concatenate the keys including 'section_id','sub_section_id','schedule_id','part_id','list_id','definition' based on 'text_element'
    '''
    sec_id, sub_sec_id, schedule_id, part_id, list_id, definition, txt_ele = row[['section_id', 'sub_section_id', 'schedule_id', 'part_id', 'list_id', 'definition', 'text_element']]
    if sec_id and schedule_id in [None, np.nan, 'None', 'nan']:
        if sec_id == '0' and str(definition) not in ['nan', '', 'None']:
            identifier = 'Parties-' + definition
        elif sec_id == '0' and str(definition) in ['nan', '', 'None']:
            identifier = 'Parties'
        else:
            if txt_ele == 'section':
                target_ids = [sec_id]
            else:
                target_ids = [sec_id, sub_sec_id]
            identifier = 'Cl_' + '.'.join([re.sub(TRAILING_DOT_ZERO, '', str(i)) for i in target_ids if i and i != 'None'])
            if str(definition) not in ['nan', '', 'None']:
                identifier += '-' + definition
    elif schedule_id and schedule_id not in [None, np.nan, 'None', 'nan']:
        target_ids = [schedule_id, part_id] if str(part_id) not in ['nan', '', 'None'] else [schedule_id]
        id1 = '.'.join([re.sub(TRAILING_DOT_ZERO, '', str(i)) for i in target_ids if i])
        if sec_id and str(sec_id) not in ['nan', '', 'None']:
            id2 = re.sub(TRAILING_DOT_ZERO, '', str(sec_id))
            identifier = 'Sched_' + id1 + '_' + id2
        else:
            identifier = 'Sched_' + id1
    else:
        identifier = ''

    if list_id and (schedule_id or (sec_id != '0' and str(definition) not in ['nan', '', 'None']) or  # if there exists definition
                    (identifier.strip() and identifier[-1].isdigit() and list_id[0].isdigit())):  # or identifier endwith digit and list id startwith digit
        identifier += '_' + str(list_id)
    elif list_id and sec_id != '0' and str(definition) in ['nan', '', 'None']:
        identifier += str(list_id)

    return identifier

def tryfunction(value, func: Callable):
    '''
    try a function, if encounter exception or error, return a given value
    '''
    try:
        return func(value)
    except:
        return value

def is_all_words_capitalized(string: str, nlp: spacy.lang.en.English) -> bool:
    '''
    Inspect if all words in a string (except stopwords) is capitalized
    e.g. THE ORIGINAL LENDER, AND BORROWER -> True
    The Original Lender, And Borrower -> True
    the Original Lender, and Borrower -> True
    the original Lender, and borrowe -> False
    '''

    doc = nlp(string)

    trues = []
    for token in doc:
        # logger.debug(token.i, token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
        #       token.shape_, token.is_alpha, token.is_stop)
        if (not token.is_stop and token.pos_ not in ['PUNCT', 'NUM'] and token.tag_ != 'NNS') and doc[token.i-1].pos_ != 'PUNCT':
            trues.append(token.text[0].isupper())

    return all(trues)

def remove_non_alphabet(s: str) -> str:
    '''
    Remove non-alphabet character from a string, e.g. 10. Documentations: -> Documentation
    '''
    import re
    import string

    bullet_symbols = ''.join(SYMBOLS)
    punct = re.escape(re.sub(r'\[|\(', '', string.punctuation))

    # remove leading non-alphabet character, e.g. 1. Borrower -> Borrower
    if re.match(r'^([0-9' + punct + bullet_symbols + r'\s]+)(.+)', s):
        s = re.match(r'^([0-9' + punct + bullet_symbols + r'\s]+)(.+)', s).groups()[1]

    # remove trailing non-alphabet character, e.g. Borrower: -> Borrower
    if re.match(r'([a-zA-Z ]+)([' + punct + bullet_symbols + r'\s]+)$', s):
        s = re.match(r'([a-zA-Z ]+)([' + punct + bullet_symbols + r'\s]+)$', s).groups()[0]

    # replace multiple whitespaces into one whitespace
    s = re.sub(' +',' ',s)
    return s

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

def get_parent_list_item(df_grp: pd.DataFrame) -> pd.DataFrame:
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

def docparse2table(input_tuple: Iterable[Union[str, dict], str]) -> dict:
    '''
    input the tuple of data (either in nested dictionary or string of file path to JSON document) and document type (either 'TS' or 'FA')
    extract all keys, terms and clauses from the content
    '''
    data, doc_type = input_tuple
    if isinstance(data, str) and '.json' in data:
        fpath = data
        if not os.path.exists(fpath):
            raise FileNotFoundError(f'{fpath} does not exists.')
        fname = os.path.basename(fpath)
        logger.info(f'processing file: {fname}')
        # load the docparse result in JSON
        with open(fpath, 'r') as f:
            d = json.load(f)
    elif isinstance(data, dict):
        d = data
    else:
        raise TypeError('Input data must either a string or a dictionary.')
    filename = d['filename']
    process_datetime = d['process_datetime']
    block2ele2order = {}
    block_ele2bbox_page = {}
    data = []
    # extract the content only and create block2ele2order for mapping text element reading order
    for content in d['content']:
        block_id = content['id']
        content_dic = {k: v for k, v in content.items() if not re.match(r'page_id|id|bboxes_pageid|block_bbox|parent.*|child.*', k)}
        # create dictionary of text_element mapping to text_element order and (block_id, text_element) mapping to (bbox, page_id)
        counter = 0
        ele2order = {}
        for k, v in content_dic.items():
            if isinstance(v, dict):
                for k1, v1 in v.items():
                    ele2order[k+'_'+k1] = counter
                    block_ele2bbox_page[(block_id, k+'_'+k1)] = content['bboxes_pageid'][k]
                    counter += 1
            else:
                ele2order[k] = counter
                block_ele2bbox_page[(block_id, k)] = content['bboxes_pageid'][k]
                counter += 1
        # create dictionary of block_id mapping to text_element-to-order dict
        block2ele2order[block_id] = ele2order

        flatten_dic = flatten(content)
        dic = {k: v for k, v in flatten_dic.items() if not re.match(r'page_id|id|bboxes_pageid|block_bbox|parent.*|child.*', k)}

        data.append(dic)

    if DEV_MODE:
        if not os.path.exists(os.path.join(OUTPUT_DOCPARSE_CSV, doc_type)):
            # Create the sub-directory under output directory because it does not exist
            os.makedirs(os.path.join(OUTPUT_DOCPARSE_CSV, doc_type))
        outpath = os.path.join(os.path.join(OUTPUT_DOCPARSE_CSV, doc_type), filename+'_docparse.csv')
    df = pd.DataFrame(data)
    # reshaped DataFrame having a multi-level index
    df = df.stack().reset_index()
    df = df.rename(columns={'level_0': 'text_block_id','level_1': 'text_element', 0: 'text'})
    # df.loc[df.text_element.str.contains('list'),'text'] = df[df.text_element.str.contains('list')]['text'].map(lambda x: ''.join(x) if isinstance(x,list) else x)
    df['index'] = df[['text_block_id', 'text_element']].apply(lambda x: block2ele2order.get(x[0]).get(x[1], None), axis=1)
    df = df.sort_values(by=['text_block_id', 'index'])
    df['index'] = df.groupby(['text_block_id', 'index'])['index'].transform(lambda x: x != x.shift()).cumsum()-1
    df['bbox_pageid'] = df[['text_block_id', 'text_element']].apply(lambda x: block_ele2bbox_page.get((int(x[0]), x[1]), None), axis=1)
    df['bbox'] = df['bbox_pageid'].apply(lambda x: x[0])
    df['page_id'] = df['bbox_pageid'].apply(lambda x: x[1])
    df.drop(columns=["bbox_pageid"], inplace=True)
    df.loc[df['text_element'].str.contains('table'), 'text'] = df[df['text_element'].str.contains('table')]['text'].astype(str)
    df = df.explode('text')  # explode value if it is a list
    df = df.sort_values(by=['text_block_id', 'index']).reset_index(drop=True)
    clause_index = pd.Series(data=[True]*len(df))
    isSec = df.text_element.eq('section')
    isSubSec = df.text_element.str.contains('^sub_section|^sub_sub_section')
    isListPara = df.text_element.str.contains('^list|^paragraph')
    # get list_id from list or paragraph element tag
    df['list_id'] = df[isListPara]['text_element'].map(lambda x: x.split('_')[-1] if len(x.split('_')) > 1 and not str(x.split('_')[-1]).isnumeric() else None).astype(str).map(str.strip)
    # rename the element key by removing the list_id suffix
    df.loc[isListPara, 'text_element'] = df[isListPara]['text_element'].map(lambda x: '_'.join(x.split('_')[:-1]) if re.match('\(*.*\)|.*\.', str(x.split('_')[-1])) else '_'.join(x.split('_')[:]) if len(x.split('_')) == 2 else x)
    df['text'] = df.text.astype(str)
    df.loc[df.text.str.contains('|'.join([f'^{i}' for i in SYMBOLS])),'list_id'] = None
    # remove residual pure punctuations and digits due to cleaning of Chinese characters
    df = df[~df.text.str.contains(PURE_PUNCT_DIGIT)]
    # df['text'] = df.text.map(join_seperated_letter)

    if doc_type == 'FA':
        isTableContent = df.text.str.contains(TABLECONTENT_BEGIN_PATTERN, na=False, case=False)
        isPartiesStart = (df.text.str.contains(PARTIES_BEGIN_PATTERN, na=False, case=False))
        isPartiesEnd = df.text.str.contains(PARTIES_END_PATTERN, na=False, case=False)
        try:
            if isTableContent.any():
                TableContentBeginID = df[isTableContent]['index'].values[0]
            else:
                TableContentBeginID = 0
        except IndexError as e:
            logger.error(f"filename: {filename}, Table of content starting position couldn't be located. Please review the if table of content opening exists or if the regular expression for table of content extraction is correct.")
            return {
                'success': False,
                'errorMessage': "Table of content starting position couldn't be located. Please review the if table of content opening exists or if the regular expression for table of content extraction is correct."
            }
        try:
            partiesBeginID = df[isPartiesStart]['index'].values[0]
        except IndexError as e:
            logger.error(f"filename: {filename}, Parties clause starting position couldn't be located. Please review the if the parties clauses opening sentence exists or if the regular expression for parties clause extraction is correct.")
            return {
                'success': False,
                'errorMessage': "Parties clause starting position couldn't be located. Please review the if the parties clauses opening sentence exists or if the regular expression for parties clause extraction is correct."
            }
        try:
            partiesEndID = df[isPartiesEnd]['index'].values[0] - 1
        except IndexError as e:
            logger.error(f"filename: {filename}, Parties clause ending position couldn't be located. Please review the if the parties clauses ending sentence exists or if the regular expression for parties clause extraction is correct.")
            return {
                'success': False,
                'errorMessage': "Parties clause ending position couldn't be located. Please review the if the parties clauses ending sentence exists or if the regular expression for parties clause extraction is correct."
            }

        # split section and sub_section text and insert new rows if text pattern '1. XXX 1.1 YYY' is found
        df[['section_text', 'sub_section_text']] = df.text.str.extract(SEC_N_SUBSEC_PATTERN2, expand=True)
        df = df.melt(id_vars=['text_block_id', 'text_element', 'text', 'page_id', 'bbox', 'index', 'list_id'],
                     var_name="section_type",
                     value_vars=["section_text", "sub_section_text"],
                     value_name="sec_text").sort_values(['index', 'section_type'])

        # extract parties clauses df
        tableCotent_id = df['index'].between(TableContentBeginID, partiesBeginID-1)
        parties_clause_id = df['index'].between(partiesBeginID, partiesEndID)

        df.loc[df.sec_text.notnull() & (~tableCotent_id | ~parties_clause_id),'text'] = df[df.sec_text.notnull() & (~tableCotent_id | ~parties_clause_id)]['sec_text']
        df.loc[(df.sec_text.notnull() & (~tableCotent_id | ~parties_clause_id)), 'text_element'] = df[(df.sec_text.notnull() & (~tableCotent_id | ~parties_clause_id))]['section_type'].str.extract('(.*)_text', expand=False)
        df = df.drop_duplicates(subset=['index', 'text_element', 'list_id', 'text'], keep='first').reset_index(drop=True).drop(['index', 'section_type', 'sec_text'], axis=1)
        df = df.rename_axis('index').reset_index()
        df = df.replace({np.nan: None, 'nan': None, 'None': None})

        # pattern of . X.YY contains in a string
        sec_embed_pattern = r"\. \d{1,2}[a-zA-Z]*\.\d{0,2}\.*\d{0,2}\.*\d{0,2}"
        # split X.YY as new sentence if such pattern is found in list item
        df.loc[df.text_element.str.contains('^list') & (~tableCotent_id | ~parties_clause_id), 'text'] = df[df.text_element.str.contains('^list') & (~tableCotent_id | ~parties_clause_id)]['text'].map(lambda x: phrase_tokenize(x, nlp) if x and re.search(sec_embed_pattern, x) else x)
        df = df.explode('text')
        df.reset_index(drop=True, inplace=True)
        df['index'] = (df.text_element!=df.text_element.shift()).cumsum()-1

        isSec = df.text_element.eq('section')
        isSubSec = df.text_element.str.contains('^sub_section|^sub_sub_section')
        isListPara = df.text_element.str.contains('^list|^paragraph')

        # redo indexing the parties and table of content
        isTableContent = df.text.str.contains(TABLECONTENT_BEGIN_PATTERN, na=False, case=False)
        isPartiesStart = (df.text.str.contains(PARTIES_BEGIN_PATTERN, na=False, case=False))
        isPartiesEnd = df.text.str.contains(PARTIES_END_PATTERN, na=False, case=False)

        # extract parties clauses df
        try:
            if isTableContent.any():
                TableContentBeginID = df[isTableContent]['index'].values[0]
            else:
                TableContentBeginID = 0
        except IndexError as e:
            logger.error(f"filename: {filename}, Table of content starting position couldn't be located. Please review the if table of content opening exists or if the regular expression for table of content extraction is correct.")
            return {
                'success': False,
                'errorMessage': "Table of content starting position couldn't be located. Please review the if table of content opening exists or if the regular expression for table of content extraction is correct."
            }
        try:
            partiesBeginID = df[isPartiesStart]['index'].values[0]
        except IndexError as e:
            logger.error(f"filename: {filename}, Parties clause starting position couldn't be located. Please review the if the parties clauses opening sentence exists or if the regular expression for parties clause extraction is correct.")
            return {
                'success': False,
                'errorMessage': "Parties clause starting position couldn't be located. Please review the if the parties clauses opening sentence exists or if the regular expression for parties clause extraction is correct."
            }
        try:
            partiesEndID = df[isPartiesEnd]['index'].values[0] - 1
        except IndexError as e:
            logger.error(f"filename: {filename}, Parties clause ending position couldn't be located. Please review the if the parties clauses ending sentence exists or if the regular expression for parties clause extraction is correct.")
            return {
                'success': False,
                'errorMessage': "Parties clause ending position couldn't be located. Please review the if the parties clauses ending sentence exists or if the regular expression for parties clause extraction is correct."
            }
        tableCotent_id = df['index'].between(TableContentBeginID, partiesBeginID-1)
        parties_clause_id = df['index'].between(partiesBeginID, partiesEndID)

        if not parties_clause_id.any():
            logger.error(f"filename: {filename}, Parties clause couldn't be located. Please review the if the parties clauses opening sentence exists or if the regular expression for parties clause extraction is correct.")
            return {
                'success': False,
                'errorMessage': "Parties clause couldn't be located. Please review the if the parties clauses opening sentence exists or if the regular expression for parties clause extraction is correct."
            }
        try:
            schedule1_title = df[tableCotent_id & df.text_element.eq('table')].text.str.extract(TABLECONTENT_SCHED_1_PATTERN, flags=re.IGNORECASE).values[-1][0].strip()
            logger.info(f'filename: {filename}, schedule1_title_0: {schedule1_title}')
        except:
            try:
                schedule1_title = df[tableCotent_id & df.text_element.eq('table') & df.text_element.shift(1).eq('section') & df.text.shift(1).str.contains(r'schedule[s]*', case=False)].text.str.extract(r'.*1\.*:*\s([A-Za-z\s]*).*', flags=re.IGNORECASE).values[-1][0].strip()
                logger.info(f'filename: {filename}, schedule1_title_1: {schedule1_title}')
            except:
                try:
                    schedule1_title = df[tableCotent_id & df.text_element.eq('table')].text.str.extract(TABLECONTENT_SCHED_1_PATTERN2, flags=re.IGNORECASE).values[-1][0].strip()
                    logger.info(f'filename: {filename}, schedule1_title_2: {schedule1_title}')
                except:
                    schedule1_title = None
        df.loc[parties_clause_id, 'definition'] = df[parties_clause_id]['text'].map(extract_parties)
        if schedule1_title not in [np.nan, 'nan', None, '']:
            sched_title = rf'^S\s*CHEDULE[S]*\s*[1|I]\b|^S\s*CHEDULE[S]*\s*[1|I]\D+|^THE FIRST SCHEDULE|^{schedule1_title}$|^THE ORIGINAL LENDERS$|^THE PARTIES$|^PART A|^EXECUTION:-*\s*S\s*CHEDULE[S]*\s*[1|I]'
        else:
            sched_title = SCHED_TITLE
        try:
            schedBeginID = df[(df.text_element.str.contains('section')) & (df.text.str.contains(sched_title, na=False, case=False)) & (~df.text.str.contains(EXCLUD_STR_IN_SCHED_TITLE, na=False, case=False)) & (~tableCotent_id | ~parties_clause_id)]['index'].values[0]
        except IndexError as e:
            logger.error(f"filename: {filename}, Schedule clause could not be recognized. Please review the if schedule opening sentence or heading exists or if the regular expression for schedule clause extraction is correct.")
            return {
                'success': False,
                'errorMessage': "Schedule clause could not be recognized. Please review the if schedule opening sentence or heading exists or if the regular expression for schedule clause extraction is correct."
            }
        if schedBeginID == 0:
            logger.error(f"filename: {filename}, Schedule clause was recognized as the beginning of the agreement, which it should\'nt be. Please review the if schedule opening sentence or heading exists or if the regular expression for schedule clause extraction is correct.")
            return {
                'success': False,
                'errorMessage': "Schedule clause was recognized as the beginning of the agreement, which it should\'nt be. Please review the if schedule opening sentence or heading exists or if the regular expression for schedule clause extraction is correct."
            }
        schedEndID = df['index'].iloc[-1]
        clause_index = df['index'].between(0, schedBeginID - 1)

    # create dictionary of text_block_id to section
    id2sec = dict(df[isSec & clause_index][['index', 'text']].values)
    sec2id = {v: k for k, v in id2sec.items()}

    if doc_type == 'FA':
        isSchedInClause = [re.match('^S\s*CHEDULE[S]*\s*\d+.*', i, flags=re.IGNORECASE) for i in id2sec.values()]
        if any(isSchedInClause):
            wrong_sched = [i[0] for i in isSchedInClause if i]
            for sch in wrong_sched:
                del id2sec[sec2id[sch]]

    df['section'] = df['index'].map(id2sec)
    # df.loc[~df.section.isnull(), 'section'] = df[~df.section.isnull()]['section'].map(lambda x: re.sub(r'[\[|\(].*[\]|\)]', '',x))  # replace all string with [] or () in section keys, e.g. 'Quotation Day ["Quotation Day"]' become 'Quotation Day'
    # df.loc[~df.section.isnull(), 'section'] = df[~df.section.isnull()]['section'].map(lambda x: re.sub(r'[\[|\(\]|\)]', '', x).strip() if x else x)
    df['next_text'] = df.text.shift(-1)
    df['next_text_element'] = df.text_element.shift(-1)

    if doc_type == 'FA':

        schedIsNotNull = df['index'].between(schedBeginID, schedEndID)
        blockid2sched = dict(df[isSec & schedIsNotNull]
                             [['text_block_id', 'text']].values)
        df['schedule'] = df.text_block_id.map(blockid2sched)

        # extract keys in schedule parts
        clause_key = ['section_id', 'sub_section_id', 'schedule_id', 'part_id', 'section', 'sub_section', 'schedule', 'part']
        df.loc[schedIsNotNull, clause_key] = df[schedIsNotNull].apply(lambda x: pd.Series(split_section_id(x), index=clause_key), axis=1, result_type='expand')
        df.loc[schedIsNotNull, ['sub_section_id', 'sub_section']] = None

        df.loc[schedIsNotNull, 'schedule'] = df[schedIsNotNull].schedule.map(lambda x: join_seperated_letter(x) if x else x)
        df.loc[schedIsNotNull, 'part'] = df[schedIsNotNull].part.map(lambda x: join_seperated_letter(x) if x else x)
        df = df.replace({np.nan: None, 'nan': None, 'None': None, '': None})
        # forward fill all empty cell in 'schedule_id', 'schedule', 'part_id', 'part' to indicate the in-between empty cell are under the same schedule/part id
        df.loc[schedIsNotNull, ['schedule_id', 'schedule']] = df[schedIsNotNull][['schedule_id', 'schedule']].ffill()
        if df[~df.schedule_id.isnull()].empty:
            logger.warning(f'filename: {filename}, Schedule heading couldn\'t be found or recognized. Please check if the schedule heading is missing or being converted into image when convert to PDF.')
            return {
                'success': False,
                'errorMessage': 'Schedule heading couldn\'t be found or recognized. Please check if the schedule heading is missing or being converted into image when convert to PDF.'
            }
        df.loc[schedIsNotNull & ~df.schedule_id.isnull(), ['part_id', 'part']] = df[schedIsNotNull & ~df.schedule_id.isnull()].groupby(['schedule_id'])[['part_id', 'part']].transform(lambda x: x.ffill())
        schedIsNull = df.schedule_id.isna()

        # extract section keys from non-section elements
        backslash_char = "\\"
        pos_lookbehind = rf'(?!{")(?!".join(NLA_NUM_DOT+[rf"{backslash_char}."+i for i in NLA_NUM_DOT])})'
        sub_section_pattern = rf'^([1-9]+[a-zA-Z]*)\.({pos_lookbehind}\d+\.*)\s*.*'

        df.loc[df.text_element.str.contains('^list|^sub_section|^caption|^paragraph|^table') & (~tableCotent_id | ~parties_clause_id), 'section_id'] = df[df.text_element.str.contains('^list|^sub_section|^caption|^paragraph|^table') & (~tableCotent_id | ~parties_clause_id)]['text'].map(lambda x: re.match(sub_section_pattern, str(x)).groups()[0] if re.match(sub_section_pattern, str(x)) else None)
        df.loc[df.text_element.str.contains('^list|^caption|^paragraph|^table') | (~tableCotent_id & ~parties_clause_id), 'sub_section_id'] = df[df.text_element.str.contains('^list|^sub_section|^caption|^paragraph|^table') & (~tableCotent_id | ~parties_clause_id)]['text'].map(lambda x: re.match(sub_section_pattern, str(x)).groups()[1] if re.match(sub_section_pattern, str(x)) else None)
        df.loc[df.section_id.notnull(), 'text_element'] = 'section'
        df.loc[df.sub_section_id.notnull(), 'text_element'] = 'sub_section'
        df.loc[df.text_element.eq('section') & df.text.str.contains(sub_section_pattern), 'text_element'] = 'sub_section'
        df = df.replace({np.nan: None, 'nan': None, 'None': None})

        isSec = df.text_element.eq('section')
        isSubSec = df.text_element.str.contains('^sub_section|^sub_sub_section')

        # extract section keys from section, sub-section elements
        df.loc[(isSec | isSubSec), 'sub_section_id'] = df[(isSec | isSubSec)].apply(lambda x: split_section_id(x)[1], axis=1).astype(str)
        df.loc[(isSec | isSubSec), 'sub_section'] = df[(isSec | isSubSec)].apply(lambda x: split_section_id(x)[5], axis=1)
        # to extract sub-section pattern with XXX. YYYY as XXX only
        df.loc[(isSec | isSubSec), 'sub_section'] = df[(isSec | isSubSec)]['sub_section'].map(lambda x: phrase_tokenize(x, nlp)[0].rstrip('.') if x and isinstance(phrase_tokenize(x, nlp), list) else x)
        # extract and map section ids from section text
        df.loc[(isSec | isSubSec) & (~tableCotent_id | ~parties_clause_id), 'section_id'] = df[(isSec | isSubSec) & (~tableCotent_id | ~parties_clause_id)].apply(lambda x: split_section_id(x)[0], axis=1)
        df.loc[(isSec | isSubSec) & (~tableCotent_id | ~parties_clause_id), 'section'] = df[(isSec | isSubSec) & (~tableCotent_id | ~parties_clause_id)].apply(lambda x: split_section_id(x)[4], axis=1)

        # parties_clause_id
        df.loc[df['index'].eq(partiesBeginID), 'section_id'] = '0'
        df.loc[df['index'].eq(partiesBeginID), 'section'] = 'PARTIES'
        df.loc[tableCotent_id, 'section'] = 'TABLE OF CONTENTS'
        df.loc[tableCotent_id, ['section_id', 'identifier']] = None

        df = df.replace({np.nan: None, 'nan': None, 'None': None, '': None})
        df.loc[schedIsNotNull & ~df.schedule_id.isnull(), ['section_id', 'section']] = df[schedIsNotNull & ~df.schedule_id.isnull()].groupby(['schedule_id'])[['section_id', 'section']].transform(lambda x: x.ffill())
        df.loc[schedIsNull & (~tableCotent_id | ~parties_clause_id), 'section_id'] = df[schedIsNull & (~tableCotent_id | ~parties_clause_id)]['section_id'].ffill()

        df.loc[df.section.notnull(), 'section'] = df[df.section.notnull()]['section'].map(lambda x: x if str(x).isupper() or len(str(x).split()) <= 10 or is_all_words_capitalized(str(x), nlp) else None)
        df.loc[schedIsNull & df['text'].str.contains(r'^SECTION \d+', na=False, case=False), 'section_id'] = None

        df.loc[schedIsNull & (~tableCotent_id | ~parties_clause_id) & ~df.section_id.isnull(), 'section'] = df[schedIsNull & (~tableCotent_id | ~parties_clause_id) & ~df.section_id.isnull()].groupby(['section_id'])['section'].transform(lambda x: x.ffill())
        # replace all string with [] or () in sub_section keys, e.g. 'Quotation Day ["Quotation Day"]' become 'Quotation Day'
        df.loc[~df.sub_section.isnull(), 'sub_section'] = df[~df.sub_section.isnull()]['sub_section'].map(lambda x: re.sub(r'[\[|\(].*[\]|\)]', '', x))
        df.loc[~df.sub_section.isnull(), 'sub_section'] = df[~df.sub_section.isnull()]['sub_section'].map(lambda x: re.sub(r'[\[|\(].*', '', x))
        df.loc[~df.sub_section.isnull(), 'section'] = df[~df.sub_section.isnull()]['section'].map(lambda x: re.sub(r'[\[|\(\]|\)]', '', str(x)).strip() if x and x != np.nan else x)

        df.loc[df.sub_section.notnull(), 'sub_section'] = df[df.sub_section.notnull()]['sub_section'].map(lambda x: x if str(x).isupper() or len(str(x).split()) <= 10 or is_all_words_capitalized(str(x), nlp) else None)

        df = df.replace({np.nan: None, 'nan': None, 'None': None})
        df.loc[schedIsNotNull & ~df.schedule_id.isnull(), ['sub_section_id', 'sub_section']] = df[schedIsNotNull & ~df.schedule_id.isnull()].groupby(['schedule_id'])[['sub_section_id', 'sub_section']].transform(lambda x: x.ffill())
        df.loc[schedIsNull & (~tableCotent_id | ~parties_clause_id) & ~df.section_id.isnull(), ['sub_section_id', 'sub_section']] = df[schedIsNull & (~tableCotent_id | ~parties_clause_id) & ~df.section_id.isnull()].groupby(['section_id'])[['sub_section_id', 'sub_section']].transform(lambda x: x.ffill())
        inBetweenSec = df.text_element.shift(-1).eq('sub_section') & df.text_element.shift(1).eq('section')
        df.loc[isSec | inBetweenSec, ['sub_section_id', 'sub_section']] = None

        # split definition term from text if it is under section 'Definitions', column 'definition' to keep definition term and 'text' keep the meaning
        secHasDef = df['section'].str.contains('Definition|Interpretation', na=False, case=False)
        subsecHasDef = df['sub_section'].str.contains('Definition|Interpretation', na=False, case=False)
        
        df.loc[secHasDef | subsecHasDef, 'text'] = df[secHasDef | subsecHasDef]['text'].map(lambda x: re.sub(r'^" ', '"', re.sub(r'\. " ', '. "', re.sub(r'(?<!\.) " ', '"', x) if x else x) if x else x) if x else x)
        df.loc[secHasDef | subsecHasDef, 'text'] = df[secHasDef | subsecHasDef]['text'].map(lambda x: phrase_tokenize(x, nlp, delimiter_pattern=PARA_SPLIT_PATTERN) if x else x)
        df = df.explode('text')
        df.loc[secHasDef | subsecHasDef, 'definition'] = df[secHasDef | subsecHasDef]['text'].map(lambda x: extract_definition(x)[0])
        df.loc[df.text.str.contains('"(.*)" means.*',na=False), 'definition'] = df[df.text.str.contains('"(.*)" means.*',na=False)]['text'].str.extract('"(.*)" means.*', expand=False).map(lambda x: x.strip())

        df.loc[(secHasDef | subsecHasDef) & ~df.section_id.isnull() & ~df.sub_section_id.isnull(), 'definition'] = df[(secHasDef | subsecHasDef) & ~df.section_id.isnull() & ~df.sub_section_id.isnull()].groupby(['section_id', 'sub_section_id'])['definition'].transform(lambda x: x.ffill())
        df = df.replace({np.nan: None})
        df['identifier'] = df.apply(get_identifier, axis=1)
        textBeginWithSection = df['text'].str.contains(SECTION_PATTERN, na=False, case=True)
        df.loc[textBeginWithSection & (~tableCotent_id | ~parties_clause_id), ['section_id', 'section', 'identifier']] = None
        df['index'] = (df.text_element!=df.text_element.shift()).cumsum()-1

    # else document type is term sheet
    else:
        df = df.reset_index(drop=True)
        # fix those section equal ":" and replace by previous section
        if (df['section'].astype(str).str.strip().eq(":")).any():
            df['next_section'] = df['section'].shift(-1)
            for index in range(len(df)):
                next_sec_str = str(df.loc[index, 'next_section']).strip()
                curr_sec_str = str(df.loc[index, 'section']).strip()
                if next_sec_str == ':' and curr_sec_str != ':':
                    df.loc[index, 'section'] = curr_sec_str + ' ' + next_sec_str
                if curr_sec_str == ':' and index-1 >= 0:
                    df.loc[index, 'section'] = str(df.loc[index-1, 'section'])
        
        df['section'] = df['section'].astype(str)
        PREPOSITION_LIST = ['about', 'above', 'across', 'after', 'along', 'among', 'at', 'before', 'below', 'between', 'by', 'dated', 'down', 'during', 'for', 'from', 'in', 'inside', 'into', 'of', 'on', 'onto', 'outside', 'over', 'since', 'through', 'to', 'towards', 'under', 'until', 'up', 'with','Name','Signature','Accepted by']
        
        # remove section that occurs in multi-header layout TS with repeated header "Items"
        df.loc[df.text.str.contains('^Items$',case=False,na=False), 'section'] = None
        df.loc[df.text.str.contains('^Items$',case=False,na=False), 'text_element'] = 'reference'
        
        # eliminate those section is a preposition word and mistakenly deemed as section
        df.loc[df.text.str.contains('|'.join([f'^\s*{i}\s*:*$' for i in PREPOSITION_LIST]),case=False,na=False), 'section'] = None
        df.loc[df.text.str.contains('|'.join([f'^\s*{i}\s*:*$' for i in PREPOSITION_LIST]),case=False,na=False), 'text_element'] = 'caption'
        df['section'] = df['section'].astype(str)
        
        # rectify caption with pattern •XXX:YYY as list
        df.loc[df.text_element.str.contains('caption') & df.text.str.contains('|'.join([f'{i}.*:.*' for i in SYMBOLS])), 'text_element'] = 'list'
        
        # split concatenated list text with bullet point symbol •XXX•YYY
        df.loc[df.text_element.str.contains('list') & df.text.str.contains('|'.join([f'{i}.*{i}.*' for i in SYMBOLS])), 'text'] = df[df.text_element.str.contains('list') & df.text.str.contains('|'.join([f'{i}.*{i}.*' for i in SYMBOLS]))]['text'].map(lambda x: phrase_tokenize(x, nlp, delimiter_pattern=rf'({"|".join(SYMBOLS)})') if x else x)
        df = df.explode('text')
        
        # split concatenated section text (e.g XXX. YYY) into different rows of section text by sentence tokenization (split sentences at full-stop)
        df.loc[df.text_element.str.contains('section') & df.text.str.contains('.*\..*'), 'text'] = df[df.text_element.str.contains('section') & df.text.str.contains('.*\..*')]['text'].map(lambda x: phrase_tokenize(x, nlp) if x else x)
        df = df.explode('text')
        
        # If ":" is found in section (e.g. with pattern XXX: zzzz. YYY:), seperate section text into rows, keep first row and key that matches section word bank as section, then rest of successful split are classified as paragraph
        df.loc[df.text_element.str.contains('section') & df.section.str.contains(r'.*(?<!\])(?<!means)(?<!means )(?<!Part \w)(?<!Part \d)\:'), 'text'] = df[df.text_element.str.contains('section') & df.section.str.contains(r'.*(?<!\])(?<!means)(?<!means )(?<!Part \w)(?<!Part \d)\:')]['text'].map(lambda x: split_at_seperators(x))
        df = df.explode('text')
        df = df.groupby(['text_block_id']).apply(lambda x: x.assign(text_element=['paragraph' if (row['text_element']=='section' and not (idx == 0 or re.match(TS_SECTIONS, str(row['text']).strip()))) or re.match('^Tranche', str(row['text']).strip()) else row['text_element'] for idx, (index, row) in enumerate(x.iterrows())], 
                                                                    section=[None if not (row['text_element']=='section' and (idx == 0 or re.match(TS_SECTIONS, str(row['text']).strip()))) or re.match('^Tranche', str(row['text']).strip()) else row['text'] for idx, (index, row) in enumerate(x.iterrows())])).reset_index(drop=True)
        
        df['section'] = df['section'].str.replace(":", "")
        # split section and sub_section from section text and insert new rows if text pattern '1. XXX 1.1 YYY' is found
        df.loc[df.text_element.str.contains('section') & df.text.str.contains(SEC_N_SUBSEC_PATTERN2), 'text'] = df[df.text_element.str.contains('section') & df.text.str.contains(SEC_N_SUBSEC_PATTERN2)].text.map(lambda x: re.findall(SEC_N_SUBSEC_PATTERN2, x)[0])
        df = df.explode('text')
        
        # extraction true section from any text looks like 1.1 XXX as XXX
        SEC_SUBSEC_ID = r'^\d{1,2}[a-zA-Z]*\.\d{0,2}\.*\s*'
        df.loc[df.text_element.str.contains('section') & df.text.str.contains(SEC_SUBSEC_ID), 'section'] = df[df.text_element.str.contains('section') & df.text.str.contains(SEC_SUBSEC_ID)]['text'].str.extract(SEC_SUBSEC_ID+'(.*)', expand=False)

        df = df[df['text'].notna() & ~df['text'].eq('') & ~df['text'].eq(':')]
        df['section'] = df['section'].map(lambda x: str(x).strip().rstrip(':') if x and x != '' else x)
        # extract section as XXX from section with pattern XXX:YYY (but not ends with ]:/ means:/ means :)
        df.loc[df.section.str.contains(r'.*(?<!\])(?<!means)(?<!means )(?<!Part \w)(?<!Part \d)\:') & ~df.section.isna(), 'section'] = df[df.section.str.contains(r'.*(?<!\])(?<!means)(?<!means )(?<!Part \w)(?<!Part \d)\:') & ~df.section.isna()]['section'].str.extract('^(.+)\:.*', expand=False)
        # promote caption as section if string begins with Part XX:
        df.loc[df.text_element.str.contains('caption') & df.text.str.contains('^Part .*:', case=False), 'section'] = df[df.text_element.str.contains('caption') & df.text.str.contains('^Part .*:', case=False)].text
        df.loc[df.section.notnull(), 'section'] = df[df.section.notnull()].section.map(lambda x: join_seperated_letter(x) if x else x)

        # if there exists no section text element after all attempts of section extraction that previously made, the layout might be expressed in list form. Attempt to extract section from list
        # if df.section.notnull().sum() < 1:
        #     df['section'] = df[df.text_element.str.contains('list')]['text'].map(lambda x: re.search(r'^(.*):()', str(x)).group(1).strip() if re.search(r'^(.*):()', str(x)) else None)
        #     # df.loc[df.text_element.str.contains('title'), 'section'] = df[df.text_element.str.contains('title')]['text']
        #     df['section'] = df['section'].ffill()
        #     df['text'] = df['text'].map(lambda x: [i.strip() for i in re.match('(.*:)(.*)', str(x)).groups() if i.strip()] if re.match('(.*:)(.*)', str(x)) else x)
        #     df = df.explode('text').reset_index(drop=True)
        #     df.loc[df.groupby('section')['text_element'].head(1).index, 'text_element'] = 'section'
        #     df.loc[~df.text_element.str.contains('section'), 'text'] = df[~df.text_element.str.contains('section')]['text'].map(lambda x: phrase_tokenize(x, nlp) if x else x)

        # df.loc[~df.text.str.contains(r'.*(?<!\])(?<!means)(?<!means )\:$'),'section'] = df[~df.text.str.contains(r'.*(?<!\])(?<!means)(?<!means )\:$')]['section'].ffill()
        
        # inspect if list text with list_id like 1. (but not ends with ]:/ means:/ means : or starts with • For/ • Tranche/ Note/ [Note/ • WHT xxx Lender) and section doesn't exists among text block, then promote text as section if text with pattern XXX:
        constraint_A = (df.text_element.str.contains('list') &
            (df.list_id.str.contains(r'\d+\.$') &
            df.text.str.contains(r'.*(?<!\])(?<!means)(?<!means )\:*$') | df.text.str.contains(r'^\d+\..*(?<!\])(?<!means)(?<!means )\:*$')) &
            ~df.text.str.contains(r'^•*\s*\bFor\b|^•*\s*Tranche|^Note|^\[Note|^•*\s*WHT.*Lender') &
            (df.text.str.count(' ') + 1 <= 5))
        
        has_no_section_in_block = (df[constraint_A].groupby('text_block_id')['section'].first().isnull())
        constraint_B = (df[constraint_A].text.str.contains(TS_SECTIONS))
            
        df.loc[constraint_A & constraint_B, 'section'] = df[constraint_A & constraint_B]['text'].map(lambda x: str(x).replace(":", ""))
        df.loc[constraint_A & constraint_B, 'text_element'] = 'section'

        # inspect if list text with pattern XXX: YYY (but not ends with ]:/ means:/ means : or starts with • For/ • Tranche/ Note/ [Note/ • WHT xxx Lender) and section doesn't exists among text block, then extract text XXX as section, text YYY as text
        # list out all the constraints
        less_than_seven = (df.text.str.split(':').map(lambda x: len(x[0].split())) <= 7)
        all_capitalized_words = (df.text.map(lambda x: bool(re.match(r'^[A-Z ]+$', str(x)))))
        contains_list = (df.text_element.str.contains('list'))
        not_prev_eq_sect = (~df.text_element.shift(1).eq('section'))

        constraint2 = (df.section.isnull() & ~df.list_id.astype(str).str.contains(r'\)$') & contains_list & not_prev_eq_sect &
                df.text.str.contains(r'•*.*(?<!\])(?<!means)(?<!means )(?<!Part \w)(?<!Part \d)\:.*\:*.*') &
            ~df.text.str.contains(r'^•*\s*\bFor\b|^•*\s*Tranche|^Note|^\[Note|^•*\s*WHT.*Lender') & less_than_seven)
        
        has_no_section_in_block2 = (df[constraint2].groupby('text_block_id')['section'].first().isnull())
        is_match_section_wordbank = (df[constraint2].text.map(lambda x: bool(re.match(TS_SECTIONS, re.findall('^•*(.*):.*\:*.*', str(x))[0].strip()))))
        
        df.loc[contains_list & all_capitalized_words, 'section'] = df[contains_list & all_capitalized_words]['text']
        df.loc[constraint2 & is_match_section_wordbank, 'section'] = df[constraint2 & is_match_section_wordbank]['text'].str.extract('^•*\s*(.*):.*\:*.*', expand=False)
        df.loc[constraint2 & is_match_section_wordbank, 'text'] = df[constraint2 & is_match_section_wordbank]['text'].str.extract('^•*\s*.*\:(.*):*.*', expand=False).map(lambda x: str(x).strip())

        df['section'] = df['section'].ffill()
        df = df.replace({np.nan: None, 'nan': None, 'None': None})
        
        df['index'] = (df.text_element!=df.text_element.shift()).cumsum()-1
        df['text_block_id'] = (df.section!=df.section.shift()).cumsum()-1
        
        # Replace section ordinal number to words, e.g. 1st -> First, 2nd -> Second
        df.loc[df.section.notnull(), 'section'] = df[df.section.notnull()]['section'].map(replace_ordinal_numbers)
        df.loc[df.section.notnull(), 'section'] = df[df.section.notnull()]['section'].map(remove_non_alphabet)
        df.loc[df.section.isin(SECTION_SPLIT_AT_COMMA),'text'] = df.loc[df.section.isin(SECTION_SPLIT_AT_COMMA)]['text'].str.split(',')

    # replace None after all operations
    df = df.replace({np.nan: None})
    # explode list items in column text into different rows
    df = df.explode('text').reset_index(drop=True)
    # filter out null or empty text
    df['text'] = df['text'].map(lambda x: x.lstrip(':').lstrip('•').strip() if x else x)
    df = df[~(df['text'].isnull()) & ~(df['text'].str.len() == 0) & ~(df['text'] == ' ')] 
    df = df[~df.text.str.contains('^\W+$', na=False)].reset_index(drop=True)
    # trace back with child list id and provide parent list id and content
    df_parent_list = df.groupby(['text_block_id'], dropna=False).apply(get_parent_list_item).reset_index(drop=True)
    df = pd.concat([df, df_parent_list], axis=1)
    # trace back to get parent caption
    df['prev_text_element'] = df['text_element'].shift()
    df.loc[((df['prev_text_element'].str.contains('caption'))==True) & (df.section==df.section.shift()), 'parent_caption'] = df.text.shift()
    df['parent_caption'] = df.groupby(['index','text_block_id'])['parent_caption'].transform('first')
    df['phrase_id'] = df.groupby(['index']).cumcount()

    if doc_type == 'TS':
        df['previous_text'] = df.text.shift(1)
        target_cols = ['index', 'text_block_id', 'page_id', 'phrase_id', 'section', 'text_element', 'list_id', 'parent_caption', 'parent_list_ids', 'parent_list', 'text']
        df = df[target_cols]
        df = df.sort_values(by=['index','phrase_id'])
    else:
        df[['section_id','schedule_id','part_id']] = df[['section_id','schedule_id','part_id']].map(lambda x: str(int(x)) if isinstance(x,float) else str(x))
        df['sub_section_id'] = df['sub_section_id'].astype(str)
        # reorder columns for FA
        df = df[['index', 'text_block_id', 'phrase_id','page_id', 'bbox',
                 'section_id', 'section', 'sub_section_id', 'sub_section',
                 'schedule_id', 'schedule', 'part_id', 'part', 'text_element',
                 'list_id', 'definition', 'identifier', 'text', 'parent_caption', 'parent_list_ids', 'parent_list']]

    df = df.replace({np.nan: None, 'nan': None, 'None': None, '': None})
    df['text_block_id'] = (df.section!=df.section.shift()).cumsum()-1
    
    # df.loc[df['text_element'].str.contains('table'), 'text'] = df[df['text_element'].str.contains('table')]['text'].map(lambda x: tryfunction(x, ast.literal_eval) if isinstance(x, str) and x.startswith('[') and x.endswith(']') else x)
    if DEV_MODE:
        df.to_csv(outpath, index=False, encoding='utf-8-sig')
        logger.info(f'Term extraction CSV is output into {outpath}')

    df['page_id'] = df['page_id'].astype(str)

    df = df.rename(columns={
        'index': 'indexId',
        'text_block_id': 'textBlockId',
        'page_id': 'pageId',
        'phrase_id': 'phraseId',
        'section_id': 'sectionId',
        'section': 'sectionContent',
        'sub_section_id': 'subSectionId',
        'sub_section': 'subSection',
        'schedule_id': 'scheduleId',
        'part_id': 'partId',
        'text_element': 'textElement',
        'list_id': 'listId',
        'text': 'textContent',
        'parent_list_ids': 'parentListIds',
        'parent_caption': 'parentCaption',
        'parent_list': 'parentList' 
    })

    return df.to_dict('records')


def batch_docparse2table(input_list: Iterable[Iterable[Union[str, dict], str]]):
    all_df_dict = []
    for input_tuple in input_list:
        df_dict = docparse2table(input_tuple)
        all_df_dict.append(df_dict)
    return all_df_dict


if __name__ == "__main__":
    import os
    from tqdm import tqdm
    import argparse
    import warnings
    import sys
    import logging
    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
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

    doc_type = 'FA'

    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--json_dir",
        default=DOCPARSE_OUTPUT_JSON_DIR,
        type=str,
        required=False,
        help="The document parsed JSON directory",
    )
    parser.add_argument(
        "--doc_type",
        default=doc_type,
        type=str,
        required=False,
        help="The document type for term extraction. TS for term sheet; FA for facility agreement",
    )
    parser.add_argument(
        "--target_json_list",
        default=None,
        type=list,
        required=False,
        help="Target list of JSON filename in json_dir",
    )
    args = parser.parse_args()
    key1 = ['TS', 'term sheet']
    key2 = ['FA', 'facility agreement', 'facilities agreement']
    # search files in all sub folders under the target docparse result folder
    fpath_typeTS = []
    fpath_typeFA = []

    for dirpath, subdirs, files in tqdm(os.walk(args.json_dir)):
        files.sort()

        for x in files:
            fpath = os.path.join(dirpath, x)
            fname = os.path.basename(fpath)
            # search 'TS' or 'FA' from filename to distinguish the result doc type
            if re.match(r'.*' + r'.*|.*'.join(key1), fname, flags=re.IGNORECASE):
                doc_type = 'TS'
            elif re.match(r'.*' + r'.*|.*'.join(key2), fname, flags=re.IGNORECASE):
                doc_type = 'FA'
            else:
                doc_type = ''
            # consider file only with '.json' file extension
            if args.target_json_list:
                if fname.endswith('.json') and doc_type == 'FA' and fname in args.target_json_list:
                    fpath_typeFA.append((fpath, doc_type))
                elif fname.endswith('.json') and doc_type == 'TS' and fname in args.target_json_list:
                    fpath_typeTS.append((fpath, doc_type))
            else:
                if fname.endswith('.json') and doc_type == 'FA':
                    fpath_typeFA.append((fpath, doc_type))
                elif fname.endswith('.json') and doc_type == 'TS':
                    fpath_typeTS.append((fpath, doc_type))

    if args.doc_type == 'FA':
        # for fpath, doc_type in fpath_typeFA:
        #     docparse2table((fpath, doc_type))
        multiprocess(batch_docparse2table, fpath_typeFA)
    elif args.doc_type == 'TS':
        # for fpath, doc_type in fpath_typeTS:
        #     docparse2table((fpath, doc_type))
        multiprocess(batch_docparse2table, fpath_typeTS)
