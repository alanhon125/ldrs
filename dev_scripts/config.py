import re
import string

PROJ_ROOT_DIR = '/home/data/ldrs_analytics'
DATA_DIR = '/home/data/ldrs_analytics/data'
# path to docparse results in JSON of TS and FA
DOCPARSE_OUTPUT_JSON_DIR = f'{PROJ_ROOT_DIR}/data/docparse_json/'
# path to output the results in CSV
OUTPUT_DOCPARSE_CSV = f'{PROJ_ROOT_DIR}/data/docparse_csv/'
# path to output the log in CSV
LOG_DIR = f'{PROJ_ROOT_DIR}/data/log/'
# path to folder of annotated docparse TS in CSV
ANTD_TS_CSV = f'{DATA_DIR}/reviewed_antd_ts/bfilled/'
# path to folder of docparse TS in CSV
TS_CSV = f'{DATA_DIR}/docparse_csv/TS/'
# path to folder of docparse FA in CSV
FA_CSV = f'{DATA_DIR}/docparse_csv/FA/'
MATCHED_FA_CSV = f'{DATA_DIR}/term_matching_csv/20230925_tm7_wb_finetune_top10/'
# path to folder of storing merge or match table
OUTPUT_MERGE_CSV_DIR = f'{DATA_DIR}/antd_ts_merge_fa/merge_new/'
OUTPUT_MERGE_MATCH_CSV_DIR = f'{DATA_DIR}/antd_ts_merge_fa/20230925_tm7_wb_finetune_top10/'
OUTPUT_MERGE_MATCH_CSV_DIR2 = f'{DATA_DIR}/antd_ts_merge_fa/merge&internalMatch/'
# 'sentence-transformers/all-MiniLM-L6-v2'
SENTBERT_MODEL_PATH = '/home/data/ldrs_analytics/models/all-MiniLM-L6-v2'

# Regular Expression for information retrievel and string checking in keys and clauses extraction
MEANING_IN_QUOTE = r'.*"\s*(.*)\s*".*'

ORDINAL2NUM = {'first': 1, 'second': 2, 'third': 3, 'fourth': 4, 'fifth': 5, 'sixth': 6, 'seventh': 7, 'eighth': 8,
               'ninth': 9, 'tenth': 10}
CAP_SIMPLE_ROMAN_PATTERN = r'[MDCLXVI]+'
PUNCTUATION = r'\!\#\$\%\&\'\(\)\*\+\,\/\:\;\<\=\>\?\@\[\\\]\^\_\`\~\.\s\d'
SEC_SUBSEC_PATTERN = r'(^\d{1,2}[a-zA-Z]*)\s*'+rf'([^{PUNCTUATION}].*)\n*' + '|' + \
    r'(^\d{1,2}[a-zA-Z]*)\.(\d{0,2}\.*\d{0,2}\.*\d{0,2})\s*' + \
    rf'([^{PUNCTUATION}].*)\n*'
SECTION_PATTERN = r'^SECTION\s*(\d*[a-zA-Z]*)(.*)(\d+[a-zA-Z]*)\.*(.*)\n*'
SCHED_DIGIT_PATTERN = r'^S\s*CHEDULE\s*(\d+)\s*(.*)'
SCHED_ROMAN_PATTERN = rf'^S\s*CHEDULE\s*({CAP_SIMPLE_ROMAN_PATTERN})\s*(.*)'
SCHED_ORDINAL_PATTERN = rf'^THE ({"|".join(ORDINAL2NUM.keys())}) SCHEDULE [a-zA-Z]+ REFERRED TO\s*(.*)'
SCHED_ORDINAL_PATTERN2 = rf'THE ({"|".join(ORDINAL2NUM.keys())}) SCHEDULE:*\s*-*(.*)'
PART_DIGIT_PATTERN = r'^PART\s(\d+):*\s*-*(.*)'
PART_ROMAN_PATTERN = rf'^PART\s({CAP_SIMPLE_ROMAN_PATTERN}):*\s*-*(.*)'
PART_ALPHA_PATTERN = r'^PART\s([a-zA-Z]):*\s*-*(.*)'

TABLECONTENT_BEGIN_PATTERN = r'CONTEN\s*T[S]*|TABLE OF CONTENT[S]*|INDEX'
PARTIES_BEGIN_PATTERN = r'^THIS AGREEMENT is|is made on|^PARTIES|BETWEEN\s*:*'
PARTIES_END_PATTERN = r'IT IS AGREED*:*|AGREED* as follows|AGREED* that'
TABLECONTENT_SCHED_1_PATTERN = r'.*Schedule[s]*\s*1\.*:*\s([A-Za-z\s]*)\.+.*'
TABLECONTENT_SCHED_1_PATTERN2 = r'THE FIRST SCHEDULE*:*\s[HEREINBEFORE REFERRED TO ]*([A-Za-z\s]*)\.+.*'
EXCLUD_STR_IN_SCHED_TITLE = r'S\s*CHEDULE[S]*\s*[2|II]\b|^S\s*CHEDULE[S]*\s*[2|II]\D+|^THE SECOND SCHEDULE'

# Regular Expression for information retrievel and string checking in TS merge (or match) FA
ROMAN_NUMERALS = r'''
                    ^M{0,3}
                    (CM|CD|D?C{0,3})?
                    (XC|XL|L?X{0,3})?
                    (IX|IV|V?I{0,3})?$
                '''
CLAUSE_SEC_LIST = r"(\d{1,2}[a-zA-Z]+|\d{1,2}|[a-zA-Z]+)(\d{1,2}\(.*\)|\(.*\)|\d{1,2}\)*)*"
# 1.1.2_3(a) -> section_id =1, sub_section_id =1.2, list_id = 3(a)
CLAUSE_SEC_SUBSEC_LIST = r"(\d{1,2}[a-zA-Z]+|\d{1,2}|[a-zA-Z]+)\.(\d{1,2}\.*\d*|[a-zA-Z])*(\d{1,2}\(.*\)|\(.*\)|\d{1,2}\)*)*"
# 1.A_2_3(a) -> schedule_id =1, part_id =A, section_id=2, list_id = 3(a)
SCHED_SEC_LIST = r"(\d{1,2}|\d{1,2}[a-zA-Z]+|[a-zA-Z]+)\.*(\d{1,2}\.*\d*|\d{1,2}\.*[a-zA-Z])*_*(\d{1,2}\(.*\)|\(.*\)|\d{1,2}\)*)*"
neg_lookahead_punct = rf'(?!{")(?!".join([re.escape(i) for i in string.punctuation])})'
ARTICLES = rf'\b(the|The|a{neg_lookahead_punct}|an{neg_lookahead_punct}|An)\b|\.'
TRAILING_DOT_ZERO = r'\.0$'  # 1.0, 2.5.0 etc.

PARTIES_CLAUSE_PATTERN = r'^Parties-.*$'
CLAUSE_PATTERN = r'^Cl_\d+\..*$'
CLAUSE_SECTION_PATTERN = r'^Cl_\d+$'
SCHED_PATTERN = r'^Sched_\d+$'
SCHED_PART_PATTERN = r'^Sched_\d+\.[\d+|\w]$'
PARAG_SCHED_PATTERN = r'^Sched_\d+.*_.*$|^Sched_\d+.*_.*_.*$'
DEF_CLAUSE_PATTERN = r'^Cl_\d+\.\d+-.*$'

SCHED_TITLE = r"^S\s*CHEDULE[S]*\s*[1|I]\b|^S\s*CHEDULE[S]*\s*[1|I]\D+|^THE FIRST SCHEDULE|^THE ORIGINAL LENDERS$|^THE PARTIES$|^PART A|^EXECUTION:-*\s*S\s*CHEDULE[S]*\s*[1|I]"
EXCLUD_STR_IN_SCHED_TITLE = r"S\s*CHEDULE[S]*\s*[2|II]\b|^S\s*CHEDULE[S]*\s*[2|II]\D+|^THE SECOND SCHEDULE"

PARTIES_LIST = []
