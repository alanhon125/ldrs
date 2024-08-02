import inflect
import re
import string
import csv
import os

FILE_ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
# reading TS section word bank from a csv file 'TS_section.csv'
TS_SECTION_BANK_CSV = os.path.join(FILE_ROOT_DIR, 'TS_section.csv')
with open(TS_SECTION_BANK_CSV, newline='') as file:
    reader = csv.reader(file, delimiter=',')
    TS_SECTIONS = []
    for row in reader:
        TS_SECTIONS.append(row[0])
TS_SECTIONS_LIST = sorted([re.escape(i) for i in TS_SECTIONS], key=len, reverse=True)
TS_SECTIONS = '^\d*\.*\s*' + r'\s*:*\s*$|^\d*\.*\s*'.join(TS_SECTIONS_LIST) + '\s*:*\s*$'

# Character blocks
DIACRITICS = r'[\u0300-\u036F]+'
FULLWIDTH_ASCII_VARIANTS = r'[\uff01-\uff5e]+'
CJK = r'[\u4e00-\u9fff]+'

# Regular Expression for sentence or phrase splits on paragraph in document parsing
COMMA_SPLIT_PATTERN = r'((?<!\d{3}),(?! \bwhich\b)(?! \band\b)(?! \bincluding\b)(?! etc.)|(?<!\blaws\b) and | or (?!\botherwise\b)|not limited to)'
PARA_SPLIT_PATTERN = r'( and/or;| ; and/or|; and/or|;and/or|; and then| and;| ; and|; and|;and| or;| ; or|; or|;or|; plus|;plus|;)'
SECTION_SPLIT_AT_COMMA = ["Documentation", "Amendments and Waivers", "Miscellaneous Provisions", "Other Terms", "Undertakings and Other Terms", "Other Terms and Conditions"]
SECTIONS_CONDITIONAL_SPLIT = ["Documentation","Undertakings", "General Undertaking", "Other Covenants & Undertaking", "Other Undertakings", "General Covenants", "Covenants","Covenants / Undertakings","Covenants and Undertakings","Representations", "Events of Default"]

# Regular Expression for information retrievel and string checking in keys and clauses extraction
MEANING_IN_QUOTE = r'.*"\s*(.*)\s*".*'
TERMS_IN_QUOTE = r'["|“]([^"“”]*)["|”]'
ORDINAL2NUM = {'first': 1, 'second': 2, 'third': 3, 'fourth': 4, 'fifth': 5, 'sixth': 6, 'seventh': 7, 'eighth': 8,
               'ninth': 9, 'tenth': 10}
CAP_SIMPLE_ROMAN_PATTERN = r'[MDCLXVI]+'
PUNCTUATION = r'\!\#\$\%\&\'\(\)\+\,\/\:\;\<\=\>\?\@\\\]\^\_\`\~\.\s\d'
# text pattern like '1. XXX 1.1 YYY'
SEC_N_SUBSEC_PATTERN = r'(^\d{1,2}[a-zA-Z]*\.\D+)' + r' (\d{1,2}[a-zA-Z]*\.\d{0,2}\.*\d{0,2}\.*\d{0,2}\s*.*)'
SEC_N_SUBSEC_PATTERN2 = r'(^\d{1,2}[a-zA-Z]*\.\D+)' + r'((?<!Clause )(?<!clause )\d{1,2}[a-zA-Z]*\.\d{0,2}\.*\d{0,2}\.*\d{0,2}\s*.*)'  # text pattern like '1. XXX 1.1 YYY'
SEC_SUBSEC_PATTERN = r'(^\d{1,2}[a-zA-Z]+)\s*' + rf'([^{PUNCTUATION}].*)|' + \
    r'(^\d{1,2}[a-zA-Z]+)\.(\d{0,2}\.*\d{0,2}\.*\d{0,2})\s*' + rf'([^{PUNCTUATION}].*)|' +\
    r'(^\d{1,2})\s*' + rf'([^{PUNCTUATION}].*)|' + \
    r'(^\d{1,2})\.(\d{0,2}\.*\d{0,2}\.*\d{0,2})\s*' + \
    rf'([^{PUNCTUATION}].*)'
SEC_SUBSEC_ID_PATTERN = r'(^\d{1,2}[a-zA-Z]+)\s*$|' + \
    r'(^\d{1,2}[a-zA-Z]+)\.(\d{0,2}\.*\d{0,2}\.*\d{0,2})\s*$|' + \
    r'(^\d{1,2})\s*$|' + \
    r'(^\d{1,2})\.(\d{0,2}\.*\d{0,2}\.*\d{0,2})\s*$'
SECTION_PATTERN = r'^SECTION\s*(\d*[a-zA-Z]*)(.*)(\d+[a-zA-Z]*)\.*(.*)\n*'
SCHED_DIGIT_PATTERN = r'^S\s*CHEDULE\s*(\d+)\s*(.*)'
SCHED_ROMAN_PATTERN = rf'^S\s*CHEDULE\s*({CAP_SIMPLE_ROMAN_PATTERN})\s*(.*)'
SCHED_ORDINAL_PATTERN = rf'^THE ({"|".join(ORDINAL2NUM.keys())}) SCHEDULE [a-zA-Z]+ REFERRED TO\s*(.*)'
SCHED_ORDINAL_PATTERN2 = rf'THE ({"|".join(ORDINAL2NUM.keys())}) SCHEDULE:*\s*-*(.*)'
PART_DIGIT_PATTERN = r'^P\s*ART\s(\d+):*\s*-*(.*)'
PART_ROMAN_PATTERN = rf'^P\s*ART\s({CAP_SIMPLE_ROMAN_PATTERN}):*\s*-*(.*)'
PART_ALPHA_PATTERN = r'^P\s*ART\s([a-zA-Z]):*\s*-*(.*)'

TABLECONTENT_BEGIN_PATTERN = r'C\s*O\s*N\s*T\s*E\s*N\s*T\s*[S]*|TABLE OF CONTENT[S]*|I\s*N\s*D\s*E\s*X'
PARTIES_BEGIN_PATTERN = r'^THIS AGREEMENT is|is made on|^THIS AGREEMENT Between|is dated .* and made between|^PARTIES|^BETWEEN\s*:*|^AMONG:|^\"Borrower\"\)|^Date: Parties'  # |.*Parties$'
PARTIES_END_PATTERN = r'IT IS AGREED*:*|AGREED* as follows|AGREED* that|The parties agreed*(?!,)'
TABLECONTENT_SCHED_1_PATTERN = r'.*Schedule[s]*\s*1\.*:*\s([A-Za-z\s]*)\.+.*'
TABLECONTENT_SCHED_1_PATTERN2 = r'THE FIRST SCHEDULE*:*\s[HEREINBEFORE REFERRED TO ]*([A-Za-z\s]*)\.+.*'
EXCLUD_STR_IN_SCHED_TITLE = r'S\s*CHEDULE[S]*\s*[2|II]\b|^S\s*CHEDULE[S]*\s*[2|II]\D+|^THE SECOND SCHEDULE'

# Regular Expression for information retrievel and string checking in TS merge (or match) FA
ROMAN_NUMERALS = r"^M{0,3}(CM|CD|D?C{0,3})?(XC|XL|L?X{0,3})?(IX|IV|V?I{0,3})?$"
CLAUSE_SEC_LIST = r"(\d{1,2}[a-zA-Z]+|\d{1,2}|[a-zA-Z]+)(\d{1,2}\(.*\)|\(.*\)|\d{1,2}\)*)*"
# 1.1.2_3(a) -> section_id =1, sub_section_id =1.2, list_id = 3(a)
CLAUSE_SEC_SUBSEC_LIST = r"(\d{1,2}[a-zA-Z]+|\d{1,2}|[a-zA-Z]+)\.(\d{1,2}\.*\d*\.*|[a-zA-Z])*(\d{1,2}\(.*\)|\(.*\)|\d{1,2}\)*)*"
# 1.A_2_3(a) -> schedule_id =1, part_id =A, section_id=2, list_id = 3(a)
SCHED_SEC_LIST = r"(\d{1,2}[a-zA-Z]+|\d{1,2}|[a-zA-Z]+)\.*(\d{1,2}\.*\d*\.*|\d{1,2}\.*[a-zA-Z])*_(\d{1,2}\(.*\)|\(.*\)|\d{1,2}\)*)*"
neg_lookahead_punct = rf"(?!{')(?!'.join([re.escape(i) for i in string.punctuation])})"
ARTICLES = rf'\b(the|The|a{neg_lookahead_punct}|an{neg_lookahead_punct}|An)\b|\.'
TRAILING_DOT_ZERO = r'\.0$'  # 1.0, 2.5.0 etc.

PARTIES_CLAUSE_PATTERN = r'^Parties-.*$'
CLAUSE_PATTERN = r'^Cl_\d+\..*$'
CLAUSE_SECTION_PATTERN = r'^Cl_\d+$'
SCHED_PATTERN = r'^Sched_\d+$'
SCHED_PART_PATTERN = r'^Sched_\d+\.[\d+|\w]$'
PARAG_SCHED_PATTERN = r'^Sched_\d+\.[\d+|\w]_.*$'
DEF_CLAUSE_PATTERN = r'^Cl_\d+\.\d+-.*$'

SCHED_TITLE = r"^S\s*CHEDULE[S]*\s*[1|I]\b|^S\s*CHEDULE[S]*\s*[1|I]\D+|^THE FIRST SCHEDULE|^THE ORIGINAL LENDERS$|^THE PARTIES$|^PART A|^EXECUTION:-*\s*S\s*CHEDULE[S]*\s*[1|I]"
EXCLUD_STR_IN_SCHED_TITLE = r"S\s*CHEDULE[S]*\s*[2|II]\b|^S\s*CHEDULE[S]*\s*[2|II]\D+|^THE SECOND SCHEDULE"

# Parties Extraction
PARTIES_PATTERN_TGT = r'.*\(.*[together|collectively],*.*the (.*);*.*\)'
PARTIES_PATTERN_TGT_2 = r'.*\(.*[together|collectively],*.* (.*);*.*\)'
PARTIES_PATTERN_BRACK = r'.*\((.*)\).*'
PARTIES_PATTERN_AS = r'.* as (?!of)\s*(.*)\s*(?!of)'
PARTIES_PATTERN_AS_OF_BRACK = r'.* as (?!of)\s*(.*) of.*\('
PARTIES_PATTERN_AS_BRACK = r'.* as (?!of)\s*(.*)\s*(?!of).*\('

# Definition Extraction
DEF_KEYS = [
    'means,',
    '" means',
    'means (?!of)',
    'means:',
    'has the meaning given to',
    'shall be construed as',
    'shall have the meaning given to',
    'is a reference to',
    'includes a reference to',
    'denote',
    'denotes',
    'has the same meaning',
    '".*"'
]  # (?!(?:(arithmetic))\b)
DEF_KEYS_2 = ['include', 'includes']
BASEL2_PATTERN = r'^(["“”]Basel II["“”]|["“”]Basel 2["“”]|Basel II [^"“”]*|Basel 2[^"“”]*) means (.*)$'

# Regex for extraction and split numbering from text into key:value pairs

XYZ = r'\b[xyz]\b'
CAPITAL_XYZ = r'\b[XYZ]\b'
CAPITAL_ROMAN_NUM = r"\b(?=[XVI])M*(X[L]|L?X{0,2})(I[XV]|V?I{0,3})\b"
ROMAN_NUM = r"\b(?=[xvi])m*(x[l]|l?x{0,2})(i[xv]|v?i{0,3})\b"
CAPITAL_ALPHABET = r'\b[^\d\sIVXivxa-zXYZ\W_]{1}\b'
LOWER_ALPHABET = r'\b[^\d\sIVXivxA-Zxyz\W_]{1}\b'
TWO_CAPITAL_ALPHABET = r'\b(AA|BB|CC|DD|EE|FF|GG|HH|JJ|KK|LL|MM|NN|OO|PP|QQ|RR|SS|TT|UU|VV|WW|XX|YY|ZZ)\b'
TWO_LOWER_ALPHABET = r'\b(aa|bb|cc|dd|ee|ff|gg|hh|jj|kk|ll|mm|nn|oo|pp|qq|rr|ss|tt|uu|vv|ww|xx|yy|zz)\b'
ONE_DIGIT = r'\b[1-9]\d{0,1}\b'
TWO_DOTS_DIGIT = rf'{ONE_DIGIT}\.{ONE_DIGIT}\.{ONE_DIGIT}'
ONE_DOT_DIGIT = rf'{ONE_DIGIT}\.{ONE_DIGIT}'
DOT_DIGIT = rf'{TWO_DOTS_DIGIT}|{ONE_DOT_DIGIT}'
DELIMITERS = [
    'to :',
    ';'
]
SYMBOLS = ['●', '•', '·', '∙', '◉', '○', '⦿', '。', '■', '□', '☐',
           '⁃', '◆', '◇', '◈', '✦', '➢', '➣', '➤', '‣', '▶', '▷', '❖', '_']
NUMBERING_LIST = [XYZ, CAPITAL_XYZ, CAPITAL_ROMAN_NUM, ROMAN_NUM,
                  CAPITAL_ALPHABET, LOWER_ALPHABET, TWO_CAPITAL_ALPHABET, TWO_LOWER_ALPHABET, TWO_DOTS_DIGIT, ONE_DOT_DIGIT, ONE_DIGIT]
ALPHABET_LIST = ['[a-z]', '[a-z]{2}', '[a-z]{3}', '[a-z]{4}',
                 '[a-z]{5}', '[A-Z]', '[A-Z]{2}', '[A-Z]{3}', '[A-Z]{4}', '[A-Z]{5}']
numWithBrack1 = ['\(' + i + '\)' for i in ALPHABET_LIST]
numWithBrack2 = ['\(' + i + '\) and' for i in ALPHABET_LIST]
numWithBrack3 = ['\(' + i + '\),' for i in ALPHABET_LIST]
numWithBrack4 = ['\(' + i + '\) to' for i in ALPHABET_LIST]
numWithBrack5 = ['\(' + i + '\) or' for i in ALPHABET_LIST]
numWithBrack = numWithBrack2 + numWithBrack3 + \
    numWithBrack4 + numWithBrack5  # numWithBrack1 +
p = inflect.engine()
numInWord = [p.number_to_words(i).capitalize() for i in range(100)] + [p.number_to_words(i) for i in range(100)] + [
    '\['+p.number_to_words(i).capitalize()+'\]' for i in range(100)] + ['\['+p.number_to_words(i)+'\]' for i in range(100)]
PREPOSITIONS = [
    ' above ',
    ' below ',
    ' above,',
    ' below,',
    ' above\.',
    ' below\.',
    ' from ',
    ' than ',
    ' and\\\or '
]

PUNCT_ALL_STR = re.escape(string.punctuation)
PURE_PUNCT_DIGIT = rf'^[{PUNCT_ALL_STR}]+$|^[\d]+$'
PUNCT_LIST = [re.escape(i)
              for i in string.punctuation if i not in ['(', '[', '"', '$', ':']]

NLB_COMMON = [
    "<",
    ">",
    "\&",
    "\(\d{1}\) to ",
    "\(\d{2}\) to ",
    "\(\w{1}\) to ",
    "\(\w{2}\) to ",
    "added",
    "Agreement",
    "agreement",
    "aircraft",
    "Aircraft",
    "Article",
    "article",
    "Articles",
    "articles",
    "Basel",
    "Basel",
    "BORROWER",
    # "Borrower",
    "clause",
    "Clause",
    "clause\(s\)",
    "clauses",
    "Clauses",
    "Column",
    "column",
    "Columns",
    "columns",
    "Company",
    "company",
    "Counsel",
    "counsel",
    "CRD",
    # "equal to",
    "Facility",
    "facility",
    "Facilities",
    "facilities",
    "General",
    "general",
    "greater than",
    "Guarantor",
    "guarantor",
    "items",
    "Items",
    "item",
    "Item",
    "in the case of",
    "In the case of",
    "KPI",
    "less than",
    "limbs",
    "Or\.",
    "ordinance",
    "Ordinance",
    "para",
    "para\.",
    "Para\.",
    "Paragraph",
    "paragraph",
    "Paragraph\(s\)",
    "paragraph\(s\)",
    "Paragraphs",
    "paragraphs",
    "Plan",
    "plan",
    "Premium",
    "premium",
    "Property",
    "property",
    "Proviso",
    "proviso",
    "referred to",
    "Sect",
    "Sect\.",
    "Section",
    "section",
    "Sections",
    "sections",
    "Shares",
    "shares",
    "Ship",
    "ship",
    "sub-paragraph",
    "Sub-paragraph",
    "sub-paragraphs",
    "Sub-paragraphs",
    "Tranche",
    "tranche",
    "Unit",
    "unit",
    "Vessel",
    "vessel",
    "°"
]

# negative lookbehind list (list of patterns that should not be preceded by numbering) for numbering with bracket,
# e.g not to extract numbering with pattern Clause (a), Paragraph (e) etc.
NLB_NUM_BRACK = [
    "\(\w[\)|\.] and ",
    "\w[\)|\.] and ",
    "within"
] + NLB_COMMON

NLB_BRACK = NLB_NUM_BRACK + numInWord + numWithBrack

# negative lookbehind list (list of patterns that should not be preceded by numbering) for numbering with dot,
# e.g not to extract numbering with pattern Clause 1., Paragraph e. etc.
NLB_NUM_DOT = [
    #   "\[",
    "\d",
    '[A-Z]\.',
    "at least",
    "at most",
    "exceed"
] + NLB_COMMON

NLB_DOT = NLB_NUM_DOT + numInWord + numWithBrack

# negative lookahead list (list of patterns that is not present immediately after numbering) for numbering with dot, e.g not to extract and split numbering with pattern 1.0, 2.1 years etc.
# Asserts that what immediately follows the current position in the string is not in this list
TMP_NLA_NUM_DOT = [
    ' above',
    'Sc',
    'Sc\.',
    '[a-zA-Z]\.',
    # '\d{1}',
    '\d{1} \%',
    '\d{1}\%',
    '\d{1} \(',
    '\d{1} per cent',
    '\d{1} percent',
    '\d{1} Years',
    '\d{1} years',
    '\d{1} yrs',
    '\d{1}\([a-zA-Z]\)',
    '\d{1}x',
    # '\d{2}',
    '\d{2} \%',
    '\d{2}\%',
    '\d{2} \(',
    '\d{2} per cent',
    '\d{2} percent',
    '\d{2} Years',
    '\d{2} years',
    '\d{2} yrs',
    # '\d{2}\([a-zA-Z]\)',
    'x',
    'a\.']
TMP_NLA_NUM_BRACK = [
    ' and \([a-zA-Z]\)',
    ',\([a-zA-Z]\)',
    ', \([a-zA-Z]\)',
    ' and \([a-zA-Z]{2}\)',
    ',\([a-zA-Z]{2}\)',
    ', \([a-zA-Z]{2}\)',
    ' and \([a-zA-Z]{3}\)',
    ',\([a-zA-Z]{3}\)',
    ', \([a-zA-Z]{3}\)',
    ' and \([a-zA-Z]{3}\)'
]
NLA_NUM_DOT = TMP_NLA_NUM_DOT + PUNCT_LIST
NLA_NUM_DOT2 = [i + '.*' for i in TMP_NLA_NUM_DOT] + [re.escape(
    i) + '$' for i in string.punctuation if i not in ['(', '[', '"', '$', ':']]
# negative lookahead list (list of patterns that is not present immediately after numbering) for numbering with bracket, e.g not to extract and split numbering with pattern 1)., (2)& etc.
NLA_NUM_BRACK = TMP_NLA_NUM_BRACK + PUNCT_LIST + PREPOSITIONS
NLA_NUM_BRACK2 = [re.escape(i) + '$' for i in string.punctuation if i not in [
    '(', '[', '"', '$', ':']] + [i+'.*' for i in PREPOSITIONS + TMP_NLA_NUM_BRACK]

# rules and regular expression applied to term matching strategy

# tuple of (section, text, definition)
DEF_RULES = [
    ('^interest rate$', 'HIBOR', '^HIBOR$|^margin$'),
    ('^interest rate$|Margin', '', '^margin$'),
    ('Margin', '', '^margin$'),
    ('', 'material adverse change', '^material adverse change'),
    ('', 'break cost', '^break cost'),
    ('', 'break-funding cost', '^break cost'),
    ('^facilit', 'up to', 'total commitment'),
    ('first repayment date', '', 'grace period'),
    ('Availability Period', 'Final Maturity Date', 'Final Maturity Date'),
    ('^facilities', 'Revolving Credit Facility', 'Initial RCF'),
    ('', 'Permitted Transactions', 'Permitted Transactions'),
    ('', 'Consolidated Interest Expense', 'Consolidated Interest Expense'),
    ('Insurances', 'agreed value', 'Agreed Value'),
    ('Interest Coverage Ratio', 'Calculation Period', 'Calculation Period'),
    ('HK Account Bank', 'HK Account Bank', 'Account Bank'),
    ('currency', 'USD|HKD', 'base currency'),
    ('Business Days', 'day', 'Business Days'),
    ('Material Subsidiaries', '', 'Material Subsidiaries'),
    ('Guarantor', 'Material Subsidiaries', 'Material Subsidiary'),
    ('Drawdown', 'prior written notice to', 'Specified Time'),
    ('average life', 'average life of', '^Termination Date'),
    ('facility amount', 'tranche A|B', '^Total Facilty A|B Commitments$'),
    ('Tax Gross Up', 'taxes', '^Taxes$'),
    ('Mandatory Prepayment', 'LTV Ratio', 'LTV Ratio Trigger Event'),
    ('Interest Rate', 'Term SOFR', '^Reference Rate$'),
    ('', 'Second Party Opinion', '^Second Party Opinion$'),
    ('^Security$', 'second ranking security', '^Second Ranking Mortgage'),
    ('Extension Option', 'Extended Participation', 'Extended Final Maturity Date'),
    ('^project$', 'Property development', 'Development'),
    ('^Finance Parties', '', '^Finance Parties$'),
    ('^security$', 'borrower', '^Initial Assignor')
]

# (section, text, schedule)
SCEHD_RULES = [
    ('^security', '', '^security document'),
    ('^conditions precedent', '', '(conditions precedent)|(Documents and evidence)|(Global Transfer Certificate)|(CONDITIONS SUBSEQUENT)'),
    ('Green Loan Provisions', '', 'GREEN LOAN PRINCIPLES MEMORANDUM')
]
# ('(General Undertakings)|(Cashflow Waterfall)|(Project Accounts)|(Drawdown)', '', 'ACCOUNTS AND CASHFLOWS')

TS_SECTIONS_WITH_TERMS = '(Documentation)|(Events of Default)|(Representations)|(Representations, Warranties and Undertakings)'