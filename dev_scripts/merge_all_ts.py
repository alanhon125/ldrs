import pandas as pd
import numpy as np
import os
import json
import argparse
from config import *

pd.options.mode.chained_assignment = None

prepositions = ['above', 'across', 'against', 'along', 'among', 'around', 'at', 'before', 'behind', 'below', 'beneath', 'beside', 'between', 'by', 'down', 'from', 'in', 'into', 'near', 'of', 'off', 'on', 'to', 'toward', 'under', 'upon', 'with' , 'within']

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

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--src_ts_csv_dir",
        default='/home/data2/ldrs_analytics/data/docparse_csv/TS',
        type=str,
        required=False,
        help="Path to folder of TS docparse source.",
    )
    parser.add_argument(
        "--output_csv_dir",
        default='/home/data2/ldrs_analytics/data/docparse_csv/',
        type=str,
        required=False,
        help="Path to output all merge .csv.",
    )
    parser.add_argument(
        "--ts_version",
        default='_v5.4.1',
        type=str,
        required=False,
        help="FA version.",
    )
    args = parser.parse_args()
    args.src_ts_csv_dir = args.src_ts_csv_dir + args.ts_version
    ts_fnames = sorted(os.listdir(args.src_ts_csv_dir))

    DF_TS = []
    for ts_fname in ts_fnames:
        filename = ts_fname.replace('.csv', '')
        df_ts = pd.read_csv(os.path.join(args.src_ts_csv_dir,ts_fname))
        df_ts = df_ts.replace({'nan': None, np.nan: None, 'None': None})
        df_ts['filename'] = filename
        DF_TS.append(df_ts)
        
    DF_TS = pd.concat(DF_TS)
    outpath0 = os.path.join(args.output_csv_dir, f'all_TS{args.ts_version}.csv')
    DF_TS.to_csv(outpath0,index=False, encoding='utf-8-sig')
    
    # merge all reviewed ts
    # ts_fnames = [f for f in sorted(os.listdir('/home/data2/ldrs_analytics/data/reviewed_antd_ts_v3.2')) if f.endswith('.csv')]

    # DF_TS = []
    # for ts_fname in ts_fnames:
    #     filename = ts_fname.replace('.csv', '')
    #     df_ts = pd.read_csv(os.path.join('/home/data2/ldrs_analytics/data/reviewed_antd_ts_v3.2',ts_fname))
    #     df_ts = df_ts.replace({'nan': None, np.nan: None, 'None': None})
    #     df_ts['filename'] = filename
    #     DF_TS.append(df_ts)
        
    # DF_TS = pd.concat(DF_TS)
    # DF_TS['section'] = DF_TS['section'].astype(str)
    # DF_TS = DF_TS[(~DF_TS['section'].isin([i.capitalize() for i in prepositions])) & (~DF_TS['section'].str.contains(r'^\d+$'))]
    # DF_TS['section'] = DF_TS['section'].map(lambda x: remove_non_alphabet(x) if isinstance(x,str) else x)
    # DF_TS = DF_TS.drop_duplicates(subset=['section'])
    # DF_TS = DF_TS['section']
    # DF_TS = DF_TS.sort_values()
    # outpath0 = os.path.join(args.output_csv_dir, f'all_reviewed_TS_v3.2_sections.csv')
    # DF_TS.to_csv(outpath0, header=False,index=False, encoding='utf-8-sig')
