import pandas as pd
import numpy as np
import os
import json
import argparse
from config import *

pd.options.mode.chained_assignment = None

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--src_fa_csv_dir",
        default='/home/data2/ldrs_analytics/data/docparse_csv/FA_v4.1.1',
        type=str,
        required=False,
        help="Path to folder of FA docparse source.",
    )
    parser.add_argument(
        "--output_csv_dir",
        default='/home/data2/ldrs_analytics/data/docparse_csv/',
        type=str,
        required=False,
        help="Path to output definition-clauses pairs .csv.",
    )
    parser.add_argument(
        "--fa_version",
        default='_v4.1.1',
        type=str,
        required=False,
        help="FA version.",
    )
    parser.add_argument(
        "--def_clause_pair_version",
        default='_v5.1.1',
        type=str,
        required=False,
        help="definition-clauses version.",
    )
    parser.add_argument(
        "--list_version",
        default='',
        type=str,
        required=False,
        help="non-empty list items version.",
    )
    args = parser.parse_args()
    fa_fnames = sorted(os.listdir(args.src_fa_csv_dir))

    if os.path.exists("/home/data2/ldrs_analytics/data/reviewed_antd_ts/checked_list.json"):
        with open("/home/data2/ldrs_analytics/data/reviewed_antd_ts/checked_list.json", 'r') as f:
            target_files = json.load(f)
        fa_fnames = [f for f in fa_fnames if f.replace('FA','TS') in target_files]

    DF_def = []
    DF_list = []
    DF_FA = []
    for fa_fname in fa_fnames:
        filename = fa_fname.replace('.csv', '')
        df_fa = pd.read_csv(os.path.join(args.src_fa_csv_dir,fa_fname))
        df_fa['sub_section'] = df_fa['sub_section'].astype(str)
        df_fa = df_fa.replace({'nan': None, np.nan: None, 'None': None})
        
        df_list = df_fa[~df_fa.list_id.isna()]
        df_list['filename'] = filename
        DF_list.append(df_list)
        
        df_def = df_fa[~df_fa.definition.isna()]
        df_def = df_def.groupby(['section_id','sub_section_id','definition'], dropna=False).head(1).reset_index(drop=True)
        df_def['filename'] = filename
        DF_def.append(df_def)
        
        df_fa['filename'] = filename
        DF_FA.append(df_fa)
        
    DF_FA = pd.concat(DF_FA)
    outpath0 = os.path.join(args.output_csv_dir, f'all_FA{args.fa_version}.csv')
    DF_FA.to_csv(outpath0,index=False, encoding='utf-8-sig')

    DF_list = pd.concat(DF_list)
    outpath = os.path.join(args.output_csv_dir, f'all_FA_list{args.list_version}.csv')
    DF_list.to_csv(outpath,index=False, encoding='utf-8-sig')

    DF_def = pd.concat(DF_def)
    outpath2 = os.path.join(args.output_csv_dir, f'all_FA_definitions_parties{args.def_clause_pair_version}.csv')
    DF_def.to_csv(outpath2,index=False, encoding='utf-8-sig')