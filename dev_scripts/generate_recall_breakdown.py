import pandas as pd
import os
import argparse

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--src_analysis_xlsx_dir",
        default='/home/data2/ldrs_analytics/data/antd_ts_merge_fa/finetune0103_reviewed_ts_v4_all_fa_v4.4_top5_filter_human_errors/analysis/(0)/',
        type=str,
        required=False,
        help="Path to source folder of individual analysis xlsx.",
    )
    parser.add_argument(
        "--dst_analysis_xlsx_dir",
        default='/home/data2/ldrs_analytics/data/antd_ts_merge_fa/finetune0103_reviewed_ts_v4_all_fa_v4.4_top5_filter_human_errors/analysis/(0)/',
        type=str,
        required=False,
        help="Path to destination folder to export breakdown analysis xlsx.",
    )
    parser.add_argument(
        "--target_xlsx_sheet_name",
        default='grp_eval',
        type=str,
        required=False,
        help="Target Excel sheet of analysis table.",
    )
    parser.add_argument(
        "--target_xlsx_names",
        default=[
            "1_GL_SYN_TS_mkd_20221215_docparse_analysis.xlsx",
            "13_BL_SYN_TS_mkd_20220713_docparse_analysis.xlsx",
            "19_GL_SYN_TS_mkd_20220718_docparse_analysis.xlsx",
            "23_BL_SYN_TS_mkd_20220715_docparse_analysis.xlsx",
            "24_GF_PRJ_TS_mkd_20220225_docparse_analysis.xlsx",
            "25_NBFI_PRJ_TS_mkd_20220613_docparse_analysis.xlsx",
            "28_GF_PRJ_TS_mkd_20221111_docparse_analysis.xlsx",
            "29_PF_PRJ_TS_mkd_20200624_docparse_analysis.xlsx",
            "3_GF_SYN_TS_mkd_20221018_docparse_analysis.xlsx",
            "31_GL_PRJ_TS_mkd_20220630_docparse_analysis.xlsx",
            "33_BL_PRJ_TS_mkd_20200727_docparse_analysis.xlsx",
            "34_AWSP_PRJ_TS_mkd_20211230_docparse_analysis.xlsx",
            "41_AF_SYN_TS_mkd_20190307_docparse_analysis.xlsx",
            "43_AF_SYN_TS_mkd_20151101_docparse_analysis.xlsx",
            "45_AF_PRJ_TS_mkd_20170331_docparse_analysis.xlsx",
            "49_VF_PRJ_TS_mkd_20151130_docparse_analysis.xlsx",
            "54_VF_PRJ_TS_mkd_20191018_docparse_analysis.xlsx",
            "58_VF_SYN_TS_mkd_20111201_docparse_analysis.xlsx",
            "59_AWSP_SYN_TS_mkd_20210814_docparse_analysis.xlsx",
            "63_AW_SYN_TS_mkd_20221025_docparse_analysis.xlsx",
            "66_PF_SYN_TS_mkd_20230106_docparse_analysis.xlsx",
            "68_PF_SYN_TS_mkd_20221104_docparse_analysis.xlsx",
            "72_NBFI_SYN_TS_mkd_20221215_docparse_analysis.xlsx",
            "74_NBFI_SYN_TS_mkd_20220401_docparse_analysis.xlsx",
            "8_GF_SYN_TS_mkd_20230215_docparse_analysis.xlsx"
    ],
        type=list,
        required=False,
        help="Target filenames to have breakdown summary.",
    )
    args = parser.parse_args()
    folders = [f for f in os.listdir(args.src_analysis_xlsx_dir) if f.endswith('.xlsx') and f != 'all_analysis.xlsx']
    DF = []
    for file in folders:
        if args.target_xlsx_names and file not in args.target_xlsx_names:
            continue
        fname = file.replace('_docparse_analysis.xlsx','')
        df = pd.read_excel(os.path.join(args.src_analysis_xlsx_dir, file), sheet_name=args.target_xlsx_sheet_name)
        df['filename'] = fname
        DF.append(df)
    new_df = pd.concat(DF)
    new_df.to_csv(os.path.join(args.dst_analysis_xlsx_dir,'all_recall_breakdown.csv'))
