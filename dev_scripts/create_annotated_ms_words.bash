PROJ_ROOT_DIR="/home/data/ldrs_analytics"
TERM_MATCH_DIR=$PROJ_ROOT_DIR"/data/antd_ts_merge_fa/1227_finetune_raw_top5_filter_human_errors/(0)/"
FA_DIR=$PROJ_ROOT_DIR"/data/docparse_csv/FA_v4/"
OUTPUT_MSWORD_DIR=$PROJ_ROOT_DIR"/data/antd_term_match_doc"
# TARGET_FILENAME_LIST=

# calling conda source activate boc from bash script
eval "$(conda shell.bash hook)"
conda activate boc
cd $PROJ_ROOT_DIR/app/create_antd_doc/

python generate_antd_doc.py \
--term_match_folder $TERM_MATCH_DIR \
--fa_folder $FA_DIR \
--output_folder $OUTPUT_MSWORD_DIR
# --target_list $TARGET_FILENAME_LIST \
# --with_annot true

# test_set = [
#     '1_GL_SYN_TS_mkd_20221215_docparse_merge&match_FA.csv',
#     '13_BL_SYN_TS_mkd_20220713_docparse_merge&match_FA.csv',
#     '19_GL_SYN_TS_mkd_20220718_docparse_merge&match_FA.csv',
#     '23_BL_SYN_TS_mkd_20220715_docparse_merge&match_FA.csv',
#     '24_GF_PRJ_TS_mkd_20220225_docparse_merge&match_FA.csv',
#     '25_NBFI_PRJ_TS_mkd_20220613_docparse_merge&match_FA.csv',
#     '28_GF_PRJ_TS_mkd_20221111_docparse_merge&match_FA.csv',
#     '29_PF_PRJ_TS_mkd_20200624_docparse_merge&match_FA.csv',
#     '3_GF_SYN_TS_mkd_20221018_docparse_merge&match_FA.csv',
#     '31_GL_PRJ_TS_mkd_20220630_docparse_merge&match_FA.csv',
#     '33_BL_PRJ_TS_mkd_20200727_docparse_merge&match_FA.csv',
#     '34_AWSP_PRJ_TS_mkd_20211230_docparse_merge&match_FA.csv',
#     '41_AF_SYN_TS_mkd_20190307_docparse_merge&match_FA.csv',
#     '43_AF_SYN_TS_mkd_20151101_docparse_merge&match_FA.csv',
#     '45_AF_PRJ_TS_mkd_20170331_docparse_merge&match_FA.csv',
#     '49_VF_PRJ_TS_mkd_20151130_docparse_merge&match_FA.csv',
#     '54_VF_PRJ_TS_mkd_20191018_docparse_merge&match_FA.csv',
#     '58_VF_SYN_TS_mkd_20111201_docparse_merge&match_FA.csv',
#     '59_AWSP_SYN_TS_mkd_20210814_docparse_merge&match_FA.csv',
#     '63_AW_SYN_TS_mkd_20221025_docparse_merge&match_FA.csv',
#     '66_PF_SYN_TS_mkd_20230106_docparse_merge&match_FA.csv',
#     '68_PF_SYN_TS_mkd_20221104_docparse_merge&match_FA.csv',
#     '72_NBFI_SYN_TS_mkd_20221215_docparse_merge&match_FA.csv',
#     '74_NBFI_SYN_TS_mkd_20220401_docparse_merge&match_FA.csv',
#     '8_GF_SYN_TS_mkd_20230215_docparse_merge&match_FA.csv',
# ]

# unannot_test_set = [
#             "125_VF_SYN_TS_mkd_20231013_docparse_results.csv",
#             "126_VF_SYN_TS_mkd_20230601_docparse_results.csv",
#             "127_NBFI_SYN_TS_mkd_20230330_docparse_results.csv",
#             "128_NBFI_SYN_TS_mkd_20230224_docparse_results.csv",
#             "129_AW_SYN_TS_mkd_20230626_docparse_results.csv",
#             "130_BL_SYN_TS_mkd_20230710_docparse_results.csv",
#             "131_AF_SYN_TS_mkd_20230718_docparse_results.csv",
#             "132_PF_SYN_TS_mkd_20230106_docparse_results.csv",
#             "133_BL_PRJ_TS_mkd_20230926_docparse_results.csv",
#             "134_GF_PRJ_TS_mkd_20230829_docparse_results.csv",
#             "135_GL_PRJ_TS_mkd_20231004_docparse_results.csv",
#             "136_BL_SYN_TS_mkd_20230418_docparse_results.csv",
#             "137_GL_SYN_TS_mkd_20231121_docparse_results.csv",
#             "138_GF_SYN_TS_mkd_20230130_docparse_results.csv",
#             "139_NBFI_PRJ_TS_mkd_20231122_docparse_results.csv"
# ]