import os
import json

UAT_LIST = [
    "1_GL_SYN_TS_mkd_20221215.json",
    "13_BL_SYN_TS_mkd_20220713.json",
    "19_GL_SYN_TS_mkd_20220718.json",
    "23_BL_SYN_TS_mkd_20220715.json",
    "24_GF_PRJ_TS_mkd_20220225.json",
    "25_NBFI_PRJ_TS_mkd_20220613.json",
    "28_GF_PRJ_TS_mkd_20221111.json",
    "29_PF_PRJ_TS_mkd_20200624.json",
    "3_GF_SYN_TS_mkd_20221018.json",
    "31_GL_PRJ_TS_mkd_20220630.json",
    "33_BL_PRJ_TS_mkd_20200727.json",
    "34_AWSP_PRJ_TS_mkd_20211230.json",
    "41_AF_SYN_TS_mkd_20190307.json",
    "43_AF_SYN_TS_mkd_20151101.json",
    "45_AF_PRJ_TS_mkd_20170331.json",
    "49_VF_PRJ_TS_mkd_20151130.json",
    "54_VF_PRJ_TS_mkd_20191018.json",
    "58_VF_SYN_TS_mkd_20111201.json",
    "59_AWSP_SYN_TS_mkd_20210814.json",
    "63_AW_SYN_TS_mkd_20221025.json",
    "66_PF_SYN_TS_mkd_20230106.json",
    "68_PF_SYN_TS_mkd_20221104.json",
    "72_NBFI_SYN_TS_mkd_20221215.json",
    "74_NBFI_SYN_TS_mkd_20220401.json",
    "8_GF_SYN_TS_mkd_20230215.json",
]

target_folders = ['/home/data2/ldrs_analytics/data/docparse_json/TS',
                  '/home/data2/ldrs_analytics/data/layoutlm_input_data',
                  '/home/data2/ldrs_analytics/data/layoutlm_output_data']

for target_f in target_folders:
    files = os.listdir(target_f)
    files = [f for f in files if f not in UAT_LIST]
    for f in files:
        os.remove(os.path.join(target_f, f))