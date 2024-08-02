# define variables

# reference doc for mapping additional information about document. e.g. filename-to-facility_type, filename-to-dataset_split, filename-to-indeterminate_id etc.
FILENAME2FACILITY_TYPE="/home/data/ldrs_analytics/data/doc/filename_facility_type.csv"
FILENAME2FACILITY_TYPE_BATCH_ID="/home/data/ldrs_analytics/data/doc/filename_facility_type_batch.csv"
FILENAME2DATASET_SPLIT="/home/data/ldrs_analytics/data/doc/filename_dataset_split.csv"
FILENAME2INDETERMINATE_ID="/home/data/ldrs_analytics/data/doc/not_exist_fa_identifier.csv"

TM_STRATEGY_VER=""
TS_VER=""
# FA_VER="_v4.4"
# REVIEW_ANTD_TS_VER="_v4"
REVIEW_ANTD_TS_VER="_v4_0229"

PROJ_ROOT_DIR="/home/data/ldrs_analytics"
DATA_DIR="${PROJ_ROOT_DIR}/data"
# FA_CSV="${DATA_DIR}/docparse_csv/FA${FA_VER}"
TS_CSV="${DATA_DIR}/docparse_csv/TS${TS_VER}"

ANTD_CSV="${DATA_DIR}/reviewed_antd_ts${REVIEW_ANTD_TS_VER}/"
RESULT_TAG="tm_reviewed_antd_ts_v4_0229_fa_0325"
OUTPUT_CSV_DIR="${DATA_DIR}/antd_ts_merge_fa/${RESULT_TAG}"
# ANTD_CSV="${DATA_DIR}/antd_ts_merge_ts/antd${REVIEW_ANTD_TS_VER}_merge_ts${TS_VER}/"
# ANTD_BFILLED_TS_CSV="${DATA_DIR}/reviewed_antd_ts${REVIEW_ANTD_TS_VER}/bfilled/"
# ANTD_BFILLED_TS_CSV="${DATA_DIR}/antd_ts_merge_ts/antd${REVIEW_ANTD_TS_VER}_merge_ts${TS_VER}/bfilled/"



python analyze_results.py \
    --result_dir ${DATA_DIR}"/antd_ts_merge_fa/tm_reviewed_antd_ts_v4_0229_fa_0325_top5_filter_human_errors/(0)" \
    --output_dir ${DATA_DIR}"/antd_ts_merge_fa/tm_reviewed_antd_ts_v4_0229_fa_0325_top5_filter_human_errors/(0)" \
    --result_flag ${RESULT_TAG} \
    --target_input_result_filename "all_merge&match_results.csv" \
    --file_dataset_doc $FILENAME2DATASET_SPLIT \
    --filename_facility_type_doc $FILENAME2FACILITY_TYPE \
    --analyze_set "test"
