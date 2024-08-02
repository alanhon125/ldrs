# define variables

# reference doc for mapping additional information about document. e.g. filename-to-facility_type, filename-to-dataset_split, filename-to-indeterminate_id etc.
FILENAME2FACILITY_TYPE="/home/data/ldrs_analytics/data/doc/filename_facility_type.csv"
FILENAME2FACILITY_TYPE_BATCH_ID="/home/data/ldrs_analytics/data/doc/filename_facility_type_batch.csv"
FILENAME2DATASET_SPLIT="/home/data/ldrs_analytics/data/doc/filename_dataset_split.csv"
FILENAME2INDETERMINATE_ID="/home/data/ldrs_analytics/data/doc/not_exist_fa_identifier.csv"

TM_STRATEGY_VER=""
TS_VER=""
# FA_VER="_v4.4"
REVIEW_ANTD_TS_VER="_v4"

PROJ_ROOT_DIR="/home/data/ldrs_analytics"
DATA_DIR="${PROJ_ROOT_DIR}/data"
# FA_CSV="${DATA_DIR}/docparse_csv/FA${FA_VER}"
TS_CSV="${DATA_DIR}/docparse_csv/TS${TS_VER}"

ANTD_CSV="${DATA_DIR}/reviewed_antd_ts${REVIEW_ANTD_TS_VER}/"
# ANTD_CSV="${DATA_DIR}/antd_ts_merge_ts/antd${REVIEW_ANTD_TS_VER}_merge_ts${TS_VER}/"
# ANTD_BFILLED_TS_CSV="${DATA_DIR}/reviewed_antd_ts${REVIEW_ANTD_TS_VER}/bfilled/"
# ANTD_BFILLED_TS_CSV="${DATA_DIR}/antd_ts_merge_ts/antd${REVIEW_ANTD_TS_VER}_merge_ts${TS_VER}/bfilled/"

# RESULT_TAG="tm_ts${TS_VER}_fa${FA_VER}_0325"
FA_VERS="_v4.4 _v5.1.1" # " _v5.2 _v5.2.1"

for FA_VER in $FA_VERS; do
    FA_CSV="${DATA_DIR}/docparse_csv/FA${FA_VER}"  
    RESULT_TAG="tm_ts${TS_VER}_fa${FA_VER}"

    TERM_MATCHED_CSV="${DATA_DIR}/term_matching_csv/${RESULT_TAG}/"
    ANTD_TS_MERGE_TERM_MATCHED_CSV="${DATA_DIR}/antd_ts_merge_tm/merge_antdTS${REVIEW_ANTD_TS_VER}_${RESULT_TAG}/"
    TS_MERGE_ANTD_TS_CSV="${DATA_DIR}/antd_ts_merge_ts/merge_ts${TS_VER}_antdTS${REVIEW_ANTD_TS_VER}/"

    OUTPUT_CSV_DIR="${DATA_DIR}/antd_ts_merge_fa/${RESULT_TAG}"
    DEMO_ANALY_DIR="${DATA_DIR}/output_for_demo"

    topN=5
    sim_thre=0

    # calling conda source activate boc from bash script
    eval "$(conda shell.bash hook)"
    conda activate boc
    cd $PROJ_ROOT_DIR/dev_scripts

    # backward filling annotation to text item without label within the same text component
    # python backward_fill_label.py --annotated_csv_dir ${ANTD_CSV}

    # [Uncomment these line when perform Raw parse TS evaluation] annotated TS left join with term matching on docparse TS, so that preserve reviewed annotated TS content & labels
    python merge_left_right_on_text.py \
    --src_left_table_csv_dir $ANTD_CSV \
    --src_right_table_csv_dir $TERM_MATCHED_CSV \
    --output_merge_dir $ANTD_TS_MERGE_TERM_MATCHED_CSV \
    --is_merge_tm true

    # merge matched results with annotated TS that backward filled, output performance evaluation based on comparing ground-truth FA identifier and matched result FA identifiers

    python ts_merge_fa.py \
    --do_merge true \
    --do_match true \
    --apply_ext_sim_match true \
    --src_ts_csv_dir $ANTD_CSV \
    --src_fa_csv_dir $FA_CSV \
    --matched_result_csv_dir $ANTD_TS_MERGE_TERM_MATCHED_CSV \
    --output_csv_dir $OUTPUT_CSV_DIR \
    --sim_threshold $sim_thre \
    --topN $topN \
    --eval_filter_human_errors true \
    --filename_facility_type_doc $FILENAME2FACILITY_TYPE \
    --filename_facility_type_batch_doc $FILENAME2FACILITY_TYPE_BATCH_ID \
    --file_dataset_doc $FILENAME2DATASET_SPLIT \
    --indeterminate_doc $FILENAME2INDETERMINATE_ID

    python generate_recall_breakdown.py \
    --src_analysis_xlsx_dir ${OUTPUT_CSV_DIR}"_top"${topN}"_filter_human_errors/analysis/("${sim_thre}")/" \
    --dst_analysis_xlsx_dir ${OUTPUT_CSV_DIR}"_top"${topN}"_filter_human_errors/analysis/("${sim_thre}")/" \
    --target_xlsx_sheet_name "grp_eval"

    # python analyze_results.py \
    --result_dir ${OUTPUT_CSV_DIR}"_top"${topN}"_filter_human_errors/("${sim_thre}")/" \
    --output_dir ${OUTPUT_CSV_DIR}"_top"${topN}"_filter_human_errors/analysis/("${sim_thre}")/" \
    --result_flag ${RESULT_TAG} \
    --target_input_result_filename "all_merge&match_results.csv" \
    --file_dataset_doc $FILENAME2DATASET_SPLIT \
    --filename_facility_type_doc $FILENAME2FACILITY_TYPE \
    --analyze_set "test"

    # Transform all merge & match data into demo system table format and perform analysis applied in demo system
    python transform_all_merge_match.py \
    --topN $topN \
    --output_folder ${DEMO_ANALY_DIR} \
    --src_ts_folder ${TS_CSV} \
    --src_fa_folder ${FA_CSV} \
    --name_tag ${RESULT_TAG}

done
