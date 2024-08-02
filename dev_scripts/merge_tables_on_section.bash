REVIEW_ANTD_TS_VER="_v4"
TS_VER="_v5.5"
FA_VER=""
TM_TAG="tm_ts_fa_0319"

PROJ_ROOT_DIR="/home/data2/ldrs_analytics"
DATA_DIR="${PROJ_ROOT_DIR}/data"
OUTPUT_DOCPARSE_CSV="${DATA_DIR}/docparse_csv/"
TS_CSV="${OUTPUT_DOCPARSE_CSV}TS${TS_VER}"
FA_CSV="${OUTPUT_DOCPARSE_CSV}FA${FA_VER}"
ANTD_CSV="${DATA_DIR}/reviewed_antd_ts${REVIEW_ANTD_TS_VER}/"
ANTD_BFILLED_TS_CSV="${ANTD_CSV}bfilled/"

TM_CSV="${DATA_DIR}/term_matching_csv/${TM_TAG}"
OUTPUT_TS_MERGE_TM_DIR="${DATA_DIR}/antd_ts_merge_tm/merge_antdTS${REVIEW_ANTD_TS_VER}_${TM_TAG}/"

OUTPUT_TS_MERGE_FA_DIR="${DATA_DIR}/antd_ts_merge_fa/merge_antdTS${REVIEW_ANTD_TS_VER}_FA${FA_VER}/"
OUTPUT_ANTD_TS_MERGE_TS_DIR="${DATA_DIR}/antd_ts_merge_ts/antd${REVIEW_ANTD_TS_VER}_merge_ts${TS_VER}"


# calling conda source activate boc from bash script
eval "$(conda shell.bash hook)"
conda activate boc
cd $PROJ_ROOT_DIR/dev_scripts

# backward filling annotation to text item without label within the same text component
# python backward_fill_label.py --annotated_csv_dir ${ANTD_CSV}

# docparse TS left join with annotated TS
# python merge_old_new_ts.py \
# --src_antd_ts_dir $ANTD_CSV \
# --src_ts_csv_dir $TS_CSV \
# --output_merge_dir $OUTPUT_ANTD_TS_MERGE_TS_DIR

# annotated TS left join with term matching on docparse TS
python merge_old_new_ts.py \
--src_antd_ts_dir $TM_CSV \
--src_ts_csv_dir $ANTD_CSV \
--output_merge_dir $OUTPUT_TS_MERGE_TM_DIR \
--is_merge_tm true