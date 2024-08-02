REVIEW_ANTD_TS_VER="_v4"
TS_VER=""
FA_VER=""
SENT_PAIRS_VER="_v12"
DEF_CL_PAIRS_VER="_v7"

PROJ_ROOT_DIR="/home/data2/ldrs_analytics"
DATA_DIR="${PROJ_ROOT_DIR}/data"
OUTPUT_DOCPARSE_CSV="${DATA_DIR}/docparse_csv/"
TS_CSV="${OUTPUT_DOCPARSE_CSV}TS${TS_VER}"
FA_CSV="${OUTPUT_DOCPARSE_CSV}FA${FA_VER}"

OUTPUT_TS_MERGE_FA_DIR="${DATA_DIR}/antd_ts_merge_fa/merge_antdTS${REVIEW_ANTD_TS_VER}_FA${FA_VER}/"
OUTPUT_ANTD_TS_MERGE_TS_DIR="${DATA_DIR}/antd_ts_merge_ts/antd${REVIEW_ANTD_TS_VER}_merge_ts${TS_VER}"
INPUT_ALL_MERGE_CSV_PATH="${OUTPUT_TS_MERGE_FA_DIR}all_merge_results.csv"
ANTD_CSV="${DATA_DIR}/reviewed_antd_ts${REVIEW_ANTD_TS_VER}/"
ANTD_BFILLED_TS_CSV="${ANTD_CSV}bfilled/"

# calling conda source activate boc from bash script
eval "$(conda shell.bash hook)"
conda activate boc
cd $PROJ_ROOT_DIR/dev_scripts

# backward filling annotation to text item without label within the same text component
# python backward_fill_label.py --annotated_csv_dir ${ANTD_CSV}

# docparse TS left join with annotated TS
python merge_old_new_ts.py \
--src_antd_ts_dir $ANTD_CSV \
--src_ts_csv_dir $TS_CSV \
--output_merge_dir $OUTPUT_ANTD_TS_MERGE_TS_DIR

# # backward filling annotation to text item without label within the same text component
# python backward_fill_label.py --annotated_csv_dir ${OUTPUT_ANTD_TS_MERGE_TS_DIR}

# merge annotated, backward filled TS with FA by FA identifier, output sentence-pair dataframe
python ts_merge_fa.py \
--do_merge true \
--src_ts_csv_dir $ANTD_CSV \
--src_fa_csv_dir $FA_CSV \
--output_csv_dir $OUTPUT_TS_MERGE_FA_DIR

# prepare sentence-pair datasets
python ts_to_pairs.py \
--input_csv_path $INPUT_ALL_MERGE_CSV_PATH \
--output_csv_dir $OUTPUT_TS_MERGE_FA_DIR \
--version $SENT_PAIRS_VER

# prepare definition-clauses pairs dataset
python extract_all_fa_definitions_parties.py \
--src_fa_csv_dir $FA_CSV \
--output_csv_dir $OUTPUT_DOCPARSE_CSV \
--def_clause_pair_version $DEF_CL_PAIRS_VER \
--fa_version "_v5.1"
# 
# # # send sentence-pairs to 10.6.55.2
# # sshpass -p "1q2w3e4r" scp ${OUTPUT_TS_MERGE_FA_DIR}/sentence_pairs${SENT_PAIRS_VER}.csv data@10.6.55.2:/home/data/data/owen/boc_data
# # sshpass -p "1q2w3e4r" scp ${OUTPUT_DOCPARSE_CSV}all_FA_definitions_parties${DEF_CL_PAIRS_VER}.csv data@10.6.55.2:/home/data/data/owen/boc_data