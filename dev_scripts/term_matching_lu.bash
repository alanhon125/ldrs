PROJ_ROOT_DIR="/home/data2/ldrs_analytics"
REVIEWED_TS_VER="_v4"
FA_VER="_v5.1.1"
# TS_VER="_reviewed_v4_0229"
MODEL_PATH="${PROJ_ROOT_DIR}/models/gte-base-train_nli-boc_epoch_20_lr_1e-05_batch_32_0130" # "thenlper/gte-base"
FA_CSV="${PROJ_ROOT_DIR}/data/docparse_csv/FA${FA_VER}"
# TS_CSV="${PROJ_ROOT_DIR}/data/docparse_csv/TS${TS_VER}"
TS_CSV="${PROJ_ROOT_DIR}/data/reviewed_antd_ts_v4_0229"
OUTPUT_TERM_MATCH="${PROJ_ROOT_DIR}/data/term_matching_csv/tm_ts_reviewed_fa${FA_VER}_0326"

# calling conda source activate boc from bash script
eval "$(conda shell.bash hook)"
conda activate boc
cd $PROJ_ROOT_DIR/app/term_match/

python term_match.py \
--src_fa_folder $FA_CSV \
--src_ts_folder $TS_CSV \
--model_path $MODEL_PATH \
--output_term_match_folder $OUTPUT_TERM_MATCH