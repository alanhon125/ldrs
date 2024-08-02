PROJ_ROOT_DIR="/home/data/ldrs_analytics"
# REVIEWED_TS_VER="_v4_0229"
# FA_VER="_v5.2"
TS_VER=""
MODEL_PATH="${PROJ_ROOT_DIR}/models/gte-base-train_nli-boc_epoch_20_lr_1e-05_batch_32_0130" # "thenlper/gte-base"
# FA_CSV="${PROJ_ROOT_DIR}/data/docparse_csv/FA${FA_VER}"
TS_CSV="${PROJ_ROOT_DIR}/data/docparse_csv/TS${TS_VER}"
# REVIEWED_TS_CSV="${PROJ_ROOT_DIR}/data/reviewed_antd_ts${REVIEWED_TS_VER}"
# OUTPUT_TERM_MATCH="${PROJ_ROOT_DIR}/data/term_matching_csv/tm_reviewed_antd_ts${REVIEWED_TS_VER}_fa${FA_VER}"


# calling conda source activate boc from bash script
eval "$(conda shell.bash hook)"
conda activate boc
cd $PROJ_ROOT_DIR/app/term_match/

FA_VERS="_v5.2 _v5.2.1"

for FA_VER in $FA_VERS; do
    FA_CSV="${PROJ_ROOT_DIR}/data/docparse_csv/FA${FA_VER}"  
    RESULT_TAG="tm_ts${TS_VER}_fa${FA_VER}"
    OUTPUT_TERM_MATCH="${PROJ_ROOT_DIR}/data/term_matching_csv/tm_ts${TS_VER}_fa${FA_VER}"

    python3 term_match.py \
    --src_fa_folder $FA_CSV \
    --src_ts_folder $TS_CSV \
    --model_path $MODEL_PATH \
    --output_term_match_folder $OUTPUT_TERM_MATCH

done