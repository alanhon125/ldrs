# define variables
PROJ_ROOT_DIR="/home/data2/ldrs_analytics"
DATA_DIR="${PROJ_ROOT_DIR}/data"
TS_VERS="_v4 _v4.1 _v4.2 _v4.3 _v4.4 _v5 _v5.1 _v5.2 _v5.3 _v5.4 _v5.4.1"
REVIEW_ANTD_TS_VER="_v4"

REVIEWED_ANTD_TS_CSV="${DATA_DIR}/reviewed_antd_ts${REVIEW_ANTD_TS_VER}/"
ANTD_BFILLED_TS_CSV="${DATA_DIR}/reviewed_antd_ts${REVIEW_ANTD_TS_VER}/bfilled/"

# calling conda source activate boc from bash script
eval "$(conda shell.bash hook)"
conda activate boc
cd $PROJ_ROOT_DIR/dev_scripts

for TS_VER in $TS_VERS; do
    # TS_VER="_v5.3"
    TS_CSV="${DATA_DIR}/docparse_csv/TS${TS_VER}"

    OUTPUT_EVAL_FOLDER="${DATA_DIR}/ts_evaluation_output/antd${REVIEW_ANTD_TS_VER}_TS${TS_VER}"

    python ts_evaluation.py \
    --antd_folder $REVIEWED_ANTD_TS_CSV \
    --ts_folder $TS_CSV \
    --output_folder $OUTPUT_EVAL_FOLDER
done

python ts_evaluation_draw.py
