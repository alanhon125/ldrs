import os

DEV_MODE = False

PROJ_ROOT_DIR = os.path.abspath(os.path.dirname(__name__))

# Input Directory
DOCPARSE_OUTPUT_JSON_DIR = 'data/docparse_json/'
DOC_DIR = 'data/doc'
PDF_DIR = 'data/pdf/'

# Output Directories
OUTPUT_ANNOT_PDF_DIR = 'data/docparse_annotation/'
OUTPUT_DOCPARSE_CSV = 'data/docparse_csv/'
DOCPARSE_TS_CSV = 'data/docparse_csv/TS/'
DOCPARSE_FA_CSV = 'data/docparse_csv/FA/'
OUTPUT_IMG_DIR = 'data/image_share/'
REMOTE_PROJ_DIR = '/home/data/ldrs_model'
REMOTE_OUTPUT_IMG_DIR = f'{REMOTE_PROJ_DIR}/data/image_share'
OUTPUT_LAYOUTLM_INPUT_DIR = 'data/layoutlm_input_data/'
OUTPUT_LAYOUTLM_OUTPUT_DIR = 'data/layoutlm_output_data/'
OUTPUT_ANNOT_DOC_DIR = 'data/antd_term_match_doc/'
OUTPUT_ANNOT_DOC_TYPE = ".zip"
OUTPUT_TERM_MATCH_CSV = 'data/term_matching_csv/'
LOG_DIR = 'data/log/'
LOG_FILEPATH = 'ldrs_analytics.log'
LOG_HANDLERS = ['console', 'file']

# Document Parsing output configuration
DOCPARSE_PDF_ANNOTATE = False
DOCPARSE_DOCUMENT_TYPE = 'termSheet'

# Models and Model setting
USE_MODEL = True
USE_OCR = False
DOCPARSE_MODELV3_CLAUSE_PATH = f'models/layoutlm/layoutlmv3_base_500k_docbank_epoch_2_lr_1e-5_511_agreement_epoch_2000_lr_1e-5'
DOCPARSE_MODELV3_TS_PATH = 'models/layoutlm/layoutlmv3_base_500k_docbank_epoch_2_lr_1e-5_511_agreement_epoch_2000_lr_1e-5_760_termsheet_epoch_2000_lr_1e-5'
SENT_BERT_MODEL_PATH = 'models/gte-base-train_nli-boc_epoch_20_lr_1e-05_batch_32_0130'
LAYOUTLM_BATCH_SIZE = 16
SENT_EMBEDDINGS_BATCH_SIZE = 16
TOP_N_RESULTS = 5
DOCPARSE_GPU_ID = "0,1,2,3"

# Data insert, update and query API URL from/to project database
ANALYTICS_ADDR = 'localhost:8000'
BACKEND_ADDR = '10.6.55.12:8080'
GPU_SERVER_ADDR = '10.6.55.3:8088'

INSERT_FA_URL = f"http://{BACKEND_ADDR}/ldrs/dataStorage/insertFADataByBatch"
INSERT_TS_URL = f"http://{BACKEND_ADDR}/ldrs/dataStorage/insertTSDataByBatch"
INSERT_TERM_MATCH_URL = f"http://{BACKEND_ADDR}/ldrs/dataStorage/insertTermMatchDataByBatch"
INSERT_ANALYTICS_URL = f"http://{BACKEND_ADDR}/ldrs/analytics/insertAccuracyData"
UPDATE_DOC_STATUS_URL = f"http://{BACKEND_ADDR}/ldrs/dataStorage/updateDocument"
UPDATE_TASK_URL = f"http://{BACKEND_ADDR}/ldrs/dataStorage/updateTask"
QUERY_TERM_MATCH_URL = f"http://{BACKEND_ADDR}/ldrs/dataStorage/queryTermMatchData"
QUERY_DOC_URL = f"http://{BACKEND_ADDR}/ldrs/dataStorage/queryDocument"
QUERY_TASK_URL = f"http://{BACKEND_ADDR}/ldrs/dataStorage/queryTask"
QUERY_FA_URL = f"http://{BACKEND_ADDR}/ldrs/dataStorage/queryFAData"
QUERY_TS_URL = f"http://{BACKEND_ADDR}/ldrs/dataStorage/queryTSData"
QUERY_TERM_MATCH_EVAL_URL = f"http://{BACKEND_ADDR}/ldrs/dataStorage/queryTermMatchEval"

LAYOUTLMV3_PREDICT_URL = f'http://{GPU_SERVER_ADDR}/api/layoutlm_predict'
COS_SIM_URL = f'http://{GPU_SERVER_ADDR}/api/cosine_sim'

INSERT_DATA_BATCH_SIZE = 1500
QUERY_DATA_BATCH_SIZE = 1500
