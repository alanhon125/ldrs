import pandas as pd
from ts_merge_fa import get_similarity_sentbert
from sentence_transformers import SentenceTransformer, util

def get_similarity_pairwise(
    ts_string_list,
    fa_string_list,
    model_path='sentence-transformers/all-MiniLM-L6-v2',
):
    assert len(ts_string_list)==len(fa_string_list), 'input lists of string must have the same length.'
    model = SentenceTransformer(model_path)

    embeddings1 = model.encode(ts_string_list, convert_to_tensor=True)
    embeddings2 = model.encode(fa_string_list, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)

    pair2score = dict()
    for i in range(len(ts_string_list)):
        all_score = list(cosine_scores[i])
        score = all_score[i].item()
        pair2score.update({(ts_string_list[i],fa_string_list[i]) : score})
    return pair2score


OUTPUT_DIR = '/home/data/ldrs_analytics/data/antd_ts_merge_fa/merge/'
SENT_PAIR_CSV_PATH = f'{OUTPUT_DIR}sentence_pairs.csv'
FINE_TUNE_MODEL_PATH = '/home/data/ldrs_analytics/models/all-MiniLM-L6-v2-train_sts-boc_17398_epoch_100_lr_3e-05'

df = pd.read_csv(SENT_PAIR_CSV_PATH)
ts_string_list = df['ts_text'].tolist()
fa_string_list = df['fa_text'].tolist()
pair2sim = get_similarity_pairwise(ts_string_list,fa_string_list) # ,model_path=FINE_TUNE_MODEL_PATH
df['sim_score'] = df[['ts_text','fa_text']].apply(lambda x: pair2sim[(x['ts_text'],x['fa_text'])],axis=1)
# df['sim_score'] = df[['ts_text','fa_text']].apply(lambda x: get_similarity_sentbert(x['ts_text'],x['fa_text'],sim_threshold=0,model_path='/home/data/ldrs_analytics/models/all-MiniLM-L6-v2-train_sts-boc_17398_epoch_100_lr_1e-05')[1],axis=1)
df = df[['split','label','ts_text','fa_text','sim_score','fa_section','fa_sub_section','fa_identifier','fa_text_length','filename']]
df.to_csv(OUTPUT_DIR+'sentence_pairs_sts_3e-05_scores.csv', index=False, encoding='utf-8-sig')