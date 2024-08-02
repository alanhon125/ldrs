import pandas as pd
import os
import re

if __name__=="__main__":
    PROJ_DIR = '/home/data/ldrs_analytics'
    OUTPUT_folderpath = f"{PROJ_DIR}/data/term_matching_csv"
    
    check_csv_path = os.path.join(OUTPUT_folderpath, '20230925_tm7_wb_finetune', 'check/')
    topN = 10
    for f in os.listdir(check_csv_path):
        print(f)
        df_match = pd.read_csv(check_csv_path+f)
        flag = 0
        final = []
        for key in list(set(df_match.ts_key)):
            df_s = df_match[df_match.ts_key==key]
            for text in list(set(df_s.ts_value)):
                df_ss = df_s[df_s.ts_value==text]
                try:
                    final.append({
                        'index': list(df_ss['index'])[0],
                        'text_block_id': list(df_ss['text_block_id'])[0],
                        'page_id': list(df_ss['page_id'])[0],
                        'phrase_id': list(df_ss['phrase_id'])[0],
                        'TS_term': key,
                        'TS_text': text,
                        'match_term_list': list(df_ss['fa_text'])[:topN],
                        'identifier_list': list(df_ss['identifier'])[:topN],
                        'similarity_list': list(df_ss['similarity'])[:topN],
                        'match_type_list': list(df_ss['match_type'])[:topN]
                    })
                except Exception as e:
                    flag = 1
                    print(f'TS_term: {term}, TS_text: {text}, Error: {e}')

        df_final = pd.DataFrame(data=final)
        df_final = df_final.sort_values(by=['index'])

        save_f = re.sub('.csv', '_results.csv', f)
        results_csv_path = os.path.join(OUTPUT_folderpath, '20230925_tm7_wb_finetune_top10', save_f)
        df_final.to_csv(results_csv_path, index=False)