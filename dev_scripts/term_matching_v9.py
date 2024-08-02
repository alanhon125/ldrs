import re
import pandas as pd
import json
import numpy as np
import os
import time
from sentence_transformers import SentenceTransformer, util


# model_name = 'thenlper/gte-base'
model_name = 'models/gte-base-train_nli-boc_epoch_20_lr_1e-05_batch_32_0103'
model = SentenceTransformer(model_name)
print('max sentence length: {}'.format(model.get_max_seq_length()))

def similarity(embeddings1, sentences2):
    #Compute embedding for both lists
    #embeddings1 = model.encode(sentences1, convert_to_tensor=True)
    embeddings2 = model.encode(sentences2, convert_to_tensor=True)

    #Compute cosine-similarits
    cosine_scores = util.cos_sim(embeddings1, embeddings2)
    #cosine_scores = util.dot_score(embeddings1, embeddings2)
    return cosine_scores

def match(ts_sents, agr_sents, ts_dict, agr_keys, type_str):
    scores = similarity(ts_sents, agr_sents)
    for term_index, (term_key, term_item) in enumerate(ts_dict):
        rank_list = []
        for agr_index, agr_key in enumerate(agr_keys):
            rank_list.append(["Clause: " + agr_key, scores[term_index][agr_index].item()])
        rank_list = list(sorted(rank_list, key=lambda x: x[1], reverse=True)) # [:5]
        if len(rank_list) > 0:
            term_sheet_result_dict[term_key][term_item][type_str] = rank_list

def get_split_list(key, text):
    if text == "":
        text = key
        text = re.sub(r"([1-9]\d*\.?\d*)|(0\.\d*[1-9])", "", text).strip()
    split_list = []
    temp_text = ""
    if text:
        for x in filter(lambda x: x!= "", text.split(".")):
            x += "."
            if len((temp_text + x).split(" ")) > 128:
                split_list.append(temp_text.strip())
                temp_text = x
            else:
                temp_text += x
        split_list.append(temp_text.strip())
        split_list = list(filter(lambda x: x.strip()!= "", split_list))
    return split_list

def gen_match_key(match_place):
    [p, t] = match_place.split("]: ")[:2]
    # p = p.split("_to_")[1].split("_")[0]
    return t.strip()

def key_or_value(match_place):
    return match_place.split("_")[0].replace("[", "")

def filter_annotation(anno):
    return any([True if x in anno.lower() else False for x in ["definition", "parties", "clause", "schedule"]])

def cal_score(kv_list):
    max_score = 0
    for x in kv_list:
        if x[1] > max_score:
            max_score = x[1]
    return max_score

def cal_score_clause(kv_list):
    sum_score = 1
    for x in kv_list:
        if x[0] == "key":
            sum_score *= x[1]
        elif x[0] == "value":
            sum_score *= x[1]
        elif x[0] == "value2":
            sum_score *= x[1]
    return sum_score

def cal_score_clause_key(kv_list):
    max_score = 0
    for x in kv_list:
        if x[0] == "key":
            if x[1] > max_score:
                max_score = x[1]
    return max_score

def cal_score_clause_value(kv_list):
    max_score = 0
    for x in kv_list:
        if x[0] == "value":
            if x[1] > max_score:
                max_score = x[1]
        elif x[0] == "value2":
            if x[1] > max_score:
                max_score = x[1]
    return max_score

def get_fatext(match_place):
    return re.sub('Clause: ', '', match_place)

def find_syn(s):
    if s:
        s = s.lower()
        for idx, row in df_wb.iterrows():
            row = [r for r in row if r]
            if len(row) > 1:
                for i in range(1, len(row)):
                    if re.search(row[i], s):
                        find = re.findall(row[i], s)[0]
                        if len(find) == len(s):
                            s = re.sub(row[i], row[0], s)
                            break
    return s

def concat_features_to_text(row, doc_type):
    '''
    @param row: pandas.Series object with columns ['text', 'section', 'fa_text', 'fa_section', 'fa_sub_section', 'parent_caption', 'parent_list']
    @param doc_type: either 'TS' or 'FA'
    @return: return a concatenated string of useful features (section, sub-section, caption, list item and main text) as a complete content
    @rtype: str
    '''
    parent_caption = parent_list = None
    if 'parent_caption' in row.columns.tolist():
        parent_caption = row['parent_caption']
    if 'parent_list' in row.columns.tolist():
        parent_list = row['parent_list']
        
    text = row['text']
    if doc_type == 'FA':
        fa_section = row['section']
        fa_sub_section = row['sub_section']
        if fa_sub_section and fa_section and text:
            if parent_caption and parent_list:
                return fa_section + ' - ' + fa_sub_section + ': ' + parent_caption + ' ' + parent_list + ': ' + text
            elif parent_caption:
                return fa_section + ' - ' + fa_sub_section + ': ' + parent_caption + ' ' +  text
            elif parent_list:
                return fa_section + ' - ' + fa_sub_section + ': ' + parent_list + ': ' + text
            else:
                return fa_section + ' - ' + fa_sub_section + ': ' + text
        elif fa_sub_section and not fa_section and text:
            if parent_caption and parent_list:
                return fa_sub_section + ': ' + parent_caption + ' ' + parent_list + ': ' + text
            elif parent_caption:
                return fa_sub_section + ': ' + parent_caption + ' ' + text
            elif parent_list:
                return fa_sub_section + ': ' + parent_list + ': ' + text
            else:
                return fa_sub_section + ': ' + text
        elif not fa_sub_section and fa_section and text:
            if parent_caption and parent_list:
                return fa_section + ': ' + parent_caption + ' ' + parent_list + ': ' + text
            elif parent_caption:
                return fa_section + ': ' + parent_caption + ' ' + text
            elif parent_list:
                return fa_section + ': ' + parent_list + ': ' + text
            else:
                return fa_section + ': ' + text
        else:
            if parent_caption and parent_list:
                return parent_caption + ' ' + parent_list + ': ' + text
            elif parent_caption:
                return parent_caption + ' ' + text
            elif parent_list:
                return parent_list + ': ' + text
            else:
                return text

    elif doc_type == 'TS':

        section = row['section']
        if section and text:
            if parent_caption and parent_list:
                return section + ': ' + parent_caption + ' ' + parent_list + ': ' + text
            elif parent_caption:
                return section + ': ' + parent_caption + ' ' + text
            elif parent_list:
                return section + ': ' + parent_list + ': ' + text
            else:
                return section + ': ' + text
        else:
            if parent_caption and parent_list:
                return parent_caption + ' ' + parent_list + ': ' + text
            elif parent_caption:
                return parent_caption + ' ' + text
            elif parent_list:
                return parent_list + ': ' + text
            else:
                return text

def multiprocess(function, input_list):
    '''
    multiprocessing the function at a time

    @param function: a function
    @type function: def
    @param input_list: a list of input that accept by the function
    @type input_list: list
    '''
    import multiprocessing

    input_length = len(input_list)
    num_processor = multiprocessing.cpu_count()
    print(f'there are {num_processor} CPU cores')
    batch_size = max(input_length // num_processor, 1)
    num_batch = int(input_length / batch_size) + \
        (input_length % batch_size > 0)
    pool = multiprocessing.Pool(num_processor)
    processes = [pool.apply_async(function, (input_list[idx * batch_size:(idx + 1) * batch_size],)) for idx in
                 range(num_batch)]
    results = [p.get() for p in processes]

    if all(r for r in results):
        return list(zip(*results))
    
if __name__=="__main__":
    PROJ_DIR = 'ldrs_analytics'
    
    APPLY_WORDBANK = False
    EXPORT_DONE_LIST = True
    top_N = 20
    
    TS_version = '_v4.3'
    
    FA_version = '_v4.4'
    check_date = f"tm_ts{TS_version}_fa{FA_version}_0116"
    
    DISPLAY = True
    if DISPLAY:
        TS_folderpath = f"{PROJ_DIR}/data/docparse_csv/TS" + TS_version
        print('ts folder: ', TS_folderpath)
    else:
        TS_folderpath = f"{PROJ_DIR}/data/reviewed_antd_ts_v3.1/bfilled"
    
    FA_folderpath = f"{PROJ_DIR}/data/docparse_csv/FA" + FA_version
    
    OUTPUT_folderpath = f"{PROJ_DIR}/data/term_matching_csv"
    CHECK_CSV_FOLDER = os.path.join(OUTPUT_folderpath, check_date, 'check')
    RESULTS_CSV_FOLDER = os.path.join(OUTPUT_folderpath, check_date)
    
    if not os.path.exists(RESULTS_CSV_FOLDER):
        os.mkdir(RESULTS_CSV_FOLDER)
    if not os.path.exists(CHECK_CSV_FOLDER):
        os.mkdir(CHECK_CSV_FOLDER)
        
    # if APPLY_WORDBANK:
    #     df_wb = pd.read_excel(PROJ_DIR+'/data/wordbank.xlsx', header=None)
    #     df_wb = df_wb.fillna('')
    #     df_wb = df_wb.apply(lambda col: col.str.lower())
        
    if EXPORT_DONE_LIST:
        done_list = []
    if os.path.exists(os.path.join(OUTPUT_folderpath, f"done_term_matching_list_{check_date}.json")):
        with open(os.path.join(OUTPUT_folderpath, f"done_term_matching_list_{check_date}.json"), 'r') as f:
            done_list = json.load(f)
        done_files = [item['file'] for item in done_list]
    else:
        done_files = []
        done_list = []
    print(f'completed term matching file list: {done_files}\nThe above list will be excluded in performing term matching')
    
    ts_files = []
    for file in os.listdir(TS_folderpath):
        if file.endswith('.csv'):
            # print(file)
            ts_files.append(file)

    # load checked reviewed_antd_ts filename list from json
    # if os.path.exists("/home/a2383/boc_finetune/BOC/analytics/data/reviewed_antd_ts/checked_list.json"):
    #     with open("/home/a2383/boc_finetune/BOC/analytics/data/reviewed_antd_ts/checked_list.json", 'r') as f:
    #         ts_files = json.load(f)
    
    # ts_files = [
    #     "15_AW_SYN_TS_mkd_20220826_docparse.csv",
    #     "76_NBFI_SYN_TS_mkd_20211231_docparse.csv"
    # ]
    
    exceptions = []
    
    for f in ts_files:
        if f not in done_files + exceptions:
            print('Processing: ', f)
            start_time = time.perf_counter()
            ts_file = os.path.join(TS_folderpath, f)
            fa_f = re.sub('_TS_', '_FA_', f)
            fa_f = re.sub('TS_', 'FA_', fa_f)
            if DISPLAY:
                if fa_f == '17_NBFI_SYN_FA_mkd_20221123_docparse.csv':
                    fa_f = '17_NBFI_SYN_FA_mktd_20221123_docparse.csv'
            fa_file = os.path.join(FA_folderpath, fa_f)

            # TS file
            df_ts = pd.read_csv(ts_file)
            df_ts = df_ts.replace({np.nan: None, 'nan': None, 'None': None})
            df_ts["processed_section"] = df_ts["section"] #.apply(lambda i: process_ts_section(i))
            df_ts['processed_text'] = df_ts.apply(
                lambda i: concat_features_to_text(i, 'TS'),
                axis=1
            )
            df_ts = df_ts[~df_ts.processed_text.isna()]
            
            # FA file
            df_fa = pd.read_csv(fa_file).astype(str)
            df_fa['index'] = df_fa['index'].astype(int)
            df_fa = df_fa.replace({np.nan: None, 'nan': None, 'None': None})
            df_fa['processed_fa_text'] = df_fa.apply(lambda i: concat_features_to_text(i, 'FA'), axis=1)
            
            # parties
            isPartiesStart = df_fa.text.str.contains('^THIS AGREEMENT is|is made on|^PARTIES|Between:*', na=False, case=False)
            isPartiesEnd = df_fa.text.str.contains('IT IS AGREED*:*|AGREED* as follows|AGREED* that', na=False, case=False)
            partiesBeginID = df_fa[isPartiesStart]['index'].values[0] + 1
            partiesEndID = df_fa[isPartiesEnd]['index'].values[0] - 1
            parties_clause_id = df_fa['index'].between(isPartiesStart, isPartiesEnd)
            df_fa.loc[parties_clause_id,'section'] = 'PARTIES'
            df_fa.loc[parties_clause_id, 'section_id'] = '0'

            df_fa = df_fa.replace({np.nan: None, 'nan': None, 'None': None})

            # df_parties = df_fa[(df_fa.section_id == "0") | (df_fa.section_id == 0)] # cols: definition + text
            df_parties = df_fa[df_fa.section=='PARTIES'] # line 149 to string, some section ids are like 0.0, not int.
            df_parties = df_parties[~df_parties.definition.isna()]
            # definition
            df_def = df_fa[~df_fa.definition.isna()]
            df_def = df_def.drop_duplicates(subset=['section_id','sub_section_id','definition'])

            # exclude parties & definition clause
            df_others = df_fa[
                ~(df_fa.section.str.contains("INTERPRETATION", na=False, case=False)) & (df_fa.section_id  != "0") & (df_fa.section!='PARTIES')&(df_fa.section!='TABLE OF CONTENTS')
            ]
            # schedule
            df_sched = df_others.loc[df_others.schedule.notnull()]
            # main clause
            df_clause = df_others.loc[~df_others.schedule.notnull()]
            df_clause = df_clause[~df_clause.section.isna()]
            
            FA_TEXT_DICT = dict(zip(
                df_fa.processed_fa_text,
                df_fa.text
            ))
            
            df_map = df_fa[['processed_fa_text', 'identifier']].drop_duplicates()

            df_def_map = df_def[['definition', 'identifier']].drop_duplicates().rename(columns={'definition': 'processed_fa_text'})
            df_parties_map = df_parties[['definition', 'identifier']].drop_duplicates().rename(columns={'definition': 'processed_fa_text'})
            df_map_short = pd.concat([df_def_map, df_parties_map])

            df_map_all = pd.concat([df_map, df_map_short])

            # compute cosine similarity scores
            l1 = list(df_ts.processed_text)
            l2 = list(set(df_map_all.processed_fa_text))
            l3 = list(set(df_map_short.processed_fa_text))

            l2 = [t for t in l2 if t]
            l3 = [t for t in l3 if t]

            e1 = model.encode(l1, convert_to_tensor=True)
            e2 = model.encode(l2, convert_to_tensor=True)
            e3 = model.encode(l3, convert_to_tensor=True)
            
            cosine_scores = util.cos_sim(e1, e2)
            cosine_scores_short = util.cos_sim(e1, e3)
            
            added_cols = [
                'section',
                'processed_text',
                'fa_text',
                'identifier',
                'similarity',
                'match_type'
            ]

            ts_cols = [
                'index',
                'text_block_id',
                'page_id',
                'phrase_id',
                'list_id',
                'text_element',
            ]
            
            data = {k: [] for k in ts_cols + added_cols}
            count = 0
            top_N_short = 20
            for i, row in df_ts.iterrows():
                if row['text_element'] == 'section':
                    sims = cosine_scores_short[count]
                    sims = sims.cpu().numpy()
                    # print(sims.cpu())
                    top_sims_idx = np.array(sims).argsort()[-1:-top_N_short-1:-1]
                    #top_sims_idx = np.array(sims).argsort()[-1::-1]
                    top_scores = [sims[idx].item() for idx in top_sims_idx]
                    top_fa_text = [l3[idx] for idx in top_sims_idx]
                    
                    all_top_idf = []
                    all_top_scores = []
                    all_top_fa_text = []
                    for j in range(len(top_fa_text)):
                        fa_text = top_fa_text[j]
                        idf = df_map_short[df_map_short.processed_fa_text==fa_text].identifier
                        if not idf.empty:
                            no_of_idf = len(list(idf))
                            all_top_idf.extend(list(idf))
                            all_top_fa_text.extend([fa_text]*no_of_idf)
                            all_top_scores.extend([top_scores[j]]*no_of_idf)
                        else:
                            all_top_idf.append(None)
                            all_top_fa_text.append(fa_text)
                            all_top_scores.append(top_scores[j])

                    length = len(all_top_idf)
                    for col in ts_cols:
                        data[col].extend([row[col]]*length)
                    data['section'].extend([row['section']]*length)
                    data['processed_text'].extend([row['processed_text']]*length)
                    data['fa_text'].extend(all_top_fa_text)
                    data['identifier'].extend(all_top_idf)
                    data['similarity'].extend(all_top_scores)
                    data['match_type'].extend(['dummy']*length)
                count += 1

            df_match_s = pd.DataFrame(data=data)
            df_match_s = df_match_s.sort_values(by=['section', 'processed_text', 'similarity'], ascending=False)
            df_match_s = df_match_s.drop_duplicates()
            df_match_s = df_match_s[~df_match_s.identifier.isna()].drop_duplicates(subset=['section', 'processed_text', 'identifier'])
            df_match_s['fa_text'] = df_match_s['fa_text'].apply(lambda i: FA_TEXT_DICT[i] if FA_TEXT_DICT.get(i) else i)
                    
            data = {k: [] for k in ts_cols + added_cols}
            count = 0
            for i, row in df_ts.iterrows():
                sims = cosine_scores[count]
                sims = sims.cpu().numpy()
                top_sims_idx = np.array(sims).argsort()[-1:-top_N-1:-1]
                # top_sims_idx = np.array(sims).argsort()[-1::-1]
                top_scores = [sims[idx].item() for idx in top_sims_idx]
                top_fa_text = [l2[idx] for idx in top_sims_idx]
                
                all_top_idf = []
                all_top_scores = []
                all_top_fa_text = []
                for j in range(len(top_fa_text)):
                    fa_text = top_fa_text[j]
                    idf = df_map_all[df_map_all.processed_fa_text==fa_text].identifier
                    if not idf.empty:
                        no_of_idf = len(list(idf))
                        all_top_idf.extend(list(idf))
                        all_top_fa_text.extend([fa_text]*no_of_idf)
                        all_top_scores.extend([top_scores[j]]*no_of_idf)
                    else:
                        all_top_idf.append(None)
                        all_top_fa_text.append(fa_text)
                        all_top_scores.append(top_scores[j])
                
                length = len(all_top_idf)
                for col in ts_cols:
                    data[col].extend([row[col]]*length)
                data['section'].extend([row['section']]*length)
                data['processed_text'].extend([row['processed_text']]*length)
                data['fa_text'].extend(all_top_fa_text)
                data['identifier'].extend(all_top_idf)
                data['similarity'].extend(all_top_scores)
                data['match_type'].extend(['dummy']*length)
                count += 1

            df_match = pd.DataFrame(data=data)
            df_match = df_match.sort_values(by=['section', 'processed_text', 'similarity'], ascending=False)
            df_match = df_match.drop_duplicates()
            df_match = df_match[~df_match.identifier.isna()].drop_duplicates(subset=['section', 'processed_text', 'identifier'])
            df_match['fa_text'] = df_match['fa_text'].apply(lambda i: FA_TEXT_DICT[i] if FA_TEXT_DICT.get(i) else i)
            
            final_data = []
            for idx, row in df_ts.iterrows():
                section = row['section']
                processed_text = row['processed_text']
                df_s = df_match[(df_match.section==section)&(df_match.processed_text==processed_text)]
                df_s2 = df_match_s[df_match_s.section==section]
                df_s = pd.concat([df_s, df_s2])
                df_s = df_s.sort_values(by=['similarity'], ascending=False)
                df_s = df_s.drop_duplicates(subset=['identifier'])
                
                row['TS_term'] = section
                row['TS_text'] = row['text']
                row['match_term_list'] = [None]
                row['identifier_list'] = [None]
                row['similarity_list'] = [None]
                row['match_type_list'] = [None]
                if not df_s.empty and row['text_element'] != 'section':
                    row['match_term_list'] = list(df_s['fa_text'])
                    row['identifier_list'] = list(df_s['identifier'])
                    row['similarity_list'] = list(df_s['similarity'])
                    row['match_type_list'] = list(df_s['match_type'])
                final_data.append(row)

            cols = [
                'index',
                'text_block_id',
                'page_id',
                'phrase_id',
                'list_id',
                'text_element',
                'TS_term',
                'TS_text',
                'match_term_list',
                'identifier_list',
                'similarity_list',
                'match_type_list'
            ]

            df_final = pd.DataFrame(data=final_data)[cols]
            df_final = df_final.sort_values(by=['index','phrase_id'])
            
            check_csv_path = os.path.join(OUTPUT_folderpath, check_date, 'check', f)
            df_match.to_csv(check_csv_path, index=False)
            save_f = re.sub('.csv', '_results.csv', f)
            results_csv_path = os.path.join(OUTPUT_folderpath, check_date, save_f)
            df_final.to_csv(results_csv_path, index=False)
            end_time = time.perf_counter()
            total_time = end_time - start_time
            print(f'{f} matching time: {total_time} s\n')
            if EXPORT_DONE_LIST:
                done_list.append({"file": f, 
                                  "time": total_time, 
                                  "length_of_ts_text": len(l1), 
                                  "length_of_fa_text": len(l2), 
                                  "average_time_per_pair": total_time/(len(l1)*len(l2))})
            
                with open(os.path.join(OUTPUT_folderpath, f"done_term_matching_list_{check_date}.json"), "w") as outfile:
                    outfile.write(json.dumps(done_list,indent=4))