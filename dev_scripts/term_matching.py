import pandas as pd
import numpy as np
import re
import os
from sentence_transformers import SentenceTransformer, util
from collections import defaultdict
import time
import json

MODELS = [
    "/home/data/ldrs_analytics/models/all-MiniLM-L6-v2",
    "/home/data/ldrs_analytics/models/all-MiniLM-L6-v2-train_nli-boc_17398_epoch_100_lr_1e-05",
    "/home/data/ldrs_analytics/models/all-MiniLM-L6-v2-train_sts-boc_17398_epoch_100_lr_1e-05"
]
MODEL_PATH = MODELS[2]
SIM_MODEL = SentenceTransformer(MODEL_PATH)

# TODO: add config for similarity threshold in each strategy.

def timeit(func):
    from functools import wraps
    import time
    LOG_DIR = '/home/data/ldrs_analytics/data/log'

    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        row = {'task': func.__name__,
               'filename': args[0].fname,
               'runtime': total_time}
        log2csv(LOG_DIR + '/log_term_matching.csv', row)
        # print(f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
        return result
    return timeit_wrapper

@timeit
def log2csv(csv_name, row):
    '''
    log the record 'row' into path 'csv_name'
    '''
    import csv
    import os.path

    file_exists = os.path.isfile(csv_name)
    # Open the CSV file in "append" mode
    with open(csv_name, 'a', newline='') as f:
        # Create a dictionary writer with the dict keys as column fieldnames
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader() # file doesn't exist yet, write a header
            # Append single row to CSV
        writer.writerow(row)

def process_str(s):
    if not s or s == np.nan:
        return None
    s = re.sub('<s>', '', s)
    s = s.strip()
    return s

def process_ts_section(section):
    ''' clean parsed term in TS
    '''
    processed_section = ''
    try:
        processed_section = process_str(re.split('  ', section)[0])
    except Exception as e:
        print(section, e)
    return processed_section


def get_similarity(
    ts_string_list,
    fa_string_list,
    map_type,
    sim_threshold=0.6,
    model_path=MODEL_PATH,
    pretrained=SIM_MODEL,
    top_N=5
):
    from sentence_transformers import SentenceTransformer, util
    import torch
    
    ts_string_list = [str(i) for i in ts_string_list]
    fa_string_list = [str(i) for i in fa_string_list]
    
    if pretrained is not None:
        # model = SentenceTransformer(pretrained, device=torch.device("cuda", 2))
        model = pretrained
    else:
        model = SentenceTransformer(model_path, device=torch.device("cuda", 2))
        
    embeddings1 = model.encode(ts_string_list, convert_to_tensor=True)
    embeddings2 = model.encode(fa_string_list, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)
    
    similar_pairs = {}
    for i in range(len(ts_string_list)):
        all_score = list(cosine_scores[i])
        above_threshold_idx = [all_score.index(k) for k in [j for j in all_score if j >= sim_threshold]]
        above_threshold_sims = [j.item() for j in all_score if j >= sim_threshold]
        idx_sims = list(zip(above_threshold_idx, above_threshold_sims))

        idx_sims = sorted(idx_sims, key=lambda x: x[1], reverse=True)[:top_N]
        ref_string = []
#         sim_score = []
        
        map_results = []
        for idx, sims in idx_sims:
            string = fa_string_list[idx]
            if string not in ref_string:
                ref_string.append(string)
                map_results.append({
                    'similar_term': string,
                    'score': round(sims, 2),
                    'map_type': map_type
                })
        if ref_string:
            similar_pairs.update({
                ts_string_list[i]: map_results
            })
    return similar_pairs

def backfill_tm(tm_file):
    ''' backfill term matching results: combine the text with empty predictions under same index/phrase_id
    '''
    pass


if __name__=="__main__":
    # 17, 41
    PROJ_DIR = '/home/data/ldrs_analytics'
    check_date = "20230807_sts"
    TS_folderpath = f"{PROJ_DIR}/data/annotated_ts/bfilled"
    FA_folderpath = f"{PROJ_DIR}/data/docparse_csv/FA"
    OUTPUT_folderpath = f"{PROJ_DIR}/data/term_matching_csv"
    
    if os.path.exists(os.path.join(OUTPUT_folderpath, f"done_term_matching_list_{check_date}.json")):
        with open(os.path.join(OUTPUT_folderpath, f"done_term_matching_list_{check_date}.json"), 'r') as f:
            done_list = json.load(f)
    else:
        done_list = []
    print('done list: ', done_list)
    
    ts_files = []
    for file in os.listdir(TS_folderpath):
        if file.endswith('.csv'):
            # print(file)
            ts_files.append(file)
    
    ts_files = [
        '6_GL_SYN_TS_mkd_docparse.csv',
        '65_PF_SYN_TS_mkd_docparse.csv',
        '4_GF_SYN_TS_mkd_20221008_docparse.csv',
        '79_PF_SYN_TS_mkd_20191129_docparse.csv',
        '64_AWSP_SYN_TS_mkd_20220622_docparse.csv',
        '71_NBFI_SYN_TS_mkd_20221103_docparse.csv',
        '18_NBFI_SYN_TS_mkd_20230131_docparse.csv',
        '61_PF_SYN_TS_mkd_20210930_docparse.csv'
    ]
    for f in ts_files:
        if f not in done_list:
            print('Processing: ', f)
            start_time = time.perf_counter()
            ts_file = os.path.join(TS_folderpath, f)
            fa_f = re.sub('_TS_', '_FA_', f)
            fa_file = os.path.join(FA_folderpath, fa_f)
        
            # FA file
            df_fa = pd.read_csv(fa_file).astype(str)
            df_fa['index'] = df_fa['index'].astype(int)
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
            # definition
            df_def = df_fa[df_fa.sub_section.str.contains('Definition', na=False, case=False)] # cols: definition + text
            df_def = df_def[~df_def.definition.isnull()]
            # exclude parties & definition clause
            df_others = df_fa[
                ~(df_fa.section.str.contains("INTERPRETATION", na=False, case=False)) & (df_fa.section_id  != "0")
            ]
            # schedule
            df_schedule = df_others.loc[df_others.schedule.notnull()]
            # main clause
            df_clause = df_others.loc[~df_others.schedule.notnull()]

            # TS file
            df_ts = pd.read_csv(ts_file)
            df_ts["processed_section"] = df_ts["section"] #.apply(lambda i: process_ts_section(i))
            print('Data is loaded. Start to do term matching ...')

            # TS term section vs. FA definition
            top_N = 5

            ts_section_list = list(set(df_ts.processed_section))
            ts_section_list = [s for s in ts_section_list if s]
            def_string_list = list(set(df_def.definition))
            def_string_list = [s for s in def_string_list if s]

            pairs_def = dict()
            if ts_section_list and def_string_list:
                pairs_def = get_similarity(
                    ts_section_list,
                    def_string_list,
                    "sec_to_def",
                    sim_threshold=0.9
                )
            print('Completed: term to def')

            # TS term section vs. FA parties
            parties_string_list = []
            for s in list(set(df_parties.definition)):
                if s:
                    if s not in parties_string_list:
                        if isinstance(s, str):
                            parties_string_list.append(s)
            pairs_parties = dict()
            if ts_section_list and parties_string_list:
                pairs_parties = get_similarity(
                    ts_section_list,
                    parties_string_list,
                    "sec_to_parties",
                    sim_threshold=0.5
                )
            print('Completed: term to parties')

            # TS term v.s. FA sub section
            sub_sec_list = list(set(df_clause[df_clause.text_element == "sub_section"].sub_section))
            sub_sec_list = [s for s in sub_sec_list if s]
            pairs_sec_to_sub_sec = dict()
            if ts_section_list and sub_sec_list:
                pairs_sec_to_sub_sec = get_similarity(
                    ts_section_list,
                    sub_sec_list,
                    "sec_to_sub_sec",
                    sim_threshold=0
                )
            print('Completed: term to sub section')
            
            # TS term + text vs. FA clause
            clause_section_list = list(set(df_clause[df_clause.text_element == "section"].section))
            clause_section_list = [s for s in clause_section_list if s]
            
            # TS section v.s. FA clause section -> select potential FA clause section
            
            total_pairs_clause = []
            pairs_clause_section = dict()
            if ts_section_list and clause_section_list:
                pairs_clause_section = get_similarity(
                    ts_section_list,
                    clause_section_list,
                    "clause_section",
                    sim_threshold=0.3
                )
            
                # check under the section candidates
                for k, v in pairs_clause_section.items():
                    df_ts_sub = df_ts[df_ts.processed_section == k]    
                    # TODO: improve this part, duplicates         
                    ts_text_list = [] # process nan value
                    for s in list(set(df_ts_sub[df_ts_sub.text_element!='section'].text)):
                        if s:
                            if s not in ts_text_list:
                                if isinstance(s, str):
                                    ts_text_list.append(s)
                    sub_ts_section_list = list(set(df_ts_sub.processed_section))

                    candidates = [item['similar_term'] for item in v]
                    df_clause_sub = df_clause[df_clause.section.isin(candidates)]
                    # print(f'ts_section: {k}, ts_section_length: {len(df_ts_sub)}, ts_text_length: {len(ts_text_list)}, fa_section_length: {len(df_clause_sub)}')

                    # TODO: improve this part
                    sub_section_list = [] # process nan value
                    for s in list(set(df_clause_sub[df_clause_sub.text_element == "sub_section"].sub_section)):
                        if s:
                            if s not in sub_section_list:
                                if isinstance(s, str):
                                    sub_section_list.append(s)
                    clause_string_list = list(set(
                        df_clause_sub[(df_clause_sub.text_element != "section") & (df_clause_sub.text_element != "sub_section")].text
                    ))

                    pairs_sub_section = dict()
                    if sub_section_list:
                        if ts_text_list:
                            pairs_sub_section = get_similarity(
                                ts_text_list,
                                sub_section_list,
                                "text_to_sub_sec",
                                sim_threshold=0
                            )
                    pairs_clause = dict()
                    if ts_text_list and clause_string_list:
                        pairs_clause = get_similarity(
                            ts_text_list,
                            clause_string_list,
                            "text_to_clause_text",
                            sim_threshold=0
                        )

                    if pairs_sub_section:
                        # case: same text under different terms
                        total_pairs_clause.append({
                            k: pairs_sub_section
                        })
                    if pairs_clause:
                        total_pairs_clause.append({k: pairs_clause})
            print('Completed: clause')
            
            ### add TS term + text vs. FA schedule
            # 0729: add "part" in schedule; TS term vs. FA schedule part
            ts_section_list = list(set(df_ts.processed_section))
            ts_section_list = [s for s in ts_section_list if s]
            schedule_section_list = list(set(df_schedule.schedule))
            schedule_part_list = []

            for s in list(set(df_schedule.part)):
                if s:
                    if s not in schedule_part_list:
                        if isinstance(s, str):
                            schedule_part_list.append(s)

            pairs_schedule_section = dict()
            if ts_section_list and schedule_section_list:
                pairs_schedule_section = get_similarity(
                    ts_section_list,
                    schedule_section_list,
                    "schedule_section",
                    sim_threshold=0.8
                )
            
            pairs_schedule_part = dict()
            if ts_section_list and schedule_part_list:
                pairs_schedule_part = get_similarity(
                    ts_section_list,
                    schedule_part_list,
                    "schedule_part",
                    sim_threshold=0
                )
            
            # TODO: combine sections based on same schedule section -> save time
            total_pairs_sched = []
            if ts_section_list and schedule_section_list:
                print("pairs_schedule_section", len(pairs_schedule_section))
                print(pairs_schedule_section)
                # check under the section candidates
                for k, v in pairs_schedule_section.items():
                    df_ts_sub = df_ts[df_ts.processed_section == k]
                    ts_text_list = list(set(df_ts_sub[df_ts_sub.text_element!='section'].text))
                    
                    candidates = [item['similar_term'] for item in v]
                    df_sched_sub = df_schedule[df_schedule.schedule.isin(candidates)]
                    sched_text_list = list(set(df_sched_sub.text))
                    pairs_sched = dict()
                    print(f'ts_text_length: {len(ts_text_list)}, schedule_text_length: {len(sched_text_list)}')
                    if ts_text_list and sched_text_list:
                        pairs_sched = get_similarity(
                            ts_text_list,
                            sched_text_list,
                            "text_to_schedule_text",
                            sim_threshold=0
                        )
                    if pairs_sched:
                        total_pairs_sched.append({k: pairs_sched})
            print('Completed: schedule')

            # summarize all results

            df_ts['similar_def'] = df_ts['processed_section'].apply(
                lambda i: pairs_def.get(i)
            )
            df_ts['similar_parties'] = df_ts['processed_section'].apply(
                lambda i: pairs_parties.get(i)
            )
            df_ts['similar_sub_section'] = df_ts['processed_section'].apply(
                lambda i: pairs_sec_to_sub_sec.get(i) if sub_sec_list else None
            )
            df_ts['similar_schedule'] = df_ts['processed_section'].apply(
                lambda i: pairs_schedule_part.get(i)
            )
            df_ts['similar_sched_section'] = df_ts['processed_section'].apply(
                lambda i: pairs_schedule_section.get(i)
            )

            df_ts_def = df_ts[~df_ts.similar_def.isna()][['section', 'processed_section', 'text','similar_def']]
            df_ts_parties = df_ts[~df_ts.similar_parties.isna()][['section', 'processed_section',  'text','similar_parties']]
            df_ts_sub_sec = df_ts[~df_ts.similar_sub_section.isna()][['section', 'processed_section',  'text', 'similar_sub_section']]
            df_ts_sched = df_ts[~df_ts.similar_schedule.isna()][['section', 'processed_section',  'text', 'similar_schedule']]
            df_ts_sched_sec = df_ts[~df_ts.similar_sched_section.isna()][['section', 'processed_section',  'text', 'similar_sched_section']]

            ### check
            total_pairs = []

            for sec in list(set(df_ts_def.processed_section)):
                sub_df = df_ts_def[df_ts_def.processed_section==sec]
                total_pairs.append({
                    sec: dict(zip(sub_df.text, sub_df.similar_def))
                })

            for sec in list(set(df_ts_parties.processed_section)):
                sub_df = df_ts_parties[df_ts_parties.processed_section==sec]
                total_pairs.append({
                    sec: dict(zip(sub_df.text, sub_df.similar_parties))
                })

            for sec in list(set(df_ts_sub_sec.processed_section)):
                sub_df = df_ts_sub_sec[df_ts_sub_sec.processed_section==sec]
                total_pairs.append({
                    sec: dict(zip(sub_df.text, sub_df.similar_sub_section))
                })
            for sec in list(set(df_ts_sched.processed_section)):
                sub_df = df_ts_sched[df_ts_sched.processed_section==sec]
                total_pairs.append({
                    sec: dict(zip(sub_df.text, sub_df.similar_schedule))
                })
            for sec in list(set(df_ts_sched_sec.processed_section)):
                sub_df = df_ts_sched_sec[df_ts_sched_sec.processed_section==sec]
                total_pairs.append({
                    sec: dict(zip(sub_df.text, sub_df.similar_sched_section))
                })
            
            total_pairs.extend(total_pairs_clause)
            total_pairs.extend(total_pairs_sched)
            

            keys = []
            for pair in total_pairs:
                k = list(pair.keys())[0]
                if k not in keys:
                    keys.append(k)


            total_pairs_updated = []
            for k in keys:
                sub_pairs = [p[k] for p in total_pairs if list(p.keys())[0] == k]
                dd = defaultdict(list)
                for p in sub_pairs:
                    for i, j in p.items():
                        dd[i].extend(j)
                
                total_pairs_updated.append({k: dd})
            
            results = []

            for pair in total_pairs_updated:
                for sec, value in pair.items():
                    for text, match in value.items():
                        for item in match:
                            results.append({
                                'TS_section': sec,
                                'TS_text': text,
                                'match_term': item['similar_term'],
                                'similarity': item['score'],
                                'match_type': item['map_type']
                            })
            
            df_results = pd.DataFrame(data=results)
            # print(json.dumps(results,indent=4))
            # df_results = df_results.sort_values(by=['TS_text', 'similarity'], ascending=False)

            ts_map = {}
            for idx, row in df_ts[['index','text_block_id','page_id', 'phrase_id', 'section','processed_section','text']].drop_duplicates().iterrows():
                ts_map[row['text']] = [
                    row['index'],
                    row['text_block_id'],
                    row['page_id'],
                    row['phrase_id']
                ]

            content2id = {
                'sec_to_def': dict(),
                'text_to_clause_text': dict(),
                'text_to_sub_sec': dict(),
                'text_to_schedule_text': dict(),
                'sec_to_parties': dict(),
                'sec_to_sub_sec': dict(),
                'schedule_part': dict(),
                'schedule_section': dict()
            }

            for idx, row in df_def[['definition', 'identifier']].drop_duplicates().iterrows():
                content2id['sec_to_def'].update({row['definition']: row['identifier']})

            for idx, row in df_fa[df_fa.text_element=='sub_section'][['sub_section', 'identifier']].drop_duplicates().iterrows():
                content2id['text_to_sub_sec'].update({row['sub_section']: row['identifier']})
            content2id['sec_to_sub_sec'] = content2id['text_to_sub_sec']
            for idx, row in df_clause[['text', 'identifier']].drop_duplicates().iterrows():
                content2id['text_to_clause_text'].update({row['text']: row['identifier']})


            for idx, row in df_schedule[['text', 'identifier']].drop_duplicates().iterrows():
                content2id['text_to_schedule_text'].update({row['text']: row['identifier']})
            for idx, row in df_schedule[['part', 'identifier']].drop_duplicates().iterrows():
                content2id['schedule_part'].update({row['part']: row['identifier']})
            for idx, row in df_schedule[df_schedule.text_element=='section'][['schedule', 'identifier']].drop_duplicates().iterrows():
                content2id['schedule_section'].update({row['schedule']: row['identifier']})

            for idx, row in df_parties[['definition', 'identifier']].drop_duplicates().iterrows():
                content2id['sec_to_parties'].update({row['definition']: row['identifier']})

            df_results['identifier'] = df_results.apply(
                lambda i: content2id[i['match_type']].get(i['match_term']),
                axis=1
            )

            df_results['index'] = df_results.apply(
                lambda i: ts_map.get(i['TS_text'])[0],
                axis=1
            )
            df_results['text_block_id'] = df_results.apply(
                lambda i: ts_map.get(i['TS_text'])[1],
                axis=1
            )
            df_results['page_id'] = df_results.apply(
                lambda i: ts_map.get(i['TS_text'])[2],
                axis=1
            )
            df_results['phrase_id'] = df_results.apply(
                lambda i: ts_map.get(i['TS_text'])[3],
                axis=1
            )

            df_results = df_results.sort_values(
                by=['TS_section', 'TS_text', 'similarity'],
                ascending=False
            )
            df_results = df_results[~df_results.match_term.isna()]
            df_results = df_results.drop_duplicates()
            check_csv_path = os.path.join(OUTPUT_folderpath, check_date, 'check', f)
            df_results.to_csv(check_csv_path, index=False)
            
            final = []

            for term in list(set(df_results.TS_section)):
                df_s = df_results[df_results.TS_section==term]
                for text in list(set(df_s.TS_text)):
                    df_ss = df_s[df_s.TS_text==text]
                    try:
                        final.append({
                            'index': list(df_ss['index'])[0],
                            'text_block_id': list(df_ss['text_block_id'])[0],
                            'page_id': list(df_ss['page_id'])[0],
                            'phrase_id': list(df_ss['phrase_id'])[0],
                            'TS_term': term,
                            'TS_text': text,
                            'match_term_list': list(df_ss['match_term'])[:5],
                            'identifier_list': list(df_ss['identifier'])[:5],
                            'similarity_list': list(df_ss['similarity'])[:5],
                            'match_type_list': list(df_ss['match_type'])[:5]
                        })
                    except Exception as e:
                        print(f'TS_term: {term}, TS_text: {text}, Error: {e}')
            df_final = pd.DataFrame(data=final)
            save_f = re.sub('.csv', '_results.csv', f)
            results_csv_path = os.path.join(OUTPUT_folderpath, check_date, save_f)
            df_final.to_csv(results_csv_path, index=False)
            # print('results are saved to: ', save_f)
            end_time = time.perf_counter()
            total_time = end_time - start_time
            print(f'{f} matching time: {total_time} s')
            done_list.append(f)
            
            with open(os.path.join(OUTPUT_folderpath, f"done_term_matching_list_{check_date}.json"), "w") as outfile:
                outfile.write(json.dumps(done_list,indent=4))             