import re
import pandas as pd
import json
import numpy as np
import os
import time
#### 2023/10/11
############ similarity function ############
from sentence_transformers import SentenceTransformer, util

# Model : max sequence length , dimension
# all-mpnet-base-v2: 384 , 768
# multi-qa-MiniLM-L6-cos-v1: 512 , 384
# all-MiniLM-L6-v2: 256 , 384
# paraphrase-MiniLM-L6-v2: 128 , 384
# model_name = 'models/all-MiniLM-L6-v2'
# model_name = 'models/all-MiniLM-L6-v2-train_nli-boc_wb_10377_epoch_20_lr_1e-05'
model_name = 'TaylorAI/gte-tiny'
model = SentenceTransformer(model_name)
# model = SentenceTransformer('msmarco-distilbert-base-tas-b')
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

def concat_section_to_text(row, doc_type):
    '''
    @param row: pandas.Series object with columns ['text', 'section', 'fa_text', 'fa_section', 'fa_sub_section']
    @param doc_type: either 'TS' or 'FA'
    '''
    text = row['text']
    if doc_type == 'FA':
        fa_section = row['section']
        fa_sub_section = row['sub_section']
        if fa_sub_section and fa_section and text:
            return fa_section + ' - ' + fa_sub_section + ': ' + text
        elif fa_sub_section and not fa_section and text:
            return fa_sub_section + ': ' + text
        elif not fa_sub_section and fa_section and text:
            return fa_section + ': ' + text
        else:
            return text
        
    elif doc_type == 'TS':
        
        section = row['section']
        if section and text:
            return section + ': ' + text
        else:
            return text

if __name__=="__main__":
    # 17, 41
    PROJ_DIR = '/home/data/ldrs_analytics'
    check_date = "ForDisplay_1020"
    DISPLAY = True
    if DISPLAY:
        TS_folderpath = f"{PROJ_DIR}/data/docparse_csv/TS_v3"
    else:
        TS_folderpath = f"{PROJ_DIR}/data/reviewed_antd_ts/bfilled"
    
    APPLY_WORDBANK = False
    
    FA_folderpath = f"{PROJ_DIR}/data/docparse_csv/FA"
    OUTPUT_folderpath = f"{PROJ_DIR}/data/term_matching_csv"
    CHECK_CSV_FOLDER = os.path.join(OUTPUT_folderpath, check_date, 'check')
    RESULTS_CSV_FOLDER = os.path.join(OUTPUT_folderpath, check_date)
    
    if not os.path.exists(RESULTS_CSV_FOLDER):
        os.mkdir(RESULTS_CSV_FOLDER)
    if not os.path.exists(CHECK_CSV_FOLDER):
        os.mkdir(CHECK_CSV_FOLDER)
    
    df_wb = pd.read_excel('data/wordbank.xlsx', header=None)
    df_wb = df_wb.fillna('')
    df_wb = df_wb.apply(lambda col: col.str.lower())

    top_N = 10
    done_list = []
    if os.path.exists(os.path.join(OUTPUT_folderpath, f"done_term_matching_list_{check_date}.json")):
        with open(os.path.join(OUTPUT_folderpath, f"done_term_matching_list_{check_date}.json"), 'r') as f:
            done_list = json.load(f)
        done_files = [item['file'] for item in done_list]
    else:
        done_files = []
    print('done list: ', done_files)
    
    ts_files = []
    for file in os.listdir(TS_folderpath):
        if file.endswith('.csv'):
            # print(file)
            ts_files.append(file)
    # ts_files = [
    #     '17_NBFI_SYN_TS_mkd_20221123_docparse.csv',
    #     '18_NBFI_SYN_TS_mkd_20230131_docparse.csv',
    #     '2_GF_SYN_TS_mkd_20221111_docparse.csv',
    #     '4_GF_SYN_TS_mkd_20221008_docparse.csv',
    #     '50_VF_PRJ_TS_mkd_20211231_docparse.csv',
    #     '51_VF_PRJ_TS_mkd_20111021_docparse.csv',
    #     '52_VF_PRJ_TS_mkd_20200225_docparse.csv',
    #     '55_VF_PRJ_TS_mkd_20081113_docparse.csv',
    #     '57_VF_PRJ_TS_mkd_20111228_docparse.csv',
    #     '5_AF_SYN_TS_mkd_20221213_docparse.csv',
    #     '60_AWSP_SYN_TS_mkd_20220628_docparse.csv',
    #     '61_PF_SYN_TS_mkd_20210930_docparse.csv',
    #     '62_PF_SYN_TS_mkd_20220331_docparse.csv',
    #     '65_PF_SYN_TS_mkd_docparse.csv',
    #     '67_PF_SYN_TS_mkd_20230328_docparse.csv',
    #     '69_PF_SYN_TS_mkd_20211216_docparse.csv',
    #     '6_GL_SYN_TS_mkd_docparse.csv',
    #     '70_PF_SYN_TS_mkd_20220520_docparse.csv',
    #     '71_NBFI_SYN_TS_mkd_20221103_docparse.csv',
    #     '73_NBFI_SYN_TS_mkd_20230224_docparse.csv',
    #     '7_GF_SYN_TS_mkd_20221130_docparse.csv',
    #     '19_GL_SYN_TS_mkd_20220718_docparse.csv',
    #     '1_GL_SYN_TS_mkd_20221215_docparse.csv',
    #     '3_GF_SYN_TS_mkd_20221018_docparse.csv',
    #     '49_VF_PRJ_TS_mkd_20151130_docparse.csv',
    #     '54_VF_PRJ_TS_mkd_20191018_docparse.csv',
    #     '58_VF_SYN_TS_mkd_20111201_docparse.csv',
    #     '59_AWSP_SYN_TS_mkd_20210814_docparse.csv',
    #     '66_PF_SYN_TS_mkd_20230106_docparse.csv',
    #     '68_PF_SYN_TS_mkd_20221104_docparse.csv',
    #     '72_NBFI_SYN_TS_mkd_20221215_docparse.csv',
    #     '74_NBFI_SYN_TS_mkd_20220401_docparse.csv',
    #     '8_GF_SYN_TS_mkd_20230215_docparse.csv',
    # ]
    
    ts_files = [
        "1_GL_SYN_TS_mkd_20221215_docparse.csv",
        "20_BL_SYN_TS_mkd_20220527_docparse.csv",
        "21_BL_SYN_TS_mktd_20220819_docparse.csv",
        "60_AWSP_SYN_TS_mkd_20220628_docparse.csv",
        "72_NBFI_SYN_TS_mkd_20221215_docparse.csv",
        "5_AF_SYN_TS_mkd_20221213_docparse.csv",
        "66_PF_SYN_TS_mkd_20230106_docparse.csv",
        "68_PF_SYN_TS_mkd_20221104_docparse.csv",
        "51_VF_PRJ_TS_mkd_20111021_docparse.csv"
    ]
    
    ts_cols = [
        'section', 'text', 'definition', 'processed_text'
    ]
    fa_cols = [
        'section', 'sub_section', 'schedule', 'part', 'definition', 'text', 'processed_fa_text'
    ]
    for f in ts_files:
        if f not in done_files:
            print('Processing: ', f)
            start_time = time.perf_counter()
            ts_file = os.path.join(TS_folderpath, f)
            fa_f = re.sub('_TS_', '_FA_', f)
            fa_file = os.path.join(FA_folderpath, fa_f)

            # TS file
            df_ts = pd.read_csv(ts_file)
            df_ts = df_ts.replace({np.nan: None, 'nan': None, 'None': None})
            df_ts["processed_section"] = df_ts["section"] #.apply(lambda i: process_ts_section(i))
            df_ts['processed_text'] = df_ts.apply(
                lambda i: concat_section_to_text(i, 'TS'),
                axis=1
            )
            
            # FA file
            df_fa = pd.read_csv(fa_file).astype(str)
            df_fa['index'] = df_fa['index'].astype(int)
            df_fa['processed_fa_text'] = df_fa.apply(lambda i: concat_section_to_text(i, 'FA'), axis=1)
            FA_TEXT_DICT = dict(zip(
                df_fa.processed_fa_text,
                df_fa.text
            ))
            
            if APPLY_WORDBANK:
                for col in ts_cols:
                    df_ts[col] = df_ts[col].apply(find_syn)
                for col in ts_cols:
                    df_fa[col] = df_fa[col].apply(find_syn)
            
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
            df_def = df_fa[
                df_fa.sub_section.str.contains('Definition', na=False, case=False)
            ] # cols: definition + text
            df_def = df_def[~df_def.definition.isnull()]
            # exclude parties & definition clause
            df_others = df_fa[
                ~(df_fa.section.str.contains("INTERPRETATION", na=False, case=False)) & (df_fa.section_id  != "0") & (df_fa.section!='PARTIES')&(df_fa.section!='TABLE OF CONTENTS')
            ]
            # schedule
            df_sched = df_others.loc[df_others.schedule.notnull()]
            # main clause
            df_clause = df_others.loc[~df_others.schedule.notnull()]
            df_clause = df_clause[~df_clause.section.isna()]

            print('Data is loaded. Start to do term matching ...')
            df_ts_not_null = df_ts[~df_ts.section.isna()]
            df_term = df_ts_not_null[df_ts_not_null.text_element!='section'][['section', 'processed_text']]
            df_term = df_term.rename(columns={'section': 'term', 'processed_text': 'content'})
            df_term = df_term[~df_term.content.isna()]
            term_list = []
            content_list = []
            annotation_list = []
            df_term = df_term.assign(
                content_split=df_term.apply(
                    lambda row: get_split_list(row["term"], row["content"]), axis=1
                )
            )
            df_agr_def = df_def[['definition', 'processed_fa_text']]
            df_agr_def = df_agr_def.rename(columns={'definition': 'key', 'processed_fa_text': 'value'})
            df_party = df_parties[~df_parties.definition.isna()][['definition', 'processed_fa_text']]
            df_party = df_party.rename(columns={'definition': 'party_key', 'processed_fa_text': 'party_value'})
            df_agr_para = df_clause[['sub_section', 'processed_fa_text', 'identifier', 'text_element']]
            df_agr_para['sub_section'] = df_agr_para['sub_section'].fillna(df_agr_para['processed_fa_text'])
            df_agr_para = df_agr_para.rename(columns={'sub_section': 'title', 'processed_fa_text': 'para'})
            df_agr_para = df_agr_para.assign(para_split=df_agr_para.apply(lambda row: get_split_list(row["title"], row["para"]), axis=1))
            df_schedule = df_sched[['schedule', 'processed_fa_text']]
            df_schedule = df_schedule.rename(columns={'processed_fa_text': 'name'})
            
            
            term_sheet_result_dict = {}
            
            for index, row in df_term.iterrows():
                term = row['term']
                item = row['content']
                if term in term_sheet_result_dict:
                    term_sheet_result_dict[term][item] = {}
                else:
                    item_result_dict = {}
                    item_result_dict[item] = {}
                    term_sheet_result_dict[term] = item_result_dict
                    
            ############ term key value
            term_sheet_keys = list(df_term["term"].values)
            term_sheet_values = list(df_term["content"].values)
            term_sheet_split_values = list(df_term["content_split"].values)
            term_sheet_dict = list(zip(term_sheet_keys, term_sheet_values))
            term_sheet_split_dict = list(zip(term_sheet_keys, term_sheet_values, term_sheet_split_values))
            
            ############ agreement definition key value
            agr_defn_keys = list(df_agr_def["key"].values)
            agr_defn_values = list(df_agr_def["value"].values)
            agr_defn_dict = dict(zip(agr_defn_keys, agr_defn_values))

            ############ agreement para key value
            agr_para_titles = list(df_agr_para["title"].values)
            agr_para_content = list(df_agr_para["para"].values)
            agr_para_content = [v for v in agr_para_content if v]
            agr_para_content_split = list(df_agr_para["para_split"].values)
            agr_para_dict = dict(zip(agr_para_titles, agr_para_content))
            agr_para_split_dict = dict(zip(agr_para_titles, agr_para_content_split))

            ############ agreement party key value
            agr_party_keys = list(df_party["party_key"].values)
            agr_party_values = list(df_party["party_value"].values)
            agr_party_dict = dict(zip(agr_party_keys, agr_party_values))

            ############ agreement schedule key value
            agr_schedule_keys = list(df_schedule["schedule"].values)
            agr_schedule_values = list(df_schedule["name"].values)
            agr_schedule_dict = dict(zip(agr_schedule_keys, agr_schedule_values))
            
            idf_map = {
                'clause_sec_text': dict(zip(df_agr_para.para, df_agr_para.identifier)),
                'clause_sub_sec_text': dict(zip(df_agr_para.para, df_agr_para.identifier)),
                'def_text': dict(zip(df_def.processed_fa_text, df_def.identifier)),
                'parties_text': dict(zip(df_parties.processed_fa_text, df_parties.identifier)),
                'schedule_text': dict(zip(df_sched.processed_fa_text, df_sched.identifier)),
                'schedule_part': dict(zip(df_sched.part, df_sched.identifier)),
                'sec_to_clause_sec': dict(zip(
                    df_agr_para[df_agr_para.text_element=='section'].title,
                    df_agr_para[df_agr_para.text_element=='section'].identifier
                )),
                'sec_to_clause_sub_sec': dict(zip(
                    df_clause[df_clause.text_element=='sub_section'].sub_section,
                    df_clause[df_clause.text_element=='sub_section'].identifier
                )),
                'sec_to_clause_sub_sub_sec': dict(zip(
                    df_clause[df_clause.text_element=='sub_sub_section'].sub_section,
                    df_clause[df_clause.text_element=='sub_sub_section'].identifier
                )),
                'sec_to_def': dict(zip(
                    df_def.drop_duplicates(subset=['definition']).definition,
                    df_def.drop_duplicates(subset=['definition']).identifier
                )),
                'sec_to_parties': dict(zip(
                    df_parties.drop_duplicates(subset=['definition']).definition,
                    df_parties.drop_duplicates(subset=['definition']).identifier
                )),
                'sec_to_schedule': dict(zip(
                    df_sched.drop_duplicates(subset=['schedule']).schedule,
                    df_sched.drop_duplicates(subset=['schedule']).identifier
                ))
            }
            
            emb_term_sheet_keys = model.encode(term_sheet_keys, convert_to_tensor=True)
            emb_term_sheet_values = model.encode(term_sheet_values, convert_to_tensor=True)

            print("compare term sheet key with agreement para title")
            agr_para_titles = [title for title in agr_para_titles if title]
            processed_titles = list([re.sub(r"([1-9]\d*\.?\d*)|(0\.\d*[1-9])", "", title).strip() for title in agr_para_titles if title])
            match(emb_term_sheet_keys, processed_titles, term_sheet_dict, agr_para_titles, "key_to_clause_title")

            print("compare term sheet value with agreement para title")
            match(emb_term_sheet_values, processed_titles, term_sheet_dict, agr_para_titles, "value2_to_clause_title")

            print("compare term sheet value with agreement para value")
            match(
                emb_term_sheet_values, agr_para_content, term_sheet_dict, 
                #agr_para_titles,
                agr_para_content,
                "value_to_clause_content"
            )
            if agr_defn_keys:
                print("compare term sheet key with agreement definition key")
                match(emb_term_sheet_keys, agr_defn_keys, term_sheet_dict, agr_defn_keys, "key_to_definition_key")

                print("compare term sheet value with agreement definition value")
                match(
                    emb_term_sheet_values, agr_defn_values, term_sheet_dict, 
                    # agr_defn_keys, 
                    agr_defn_values, 
                    "value_to_definition_value"
                )
                print("compare term sheet value with agreement definition key")
                match(emb_term_sheet_values, agr_defn_keys, term_sheet_dict, agr_defn_keys, "value_to_definition_key")
                
            if agr_party_keys:
                print("compare term sheet key with agreement party key")
                match(emb_term_sheet_keys, agr_party_keys, term_sheet_dict, agr_party_keys, "key_to_party_key")

            if agr_party_values:
                print("compare term sheet value with agreement party value")
                match(
                    emb_term_sheet_values, agr_party_values, term_sheet_dict, 
                    #agr_party_keys, 
                    agr_party_values, 
                    "value_to_party_value"
                )

            if agr_schedule_values:
                print("compare term sheet key with agreement schedule value")
                match(emb_term_sheet_keys, agr_schedule_values, term_sheet_dict, agr_schedule_keys, "key_to_schedule")

                print("compare term sheet value with agreement schedule value")
                match(
                    emb_term_sheet_values, agr_schedule_values, term_sheet_dict, 
                    #agr_schedule_keys,
                    agr_schedule_values, 
                    "value_to_schedule"
                )
            
            df_result = pd.DataFrame(columns=["term", "match_place", "score"])
            df_result_all = pd.DataFrame(columns=["term", "match_place", "score"])
            for key, item_result in term_sheet_result_dict.items():
                for item, value in item_result.items():
                    column_list = []
                    score_list = []
                    for k, v in value.items():
                        for i in v:
                            column_list.append("[{}]: {}".format(k, i[0]))
                            score_list.append(i[1])
                    term_list = [key] * len(column_list)
                    item_list = [item] * len(column_list)

                    if len(column_list) == 0:
                        term_list = [key]
                        column_list = ["not matched"]
                        score_list = [0]

                    df = pd.DataFrame({"term": term_list, "item": item_list, "match_place": column_list, "score": score_list})
                    df = df.sort_values(by=["score"], ascending=False) # [:5]
                    df_result = pd.concat([df_result, df[:10]], ignore_index=True, axis=0)
                    df_result_all = pd.concat([df_result_all, df], ignore_index=True, axis=0)
                    
            df_match = df_result_all.drop_duplicates()
            # df_match = df_result_all
            

            df_match = df_match.assign(match_place_key=df_match.apply(lambda row: gen_match_key(row["match_place"]), axis=1))
            df_match = df_match.assign(key_value=df_match.apply(lambda row: key_or_value(row["match_place"]), axis=1))
            
            L = list(zip(df_match["term"].values, df_match["item"].values))
            terms = sorted(set(L), key=L.index)
            term_dict = {"{}_{}".format(x,y): pd.DataFrame(columns=["term", "item", "match_place", "weight_score"]) for x, y in terms}
            
            df_match_weighted = pd.DataFrame(columns=["term", "item", "match_place", "weight_score"])
            for (term, item), content in df_match.groupby(["term", "item"]):
                key_list = []
                score_list = []
                key_list_clause = []
                score_list_clause = []
                for key, result in content.groupby(["match_place_key"]):
                    key_list_clause.append(key)
                    scores = list(result['score'].values)
                    kvs = list(result['key_value'].values)
                    score_list_clause.append(cal_score_clause_value(list(zip(kvs, scores))))
                    
                    key_list.append(key)
                    scores = list(result['score'].values)
                    kvs = list(result['key_value'].values)
                    score_list.append(cal_score_clause_key(list(zip(kvs, scores))))
                term_list = [term] * len(key_list)
                item_list = [item] * len(key_list)
                df_tmp = pd.DataFrame({"term": term_list, "item": item_list, "match_place": key_list, "weight_score": score_list})
                df_tmp = df_tmp.sort_values(by=["weight_score"], ascending=False)[:5]
                term_list = [term] * len(key_list_clause)
                item_list = [item] * len(key_list_clause)
                df_tmp_clause = pd.DataFrame({"term": term_list, "item": item_list, "match_place": key_list_clause, "weight_score": score_list_clause})
                df_tmp_clause = df_tmp_clause.sort_values(by=["weight_score"], ascending=False)[:5]
                df_tmp = df_tmp.append(df_tmp_clause)
                
                term_dict["{}_{}".format(term,item)] = df_tmp
                
            for key, value in term_dict.items():
                df_match_weighted = pd.concat([df_match_weighted, value], ignore_index=True, axis=0)
            df_match_weighted = df_match_weighted.assign(item=df_match_weighted.apply(lambda row: row["item"].replace('\n', ' '), axis=1))
            df_match_weighted['fa_text'] = df_match_weighted['match_place'].apply(get_fatext)
            all_map = dict()
            for k, v in idf_map.items():
                all_map.update(v)
                
            df_match_weighted['identifier'] = df_match_weighted['fa_text'].apply(lambda i: all_map.get(i))
            df_match_weighted['ts_key'] = df_match_weighted['term']
            df_match_weighted['ts_value'] = df_match_weighted['item']
            df_match_weighted['similarity'] = df_match_weighted['weight_score']
            df_match_weighted['match_type'] = 'dummy'
            # cols = [
            #     'index',
            #     'text_block_id',
            #     'page_id',
            #     'phrase_id',
            #     'list_id',
            #     'text_element',
            #     'ts_key',
            #     'ts_value',
            #     'fa_text',
            #     'identifier',
            #     'similarity',
            #     'match_type',
            #     'text',
            #     'processed_text',
            #     'section'
            # ]
            df_match = df_match_weighted.merge(df_ts, left_on=['ts_key', 'ts_value'], right_on=['section', 'processed_text'], how='right')
            df_match['fa_text'] = df_match['fa_text'].apply(lambda i: FA_TEXT_DICT[i] if FA_TEXT_DICT.get(i) else i)
            
            
            # df_match = df_match[~df_match.ts_value.isna()]
            df_match = df_match.sort_values(by=['section', 'processed_text', 'similarity'], ascending=False)
            df_match = df_match.drop_duplicates(subset=['section', 'processed_text', 'identifier'])
            df_match = df_match.replace({np.nan: None, 'nan': None, 'None': None})
            
            flag = 0
            final_data = []
            for idx, row in df_ts.iterrows():
                section = row['section']
                processed_text = row['processed_text']
                df_s = df_match[(df_match.section==section)&(df_match.processed_text==processed_text)]
                row['TS_term'] = section
                row['TS_text'] = row['text']
                if not df_s.empty:
                    row['match_term_list'] = list(df_s['fa_text'])[:top_N]
                    row['identifier_list'] = list(df_s['identifier'])[:top_N]
                    row['similarity_list'] = list(df_s['similarity'])[:top_N]
                    row['match_type_list'] = list(df_s['match_type'])[:top_N]
                else:
                    row['match_term_list'] = [None]
                    row['identifier_list'] = [None]
                    row['similarity_list'] = [None]
                    row['match_type_list'] = [None]
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
            df_final = df_final.sort_values(by=['index'])
            
            check_csv_path = os.path.join(OUTPUT_folderpath, check_date, 'check', f)
            df_match.to_csv(check_csv_path, index=False)
            save_f = re.sub('.csv', '_results.csv', f)
            results_csv_path = os.path.join(OUTPUT_folderpath, check_date, save_f)
            df_final.to_csv(results_csv_path, index=False)
            end_time = time.perf_counter()
            total_time = end_time - start_time
            print(f'{f} matching time: {total_time} s')
            if flag == 0:
                done_list.append({"file":f, "time":total_time})
            
            with open(os.path.join(OUTPUT_folderpath, f"done_term_matching_list_{check_date}.json"), "w") as outfile:
                outfile.write(json.dumps(done_list,indent=4))