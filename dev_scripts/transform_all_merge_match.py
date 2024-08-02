import os
from collections import defaultdict
import pandas as pd
import numpy as np
from ast import literal_eval
from datetime import datetime


def gp_judge(row):
    row = row.tolist()
    lst = list(zip(*[literal_eval(i) for i in row]))
    r = []
    for i in lst:
        if any(j == 'TP' for j in i):
            r.append(True)
        elif all(j == 'FP' for j in i):
            r.append(None)
        else:
            r.append(False)
    if all(i is None for i in r):
        r = False
    return r


def miss_results(row):
    judge_all = row['judge_all'].tolist()
    fa_identifier = row['fa_identifier'].tolist()
    fa_text = row['fa_text'].tolist()
    pairs = list(zip(judge_all, fa_identifier, fa_text))
    if fa_identifier:
        return pd.Series([list(set([i[1] for i in pairs if i[0] != 'TP' and i[1]])), list(set([i[2] for i in pairs if i[0] != 'TP' and i[1]]))], index=['manual_identifier_list', 'manual_content'])
    else:
        return pd.Series([None, None], index=['manual_identifier_list', 'manual_content'])


def join_parse(src_folder_path, output_folder, doc_type):
    docId2filename_csv = 'doc_id2filename.csv'
    df_map = pd.read_csv(os.path.join(output_folder,docId2filename_csv))
    df_map['filename'] = df_map['filename'].map(lambda x: x.split('.')[0])
    dic = dict(zip(df_map['filename'], df_map['id']))

    all_df = []
    files = [f for f in sorted(os.listdir(src_folder_path)) if f.endswith('.csv')]
    files = [f for f in files if f.replace('_docparse.csv', '') in dic.keys()]
    for f in files:
        fname = f.replace('_docparse.csv', '')
        df = pd.read_csv(os.path.join(src_folder_path, f))
        df = df.replace({np.nan: None, 'None': None})
        try:
            df['file_id'] = dic[fname]
            df['del_flag'] = 0
            all_df.append(df)
        except KeyError:
            continue
    all_df = pd.concat(all_df)

    if doc_type == 'ts':
        all_df['parent_list_ids'] = all_df['parent_list_ids'].map(lambda x: literal_eval(str(x)) if x else None)
        all_df = all_df.rename(columns={
            'index': 'indexId',
            'text_block_id': 'textBlockId',
            'page_id': 'pageId',
            'phrase_id': 'phraseId',
            'section': 'sectionContent',
            'text_element': 'textElement',
            'list_id': 'listId',
            'parent_caption': 'parentCaption',
            'parent_list_ids': 'parentListIds',
            'parent_list': 'parentList',
            'file_id': 'fileId',
            'del_flag': 'delFlag'
        })
    else:
        all_df[['bbox', 'parent_list_ids']] = all_df[['bbox', 'parent_list_ids']].map(
            lambda x: literal_eval(str(x)) if x else None)
        all_df = all_df.rename(columns={
            'index': 'indexId',
            'text_block_id': 'textBlockId',
            'page_id': 'pageId',
            'phrase_id': 'phraseId',
            'section_id': 'sectionId',
            'section': 'sectionContent',
            'sub_section_id': 'subSectionId',
            'sub_section': 'subSection',
            'schedule_id': 'scheduleId',
            'part_id': 'partId',
            'text_element': 'textElement',
            'text_content': 'textContent',
            'list_id': 'listId',
            'parent_caption': 'parentCaption',
            'parent_list_ids': 'parentListIds',
            'parent_list': 'parentList',
            'file_id': 'fileId',
            'del_flag': 'delFlag'
        })
    all_df.to_csv(os.path.join(output_folder, f"all_{doc_type}.csv"),index=False, encoding='utf-8-sig')
    
    return all_df

def count_miss(row):
    result = row['result']
    manual_identifier_list = row['manual_identifier_list']
    if isinstance(result, list) and isinstance(manual_identifier_list, list):
        miss = min(min(5, len([i==False for i in result])),len(manual_identifier_list))
        return miss
    else:
        return 0

def transform_term_match(src_folder_path, output_folder, name_tag):
    df = pd.read_csv(os.path.join(src_folder_path,'all_merge&match_results.csv'))
    df = df.replace({np.nan: None, 'None': None})

    df = df[
        [
            'filename',
            'facility_type',
            'index',
            'text_block_id',
            'page_id',
            'phrase_id',
            'section',
            'text_element',
            'text',
            'list_id',
            'fa_identifier',
            'fa_text',
            'match_term_list',
            'identifier_list',
            'similarity_list',
            'match_type_list',
            'judge',
            'judge_all'
        ]
    ]

    df[['index', 'text_block_id', 'phrase_id']] = df[['index', 'text_block_id', 'phrase_id']].astype(int)
    df[['match_term_list', 'identifier_list', 'similarity_list', 'match_type_list']] = df[
        ['match_term_list', 'identifier_list', 'similarity_list', 'match_type_list']].map(
        lambda x: literal_eval(str(x)) if x else None)
    df.loc[df['match_term_list'].map(is_all_null),['match_term_list', 'identifier_list', 'similarity_list', 'match_type_list']] = None
    df['task_id'] = df['filename'].map(lambda x: int(x.split('_')[0]))
    df['del_flag'] = 0
    df = df.sort_values(by=['filename', 'index', 'phrase_id']).reset_index(drop=True)

    now = datetime.now()  # current date and time
    date_time = now.strftime("%Y-%m-%d %H:%M:%S")
    df['create_time'] = date_time
    df['update_time'] = date_time
    result = df.groupby(['filename', 'index', 'phrase_id', 'text', 'fa_identifier'], dropna=False)['judge'].apply(lambda row: gp_judge(row)).reset_index().rename(columns={'judge': 'result'})
    fa_identifier = df.groupby(['filename', 'index', 'phrase_id', 'text'])['fa_identifier'].apply(lambda x: list(set(x.tolist()))).reset_index()
    miss_result = df.groupby(['filename', 'index', 'phrase_id', 'text']).apply(lambda x: miss_results(x)).reset_index()
    df = df.merge(result, how='left', on=['filename', 'index', 'phrase_id', 'text', 'fa_identifier'])
    df = df.merge(miss_result, how='left', on=['filename', 'index', 'phrase_id', 'text'])
    df = df.drop(['fa_identifier'], axis=1)
    df = df.merge(fa_identifier, how='left', on=['filename', 'index', 'phrase_id', 'text'])
    df['fa_identifier'] = df['fa_identifier'].map(lambda x: None if x in ([None],[]) else x)
    df[['manual_identifier_list', 'manual_content']] = df[['manual_identifier_list', 'manual_content']].map(lambda x: None if x in ([None],[]) else x)
    df = df.drop(['fa_text','judge','judge_all'], axis=1)
    df = df.drop_duplicates(subset=['filename', 'index', 'phrase_id', 'text'], keep='first')
    df = df[~df.task_id.isnull()]
    df['phrase_id'] = df.groupby(['index']).cumcount()
    df['hitCount'] = df['result'].map(lambda x: min(5,sum(x)) if isinstance(x,list) else 0)
    df['missCount'] = df.apply(lambda x: count_miss(x), axis=1)
    df['hitRate'] = df.apply(lambda x: x.hitCount / (x.hitCount + x.missCount) if (x.hitCount + x.missCount)>0 else None, axis=1)
    df['updateUser'] = 'UAT'
    df = df.replace({np.nan: None, 'None': None})
    df = df.rename(columns={
        "task_id": "taskId",
        "facility_type": "facilityType",
        "index": "indexId",
        'text_block_id': 'textBlockId',
        'page_id': 'pageId',
        'phrase_id': 'phraseId',
        'section': 'tsTerm',
        'list_id': 'listId',
        'del_flag': 'delFlag',
        'create_time': 'createTime',
        'update_time': 'updateTime',
        'text': 'tsText',
        'match_term_list': 'matchTermList',
        'identifier_list': 'identifierList',
        'similarity_list': 'similarityList',
        'match_type_list': 'matchTypeList',
        'manual_identifier_list': 'manualIdentifier',
        'manual_content': 'manualContent',
        'fa_identifier': 'faIdentifier'
    })

    df.to_csv(os.path.join(output_folder, f'{name_tag}_all_term_match.csv'),index=False, encoding='utf-8-sig')
    
    return df

def is_all_null(row):
    return all(i is None for i in row)

def id2idtype(identifier):
    import re
    if identifier is None:
        return None
    elif re.match('^Cl_.*-.*', identifier):
        return 'definitionClause'
    elif re.match('^Parties.*', identifier):
        return 'partiesClause'
    elif re.match('^Cl_.*', identifier):
        return 'clause'
    elif re.match('^Sched_.*', identifier):
        return 'schedule'


if __name__ == "__main__":
    import argparse
    import logging
    
    # Change root logger level from WARNING (default) to NOTSET in order for all messages to be delegated.
    logging.getLogger().setLevel(logging.NOTSET)
    logger = logging.getLogger(__name__)
    
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--topN",
        default=5,
        type=int,
        required=False,
        help="Set True if merging annotated TS on FA.",
    )
    parser.add_argument(
        "--output_folder",
        default='/home/data2/ldrs_analytics/data/output_for_demo',
        type=str,
        required=False,
        help="Set True if merging annotated TS on FA.",
    )
    parser.add_argument(
        "--src_ts_folder",
        default='/home/data2/ldrs_analytics/data/docparse_csv/TS',
        type=str,
        required=False,
        help="Set True if merging annotated TS on FA.",
    )
    parser.add_argument(
        "--src_fa_folder",
        default='/home/data2/ldrs_analytics/data/docparse_csv/FA_v5.1',
        type=str,
        required=False,
        help="Set True if merging annotated TS on FA.",
    )
    parser.add_argument(
        "--name_tag",
        default='tm_ts_fa_v4.4_0325',
        type=str,
        required=False,
        help="Set True if merging annotated TS on FA.",
    )
    args = parser.parse_args()
    args.topN = 5
    args.src_antdTS_merge_tm = f'/home/data2/ldrs_analytics/data/antd_ts_merge_fa/{args.name_tag}_top5_filter_human_errors/(0)'

    # join_parse(args.src_ts_folder, output_folder, 'ts')
    # join_parse(args.src_fa_folder, output_folder, 'fa')
    df = transform_term_match(args.src_antdTS_merge_tm, args.output_folder, args.name_tag)

    filenames = list(set(df.filename.tolist()))
    analysis = []
    tmp_df = df.copy()
    for filename in filenames:
        df = tmp_df[tmp_df.filename==filename]
        taskId = df['taskId'].values[0]
        term_match_data = {'result': {'records': None}}
        term_match_data['result']['records'] = df.to_dict(orient='records')
        # =================== analysis ===================
        hit = 0
        miss = 0
        irrelavent_instance_count = 0
        hit_instance = 0
        manual_select_instance = 0
        total_manual_select = 0
        hit_topN = defaultdict(int)
        hit_count_by_identifier_type = defaultdict(int)
        miss_count_by_identifier_type = defaultdict(int)
        miss_count_by_section = defaultdict(int)
        term_list = []
        content_list = []
        for record in term_match_data['result']['records']:
            if record["tsTerm"] and record["tsTerm"] not in term_list:
                term_list.append(record["tsTerm"])
            content_list.append(record["tsText"])
            selected = 0
            if record["result"] and record["identifierList"]:
                selected = min(sum(record["result"]), args.topN)
                indice_selected = [i for i, e in enumerate(record["result"]) if e == True]
                selected_identifiers = [e for i, e in enumerate(record["identifierList"]) if i in indice_selected]
                selected_identifier_types = [id2idtype(identifier) for identifier in selected_identifiers]
                for typ in selected_identifier_types:
                    hit_count_by_identifier_type[typ] += 1
                # unselected choices implies the results are either irrelevant or don't care
                unselected = min(len(record["result"]) - selected, args.topN)
                hit += selected
                # count hit by rank
                for i, r in enumerate(record["result"]):
                    hit_topN[i + 1] += int(r)
                # if all results are unselected and no manual-added result, imply the result is irrelevant
                if all(i == False for i in record["result"]) and "manualIdentifier" in record and record[
                    "manualIdentifier"] is None:
                    irrelavent_instance_count += 1
            # by default result = null and there exists term match result means user accept 1st result as ground-truth such that they didn't edit on the record
            elif record["result"] is None and record["identifierList"] is not None:
                selected = 1
                unselected = 0
                hit += selected
                hit_topN[1] += 1
            # when result = false, the split is irrelavent and ignore the case
            else:
                continue

            # if any selected, it means it is a hit instance
            if selected > 0:
                hit_instance += 1

            manual_selected = 0
            if "manualIdentifier" in record and record["manualIdentifier"]:
                manualIdentifier = [i for i in record["manualIdentifier"] if i and i != '']
                if manualIdentifier:
                    manual_select_instance += 1
                    manual_selected = len(manualIdentifier)
                    manual_selected_identifier_types = [id2idtype(identifier) for identifier in manualIdentifier]
                    for typ in manual_selected_identifier_types:
                        miss_count_by_identifier_type[typ] += 1
                        miss_count_by_section[record["tsTerm"]] += 1
                    total_manual_select += manual_selected

            # only consider miss count if unselected event occurs
            miss += min(unselected, manual_selected)

        assert (hit + miss) > 0, logger.error(
            'There is neither hit result nor manually-added result. There should be at least either one hit result or one manually-added result. Please check.')
        avg_accuracy = round((hit / (hit + miss)) * 100, 2)
        topN_hit_rate = {k: round((hit_topN[k] / (hit + miss)) * 100, 2) if hit > 0 else None for k in
                         range(1, args.topN + 1)}
        cumulative_upto_topN = {k: sum([hit_topN[i] for i in range(1, k + 1)]) for k in range(1, args.topN + 1)}
        if miss_count_by_section:
            section_top_miss = sorted(miss_count_by_section.items(), key=lambda x: x[1])[0][0]
        else:
            section_top_miss = None

        relavent_instance_count = len(content_list) - irrelavent_instance_count
        result = {
            "taskId": taskId,
            "avgAccuracy": avg_accuracy,
            "termCount": len(term_list),
            "termContentCount": len(content_list),
            "irrelavantContentCount": irrelavent_instance_count,
            "relavantInstanceRate": round(relavent_instance_count / len(content_list) * 100, 2),
            "hitCount": hit,
            "missCount": miss,
            "topMissTsTerm": section_top_miss,
            "hitInstanceCount": hit_instance,
            "avgHitperHitInstance": hit / hit_instance if hit_instance > 0 else None,
            "manualCount": total_manual_select,
            "manualInstanceCount": manual_select_instance,
            "missRate": round(miss / (hit + miss) * 100, 2),
            "manualInstanceRate": round(manual_select_instance / relavent_instance_count * 100,
                                        2) if relavent_instance_count > 0 else None
        }

        clause_type = ['definitionClause', 'partiesClause', 'clause', 'schedule']
        for k, v in dict(hit_count_by_identifier_type).items():
            try:
                clause_type.remove(k)
            except:
                pass
            result.update(
                {
                    f"{k}HitCount": v
                }
            )
        for k in clause_type:
            result.update(
                {
                    f"{k}HitCount": 0
                }
            )

        clause_type = ['definitionClause', 'partiesClause', 'clause', 'schedule']
        for k, v in dict(miss_count_by_identifier_type).items():
            try:
                clause_type.remove(k)
            except:
                pass
            result.update(
                {
                    f"{k}MissCount": v
                }
            )
        for k in clause_type:
            result.update(
                {
                    f"{k}MissCount": 0
                }
            )

        for i in range(1, args.topN + 1):
            result.update(
                {
                    f"top{str(i)}HitCount": hit_topN[i],
                    f"top{str(i)}HitRate": topN_hit_rate[i],
                    f"upToTop{str(i)}HitCount": cumulative_upto_topN[i],
                    f"upToTop{str(i)}HitRate": round(cumulative_upto_topN[i] / (hit + miss) * 100, 2) if hit > 0 else None
                }
            )
        result.update({'filename': filename})
        analysis.append(result)
    df_analy = pd.DataFrame(data=analysis)
    df_analy.to_csv(os.path.join(args.output_folder, f'{args.name_tag}_all_analysis.csv'),index=False, encoding='utf-8-sig')