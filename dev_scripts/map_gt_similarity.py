from sentence_transformers import SentenceTransformer, util
import pandas as pd
import numpy as np
import os

def concat_features_to_text(row, doc_type, off_parent_text=False):
    '''
    @param row: pandas.Series object with indices ['text', 'section', 'fa_text', 'fa_section', 'fa_sub_section', 'parent_caption', 'parent_list']
    @param doc_type: either 'TS' or 'FA'
    @return: return a concatenated string of useful features (section, sub-section, caption, list item and main text) as a complete content
    @rtype: str
    '''
    parent_caption = parent_list = None
    if doc_type == 'TS':
        if 'parent_caption' in row.index.tolist():
            parent_caption = row['parent_caption']
        if 'parent_list' in row.index.tolist():
            parent_list = row['parent_list']
    else:
        if 'fa_parent_caption' in row.index.tolist():
            parent_caption = row['fa_parent_caption']
        if 'fa_parent_list' in row.index.tolist():
            parent_list = row['fa_parent_list']
    
    if off_parent_text:
        parent_list = parent_caption = None
    
    if doc_type == 'TS':
        text = row['text']
    else:
        text = row['fa_text']
    if not text:
        return None
    
    if doc_type == 'FA':
        fa_section = row['fa_section']
        fa_sub_section = row['fa_sub_section']
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
            
def concat_features_to_split_text(row):
    '''
    @param row: pandas.Series object with indices ['text', 'section', 'fa_text', 'fa_section', 'fa_sub_section', 'parent_caption', 'parent_list']
    @param doc_type: either 'TS' or 'FA'
    @return: return a concatenated string of useful features (section, sub-section, caption, list item and main text) as a complete content
    @rtype: str
    '''
    parent_caption = parent_list = None

    if 'parent_caption' in row.index.tolist():
        parent_caption = row['parent_caption']
    if 'parent_list' in row.index.tolist():
        parent_list = row['parent_list']
    
    texts = row['fa_text_split']
    if not texts:
        return None
    
    split = []
    for text in texts:
        fa_section = row['fa_section']
        fa_sub_section = row['fa_sub_section']
        if fa_sub_section and fa_section and text:
            if parent_caption and parent_list:
                split.append(fa_section + ' - ' + fa_sub_section + ': ' + parent_caption + ' ' + parent_list + ': ' + text)
            elif parent_caption:
                split.append(fa_section + ' - ' + fa_sub_section + ': ' + parent_caption + ' ' +  text)
            elif parent_list:
                split.append(fa_section + ' - ' + fa_sub_section + ': ' + parent_list + ': ' + text)
            else:
                split.append(fa_section + ' - ' + fa_sub_section + ': ' + text)
        elif fa_sub_section and not fa_section and text:
            if parent_caption and parent_list:
                split.append(fa_sub_section + ': ' + parent_caption + ' ' + parent_list + ': ' + text)
            elif parent_caption:
                split.append(fa_sub_section + ': ' + parent_caption + ' ' + text)
            elif parent_list:
                split.append(fa_sub_section + ': ' + parent_list + ': ' + text)
            else:
                split.append(fa_sub_section + ': ' + text)
        elif not fa_sub_section and fa_section and text:
            if parent_caption and parent_list:
                split.append(fa_section + ': ' + parent_caption + ' ' + parent_list + ': ' + text)
            elif parent_caption:
                split.append(fa_section + ': ' + parent_caption + ' ' + text)
            elif parent_list:
                split.append(fa_section + ': ' + parent_list + ': ' + text)
            else:
                split.append(fa_section + ': ' + text)
        else:
            if parent_caption and parent_list:
                split.append(parent_caption + ' ' + parent_list + ': ' + text)
            elif parent_caption:
                split.append(parent_caption + ' ' + text)
            elif parent_list:
                split.append(parent_list + ': ' + text)
            else:
                split.append(text)
    return split

def flatten(lst):
    for el in lst:
        if isinstance(el, tuple):
            el = list(el)
        if isinstance(el, list):
            # recurse
            yield from flatten(el)
        else:
            # generate
            yield el

if __name__=="__main__":
    UAT = [
        "1_GL_SYN_TS_mkd_20221215_docparse_merge&match_FA.csv",
        "13_BL_SYN_TS_mkd_20220713_docparse_merge&match_FA.csv",
        "19_GL_SYN_TS_mkd_20220718_docparse_merge&match_FA.csv",
        "23_BL_SYN_TS_mkd_20220715_docparse_merge&match_FA.csv",
        "24_GF_PRJ_TS_mkd_20220225_docparse_merge&match_FA.csv",
        "25_NBFI_PRJ_TS_mkd_20220613_docparse_merge&match_FA.csv",
        "28_GF_PRJ_TS_mkd_20221111_docparse_merge&match_FA.csv",
        "29_PF_PRJ_TS_mkd_20200624_docparse_merge&match_FA.csv",
        "3_GF_SYN_TS_mkd_20221018_docparse_merge&match_FA.csv",
        "31_GL_PRJ_TS_mkd_20220630_docparse_merge&match_FA.csv",
        "33_BL_PRJ_TS_mkd_20200727_docparse_merge&match_FA.csv",
        "34_AWSP_PRJ_TS_mkd_20211230_docparse_merge&match_FA.csv",
        "41_AF_SYN_TS_mkd_20190307_docparse_merge&match_FA.csv",
        "43_AF_SYN_TS_mkd_20151101_docparse_merge&match_FA.csv",
        "45_AF_PRJ_TS_mkd_20170331_docparse_merge&match_FA.csv",
        "49_VF_PRJ_TS_mkd_20151130_docparse_merge&match_FA.csv",
        "54_VF_PRJ_TS_mkd_20191018_docparse_merge&match_FA.csv",
        "58_VF_SYN_TS_mkd_20111201_docparse_merge&match_FA.csv",
        "59_AWSP_SYN_TS_mkd_20210814_docparse_merge&match_FA.csv",
        "63_AW_SYN_TS_mkd_20221025_docparse_merge&match_FA.csv",
        "66_PF_SYN_TS_mkd_20230106_docparse_merge&match_FA.csv",
        "68_PF_SYN_TS_mkd_20221104_docparse_merge&match_FA.csv",
        "72_NBFI_SYN_TS_mkd_20221215_docparse_merge&match_FA.csv",
        "74_NBFI_SYN_TS_mkd_20220401_docparse_merge&match_FA.csv",
        "8_GF_SYN_TS_mkd_20230215_docparse_merge&match_FA.csv"
    ]
    
    folder = '/home/data2/ldrs_analytics/data/antd_ts_merge_fa/finetune0130_reviewed_ts_v4_all_fa_v4.4_top5_filter_human_errors/(0)'
    files = sorted([f for f in os.listdir(folder) if f.endswith('csv') and f != 'all_merge&match_results.csv'])
    model_path = '/home/data2/ldrs_analytics/models/gte-base-train_nli-boc_epoch_20_lr_1e-05_batch_32_0130'
    model = SentenceTransformer(model_path)
    # Start the multi-process pool on all available CUDA devices
    # pool = model.start_multi_process_pool()

    for f in files:
        print(f'processing: {f}')
        df = pd.read_csv(os.path.join(folder,f))
        df = df.replace({np.nan:None, 'nan':None})
        
        df['processed_ts_text'] = df.apply(lambda x: concat_features_to_text(x, 'TS'), axis=1)
        df['processed_fa_text'] = df.apply(lambda x: concat_features_to_text(x, 'FA'), axis=1)
        # df.loc[~df.fa_text.isnull(),'fa_text_splits'] = df[~df.fa_text.isnull()]['fa_text'].split(';- ')
        # df.loc[~df.fa_text.isnull(),'fa_text_splits'] = df[~df.fa_text.isnull()]['fa_text_splits'].apply(lambda x: concat_features_to_split_text(x), axis=1)
        
        ts_text_list = list(set(df[~df.fa_text.isnull()].processed_ts_text.values.tolist()))
        fa_text_list = list(set(df[~df.fa_text.isnull()].processed_fa_text.values.tolist()))
        fa_split_text_list = [] # list(set(flatten(df[~df.fa_text.isnull()].fa_text_splits.values.tolist())))
        
        fa_all_text = fa_text_list + fa_split_text_list
        # ts_text_list = ['Interest Margin: determination by the Lenders that the requisite No. of Sustainability Performance Targets are achieved for the Relevant Sustainability Performance Period.']
        # fa_text_list = ['INFORMATION UNDERTAKINGS - Sustainability Performance Certificate: such Sustainability Performance Certificate shall']
        # Compute embedding for both lists
        embeddings1 = model.encode(ts_text_list, convert_to_tensor=True)
        embeddings2 = model.encode(fa_all_text, convert_to_tensor=True)
        
        # # Compute the embeddings using the multi-process pool
        # embeddings1 = model.encode_multi_process(ts_text_list, pool)
        # embeddings2 = model.encode_multi_process(fa_all_text, pool)

        cos_scores = util.cos_sim(embeddings1, embeddings2).cpu().numpy()
        # print(cos_scores)
        
        df.loc[~df.fa_text.isnull(), 'sim_annotation_fa_text(with_parent_text)'] = df[~df.fa_text.isnull()].apply(lambda x: cos_scores[ts_text_list.index(x.processed_ts_text)][fa_all_text.index(x.processed_fa_text)], axis=1)
        # df.loc[~df.fa_text.isnull(), 'sim_annotation_split_fa_text'] = df[~df.fa_text.isnull()].apply(lambda x: [cos_scores[ts_text_list.index(x.processed_ts_text)][fa_all_text.index(i)] for i in x.fa_text_splits], axis=1)
        
        # ===================================================================================================================================================================================================
        
        df['processed_ts_text2'] = df.apply(lambda x: concat_features_to_text(x, 'TS' ,off_parent_text=True), axis=1)
        df['processed_fa_text2'] = df.apply(lambda x: concat_features_to_text(x, 'FA' ,off_parent_text=True), axis=1)
        # df.loc[~df.fa_text.isnull(),'fa_text_splits'] = df[~df.fa_text.isnull()]['fa_text'].split(';- ')
        # df.loc[~df.fa_text.isnull(),'fa_text_splits'] = df[~df.fa_text.isnull()]['fa_text_splits'].apply(lambda x: concat_features_to_split_text(x), axis=1)
        
        ts_text_list = list(set(df[~df.fa_text.isnull()].processed_ts_text2.values.tolist()))
        fa_text_list = list(set(df[~df.fa_text.isnull()].processed_fa_text2.values.tolist()))
        fa_split_text_list = [] # list(set(flatten(df[~df.fa_text.isnull()].fa_text_splits.values.tolist())))
        
        fa_all_text = fa_text_list + fa_split_text_list
        # ts_text_list = ['Interest Margin: determination by the Lenders that the requisite No. of Sustainability Performance Targets are achieved for the Relevant Sustainability Performance Period.']
        # fa_text_list = ['INFORMATION UNDERTAKINGS - Sustainability Performance Certificate: such Sustainability Performance Certificate shall']
        # Compute embedding for both lists
        embeddings1 = model.encode(ts_text_list, convert_to_tensor=True)
        embeddings2 = model.encode(fa_all_text, convert_to_tensor=True)
        
        # # Compute the embeddings using the multi-process pool
        # embeddings1 = model.encode_multi_process(ts_text_list, pool)
        # embeddings2 = model.encode_multi_process(fa_all_text, pool)

        cos_scores = util.cos_sim(embeddings1, embeddings2).cpu().numpy()
        # print(cos_scores)
        
        df.loc[~df.fa_text.isnull(), 'sim_annotation_fa_text(no_parent_text)'] = df[~df.fa_text.isnull()].apply(lambda x: cos_scores[ts_text_list.index(x.processed_ts_text2)][fa_all_text.index(x.processed_fa_text2)], axis=1)
        # df.loc[~df.fa_text.isnull(), 'sim_annotation_split_fa_text'] = df[~df.fa_text.isnull()].apply(lambda x: [cos_scores[ts_text_list.index(x.processed_ts_text)][fa_all_text.index(i)] for i in x.fa_text_splits], axis=1)
        
        
        # df = df.drop(columns=['processed_ts_text','processed_fa_text'])
        df.to_csv(os.path.join(folder,f),index=False,encoding='utf-8-sig')
        print(f'completed processing: {f} \n')

    # Optional: Stop the processes in the pool
    # model.stop_multi_process_pool(pool)