import pandas as pd
import numpy as np
import csv
import os
import string
import ast
from string import digits

DATA_DIR = '/home/data/ldrs_analytics'
ts_annot_csv_dir = f'{DATA_DIR}/docparse_csv_annotated/'

label_cols=['clause_id','definition','schedule_id']
df_info = pd.read_csv(os.path.join(DATA_DIR,'doc_info.csv'))

def all_same_label(df, label_cols=['clause_id','definition','schedule_id']):

    if all([(df[i] == df[i].values[0]).all() for i in label_cols]):
        return True
    else:
        return False

def has_label(df, label_cols=['clause_id','definition','schedule_id']):
    if any([(df[i] != 'None').any() for i in label_cols]):
        return True
    else:
        return False

def count_labels(df, label_cols=['clause_id','definition','schedule_id']):
    if (df['clause_id'] == 'None').all() and (df['schedule_id'] == 'None').all():
        return 0
    else:
        count_clauses = df['clause_id'].map(lambda x: sum([len(i.split('\n')) for i in x.split(',')]) if x != 'None' else 0).sum() / len(df)
        count_schedule = df['schedule_id'].map(
            lambda x: sum([len(i.split('\n')) for i in x.split(',')]) if x != 'None' else 0).sum() / len(df)
        return count_clauses + count_schedule

list_doc_analysis = []
for file in os.listdir(ts_annot_csv_dir):
    if file.endswith('_docparse.csv') and not file.startswith("~$"):
        print('filename: ',file)
        df = pd.read_csv(ts_annot_csv_dir+file)
        df = df.replace({np.nan: 'None'})
        df_gp = df.groupby(by=["index"])
        df_analysis = pd.DataFrame(data={
            'index': df_gp.index.first(),
            'section': df_gp.section.first(),
            'text': df_gp.text.apply(lambda x: ' '.join(x)),
            'text_element': df_gp.text_element.first().map(lambda x: x.split('_')[0]),
            'phrase_count': df_gp.index.size(),
            'all_same_label': df_gp.apply(lambda x: all_same_label(x)),
            'has_label': df_gp.apply(lambda x: has_label(x,label_cols=['annotation'])),
            'label_count': df_gp.apply(lambda x: count_labels(x)),
        })
        def label_split_type(df):

            # phrase_x_label = df['phrase_count'] * df['label_count']
            # isinteger = phrase_x_label.is_integer() or phrase_x_label==0

            if df['label_count'] == 0:
                label_count_over_1 = 0
            elif df['label_count'] == 1:
                label_count_over_1 = False
            elif df['label_count'] < 1:
                label_count_over_1 = None
            elif df['label_count'] >1:
                label_count_over_1 = True

            tup = (df['text_element'],df['all_same_label'],df['has_label'],label_count_over_1) # ,isinteger

            cases = {
                        ('paragraph',False,False,0) : 'phrase-level, no label',
                        ('paragraph', False, True, False): 'phrase-level, single label',
                        ('list', False, True, False): 'phrase-level, single label',
                        ('paragraph',False,True,None) : 'phrase-level, single label',
                        ('list', False, True, None) : 'phrase-level, single label',
                        ('paragraph',False,True,True) : 'phrase-level, multiple label',
                        ('list', False, True, True) : 'phrase-level, multiple label',
                        ('paragraph',True,False,0) : 'paragraph-level, no label',
                        ('paragraph', True, True, False): 'paragraph-level, single label',
                        ('paragraph',True,True,None) : 'paragraph-level, single label',
                        ('paragraph',True,True,True) : 'paragraph-level, multiple label',
                        ('title', True, False, 0): 'section-level, no label',
                        ('section',True,False,0) : 'section-level, no label',
                        ('title', True, True, 0): 'section-level, single label',
                        ('section', True, True, 0): 'section-level, single label',
                        ('section',True,True,None) : 'section-level, single label',
                        ('section',True,True,True) : 'section-level, multiple label',
                        ('list',True,False,0) : 'list-level, no label',
                        ('list', True, True, False): 'list-level, single label',
                        ('list',True,True,None) : 'list-level, single label',
                        ('list',True,True,True) : 'list-level, multiple label',
                        ('caption',True,False,0) : 'caption-level, no label',
                        ('caption', True, True, False): 'caption-level, single label',
                        ('caption',True,True,None) : 'caption-level, single label',
                        ('caption',True,True,True) : 'caption-level, multiple label',
                        ('table', True, False, 0): 'other, no label',
                        ('reference', True, False, 0): 'other, no label'
                     }
            return cases[tup]

        df_analysis['annotation_level_type'] = df_analysis.apply(label_split_type, axis=1)
        annotation_level_type = sorted(df_analysis['annotation_level_type'].unique().tolist())
        fname = file.split('_docparse.csv')[0]
        facility_type = df_info[df_info['filename']==fname]['facility_type'].values[0]
        df_analysis2 = pd.DataFrame(
                                    {'filename': [fname],
                                     'facility_type': [facility_type],
                                    'paragraph-level, no label': [None],
                                    'sections in paragraph-level, no label': [None],
                                    'paragraph-level, single label': [None],
                                    'sections in paragraph-level, single label': [None],
                                    'paragraph-level, multiple label': [None],
                                    'sections in paragraph-level, multiple label': [None],
                                    'phrase-level, single label': [None],
                                    'sections in phrase-level, single label': [None],
                                    'phrase-level, multiple label': [None],
                                    'sections in phrase-level, multiple label': [None],
                                    'list-level, no label': [None],
                                    'sections in list-level, no label': [None],
                                    'list-level, single label': [None],
                                    'sections in list-level, single label': [None],
                                    'list-level, multiple label': [None],
                                    'sections in list-level, multiple label': [None],
                                    'caption-level, no label': [None],
                                    'sections in caption-level, no label': [None],
                                    'caption-level, single label': [None],
                                    'sections in caption-level, single label': [None],
                                    'caption-level, multiple label': [None],
                                    'sections in caption-level, multiple label': [None],
                                    'section-level, no label': [None],
                                    'sections in section-level, no label': [None],
                                    'section-level, single label': [None],
                                    'sections in section-level, single label': [None],
                                    'other, no label': [None],
                                    'sections in other, no label': [None]
                                     }
                                    )
        for i in annotation_level_type:
            df_analysis2[i] = df_analysis['annotation_level_type'].value_counts()[i]
            df_analysis2['sections in '+i] = str(df_analysis[df_analysis['annotation_level_type'] == i]['section'].unique().tolist())
            df_analysis2['sections in ' + i] = df_analysis2['sections in '+i].str.replace(',', '\n', regex=False)
        df_analysis2['text_element_count'] = df_analysis2[annotation_level_type].sum(axis=1)
        for i in annotation_level_type:
            df_analysis2[i] = df_analysis2[i]/df_analysis2['text_element_count']
            df_analysis2[i] = df_analysis2[i].map("{0:,.2%}".format)

        df_analysis2 = df_analysis2[['filename',
                                     'facility_type',
                                     'text_element_count',
                                     'paragraph-level, no label',
                                     'sections in paragraph-level, no label',
                                     'paragraph-level, single label',
                                     'sections in paragraph-level, single label',
                                     'paragraph-level, multiple label',
                                     'sections in paragraph-level, multiple label',
                                     'phrase-level, single label',
                                     'sections in phrase-level, single label',
                                     'phrase-level, multiple label',
                                     'sections in phrase-level, multiple label',
                                     'list-level, no label',
                                     'sections in list-level, no label',
                                     'list-level, single label',
                                     'sections in list-level, single label',
                                     'list-level, multiple label',
                                     'sections in list-level, multiple label',
                                     'caption-level, no label',
                                     'sections in caption-level, no label',
                                     'caption-level, single label',
                                     'sections in caption-level, single label',
                                     'caption-level, multiple label',
                                     'sections in caption-level, multiple label',
                                     'section-level, no label',
                                     'sections in section-level, no label',
                                     'section-level, single label',
                                     'sections in section-level, single label',
                                    'other, no label',
                                     'sections in other, no label'
        ]]
        df_analysis.to_csv(ts_annot_csv_dir+f'{fname}_label_stat.csv')
        list_doc_analysis.append(df_analysis2)

def extract_keys_by_granularity(df):
    section_lv = {
    'paragraph-level': [],
    'phrase-level': [],
    'list-level': [],
    'caption-level': []
    }
    columns = [col for col in df if col.startswith('sections') and not col.endswith('no label')]
    df = df[columns]
    df = df.applymap(lambda x: ast.literal_eval(x.replace('\n',',')) if x else [None])
    for lv in section_lv.keys():
        for index, row in df.iterrows():
            cols = [col for col in df.columns if lv in col]
            for col in cols:
                section_lv[lv].append(row[col])
    section_lv = {k: list(set([item for sublist in v for item in sublist])) if v != [None] else None for k,v in section_lv.items()}
    section_lv = {k: str([i for i in v if i is not None and i != 'None']) for k,v in section_lv.items()}
    section_lv = {k: [v.replace(',','\n')] for k, v in section_lv.items()}
    df2 = pd.DataFrame(section_lv)
    return df2

df_analysis_all = pd.concat(list_doc_analysis)
df_sections = extract_keys_by_granularity(df_analysis_all)
df_analysis_all.to_csv(ts_annot_csv_dir + f'label_lv_type_stat_summary.csv')
df_sections.to_csv(ts_annot_csv_dir + f'label_lv_type_sections_summary.csv')