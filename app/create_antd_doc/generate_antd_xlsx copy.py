import ast
import os
import pandas as pd
import numpy as np
import sys

try:
    from config import TOP_N_RESULTS
except ModuleNotFoundError:
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    currentdir = os.path.dirname(os.path.abspath(__file__))
    parentdir = os.path.dirname(currentdir)
    parentparentdir = os.path.dirname(parentdir)
    sys.path.insert(0, parentdir)
    sys.path.insert(0, parentparentdir)
    from config import TOP_N_RESULTS

def get_col_widths(dataframe):
    # First we find the maximum length of the index column
    idx_max = max([len(str(s)) for s in dataframe.index.values] + [len(str(dataframe.index.name))])
    # Then, we concatenate this to the max of the lengths of column name and its values for each column, left to right
    return [idx_max] + [max([len(str(s)) for s in dataframe[col].values] + [len(col)]) for col in dataframe.columns]

def float2percent(string):
    import ast
    if isinstance(string, str):
        s = ast.literal_eval(string)
    else:
        s = string
    s = tuple(["{0:.2f}%".format(i*100) if i and isinstance(i,(float, int)) else i for i in s])
    return s

def tryconvert(value, default, *types):
    for t in types:
        try:
            return t(value)
        except (ValueError, TypeError):
            continue
    return default

def list2string_list(lst):
    if not lst:
        return lst
    s = ''
    for i, item in enumerate(lst):
        s += str(i+1) + f'. {item}\n\n'
    return s

def list2string_list2(lst):
    if not lst:
        return lst
    s = ''
    for i, item in enumerate(lst):
        s += str(i+1) + f'.\t {item}\n\n' # --------------------------------------------------------------------------------------------------------------------------------------------------------------
    return s

class RegenerateXlsx(object):
    def __init__(self, writer, term_match_data, ts_filename, fa_filename, output_path, with_annot=False):
        self.writer = writer
        self.workbook = self.writer.book
        self.df = pd.DataFrame(term_match_data)
        self.ts_filename = ts_filename
        self.fa_filename = fa_filename
        self.output_path = output_path
        self.with_annot = with_annot
        self.topN = TOP_N_RESULTS
        
    def change_word_color(self, worksheet, col_id, word_list ,color):
        # Define the color format and a default format
        cell_format_color = self.workbook.add_format({'font_color': color})
        cell_format_default = self.workbook.add_format({'bold': False})

        # Start iterating through the rows and through all of the words in the list
        for row in range(0,self.df.shape[0]):
            for word in word_list:
                cell = self.df.iloc[row,col_id]
                if cell:
                    try:
                        string_len = len(cell)
                        word_len = len(word)
                        # 1st case, specific word is at the start and there is additional text
                        if (cell.index(word) == 0) and (string_len != word_len):
                            worksheet.write_rich_string(row, 
                                                        col_id, 
                                                        word,
                                                        cell_format_color, 
                                                        cell[word_len:],
                                                        cell_format_default)

                        # 2nd case, specific word is at the middle of the string
                        elif (cell.index(word) > 0) and (cell.index(word) != string_len-word_len):
                            starting_point = cell.index(word)
                            worksheet.write_rich_string(row, 
                                                        col_id, 
                                                        cell_format_default,
                                                        cell[0:starting_point],
                                                        word, 
                                                        cell_format_color, 
                                                        cell[starting_point+word_len:],
                                                        cell_format_default)

                        # 3rd case, specific word is at the end of the string
                        elif (cell.index(word) > 0) and (cell.index(word) == string_len-word_len):
                            starting_point = cell.index(word)
                            worksheet.write_rich_string(row, 
                                                        col_id, 
                                                        cell[0:starting_point],
                                                        cell_format_default,
                                                        word,
                                                        cell_format_color)

                        # 4th case, specific word is the only one in the string
                        elif (cell.index(word) == 0) and (string_len == word_len):
                            worksheet.write(row, col_id, word, cell_format_color)
                    except ValueError:
                        continue

                else:
                    continue

    def generate(self):
        self.df = self.df.rename(columns={
            'indexId': 'index',
            'textBlockId': 'text_block_id',
            'pageId': 'page_id',
            'phraseId': 'phrase_id',
            'listId': 'list_id',
            'textElement': 'text_element',
            'tsTerm': 'TS_term',
            'tsText': 'TS_text',
            'matchTermList': 'match_term_list',
            'identifierList': 'identifier_list',
            'similarityList': 'similarity_list',
            'matchTypeList': 'match_type_list'
        })
        self.df = self.df.replace({np.nan: None})
        self.df = self.df.map(lambda x: x.replace('(None,)','').replace('(None, None)','') if isinstance(x,str) else x)
        self.df = self.df.map(lambda x: x.replace("('TN',)",'').replace("('TN', 'TN')",'') if isinstance(x,str) else x)
        self.df = self.df.map(lambda x: x.replace("None",'') if isinstance(x,str) else x)

        self.df[['identifier_list', 'match_term_list', 'similarity_list']] = self.df[
            ['identifier_list', 'match_term_list', 'similarity_list']].map(
            lambda x: tuple(ast.literal_eval(str(x)))[:TOP_N_RESULTS] if isinstance(x, str) and (x.startswith('[') or x.startswith('(')) and x != 'None' else x)
        
        self.df['similarity_list'] = self.df['similarity_list'].map(lambda x: float2percent(x) if x else x)
        if 'match_type_list' in self.df.columns:
            self.df = self.df.drop(columns=['match_type_list'])
        self.df[['identifier_list', 'similarity_list']] = self.df[['identifier_list', 'similarity_list']].map(list2string_list)
        self.df['match_term_list'] = self.df['match_term_list'].map(list2string_list2)
        if 'identifier_list_old' in self.df.columns:
            self.df[['identifier_list_old', 'similarity_list_old']] = self.df[['identifier_list_old', 'similarity_list_old']].map(list2string_list)
            self.df['match_term_list_old'] = self.df['match_term_list_old'].map(list2string_list2)
        self.df['ts_filename'] = self.ts_filename
        self.df['fa_filename'] = self.fa_filename

        # with no annotation
        if not self.with_annot:
            # self.df.loc[self.df['Old Judgement'].isna(), 'Old Judgement'] = 'N/A'
            # self.df.loc[self.df.text_element.str.contains('section'),'match_term_list'] = 'The associate TS text is a standalone section text. It is not expect to perform term matching.'
            self.df['Judgement'] = None
            self.df['Matched Count'] = None
            self.df['Expected Count'] = None
            self.df['Comment'] = None
            self.df = self.df[['index','text_block_id','page_id','phrase_id','text_element','TS_term','list_id','TS_text','match_term_list','identifier_list','similarity_list','Judgement','Matched Count','Expected Count','Comment','ts_filename','fa_filename']]
            colname2id = {i: self.df.columns.get_loc(i) + 1 for i in list(self.df.columns)}
            map_width = {colname2id['index']: 5,
                         colname2id['text_block_id']: 5,
                         colname2id['page_id']: 5,
                         colname2id['phrase_id']: 5,
                         colname2id['list_id']: 8,
                         colname2id['text_element']: 10,
                         colname2id['TS_term']: 20,
                         colname2id['TS_text']: 40,
                         colname2id['match_term_list']: 80,
                         colname2id['identifier_list']: 54,
                         colname2id['similarity_list']: 20,
                         colname2id['Judgement']: 20,
                         colname2id['Matched Count']: 13,
                         colname2id['Expected Count']: 13,
                         # colname2id['match_term_list_old']: 54,
                         # colname2id['identifier_list_old']: 54,
                         # colname2id['similarity_list_old']: 20,
                         # colname2id['Old Judgement']: 20,
                         # colname2id['Old Matched Count']: 13,
                         # colname2id['Old Expected Count']: 13,
                         colname2id['Comment']: 25}
        else:
            self.df = self.df.map(lambda x: x.replace('TP', 'Hit') if isinstance(x, str) else x)
            self.df = self.df.map(lambda x: x.replace('FP', 'False Alarm') if isinstance(x, str) else x)
            self.df = self.df.map(lambda x: x.replace('TN', 'Correct Reject') if isinstance(x, str) else x)
            self.df = self.df.map(lambda x: x.replace('FN', 'Miss') if isinstance(x, str) else x)
            self.df['clause_id'] = None
            self.df['definition'] = None
            self.df['schedule_id'] = None
            self.df['Comment'] = None
            
            self.df = self.df[['index', 'text_block_id', 'page_id', 'phrase_id', 'section', 'text_element', 'list_id', 'text',
                     'fa_identifier', 'fa_text', 'match_term_list', 'identifier_list', 'similarity_list', 'judge',
                     'judge_all']]  # 'filename', 'facility_type', 'split', 'sim_annotation_fa_text', 'sim_annotation_fa_text(with_parent_text)','sim_annotation_fa_text(no_parent_text)', 'clause_id', 'definition', 'schedule_id', 'Comment','ts_filename','fa_filename'
            
            self.df = self.df.rename(columns={'fa_identifier': 'annotation', 'fa_text': 'fa_text_of_annotation'})
            colname2id = {i: self.df.columns.get_loc(i) + 1 for i in list(self.df.columns)}
            map_width = {colname2id['section']: 30,
                         colname2id['text']: 60,
                         colname2id['annotation']: 27,
                         colname2id['fa_text_of_annotation']: 34,
                         colname2id['match_term_list']: 32,
                         colname2id['identifier_list']: 35,
                         colname2id['similarity_list']: 32,
                         colname2id['judge']: 20}
            # colname2id['Comment']: 38,
            # colname2id['sim_annotation_fa_text(with_parent_text)']: 19,
            # colname2id['sim_annotation_fa_text(no_parent_text)']: 19,
            # colname2id['fa_parent_caption']: 30,
            # colname2id['fa_parent_list']: 30,
            # colname2id['fa_section']: 27,
            # colname2id['fa_sub_section']: 27,
            # colname2id['parent_caption']: 30,
            # colname2id['parent_list']: 30,

        # If you want to stylize only the Comments column
        # Convert the dataframe to an XlsxWriter Excel object.
        self.df.to_excel(self.writer, sheet_name=self.ts_filename[:9])
        df_len = len(self.df.index)
        worksheet = self.writer.sheets[self.ts_filename[:9]]  # pull worksheet object
        
        if self.with_annot:
            hidden_col_id = [0, colname2id['index'], colname2id['text_block_id'], colname2id['page_id'],
                             colname2id['phrase_id']]
        else:
            hidden_col_id = None
        if not self.with_annot:
            enlarge_col = ['TS_term', 'list_id', 'Matched Count', 'Expected Count'] # 'Old Judgement', 'Old Matched Count', 'Old Expected Count'
            enlarge_col_id = [colname2id[i] for i in enlarge_col]
            enlarge_col2 = ['index', 'text_block_id', 'page_id', 'phrase_id', 'TS_text', 'identifier_list', 'similarity_list'] # 'identifier_list_old', 'similarity_list_old'
            enlarge_col_id2 = [colname2id[i] for i in enlarge_col2]
        else:
            enlarge_col = ['section', 'annotation', 'judge_all'] # 'sim_annotation_fa_text(with_parent_text)', 'sim_annotation_fa_text(no_parent_text)'
            enlarge_col_id = [colname2id[i] for i in enlarge_col]
            enlarge_col2 = ['index', 'text_block_id', 'page_id', 'phrase_id', 'text', 'identifier_list', 'similarity_list', 'judge'] # 'sim_annotation_fa_text'
            enlarge_col_id2 = [colname2id[i] for i in enlarge_col2]
            
        for idx, width in enumerate(get_col_widths(self.df)):  # loop through all columns
            style = dict()

            if idx in list(map_width.keys()):
                style.update({'text_wrap': True})
                width = map_width[idx]
            else:
                style.update({'text_wrap': False})

            format = self.workbook.add_format(style)
            format.set_align('vcenter')
            if idx in enlarge_col_id:
                format.set_font_size(18)
            if idx in enlarge_col_id2:
                format.set_font_size(14)

            if hidden_col_id and idx in hidden_col_id:
                col_range = chr(65 + idx) + ':' + chr(65 + idx)  # col_id =0, range = "A:A", chr(65)="A"
                worksheet.set_column(col_range, None, None, {'hidden': 1})
            else:
                worksheet.set_column(idx, idx, width, format)

        if not self.with_annot:
            worksheet.freeze_panes(1, 1)
        else:
            worksheet.freeze_panes(1, colname2id['annotation'])

        # Apply the autofilter based on the dimensions of the dataframe.
        worksheet.autofilter(0, 0, self.df.shape[0], self.df.shape[1])

        # Add an optional filter criteria. The placeholder "judge_all" in the filter
        # is ignored and can be any string that adds clarity to the expression.
        # worksheet.filter_column('P', "x = FN")
        
        # Add data validation on particular columns
        validations = {
            "Judgement": {
                "source": [
                        "All Matched", 
                        "Partially Matched", 
                        "All Not Matched", 
                        "N/A"
                    ]
                },
            "Comment": {
                "source": [
                        "Wrong section", 
                        "Wrong split on text",
                        "Fail to split text into seperate phrases",
                        "Text has no keypoints"
                    ]
                },
        }
        
        for col, source in validations.items():
            if col in self.df.columns:
                criteria = {"validate": "list"}
                row_range = chr(65 + colname2id[col]) + '2:' + chr(65 + colname2id[col]) + str(df_len+1)
                criteria.update(source)
                worksheet.data_validation(row_range, criteria)
        
        
        # Light red fill with dark red text.
        red_fill = self.workbook.add_format({'bg_color': '#FFC7CE',
                                        'font_color': '#9C0006'})

        # Light yellow fill with dark yellow text.
        yellow_fill = self.workbook.add_format({'bg_color': '#FFEB9C',
                                           'font_color': '#9C6500'})

        # Green fill with dark green text.
        green_fill = self.workbook.add_format({'bg_color': '#C6EFCE',
                                          'font_color': '#006100'})
        
        # Blue fill with dark green text.
        blue_fill = self.workbook.add_format({'bg_color': '#CFE2F3',
                                          'font_color': '#000FFF'})
        
        # Purple fill with dark green text.
        purple_fill = self.workbook.add_format({'bg_color': '#B4A7D6',
                                          'font_color': '#4700FF'})
        
        # Add condition formatting on particular columns with text containing particular text
        cond_formatting = {
            # 'text_element': [
            #     {
            #         'value': "section",
            #         'format': red_fill
            #     },
            #     {
            #         'value': "paragraph",
            #         'format': green_fill
            #     },
            #     {
            #         'value': "list",
            #         'format': yellow_fill
            #     },
            #     {
            #         'value': "caption",
            #         'format': blue_fill
            #     },
            #     {
            #         'value': "table",
            #         'format': purple_fill
            #     },
            # ],
            "Judgement": [
                {
                    'value': "All Not Matched",
                    'format': red_fill
                },
                {
                    'value': "All Matched",
                    'format': green_fill
                },
                {
                    'value': "Partially Matched",
                    'format': yellow_fill
                },
                {
                    'value': "N/A",
                    'format': blue_fill
                }
            ],
            # "judge": [
            #     {
            #         'value': "Miss",
            #         'format': red_fill
            #     },
            #     {
            #         'value': "Hit",
            #         'format': green_fill
            #     },
            #     {
            #         'value': "False Alarm",
            #         'format': yellow_fill
            #     },
            #     {
            #         'value': "Correct Reject",
            #         'format': blue_fill
            #     },
            #     {
            #         'value': "FN",
            #         'format': red_fill
            #     },
            #     {
            #         'value': "TP",
            #         'format': green_fill
            #     },
            #     {
            #         'value': "FP",
            #         'format': yellow_fill
            #     },
            #     {
            #         'value': "TN",
            #         'format': blue_fill
            #     }
            # ],
            "judge_all": [
                {
                    'value': "Miss",
                    'format': red_fill
                },
                {
                    'value': "Hit",
                    'format': green_fill
                },
                {
                    'value': "False Alarm",
                    'format': yellow_fill
                },
                {
                    'value': "Correct Reject",
                    'format': blue_fill
                },
                {
                    'value': "FN",
                    'format': red_fill
                },
                {
                    'value': "TP",
                    'format': green_fill
                },
                {
                    'value': "FP",
                    'format': yellow_fill
                },
                {
                    'value': "TN",
                    'format': blue_fill
                }
            ]
                
        }
        
        for col, format_list in cond_formatting.items():
            if col in self.df.columns:
                criteria = {
                                'type': 'text',
                                'criteria': 'containing'
                            }
                row_range = chr(65 + colname2id[col]) + '2:' + chr(65 + colname2id[col]) + str(df_len+1)
                for format in format_list:
                    criteria.update(format)
                    worksheet.conditional_format(row_range, criteria)
                    
        # # change specific word color
        # change_color_scheme = {
        #     "judge_all":
        #         [
        #             {
        #                 'words': ["Miss", "FN"],
        #                 'color': 'red'
        #             },
        #             {
        #                 'words': ["Hit", "TP"],
        #                 'color': 'green'
        #             },
        #             {
        #                 'words': ["False Alarm", "FP"],
        #                 'color': 'orange'
        #             },
        #             {
        #                 'words': ["Correct Reject", "TN"],
        #                 'color': 'blue'
        #             }
        #         ],
        #     "identifier_list": 
        #         [
        #             {
        #                 'words': ["Cl_"],
        #                 'color': 'green'
        #             },
        #             {
        #                 'words': ["Cl_1.1"],
        #                 'color': 'red'
        #             },
        #             {
        #                 'words': ["Sched_"],
        #                 'color': 'blue'
        #             },
        #             {
        #                 'words': ["Parties-"],
        #                 'color': 'orange'
        #             },
        #         ]
        # }
        
        # for col, change_scheme in change_color_scheme.items():
        #     if col in self.df.columns:
        #         col_id = colname2id[col] -1
        #         for i in change_scheme:
        #             word_list = i['words']
        #             color = i['color']
        #             self.change_word_color(worksheet, col_id, word_list ,color)


if __name__=="__main__":
    import argparse
    
    # target = [
    #     "125_VF_SYN_TS_mkd_20231013_docparse_results.csv",
    #     "126_VF_SYN_TS_mkd_20230601_docparse_results.csv",
    #     "127_NBFI_SYN_TS_mkd_20230330_docparse_results.csv",
    #     "128_NBFI_SYN_TS_mkd_20230224_docparse_results.csv",
    #     "129_AW_SYN_TS_mkd_20230626_docparse_results.csv",
    #     "130_BL_SYN_TS_mkd_20230710_docparse_results.csv",
    #     "131_AF_SYN_TS_mkd_20230718_docparse_results.csv",
    #     "132_PF_SYN_TS_mkd_20230106_docparse_results.csv",
    #     "133_BL_PRJ_TS_mkd_20230926_docparse_results.csv",
    #     "134_GF_PRJ_TS_mkd_20230829_docparse_results.csv",
    #     "135_GL_PRJ_TS_mkd_20231004_docparse_results.csv",
    #     "136_BL_SYN_TS_mkd_20230418_docparse_results.csv",
    #     "137_GL_SYN_TS_mkd_20231121_docparse_results.csv",
    #     "138_GF_SYN_TS_mkd_20230130_docparse_results.csv",
    #     "139_NBFI_PRJ_TS_mkd_20231122_docparse_results.csv",
    #     "140_NBFI_SYN_TS_mkd_20230512_docparse_results.csv",
    #     "141_GL_SYN_TS_mkd_20231221_docparse_results.csv",
    #     "142_PF_SYN_TS_mkd_20230810_docparse_results.csv",
    #     "143_GF_SYN_TS_mkd_undated_docparse_results.csv",
    #     "144_NBFI_SYN_TS_mkd_20231031_docparse_results.csv",
    #     "145_GF_SYN_TS_mkd_20231031_docparse_results.csv",
    #     "146_BL_PRJ_TS_mkd_20230629_docparse_results.csv",
    #     "147_GL_PRJ_TS_mkd_20230817_docparse_results.csv",
    #     "148_GF_PRJ_TS_mkd_20230919_docparse_results.csv",
    #     "149_BL_PRJ_TS_mkd_20231102_docparse_results.csv"
    # ]

    result_tag = 'finetune0130_reviewed_ts_v4_0304_fa_v4.4_section_top5_filter_human_errors'
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--term_match_folder",
        default='/home/data2/ldrs_analytics/data/antd_ts_merge_fa/0304_lu/finetune0130_raw_ts_v5.5_fa_v4.4_section_top5_filter_human_errors/(0)',
        type=str,
        required=False,
        help="The input term matching result CSV directory.",
    )
    parser.add_argument(
        "--output_folder",
        default='/home/data2/ldrs_analytics/data/antd_ts_merge_fa/0304_lu/finetune0130_raw_ts_v5.5_fa_v4.4_section_top5_filter_human_errors/(0)',  # f'/home/data2/ldrs_analytics/data/antd_ts_merge_fa/0304_lu/{result_tag}/(0)'
        type=str,
        required=False,
        help="The output annotated term match MS Word document directory.",
    )
    parser.add_argument(
        "--target_list",
        default=[
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
],
        type=list,
        required=False,
        help="The target term matching CSV files in term_match_folder that used to generate annotated MS document.",
    )
    parser.add_argument(
        "--with_annot",
        default=True,
        type=bool,
        required=False,
        help="True if term matching result CSV contains annotation.",
    )
    suffix = '_docparse_results.csv'
    args = parser.parse_args()
    output_fname = f'term_match_result_annotated_all.xlsx'
    output_path = os.path.join(args.output_folder, output_fname)

    if args.target_list:
        files = sorted([f for f in os.listdir(args.term_match_folder) if f.endswith('.csv') and f in args.target_list])
    else:
        files = sorted([f for f in os.listdir(args.term_match_folder) if f.endswith('.csv')])
        
    writer = pd.ExcelWriter(output_path, engine='xlsxwriter')
    for f in files:
        term_match_data = pd.read_csv(os.path.join(args.term_match_folder,f)).to_dict('records')
        ts_filename = f.replace(suffix, '')
        fa_filename = ts_filename.replace('TS', 'FA')
        print(f'Generate Annotated Term Match Results for file: {ts_filename}')
        RegenerateXlsx(writer, term_match_data, ts_filename, fa_filename, output_path, with_annot=args.with_annot).generate()
    # Close the Pandas Excel writer and output the Excel file.
    writer.close()