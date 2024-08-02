import ast
import os
import pandas as pd
import numpy as np
import sys
import asyncio
from app.analytics.views import analyze

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
    s = tuple(["{0:.2f}%".format(i * 100) if i and isinstance(i, (float, int)) else i for i in s])
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
        s += str(i + 1) + f'. {item}\n\n'
    return s


def judge_all(x):
    judge_lst = x.judge
    manual_identifier = x.manual_identifier
    match_term_list = x.match_term_list

    if judge_lst is None and not manual_identifier and not match_term_list:
        return 'N/A'
    elif judge_lst is None and not manual_identifier:
        return 'All Matched'
    elif judge_lst == False or all(i == False for i in judge_lst) and not manual_identifier:
        return 'N/A'
    elif any(i == True for i in judge_lst) and not manual_identifier:
        return 'All Matched'
    elif all(i == False for i in judge_lst) and manual_identifier:
        return 'All Not Matched'
    elif any(i == True for i in judge_lst) and manual_identifier:
        return 'Partially Matched'


class RegenerateXlsx(object):
    def __init__(self, writer, term_match_data, taskId, ts_filename, fa_filename, output_path):
        self.writer = writer
        self.workbook = self.writer.book
        self.df = pd.DataFrame(term_match_data)
        self.taskId = taskId
        self.ts_filename = ts_filename
        self.fa_filename = fa_filename
        self.output_path = output_path
        self.topN = TOP_N_RESULTS

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
            'matchTypeList': 'match_type_list',
            'result': 'judge',
            'manualIdentifier': 'manual_identifier',
            'manualContent': 'manual_content',
            'textContent': 'text_content'
        })
        self.df = self.df.replace({np.nan: None, 'None': None})
        self.df = self.df.map(
            lambda x: x.replace('(None,)', '').replace('(None, None)', '') if isinstance(x, str) else x)
        self.df = self.df.map(lambda x: x.replace("None", '') if isinstance(x, str) else x)
        self.df.loc[~self.df.match_term_list.isna(), 'judge'] = self.df[~self.df.match_term_list.isna()]['judge'].map(
            lambda x: [True] + [False] * (TOP_N_RESULTS - 1) if x is None else x)
        self.df['judge_all'] = self.df[['judge', 'manual_identifier', 'match_term_list']].apply(lambda x: judge_all(x),
                                                                                                axis=1)
        self.df.loc[(self.df.judge_all == 'All Not Matched') & (self.df.manual_identifier.isna()), 'judge_all'] = 'N/A'
        self.df.loc[~self.df.match_term_list.isna(), 'match_count'] = self.df[~self.df.match_term_list.isna()][
            'judge'].map(lambda x: sum(x))
        self.df[['identifier_list', 'match_term_list', 'similarity_list']] = self.df[
            ['identifier_list', 'match_term_list', 'similarity_list']].map(
            lambda x: tuple(ast.literal_eval(str(x)))[:TOP_N_RESULTS] if isinstance(x, str) and (
                        x.startswith('[') or x.startswith('(')) and x != 'None' else x)

        self.df['similarity_list'] = self.df['similarity_list'].map(lambda x: float2percent(x) if x else x)
        if 'match_type_list' in self.df.columns:
            self.df = self.df.drop(columns=['match_type_list'])
        self.df['manually_added_count'] = self.df['manual_identifier'].map(lambda x: len(x) if x else 0)
        self.df[
            ['identifier_list', 'similarity_list', 'match_term_list', 'judge', 'manual_identifier', 'manual_content']] = \
        self.df[['identifier_list', 'similarity_list', 'match_term_list', 'judge', 'manual_identifier',
                 'manual_content']].map(list2string_list)
        self.df['ts_filename'] = self.ts_filename
        self.df['fa_filename'] = self.fa_filename

        self.df = self.df[
            ['index', 'text_block_id', 'page_id', 'phrase_id', 'TS_term', 'text_element', 'list_id', 'TS_text',
             'match_term_list', 'identifier_list', 'similarity_list', 'judge', 'judge_all', 'match_count',
             'manual_identifier', 'manual_content', 'manually_added_count']]
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
                     colname2id['judge']: 20,
                     colname2id['judge_all']: 15,
                     colname2id['match_count']: 8,
                     colname2id['manual_identifier']: 54,
                     colname2id['manual_content']: 80,
                     colname2id['manually_added_count']: 8
                     }

        # If you want to stylize only the Comments column
        # Convert the dataframe to an XlsxWriter Excel object.
        self.df.to_excel(self.writer, sheet_name=self.ts_filename[:9])
        df_len = len(self.df.index)
        worksheet = self.writer.sheets[self.ts_filename[:9]]  # pull worksheet object

        hidden_col_id = [0,
                         colname2id['index'],
                         colname2id['text_block_id'],
                         colname2id['page_id'],
                         colname2id['phrase_id']]

        enlarge_col = ['TS_term', 'list_id', 'match_count', 'manually_added_count']
        enlarge_col_id = [colname2id[i] for i in enlarge_col]
        enlarge_col2 = ['index', 'text_block_id', 'page_id', 'phrase_id', 'TS_text', 'identifier_list',
                        'similarity_list', 'judge', 'judge_all', 'manual_identifier']
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

        worksheet.freeze_panes(1, colname2id['TS_text'] + 1)

        # Apply the autofilter based on the dimensions of the dataframe.
        worksheet.autofilter(0, 0, self.df.shape[0], self.df.shape[1])

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
            "judge_all": [
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
            ]
        }

        for col, format_list in cond_formatting.items():
            if col in self.df.columns:
                criteria = {
                    'type': 'text',
                    'criteria': 'containing'
                }
                row_range = chr(65 + colname2id[col]) + '2:' + chr(65 + colname2id[col]) + str(df_len + 1)
                for format in format_list:
                    criteria.update(format)
                    worksheet.conditional_format(row_range, criteria)

        response = asyncio.run(analyze({"taskId": self.taskId}))
        self.df_analysis = pd.DataFrame(data=[response.data["result"]], index=['Value']).T
        explanation = {
            "taskId": "task ID",
            "avgAccuracy": "average accuracy of term matching = hit ÷ (hit+miss)",
            "termCount": "no. of terms (i.e. section) occur in a TS",
            "termContentCount": "no. of sentence split on TS",
            "irrelaventContentCount": "no. of irrelavent sentence split on TS",
            "relavantInstanceRate": "rate of relavent instance = (term_content_count - irrelavent_content_count) ÷ term_content_count",
            "hitCount": "total no. of hit",
            "missCount": "total no. of miss when top 5 results are not satisfied = min(manual-selected, unselected)",
            "topMissTsTerm": "The TS section that has the most manually-added results",
            "hitInstanceCount": "total no. of hit instance (i.e. no. of sentence split that has at least one hit)",
            "avgHitPerHitInstance": "average no. of hit appears per hit instance",
            "manualCount": "no. of manually-added results",
            "manualInstanceCount": "no. of sentence split that has at least one manually-added result",
            "missRate": "rate of unsatisified miss = miss ÷ (hit+miss)",
            "manualInstanceRate": "Rate of manually-added instance = manual_instance_count ÷ (term_content_count - irrelavent_content_count)",
            "definitionClauseHitCount": "total manually-added count by clause type equal to definition clause (e.g. Cl_1.1-Borrower)",
            "partiesClauseHitCount": "total hit count by clause type equal to parties clause (e.g. Parties-Borrower)",
            "clauseHitCount": "total hit count by clause type equal to regular clause (e.g. Cl_4.2(a))",
            "scheduleHitCount": "total hit count by clause type equal to schedule (e.g. Sched_1.A)",
            "definitionClauseMissCount": "total manually-added count by clause type equal to definition clause (e.g. Cl_1.1-Borrower)",
            "partiesClauseMissCount": "total manually-added count by clause type equal to parties clause (e.g. Parties-Borrower)",
            "clauseMissCount": "total manually-added count by clause type equal to regular clause (e.g. Cl_4.2(a))",
            "scheduleMissCount": "total manually-added count by clause type equal to schedule (e.g. Sched_1.A)",
            "top1HitCount": "total no. of hit appears at top 1",
            "top1HitRate": "hit rate of term matching appears at top1 = sum of hit@top1 ÷ (hit+miss)",
            "upToTop1HitCount": "cumulative sum of hit count at top 1",
            "upToTop1HitRate": "cumulative sum of hit count at top 1 ÷ (hit+miss)",
            "top2HitCount": "total no. of hit appears at top 2",
            "top2HitRate": "hit rate of term matching appears at top2 = sum of hit@top2 ÷ (hit+miss)",
            "upToTop2HitCount": "cumulative sum of hit count up from top 1 to top 2",
            "upToTop2HitRate": "cumulative sum of hit count from top 1 to top 2 ÷ (hit+miss)",
            "top3HitCount": "total no. of hit appears at top 3",
            "top3HitRate": "hit rate of term matching appears at top3 = sum of hit@top3 ÷ (hit+miss)",
            "upToTop3HitCount": "cumulative sum of hit count from top 1 to top 3",
            "upToTop3HitRate": "cumulative sum of hit count from top 1 to top 3 ÷ (hit+miss)",
            "top4HitCount": "total no. of hit appears at top 4",
            "top4HitRate": "hit rate of term matching appears at top4 = sum of hit@top4 ÷ (hit+miss)",
            "upToTop4HitCount": "cumulative sum of hit count from top 1 to top 4",
            "upToTop4HitRate": "cumulative sum of hit count from top 1 to top 4 ÷ (hit+miss)",
            "top5HitCount": "total no. of hit appears at top 5",
            "top5HitRate": "hit rate of term matching appears at top5 = sum of hit@top5 ÷ (hit+miss)",
            "upToTop5HitCount": "cumulative sum of hit count from top 1 to top 5",
            "upToTop5HitRate": "cumulative sum of hit count from top 1 to top 5 ÷ (hit+miss)"
        }
        col_in_percent = ["avgAccuracy", "relavantInstanceRate", "missRate", "manualInstanceRate", "top1HitRate",
                          "upToTop1HitRate", "top2HitRate", "upToTop2HitRate", "top3HitRate", "upToTop3HitRate",
                          "top4HitRate", "upToTop4HitRate", "top5HitRate", "upToTop5HitRate"]
        map_unit = {i: "%" for i in col_in_percent}
        self.df_analysis['Unit'] = self.df_analysis.index.to_series().map(lambda x: map_unit.get(x, None))
        self.df_analysis['Explanation of Performance Metric'] = self.df_analysis.index.to_series().map(
            lambda x: explanation.get(x, None))
        self.df_analysis.to_excel(self.writer, sheet_name='analysis')
        worksheet2 = self.writer.sheets['analysis']  # pull worksheet object
        cell_format = self.workbook.add_format()
        cell_format.set_align('left')
        cell_format2 = self.workbook.add_format()
        cell_format2.set_align('vcenter')
        worksheet2.set_column(0, 0, 21, cell_format)
        worksheet2.set_column(1, 1, 15.83, cell_format2)
        worksheet2.set_column("D:D", 84.5)