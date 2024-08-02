import os
import re
import fire
import string
import difflib
import warnings
import pandas as pd
import numpy as np

from glob import glob
from tqdm import tqdm
from pathlib import Path
from typing import Callable, Dict, List
from num2words import num2words
from sklearn.feature_extraction.text import CountVectorizer

from pandas.errors import SettingWithCopyWarning

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

SECTION_CLEAN_PATTERN = re.compile("[\W_]+")
TEXT_CLEAN_PATTERN = re.compile("[^a-zA-Z ]")

IGNORE_LIST = [
        "62_PF_SYN_TS_mkd_20220331_docparse.csv",
        "65_PF_SYN_TS_mkd_docparse.csv",
        "Acquisition and share pledge_(1) TS_docparse.csv",
        "(11) sample TS_docparse.csv"
    ]

ANNOTATION_LIST = ["clause_id", "definition", "schedule_id"]

UAT_LIST = [
    "1_GL_SYN_TS_mkd_20221215_docparse.csv",
    "13_BL_SYN_TS_mkd_20220713_docparse.csv",
    "19_GL_SYN_TS_mkd_20220718_docparse.csv",
    "23_BL_SYN_TS_mkd_20220715_docparse.csv",
    "24_GF_PRJ_TS_mkd_20220225_docparse.csv",
    "25_NBFI_PRJ_TS_mkd_20220613_docparse.csv",
    "28_GF_PRJ_TS_mkd_20221111_docparse.csv",
    "29_PF_PRJ_TS_mkd_20200624_docparse.csv",
    "3_GF_SYN_TS_mkd_20221018_docparse.csv",
    "31_GL_PRJ_TS_mkd_20220630_docparse.csv",
    "33_BL_PRJ_TS_mkd_20200727_docparse.csv",
    "34_AWSP_PRJ_TS_mkd_20211230_docparse.csv",
    "41_AF_SYN_TS_mkd_20190307_docparse.csv",
    "43_AF_SYN_TS_mkd_20151101_docparse.csv",
    "45_AF_PRJ_TS_mkd_20170331_docparse.csv",
    "49_VF_PRJ_TS_mkd_20151130_docparse.csv",
    "54_VF_PRJ_TS_mkd_20191018_docparse.csv",
    "58_VF_SYN_TS_mkd_20111201_docparse.csv",
    "59_AWSP_SYN_TS_mkd_20210814_docparse.csv",
    "63_AW_SYN_TS_mkd_20221025_docparse.csv",
    "66_PF_SYN_TS_mkd_20230106_docparse.csv",
    "68_PF_SYN_TS_mkd_20221104_docparse.csv",
    "72_NBFI_SYN_TS_mkd_20221215_docparse.csv",
    "74_NBFI_SYN_TS_mkd_20220401_docparse.csv",
    "8_GF_SYN_TS_mkd_20230215_docparse.csv",
]


def clean_section(section: str) -> str:
    return SECTION_CLEAN_PATTERN.sub("", section)


def clean_texts(row: pd.Series):
    section, text_antd, text_parsed = row.section, row.text_antd, row.text_parsed
    text_antd = section + " " + text_antd.replace(section, "")
    text_parsed = section + " " + text_parsed.replace(section, "")
    return pd.Series(
        [
            TEXT_CLEAN_PATTERN.sub("", text_antd).strip(),
            TEXT_CLEAN_PATTERN.sub("", text_parsed).strip(),
        ]
    )


def get_uat_file_list() -> List[str]:
    with open("./UAT_list", "r") as f:
        uat_file_list = f.readlines()
    return [s.strip() for s in uat_file_list]


def get_agg_dict(df: pd.DataFrame) -> Dict[str, Callable]:
    return {
        "text": " ".join,
        **{col: lambda i: list(i) for col in df.columns if col in ANNOTATION_LIST},
        **{
            col: lambda i: set(i)
            for col in df.columns
            if col not in ["section", "text", "label_count"] + ANNOTATION_LIST
        },
        "label_count": "count",
    }


def simply_check(df_antd: pd.Series, df_parsed: pd.DataFrame) -> pd.Series:
    section_antd = df_antd.section

    for row in df_parsed.itertuples():
        section_parsed = row.section

        # Among those failure cases, check:
        # 1: whether the "section" from parsed contains that from reviewed. (Except that one of two strings is empty, but another is not)
        # 2: whether two "section" texts are same after removing whitespaces and punctuations

        if (not section_antd and section_parsed) or (
            section_antd and not section_parsed
        ):
            return pd.Series([None] * 3)

        if (
            section_antd in section_parsed
            or df_antd.section_cleaned == row.section_cleaned
        ):
            return pd.Series([row.text_parsed, section_parsed, row.Index])
    return pd.Series([None] * 3)


def remove_non_alphabet(s):
    '''
    Remove non-alphabet character from a string, e.g. 10. Documentations: -> Documentation
    '''
    import re
    import string

    SYMBOLS = ['●', '•', '·', '∙', '◉', '○', '⦿', '。', '■', '□', '☐', '⁃', '◆', '◇', '◈', '✦', '➢', '➣', '➤', '‣', '▶', '▷', '❖']
    bullet_symbols = ''.join(SYMBOLS)
    punct = re.escape(re.sub(r'\[|\(', '', string.punctuation))

    # remove leading non-alphabet character, e.g. 1. Borrower -> Borrower
    if re.match(r'^([0-9' + punct + bullet_symbols + r'\s]+)(.+)', s):
        s = re.match(r'^([0-9' + punct + bullet_symbols + r'\s]+)(.+)', s).groups()[1]

    # remove trailing non-alphabet character, e.g. Borrower1: -> Borrower
    if re.match(r'(.+[^0-9'+ punct + bullet_symbols +'])([0-9' + punct + bullet_symbols + r'\s]+)$', s):
        s = re.match(r'(.+[^0-9'+ punct + bullet_symbols +'])([0-9' + punct + bullet_symbols + r'\s]+)$', s).groups()[0]

    # replace multiple whitespaces into one whitespace
    s = re.sub(' +',' ',s)
    s = re.sub('"','',s)
    s = re.sub("'",'',s)
    
    return s


def replace_ordinal_numbers(text):
    """
    e.g. 1st -> First; 2nd -> Second
    """

    re_results = re.findall("(\d+\s*(st|nd|rd|th))", text)
    for enitre_result, suffix in re_results:
        num = int(enitre_result[: -len(suffix)])
        text = text.replace(enitre_result, num2words(num, ordinal=True).capitalize())
    return remove_non_alphabet(text)


def calculate_BoW(texts: pd.Series, count_vectorizer: CountVectorizer) -> pd.Series:
    text_antd, text_parsed = texts.text_antd_clean, texts.text_parsed_clean

    if not text_antd and not text_parsed:
        return pd.Series([0, 0])
    elif not text_antd:
        return pd.Series([1, 0])
    elif not text_parsed:
        return pd.Series([0, 1])

    matrix = count_vectorizer.fit_transform(texts)
    diff_array = matrix[0] - matrix[1]

    if matrix[0].sum() == 0:
        more_ratio = 1
        less_ratio = 0
    else:
        more_ratio = -diff_array[diff_array < 0].sum() / matrix[0].sum()
        less_ratio = diff_array[diff_array > 0].sum() / matrix[0].sum()

    return pd.Series([round(more_ratio, 4), round(less_ratio, 4)])


def compare_texts_difference(text_antd: str, text_parsed: str) -> pd.Series:
    more_count_no_space = 0
    less_count_no_space = 0

    for s in difflib.ndiff(text_antd, text_parsed):
        if s[0] == "+" and s[-1] != " ":
            more_count_no_space += 1
        elif s[0] == "-" and s[-1] != " ":
            less_count_no_space += 1

    length_of_string_no_space = sum(map(len, text_antd.split()))

    if length_of_string_no_space:
        more_ratio = more_count_no_space / length_of_string_no_space
        less_ratio = less_count_no_space / length_of_string_no_space
        return pd.Series([round(more_ratio, 4), round(less_ratio, 4)])
    else:
        return pd.Series([1, 0])


def write_to_file(output_folder: str, **dfs: Dict[str, pd.DataFrame]):
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    for name, df in dfs.items():
        df.to_csv(os.path.join(output_folder, f"{name}.csv"), index=False)


def main(antd_folder: str, ts_folder: str, output_folder: str):
    total_valid = pd.DataFrame()
    total_failed = pd.DataFrame()
    total_candidate = pd.DataFrame()

    count_vectorizer = CountVectorizer()

    for file in tqdm(glob("*.csv", root_dir=antd_folder)):

        if file in IGNORE_LIST:
            continue

        antd_df = pd.read_csv(os.path.join(antd_folder, file))
        # antd_df = antd_df[antd_df[["clause_id", "definition", "schedule_id"]].notna().any(axis=1)]    # Only has "annotation"

        parsed_df = pd.read_csv(os.path.join(ts_folder, file))

        # Clean "section" and "text"
        antd_df["section"] = antd_df["section"].apply(
            lambda i: replace_ordinal_numbers(i.strip()) if isinstance(i, str) else ""
        )
        antd_df["text"] = antd_df["text"].fillna("").replace(r"\n", "", regex=True)

        parsed_df["section"] = parsed_df["section"].apply(
            lambda i: replace_ordinal_numbers(i.strip()) if isinstance(i, str) else ""
        )
        parsed_df["text"] = parsed_df["text"].fillna("").replace(r"\n", "", regex=True)

        antd_df["label_count"] = ""
        parsed_df["label_count"] = ""

        # Group based on "section"
        antd_df_grouped = (
            antd_df.groupby(by="section").agg(get_agg_dict(antd_df)).reset_index()
        )
        parsed_df_grouped = (
            parsed_df.groupby(by="section").agg(get_agg_dict(parsed_df)).reset_index()
        )

        antd_df_grouped["valid_annotation"] = antd_df_grouped[ANNOTATION_LIST].apply(
            lambda i: any([pd.notna(i[ann]).any() for ann in ANNOTATION_LIST]), axis=1
        )  # Only check has valid "annotation"
        
        antd_df_grouped = antd_df_grouped.loc[
            antd_df_grouped.valid_annotation
        ]  # Only keep rows with valid annotation

        merged_df = pd.merge(
            antd_df_grouped,
            parsed_df_grouped,
            how="outer",
            on=["section"],
            suffixes=("_antd", "_parsed"),
        )
        merged_df[["section_parsed", "section_cleaned"]] = merged_df["section"].apply(
            lambda i: pd.Series([i, clean_section(i)])
        )

        merged_df["filename"] = file
        merged_df["UAT"] = merged_df["filename"].apply(lambda i: i in UAT_LIST)
        merged_df["from_index"] = None

        failed_cases = merged_df[merged_df["text_parsed"].isnull()]
        candidate_cases = merged_df[
            merged_df["text_antd"].isnull() & ~merged_df["text_parsed"].isnull()
        ]

        if not candidate_cases.empty and not failed_cases.empty:
            merged_df.loc[
                failed_cases.index, ["text_parsed", "section_parsed", "from_index"]
            ] = (
                failed_cases[["section", "section_cleaned"]]
                .apply(lambda i: simply_check(i, candidate_cases), axis=1)
                .values
            )

        valid_df = merged_df[
            ~merged_df.text_antd.isnull() & ~merged_df.text_parsed.isnull()
        ]

        # Remove "section" from beginnings of both "text_antd" and "text_parsed"
        valid_df[["text_antd_clean", "text_parsed_clean"]] = valid_df[
            ["section", "text_antd", "text_parsed"]
        ].apply(clean_texts, axis=1)

        # Calculate cosine similarity by Bag-Of-Words
        valid_df[["difference_BoW_more", "difference_BoW_less"]] = valid_df[
            ["text_antd_clean", "text_parsed_clean"]
        ].apply(lambda i: calculate_BoW(i, count_vectorizer), axis=1)

        # Calculate ratios of differences between to texts
        valid_df[["difference_distance_more", "difference_distance_less"]] = valid_df[
            ["text_antd_clean", "text_parsed_clean"]
        ].apply(
            lambda i: compare_texts_difference(i.text_antd_clean, i.text_parsed_clean),
            axis=1,
        )
        valid_df["partially_matched"] = valid_df["from_index"].apply(
            lambda i: not pd.isnull(i)
        )

        # Update remaining candidate_cases
        drop_index = merged_df[~merged_df["from_index"].isnull()].from_index.to_list()
        candidate_cases.drop(index=drop_index, inplace=True)

        # Update remaining failed_cases
        failed_cases = merged_df[merged_df["text_parsed"].isnull()]

        total_valid = pd.concat([total_valid, valid_df])
        total_failed = pd.concat([total_failed, failed_cases])
        total_candidate = pd.concat([total_candidate, candidate_cases])
    
    
    total_valid = total_valid.sort_values(by=['difference_BoW_less'], ascending=False)
    total_valid.loc[:,["difference_BoW_less","difference_distance_less"]] = total_valid[["difference_BoW_less","difference_distance_less"]].map(lambda x: '{:.2%}'.format(x) if x or x != np.nan else x)
    write_to_file(
        output_folder,
        total_valid=total_valid[
            [
                "filename",
                "UAT",
                "section",
                "section_parsed",
                "text_antd",
                "text_parsed",
                "partially_matched",
                "difference_BoW_less",
                "difference_distance_less",
            ]
        ],
        total_candidate=total_candidate[["filename", "section_parsed", "text_parsed"]],
        total_failed=total_failed[
            ["filename", "UAT", "section", "text_antd", "label_count_antd"]
        ],
    )


if __name__ == "__main__":
    fire.Fire(main)
