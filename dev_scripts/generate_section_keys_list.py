import pandas as pd
import csv
import os
import string
from string import digits

DATA_DIR = '/home/data/ldrs_analytics/data'
TS_CSV_DIR = f'{DATA_DIR}/docparse_csv/TS_v3/'
FA_CSV_DIR = f'{DATA_DIR}/docparse_csv/FA/'
ts_sections_csv = f'{DATA_DIR}/section_keys/ts_sections.csv'
fa_sections_csv = f'{DATA_DIR}/section_keys/fa_sections.csv'
fa_definitions_csv = f'{DATA_DIR}/section_keys/fa_definitions.csv'
fa_subsections_csv = f'{DATA_DIR}/section_keys/fa_subsections.csv'
fa_schedules_csv = f'{DATA_DIR}/section_keys/fa_schedules.csv'
fa_parts_csv = f'{DATA_DIR}/section_keys/fa_parts.csv'
fa_parties_csv = f'{DATA_DIR}/section_keys/fa_parties.csv'

def remove_non_ascii(a_str):
    ascii_chars = set(string.printable)

    return ''.join(
        filter(lambda x: x in ascii_chars, a_str)
    )

def clean_string(s):
    return remove_non_ascii(string.capwords(s).lstrip(digits).lstrip('.').strip())

def valid_text_condition(s):
    if str(s)!='nan' and remove_non_ascii(str(s)).strip() and str(s)[0].isalpha():
        return True
    else:
        return False

def write_append_excel(df, outpath, sheet_name):
    import pandas as pd
    if os.path.exists(outpath):
        mode = 'a'
    else:
        mode = 'w'
    with pd.ExcelWriter(outpath, mode=mode) as writer:
        df.to_excel(writer, sheet_name=sheet_name)

def export_keys_tocsv(doc_dir,output_csv,search_key='section',condition=None):
    l = []
    for file in os.listdir(doc_dir):
        if file.endswith('.csv'):
            doc = pd.read_csv(doc_dir+file)
            if search_key == 'parties':
                column = doc[doc['section'].eq('PARTIES')]['definition']
            else:
                if condition:
                    column = doc[doc[search_key].str.contains(condition, na=False, case=False, regex=True)][search_key]
                else:
                    column = doc[search_key]
            doc_keys = [clean_string(i) for i in column.unique().tolist() if valid_text_condition(i)]
            # exist_keys = pd.read_csv(output_csv)[search_key].tolist()
            # doc_keys = [i for i in doc_keys if i not in exist_keys]
            df = pd.DataFrame(doc_keys, columns=[search_key])
            l.append(df)
    all_df = pd.concat(l).reset_index(drop=True).drop_duplicates().sort_values(by=search_key)
    return all_df

if __name__=="__main__":
    outpath = f'{DATA_DIR}/section_keys/keys.xlsx'
    df_ts_sec = export_keys_tocsv(TS_CSV_DIR,ts_sections_csv,search_key='section')
    df_fa_sec = export_keys_tocsv(FA_CSV_DIR,fa_sections_csv,search_key='section')
    df_fa_def = export_keys_tocsv(FA_CSV_DIR,fa_definitions_csv,search_key='definition')
    df_fa_subsec = export_keys_tocsv(FA_CSV_DIR,fa_subsections_csv,search_key='sub_section')
    df_fa_sched = export_keys_tocsv(FA_CSV_DIR,fa_schedules_csv,search_key='schedule')
    df_fa_part = export_keys_tocsv(FA_CSV_DIR,fa_parts_csv,search_key='part')
    df_fa_parties = export_keys_tocsv(FA_CSV_DIR,fa_parties_csv,search_key='parties')

    write_append_excel(df_ts_sec, outpath, 'ts_sections')
    write_append_excel(df_fa_sec, outpath, 'fa_sections')
    write_append_excel(df_fa_def, outpath, 'fa_definitions')
    write_append_excel(df_fa_subsec, outpath, 'fa_subsections')
    write_append_excel(df_fa_sched, outpath, 'fa_schedules')
    write_append_excel(df_fa_part, outpath, 'fa_part')
    write_append_excel(df_fa_parties, outpath, 'fa_parties')

# record = []
# for fa_file in os.listdir(FA_CSV_DIR):
#     fa = pd.read_csv(FA_CSV_DIR+fa_file)
#     doc_fa_sections = [clean_string(i) for i in fa['section'].unique().tolist() if str(i)!='nan' and remove_non_ascii(str(i)).strip()]
#     fa_sections = pd.read_csv(fa_sections_csv)['section'].tolist()
#     doc_fa_sections = [i for i in doc_fa_sections if i not in fa_sections]
#     pd.DataFrame(doc_fa_sections, columns=['section']).to_csv(fa_sections_csv, mode='a',index=False, header=False,encoding='utf-8-sig')