import pandas as pd
import xlsxwriter
import os

if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result_name_tag1",
        default='',
        type=str,
        required=False,
        help="The path to sentence-transformer model apply for cosine similarity computation.",
    )
    parser.add_argument(
        "--result_name_tag2",
        default='',
        type=str,
        required=False,
        help="The path to sentence-transformer model apply for cosine similarity computation.",
    )
    parser.add_argument(
        "--result_folder",
        default='/home/data2/ldrs_analytics/data/antd_ts_merge_fa/',
        type=str,
        required=False,
        help="The path to sentence-transformer model apply for cosine similarity computation.",
    )
    args = parser.parse_args()
    writer = pd.ExcelWriter(os.path.join(args.result_folder,f'{args.result_name_tag1}_merge_{args.result_name_tag2}.xlsx'), engine = 'xlsxwriter')
    inpath1 = os.path.join(args.result_folder, f'{args.result_name_tag1}_top5_filter_human_errors/(0)/all_merge&match_results.csv')
    inpath2 = os.path.join(args.result_folder, f'{args.result_name_tag2}_top5_filter_human_errors/(0)/all_merge&match_results.csv')
    
    df1 = pd.read_csv(inpath1)
    df2 = pd.read_csv(inpath2)
    
    df1 = df1[['filename','facility_type','split','index','text_block_id','page_id','phrase_id','section','text_element','text','list_id','fa_identifier','fa_text','match_term_list','identifier_list','similarity_list','match_type_list','judge','judge_all']]
    df2 = df2[['filename','facility_type','split','index','text_block_id','page_id','phrase_id','section','text_element','text','list_id','fa_identifier','fa_text','match_term_list','identifier_list','similarity_list','match_type_list','judge','judge_all']]
    
    df1 = df1[['match_term_list','identifier_list','similarity_list','match_type_list','judge','judge_all']].add_suffix('_'+args.result_name_tag1)
    df2 = df2[['match_term_list','identifier_list','similarity_list','match_type_list','judge','judge_all']].add_suffix('_'+args.result_name_tag2)
    
    df = df1.merge(df2,on=['filename','facility_type','split','index','text_block_id','page_id','phrase_id','section','text_element','text','list_id','fa_identifier','fa_text'])
    
    # Convert the dataframe to an XlsxWriter Excel object.
    df.to_excel(writer, sheet_name='Sheet1')
    worksheet = writer.sheets["Sheet1"]
    for row_num in df.index[(df[f"judge_all_{args.result_name_tag1}"] != "TP") & (df[f"judge_all_{args.result_name_tag2}"] != "TP")].tolist():
        worksheet.set_row(row_num + 1, options={"hidden": True})
    