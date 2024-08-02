import pandas as pd
import numpy as np
import os

src_folder = '/home/data2/ldrs_analytics/data/reviewed_antd_ts_v3.2'
dst_folder = '/home/data2/ldrs_analytics/data/reviewed_antd_ts_v3.2'

src_files = [f for f in os.listdir(src_folder) if f.endswith('.csv')]

for filename in src_files:
    f = os.path.join(src_folder, filename)
    outpath = os.path.join(dst_folder, filename)
    df = pd.read_csv(f)
    df = df.replace({'nan': None, np.nan: None, 'None': None})
    df['text_block_id'] = (df.section!=df.section.shift()).cumsum()-1
    df['index'] = (df.text_element!=df.text_element.shift()).cumsum()-1
    df['phrase_id'] = df.groupby(['index']).cumcount()
    df.to_csv(outpath, index=False, encoding='utf-8-sig')