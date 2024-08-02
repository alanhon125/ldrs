import pandas as pd
import os
import numpy as np

folder = '/home/data2/ldrs_analytics/data/term_matching_csv/tm_finetune0130_ts_v5.3_fa_v4.4'
for f in sorted(os.listdir(folder)):
    df = pd.read_csv(os.path.join(folder, f))
    df = df.replace({np.nan: None, 'nan': None})
    df = df.sort_values(by=['index','phrase_id'])
    df.to_csv(os.path.join(folder, f), index=False, encoding='utf-8-sig')