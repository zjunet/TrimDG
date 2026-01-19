import numpy as np # type: ignore
import pandas as pd # type: ignore

df = pd.read_csv("ml_BitOtc.csv")  # 筛选label为1的行 
label_1_rows = df[df['label'] == 1.0] 
print(label_1_rows)

rows_180 = df[df['u'] == 180]
print(rows_180)


