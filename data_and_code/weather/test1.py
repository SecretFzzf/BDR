import pandas as pd
import numpy as np

df = pd.read_csv('weather.csv')
print(f'Rows: {len(df)}')
print(f'Columns: {df.shape[1]}')
print(f'Column names: {list(df.columns)}')
numeric_df = df.select_dtypes(include=[np.number])
print(f'Numeric feature count: {numeric_df.shape[1]}')
print(f'Missing rate: {numeric_df.isnull().mean().mean()*100:.2f}%')
print(f'Time span: {df.iloc[0,0]} to {df.iloc[-1,0]}')
print(numeric_df.describe().round(2))
