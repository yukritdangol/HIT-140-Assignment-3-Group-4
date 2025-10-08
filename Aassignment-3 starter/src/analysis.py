
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

DATA1 = 'dataset1.csv'
DATA2 = 'dataset2.csv'

def safe_read(path):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        print('Error reading', path, e)
        return None

def summary(df, name):
    if df is None:
        return
    print('\n===', name, 'shape:', df.shape)
    print('\n-- dtypes --')
    print(df.dtypes)
    print('\n-- head --')
    print(df.head().to_string())
    print('\n-- describe (numeric) --')
    print(df.describe(include=[np.number]).transpose())

def main():
    d1 = safe_read(DATA1)
    d2 = safe_read(DATA2)
    summary(d1, 'dataset1.csv')
    summary(d2, 'dataset2.csv')

    # Example simple analysis idea (Investigation A starter)
    if d1 is not None:
        # Check relationship between 'risk' and 'seconds_after_rat_arrival'
        if 'risk' in d1.columns and 'seconds_after_rat_arrival' in d1.columns:
            print('\nProportion of risk-taking (risk==1):',
                  d1['risk'].mean())
            grouped = d1.groupby('risk')['seconds_after_rat_arrival'].agg(['count','median','mean','std'])
            print('\nseconds_after_rat_arrival by risk:')
            print(grouped.to_string())
        else:
            print('\nRequired columns for simple analysis not found in dataset1.')

if __name__ == '__main__':
    main()
