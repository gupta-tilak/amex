import pandas as pd
import numpy as np
import argparse

# Load data dictionary
DICT_PATH = 'dataset/data_dictionary.csv'
dd = pd.read_csv(DICT_PATH)

type_map = {
    'Numerical': 'float32',
    'Categorical': 'category',
    'One hot encoded': 'int8',
    'Key': 'string',
    'Label': 'int8',
    '-': 'string',
}
# Columns that are dates/timestamps (manually identified)
date_cols = {'id4', 'id5'}

# Build column:type mapping
dd_map = dict(zip(dd['masked_column'], dd['Type']))

def get_dtype(col):
    if col in date_cols:
        return 'datetime64[ns]'
    t = dd_map.get(col, None)
    if t is None:
        return None
    return type_map.get(t, 'string')

def convert_types(df):
    for col in df.columns:
        dtype = get_dtype(col)
        if dtype is None:
            continue  # leave as is
        if dtype == 'datetime64[ns]':
            df[col] = pd.to_datetime(df[col], errors='coerce')
        elif dtype == 'category':
            df[col] = df[col].astype('category')
        elif dtype == 'float32':
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('float32')
        elif dtype == 'int8':
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int8')
        elif dtype == 'string':
            df[col] = df[col].astype('string')
    return df

def main():
    parser = argparse.ArgumentParser(description='Convert types of a parquet file according to data dictionary.')
    parser.add_argument('input', help='Input parquet file')
    parser.add_argument('output', help='Output parquet file')
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    df = convert_types(df)
    df.to_parquet(args.output, index=False)
    print(f'Converted {args.input} and saved to {args.output}')

if __name__ == '__main__':
    main() 