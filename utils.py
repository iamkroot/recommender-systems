import pandas as pd


def pivot_to_csv(df: pd.DataFrame, path: str):
    """Store a pivot table (utility matrix) as CSV"""
    df_csv = pd.DataFrame(columns=df.columns, index=[df.index.name]).append(df)
    df_csv.to_csv(path, index_label=df.columns.name)


def read_pivot_csv(path: str) -> pd.DataFrame:
    """Read a CSV file containing a pivot table (utility matrix)"""
    df = pd.read_csv(path, index_col=0, low_memory=False)
    df.columns.name = df.index.name
    df.index.name = df.index[0]
    return df.iloc[1:]
