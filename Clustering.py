import sys
import pandas as pd

def read_file(filename):
    try:
        return pd.read_excel(filename, usecols="A,D,E,F")
    except PermissionError:
        print("[Error:] Cannot read from the file while it is open. Please close it.", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print(f"[Error:] Cannot find the file '{filename}'.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"[Error:] An unknown error occurred: {e}", file=sys.stderr)
        sys.exit(1)

def z_score_norm(pandas_df):
    pandas_df["sch9/wt"] = (pandas_df["sch9/wt"] - pandas_df["sch9/wt"].mean()) / pandas_df["sch9/wt"].std()
    pandas_df["ras2/wt"] = (pandas_df["ras2/wt"] - pandas_df["ras2/wt"].mean()) / pandas_df["ras2/wt"].std()
    pandas_df["tor1/wt"] = (pandas_df["tor1/wt"] - pandas_df["tor1/wt"].mean()) / pandas_df["tor1/wt"].std()

    return pandas_df

data = read_file("Longotor1delta.xls")
normalized_data = z_score_norm(data)
print(data)