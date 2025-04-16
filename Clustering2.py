import sys
import pandas as pd

def read_file(filename):
    try:
        return pd.read_excel(filename, usecols="A,B,D,E,F")
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
    normalized_df = pandas_df.copy()

    # Z-score norm for e/a column
    for column in ["sch9/wt", "ras2/wt", "tor1/wt"]:
        normalized_df[column] = (normalized_df[column] - normalized_df[column].mean()) / normalized_df[column].std()

    return normalized_df

with open("candidate_genes_list.txt", "r") as file:
    candidate_gene = [line.strip() for line in file]

with open("longevity_genes_list.txt", "r") as file:
    longevity_gene = [line.strip() for line in file]

all_genes = read_file("Longotor1delta.xls")
# Extract candidate & longevity genes if exists in full dataset.
candidate_genes = all_genes[all_genes["Public ID"].isin(candidate_gene)]
longevity_genes = all_genes[all_genes["Public ID"].isin(longevity_gene)]

candidate_genes_normalized = z_score_norm(candidate_genes)
longevity_genes_normalized = z_score_norm(longevity_genes)

print()