import re
import pandas as pd
from typing import Dict, List, Any
from pathlib import Path

def extract_classification_tree_from_csv(csv_path: str, expected_columns: List[str] = None) -> Dict[str, Any]:
    """
    Extract hierarchical classification tree from CSV file

    Args:
        csv_path: Path to CSV file
        expected_columns: Optional list of expected column names

    Returns:
        Dictionary representing the hierarchical classification tree
    """
    try:
        df_with_headers = pd.read_csv(csv_path)
        if df_with_headers.iloc[0].dtype == 'object' and all(isinstance(val, str) for val in df_with_headers.iloc[0]):
            df = df_with_headers
            print("CSV loaded with headers")
        else:
            df = pd.read_csv(csv_path, header=None)

        if df.shape[1] > 0:
            df = df.iloc[:, 1:]
            print("Dropped the first column")
        else:
            print("Warning: No columns to drop in the DataFrame.")

        if 'Level_1' in df.columns:
            df.columns = [f"Level_{i+1}" for i in range(df.shape[1])]
            print("Re-assigned column names after dropping the first column")

    except Exception as e:
        print(f"Error loading CSV: {e}")
        return {}

    print(f"Dataset shape after dropping first column: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    df_clean = df.copy()
    duplicate_count = df_clean.duplicated().sum()
    print("Number of duplicate rows:", duplicate_count)
    print(f"Dataset shape before cleaning: {df_clean.shape}")
    df_clean = df_clean.drop_duplicates().reset_index(drop=True)
    df_clean = df_clean.fillna("NA")
    df_clean = df_clean.replace("", "NA")
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            df_clean[col] = df_clean[col].astype(str).apply(
                lambda x: re.sub(r'[\t\n\r\.]', '', x) if x != "NA" else x
            )

    df_clean = df_clean.dropna(how='all')
    print(f"Dataset shape after cleaning: {df_clean.shape}")
    duplicate_count = df_clean.duplicated().sum()
    print("Number of duplicate rows:", duplicate_count)
    classification_tree = {}
    columns = df_clean.columns.tolist()

    for _, row in df_clean.iterrows():
        current_level = classification_tree

        for i, col in enumerate(columns):
            value = str(row[col]).strip()

            if value and value != "NA":
                if i == len(columns) - 1:
                    # Last level - just ensure the key exists
                    if value not in current_level:
                        current_level[value] = {}
                else:
                    if value not in current_level:
                        current_level[value] = {}
                    current_level = current_level[value]

    return classification_tree 