"""
Data Loader Module for Intelligent Credit Risk Assessment Using Explainable AI.

Responsibilities:
1. Download original raw German Credit Data from the official UCI repository.
2. Parse space-delimited records into a pandas DataFrame with descriptive column names.
3. Validate original target labels and map them to standard convention:
   - 0: Good / Non-Risky (original label 1)
   - 1: Bad / Risky / Default (original label 2)
4. Validate schema integrity, missing values, row counts, and label distributions.
5. Save the structured, mapped dataset to data/processed/german_credit_data.csv.
"""

import sys
import urllib.request
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

# Official UCI Repository URLs
UCI_DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/144/statlog+german+credit+data.zip"

# Project Paths
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
RAW_DATA_FILE = RAW_DATA_DIR / "german.data"
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "german_credit_data.csv"

# Official 20 Attribute Names + 1 Target Column
COLUMN_NAMES = [
    "checking_status",        # A11..A14 (Categorical)
    "duration",               # Duration in months (Numerical)
    "credit_history",         # A30..A34 (Categorical)
    "purpose",                # A40..A410 (Categorical)
    "credit_amount",          # Credit amount in DM (Numerical)
    "savings_status",         # A61..A65 (Categorical)
    "employment",             # A71..A75 (Categorical)
    "installment_rate",       # Installment rate in % of disposable income (Numerical)
    "personal_status",        # A91..A95 (Categorical)
    "other_parties",          # A101..A103 (Categorical)
    "residence_since",        # Present residence since in years (Numerical)
    "property_magnitude",     # A121..A124 (Categorical)
    "age",                    # Age in years (Numerical)
    "other_payment_plans",    # A141..A143 (Categorical)
    "housing",                # A151..A153 (Categorical)
    "existing_credits",       # Number of existing credits at this bank (Numerical)
    "job",                    # A171..A174 (Categorical)
    "num_dependents",         # Number of people liable for maintenance (Numerical)
    "telephone",              # A191..A192 (Categorical)
    "foreign_worker",         # A201..A202 (Categorical)
    "credit_risk"             # Original: 1 = Good, 2 = Bad (Target)
]

NUMERICAL_COLUMNS = [
    "duration",
    "credit_amount",
    "installment_rate",
    "residence_since",
    "age",
    "existing_credits",
    "num_dependents"
]

CATEGORICAL_COLUMNS = [
    "checking_status",
    "credit_history",
    "purpose",
    "savings_status",
    "employment",
    "personal_status",
    "other_parties",
    "property_magnitude",
    "other_payment_plans",
    "housing",
    "job",
    "telephone",
    "foreign_worker"
]


def download_raw_data(raw_file_path: Path = RAW_DATA_FILE) -> Path:
    """
    Downloads original german.data from UCI repository if not already present.
    """
    raw_file_path.parent.mkdir(parents=True, exist_ok=True)

    if raw_file_path.exists() and raw_file_path.stat().st_size > 0:
        print(f"[1/5] Raw data already present at: {raw_file_path}")
        return raw_file_path

    print(f"[1/5] Downloading German Credit Data from UCI...")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    # Attempt direct download from legacy UCI mirror
    try:
        req = urllib.request.Request(UCI_DATA_URL, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as response:
            content = response.read()
            raw_file_path.write_bytes(content)
        print(f"      Successfully downloaded raw data ({len(content)} bytes) to {raw_file_path}")
        return raw_file_path
    except Exception as direct_err:
        print(f"      Direct URL failed ({direct_err}). Trying UCI archive zip endpoint...")

    # Fallback to UCI zip bundle
    try:
        req = urllib.request.Request(UCI_ZIP_URL, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as response:
            zip_bytes = response.read()
        with zipfile.ZipFile(BytesIO(zip_bytes)) as z:
            with z.open("german.data") as zf:
                content = zf.read()
                raw_file_path.write_bytes(content)
        print(f"      Successfully extracted german.data ({len(content)} bytes) to {raw_file_path}")
        return raw_file_path
    except Exception as zip_err:
        raise RuntimeError(f"Failed to download dataset from both UCI endpoints: {zip_err}") from zip_err


def parse_german_data(raw_file_path: Path = RAW_DATA_FILE) -> pd.DataFrame:
    """
    Parses space-delimited german.data into a pandas DataFrame with column names.
    Ensures categorical columns stay strings and numerical stay int64.
    """
    print(f"[2/5] Parsing raw space-delimited records...")
    dtype_spec = {col: "str" for col in CATEGORICAL_COLUMNS}
    for col in NUMERICAL_COLUMNS:
        dtype_spec[col] = "int64"
    dtype_spec["credit_risk"] = "int64"

    df = pd.read_csv(
        raw_file_path,
        sep=r"\s+",
        header=None,
        names=COLUMN_NAMES,
        dtype=dtype_spec
    )
    print(f"      Parsed {len(df)} rows and {len(df.columns)} columns.")
    return df


def validate_original_labels(df: pd.DataFrame) -> None:
    """
    Validates that original labels are strictly 1 and 2 with counts 700 and 300.
    """
    print(f"[3/5] Validating original target labels before mapping...")
    original_counts = df["credit_risk"].value_counts().to_dict()
    print(f"      Original label counts: {original_counts}")

    if set(original_counts.keys()) != {1, 2}:
        raise ValueError(f"Unexpected original target classes: {set(original_counts.keys())}, expected {{1, 2}}")
    if original_counts[1] != 700:
        raise ValueError(f"Expected exactly 700 Good (1) records, found {original_counts[1]}")
    if original_counts[2] != 300:
        raise ValueError(f"Expected exactly 300 Bad (2) records, found {original_counts[2]}")
    print("      Original target validation PASSED: 700 Good (1) and 300 Bad (2).")


def map_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Maps original target labels:
      1 (Good) -> 0 (Non-Risky)
      2 (Bad)  -> 1 (Risky/Default)
    """
    df = df.copy()
    validate_original_labels(df)

    print("      Applying target mapping: {1: 0 (Non-Risky), 2: 1 (Risky)}...")
    df["credit_risk"] = df["credit_risk"].map({1: 0, 2: 1}).astype("int64")

    mapped_counts = df["credit_risk"].value_counts().to_dict()
    print(f"      Mapped target counts: {mapped_counts}")

    if mapped_counts[0] != 700:
        raise ValueError(f"Post-mapping validation failed: expected 700 zeros, found {mapped_counts[0]}")
    if mapped_counts[1] != 300:
        raise ValueError(f"Post-mapping validation failed: expected 300 ones, found {mapped_counts[1]}")
    print("      Mapped target validation PASSED: 700 Non-Risky (0) and 300 Risky (1).")
    return df


def validate_dataset(df: pd.DataFrame) -> Dict[str, any]:
    """
    Comprehensive validation of dataset shape, columns, dtypes, and missing values.
    """
    print(f"[4/5] Running comprehensive schema & integrity verification...")

    # 1. Shape
    assert df.shape == (1000, 21), f"Unexpected shape {df.shape}, expected (1000, 21)"

    # 2. Columns
    assert list(df.columns) == COLUMN_NAMES, "Column names or order do not match expected schema"

    # 3. Missing values
    missing_count = int(df.isnull().sum().sum())
    assert missing_count == 0, f"Found {missing_count} missing values in dataset"

    # 4. Data types
    num_cols_found = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and col != "credit_risk"]
    cat_cols_found = [col for col in df.columns if pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col])]

    assert set(num_cols_found) == set(NUMERICAL_COLUMNS), f"Numerical column mismatch: {set(num_cols_found) ^ set(NUMERICAL_COLUMNS)}"
    assert set(cat_cols_found) == set(CATEGORICAL_COLUMNS), f"Categorical column mismatch: {set(cat_cols_found) ^ set(CATEGORICAL_COLUMNS)}"

    # 5. Target column integrity
    assert pd.api.types.is_integer_dtype(df["credit_risk"]), "Target column must be integer dtype"

    results = {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_values": missing_count,
        "numerical_features_count": len(num_cols_found),
        "categorical_features_count": len(cat_cols_found),
        "target_distribution": df["credit_risk"].value_counts().to_dict()
    }
    print("      Integrity verification PASSED with 0 errors.")
    return results


def save_processed_data(df: pd.DataFrame, output_path: Path = PROCESSED_DATA_FILE) -> Path:
    """
    Saves the structured DataFrame to CSV.
    """
    print(f"[5/5] Saving processed dataset...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"      Saved structured dataset to: {output_path} ({output_path.stat().st_size} bytes)")
    return output_path


def load_dataset() -> Tuple[pd.DataFrame, Dict[str, any]]:
    """
    Full pipeline runner: downloads, parses, maps, validates, and saves.
    """
    raw_path = download_raw_data()
    parsed_df = parse_german_data(raw_path)
    mapped_df = map_target(parsed_df)
    validation_results = validate_dataset(mapped_df)
    save_processed_data(mapped_df)
    return mapped_df, validation_results


if __name__ == "__main__":
    try:
        df, metrics = load_dataset()
        print("\n" + "=" * 60)
        print("DATASET SETUP SUCCESSFUL")
        print("=" * 60)
        print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"Missing Values: {metrics['missing_values']}")
        print(f"Numerical Features ({metrics['numerical_features_count']}): {NUMERICAL_COLUMNS}")
        print(f"Categorical Features ({metrics['categorical_features_count']}): {CATEGORICAL_COLUMNS}")
        print(f"Target Distribution: Non-Risky (0) = {metrics['target_distribution'][0]} ({metrics['target_distribution'][0]/len(df):.1%}), Risky (1) = {metrics['target_distribution'][1]} ({metrics['target_distribution'][1]/len(df):.1%})")
        print(f"Sample Head:\n{df.head(3)}")
    except Exception as exc:
        print(f"\n[ERROR] Dataset setup failed: {exc}", file=sys.stderr)
        sys.exit(1)
