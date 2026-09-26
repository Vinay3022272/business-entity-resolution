"""
Preprocess Dataset: Chunk-Wise Business Entity Normalization
============================================================

This script reads any business entity TSV file (Source 1, Source 2, or Source 3),
processes it chunk-by-chunk to keep memory usage minimal, and extracts the
normalized entity features:
  1. entity_id
  2. business_name_raw
  3. name_norm (Unicode preserved, lowercase, punctuation removed, legal forms stripped)
  4. name_ascii (Accent-free transliteration)
  5. legal_form (Clean canonical names like 'private_limited', 'llc', or 'missing')
  6. address_norm (Cleaned, lowercased address)
  7. country_norm (Standardized lowercase country string)

Usage:
------
    # As a Python function:
    from preprocess_dataset import preprocess_tsv_chunkwise
    preprocess_tsv_chunkwise(
        input_path="dataset/train/train_source1.tsv",
        output_path="data/pre-processing/preprocessed_train_sounce1.tsv",
        chunksize=100000
    )

    # From terminal:
    python src/data_preprocessing/preprocess_dataset.py \
        --input dataset/train/train_source1.tsv \
        --output data/pre-processing/preprocessed_train_sounce1.tsv \
        --chunksize 100000
"""

import os
import sys
import re
import csv
import time
import argparse
import unicodedata
import unidecode
from typing import Optional, List, Tuple
import pandas as pd


# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


# ============================================================================
# 1. Comprehensive Predefined Legal Forms Mapping Dictionary
# ============================================================================
LEGAL_FORMS = {
    # ------------------------------------------------------------------------
    # 1. PRIVATE LIMITED VARIANTS (English - India, UK, Commonwealth)
    # ------------------------------------------------------------------------
    "private limited": "private_limited",
    "pvt limited": "private_limited",
    "private ltd": "private_limited",
    "pvt ltd": "private_limited",
    "p limited": "private_limited",
    "p ltd": "private_limited",
    "pvtltd": "private_limited",
    "prvt ltd": "private_limited",
    "private": "private_limited",  # Truncated ending common in Source 2 and Source 3
    "pvt": "private_limited",      # Truncated suffix common in Source 2 and Source 3
    
    # ------------------------------------------------------------------------
    # 2. LIMITED LIABILITY COMPANY (LLC / PLLC - US & Global)
    # ------------------------------------------------------------------------
    "limited liability company": "llc",
    "limited liability co": "llc",
    "llc": "llc",
    "l l c": "llc",
    "professional limited liability company": "pllc",
    "pllc": "pllc",
    "p l l c": "pllc",
    
    # ------------------------------------------------------------------------
    # 3. LIMITED LIABILITY PARTNERSHIP & LIMITED PARTNERSHIP (LLP, LP, LLLP)
    # ------------------------------------------------------------------------
    "limited liability partnership": "llp",
    "llp": "llp",
    "l l p": "llp",
    "limited liability limited partnership": "lllp",
    "lllp": "lllp",
    "limited partnership": "lp",
    "lp": "lp",
    "l p": "lp",
    
    # ------------------------------------------------------------------------
    # 4. PROFESSIONAL CORPORATIONS & ASSOCIATIONS (US Healthcare / Legal)
    # ------------------------------------------------------------------------
    "professional corporation": "pc",
    "prof corp": "pc",
    "pc": "pc",
    "p c": "pc",
    "professional association": "pa",
    "professional service corporation": "psc",
    "psc": "psc",
    
    # ------------------------------------------------------------------------
    # 5. INCORPORATED & CORPORATION (US, Canada, Global)
    # ------------------------------------------------------------------------
    "incorporated": "incorporated",
    "inc": "incorporated",
    "corporation": "corporation",
    "corp": "corporation",
    
    # ------------------------------------------------------------------------
    # 6. PUBLIC LIMITED & PROPRIETARY LIMITED (UK, Australia, India)
    # ------------------------------------------------------------------------
    "public limited company": "public_limited",
    "public limited": "public_limited",
    "plc": "public_limited",
    "proprietary limited": "pty_ltd",
    "proprietary ltd": "pty_ltd",
    "pty limited": "pty_ltd",
    "pty ltd": "pty_ltd",
    "pty": "pty_ltd",
    
    # ------------------------------------------------------------------------
    # 7. SPECIALIZED INDIAN ENTITIES (OPC, Producer Co, Section 8)
    # ------------------------------------------------------------------------
    "one person company": "opc",
    "opc private limited": "opc",
    "opc pvt ltd": "opc",
    "opc": "opc",
    "producer company limited": "producer_company",
    "producer company": "producer_company",
    "section 8 company": "section_8",
    
    # ------------------------------------------------------------------------
    # 8. STANDARD LIMITED & COMPANY SUFFIXES
    # ------------------------------------------------------------------------
    "company limited": "company_limited",
    "co limited": "company_limited",
    "co ltd": "company_limited",
    "and company": "company",
    "and co": "company",
    "ltd": "limited",
    
    # ------------------------------------------------------------------------
    # 9. TRUSTS, SOCIETIES & FOUNDATIONS (India / US Non-profits)
    # ------------------------------------------------------------------------
    "charitable trust": "trust",
    "educational trust": "trust",
    "memorial trust": "trust",
    "welfare trust": "trust",
    "trust": "trust",
    "educational society": "society",
    "welfare society": "society",
    "seva samiti": "society",
    "samiti": "society",
    "society": "society",
    "foundation": "foundation",
    
    # ------------------------------------------------------------------------
    # 10. EUROPEAN CORPORATE FORMS (France, Germany, Italy, Netherlands, Spain, Poland)
    # ------------------------------------------------------------------------
    "gmbh & co kg": "gmbh_co_kg",
    "gmbh co kg": "gmbh_co_kg",
    "gmbh": "gmbh",
    "ag": "ag",
    "sarl": "sarl",
    "s.a.r.l.": "sarl",
    "sarlu": "sarl",
    "sas": "sas",
    "s.a.s.": "sas",
    "sasu": "sasu",
    "s.a.s.u.": "sasu",
    "sci": "sci",
    "s.c.i.": "sci",
    "eurl": "eurl",
    "e.u.r.l.": "eurl",
    "srl": "srl",
    "bv": "bv",
    "nv": "nv",
    "sl": "sl",
    "sp z oo": "sp_z_oo",
    "spzoo": "sp_z_oo",
    "gie": "gie",
    "selarl": "selarl",
    "societe anonyme": "sa",
    "s.a.": "sa",
    
    # ------------------------------------------------------------------------
    # 11. INDIC REGIONAL SCRIPT LEGAL FORMS (Hindi, Tamil, Telugu, etc.)
    # ------------------------------------------------------------------------
    # Hindi / Marathi (Devanagari)
    "प्राइवेट लिमिटेड": "private_limited",
    "प्रा लि": "private_limited",
    "प्रा. लि.": "private_limited",
    "प्रा.लि.": "private_limited",
    "पब्लिक लिमिटेड": "public_limited",
    "लिमिटेड": "limited",
    "लि.": "limited",
    "लि": "limited",
    "एलएलपी": "llp",
    "एल एल पी": "llp",
    "एल.एल.पी.": "llp",
    "ट्रस्ट": "trust",
    "समिति": "society",
    "सोसायटी": "society",
    "सोसाइटी": "society",
    "फाउंडेशन": "foundation",
    
    # Tamil
    "பிரைவேட் லிமிடெட்": "private_limited",
    "லிமிடெட்": "limited",
    "டிரஸ்ட்": "trust",
    
    # Telugu
    "ప్రైవేట్ లిమిటెడ్": "private_limited",
    "లిమిటెడ్": "limited",
    "ట్రస్ట్": "trust",
    
    # Kannada
    "ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್": "private_limited",
    "ಲಿಮಿಟೆಡ್": "limited",
    "ಟ್ರಸ್ಟ್": "trust",
    
    # Malayalam
    "പ്രൈവറ്റ് ലിമിറ്റഡ്": "private_limited",
    "ലിമിറ്റഡ്": "limited",
    "ട്രസ്റ്റ്": "trust",
    
    # Gujarati
    "પ્રાઇવેટ લિમિટેડ": "private_limited",
    "પ્રા લિ": "private_limited",
    "પ્રા. લિ.": "private_limited",
    "લિમિટેડ": "limited",
    "લિ.": "limited",
    "લિ": "limited",
    "ટ્રસ્ટ": "trust",
    
    # Bengali
    "প্রাইভেট লিমিটেড": "private_limited",
    "লিমিটেড": "limited",
    "ট্রাস্ট": "trust",
    
    # Odia
    "ପ୍ରାଇଭେଟ୍ ଲିମିଟେଡ୍": "private_limited",
    "ଲିମିଟେଡ୍": "limited",
    "ଟ୍ରଷ୍ଟ": "trust",
    
    # Punjabi
    "ਪ੍ਰਾਈਵੇਟ ਲਿਮਿਟੇਡ": "private_limited",
    "ਲਿਮਿਟੇਡ": "limited",
    "ਟਰੱਸਟ": "trust",
}

# Compile regex: sorted by length descending so longer phrases match first
_SORTED_PATTERNS = sorted(LEGAL_FORMS.keys(), key=len, reverse=True)
_LEGAL_REGEX = re.compile(
    r'(?:(?<=\s)|(?<=^)|(?<=\b))(' + '|'.join(re.escape(k) for k in _SORTED_PATTERNS) + r')(?:(?=\s)|(?=$)|(?=\b))',
    re.IGNORECASE
)

# Apostrophes and quotes to strip without adding spaces (Orelee's -> orelees)
_APOSTROPHE_REGEX = re.compile(r"['’`]")

# Non-alphanumeric / non-unicode characters (preserves all Indic scripts \u0900-\u0D7F)
_PUNCT_REGEX = re.compile(r'[^\w\s\u0900-\u0D7F]', re.UNICODE)

# Multi-whitespace regex
_WHITESPACE_REGEX = re.compile(r'\s+')

# Pure ASCII alphanumeric cleaning regex
_ASCII_CLEAN_REGEX = re.compile(r'[^a-z0-9\s]')

# Output columns in standardized order
OUTPUT_COLUMNS = [
    "entity_id",
    "business_name_raw",
    "name_norm",
    "name_ascii",
    "legal_form",
    "address_norm",
    "country_norm",
]


# ============================================================================
# 2. Row Normalization Functions
# ============================================================================

def normalize_name(raw_name: Optional[str]) -> Tuple[str, str, str, str]:
    """
    Normalizes a business name:
    1. Removes apostrophes directly (Orelee's -> orelees, McDonald's -> mcdonalds)
    2. Replaces punctuation with spaces while preserving native Unicode letters.
    3. Extracts legal forms using the predefined dictionary.
    4. Removes matched legal forms from the name to produce name_norm.
    5. Creates accent-free ASCII version name_ascii.
    6. Returns (raw_name, name_norm, name_ascii, legal_form_str).
       - If legal form is absent, legal_form_str is 'missing'.
    """
    if raw_name is None or pd.isna(raw_name):
        return "", "", "", "missing"

    raw_str = str(raw_name).strip()
    if not raw_str:
        return "", "", "", "missing"

    # Lowercase & strip apostrophes directly
    clean_str = _APOSTROPHE_REGEX.sub('', raw_str.lower())
    # Replace punctuation with spaces
    clean_str = _PUNCT_REGEX.sub(' ', clean_str)
    # Collapse extra spaces
    clean_str = _WHITESPACE_REGEX.sub(' ', clean_str).strip()

    # Match and extract legal forms
    matches = _LEGAL_REGEX.findall(clean_str)
    if matches:
        seen = set()
        canonical_forms = []
        for m in matches:
            canon = LEGAL_FORMS[m.lower()]
            if canon not in seen:
                seen.add(canon)
                canonical_forms.append(canon)
        legal_form_str = ', '.join(canonical_forms)
    else:
        legal_form_str = "missing"

    # Remove legal forms from the name string
    name_clean = _LEGAL_REGEX.sub(' ', clean_str)
    name_norm = _WHITESPACE_REGEX.sub(' ', name_clean).strip()
    
    # Safety fallback: if stripping legal forms emptied the name completely (e.g. "Memorial Trust"),
    # fall back to clean_str so name_norm is never empty or NaN
    if not name_norm:
        name_norm = clean_str

    # Generate transliterated ASCII version (supports Latin accents, Indic, Cyrillic, etc.)
    try:
        raw_ascii = unidecode.unidecode(name_norm).lower()
        raw_ascii = _ASCII_CLEAN_REGEX.sub(' ', raw_ascii)
        name_ascii = _WHITESPACE_REGEX.sub(' ', raw_ascii).strip()
        if not name_ascii:
            name_ascii = name_norm
    except Exception:
        name_ascii = unicodedata.normalize('NFKD', name_norm).encode('ascii', 'ignore').decode('utf-8')
        name_ascii = _WHITESPACE_REGEX.sub(' ', name_ascii).strip()
        if not name_ascii:
            name_ascii = name_norm

    return raw_str, name_norm, name_ascii, legal_form_str


def normalize_address(raw_address: Optional[str]) -> str:
    """
    Normalizes a business address:
    - Lowercase
    - Strip apostrophes
    - Replace punctuation with spaces
    - Collapse extra whitespace
    """
    if raw_address is None or pd.isna(raw_address):
        return ""
    addr = str(raw_address).strip().lower()
    addr = _APOSTROPHE_REGEX.sub('', addr)
    addr = _PUNCT_REGEX.sub(' ', addr)
    return _WHITESPACE_REGEX.sub(' ', addr).strip()


def normalize_country(raw_country: Optional[str]) -> str:
    """
    Normalizes country string: lowercase and trimmed.
    Supports open-set countries.
    """
    if raw_country is None or pd.isna(raw_country):
        return ""
    return str(raw_country).strip().lower()


# ============================================================================
# 3. Master Chunk-Wise Processing Function
# ============================================================================

def preprocess_tsv_chunkwise(
    input_path: str,
    output_path: str,
    chunksize: int = 100_000,
    max_rows: Optional[int] = None,
    verbose: bool = True
) -> dict:
    """
    Reads an input TSV file in chunks, processes every entity row-wise,
    and appends the cleaned records to output_path.

    Parameters:
    -----------
    input_path : str
        Path to raw input TSV (e.g. 'dataset/train/train_source1.tsv').
    output_path : str
        Path where cleaned TSV will be saved (e.g. 'data/pre-processing/preprocessed_train_sounce1.tsv').
    chunksize : int, default=100_000
        Number of rows to read and process in each batch to keep RAM usage low.
    max_rows : int, optional
        Maximum total rows to process (useful for quick testing/debugging).
    verbose : bool, default=True
        Whether to print real-time chunk progress and throughput statistics.

    Returns:
    --------
    dict
        Summary metrics including total rows processed, elapsed seconds, throughput, and output file size.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Ensure parent output directory exists
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if verbose:
        print("=" * 70)
        print(" CHUNK-WISE ENTITY PREPROCESSING PIPELINE (EXPANDED DICTIONARY)")
        print("=" * 70)
        print(f"Input File:   {input_path}")
        print(f"Output File:  {output_path}")
        print(f"Chunk Size:   {chunksize:,} rows")
        print(f"Legal Rules:  {len(LEGAL_FORMS)} defined patterns")
        if max_rows:
            print(f"Max Rows:     {max_rows:,} rows (TEST MODE)")
        print("-" * 70)

    start_time = time.time()
    total_processed = 0
    chunk_index = 0
    first_chunk = True

    # Read dataset in streaming chunks
    reader = pd.read_csv(
        input_path,
        sep='\t',
        quoting=csv.QUOTE_NONE,
        dtype=str,
        chunksize=chunksize
    )

    for chunk_df in reader:
        chunk_start = time.time()
        chunk_index += 1

        # Check max_rows limit if specified
        if max_rows is not None and total_processed + len(chunk_df) > max_rows:
            chunk_df = chunk_df.iloc[:max_rows - total_processed]

        # Fast row-wise processing using itertuples
        cleaned_records = []
        for row in chunk_df.itertuples(index=False):
            eid = str(row[0]) if pd.notna(row[0]) else ""
            bname = str(row[1]) if pd.notna(row[1]) else ""
            baddr = str(row[2]) if pd.notna(row[2]) else ""
            ctry = str(row[3]) if pd.notna(row[3]) else ""

            raw_name, n_norm, n_ascii, l_form = normalize_name(bname)
            addr_norm = normalize_address(baddr)
            c_norm = normalize_country(ctry)

            cleaned_records.append((
                eid,
                raw_name,
                n_norm,
                n_ascii,
                l_form,
                addr_norm,
                c_norm
            ))

        # Convert chunk to DataFrame and write to disk
        out_chunk_df = pd.DataFrame(cleaned_records, columns=OUTPUT_COLUMNS)
        out_chunk_df.to_csv(
            output_path,
            sep='\t',
            index=False,
            mode='w' if first_chunk else 'a',
            header=first_chunk,
            quoting=csv.QUOTE_MINIMAL
        )
        first_chunk = False
        total_processed += len(cleaned_records)

        chunk_elapsed = time.time() - chunk_start
        cumulative_elapsed = time.time() - start_time
        overall_rate = total_processed / cumulative_elapsed if cumulative_elapsed > 0 else 0

        if verbose:
            print(
                f"[Chunk {chunk_index:3d}] Processed {len(cleaned_records):,d} rows "
                f"in {chunk_elapsed:.2f}s | "
                f"Total: {total_processed:,d} rows "
                f"({cumulative_elapsed:.1f}s, {overall_rate:,.0f} rows/s)"
            )

        if max_rows is not None and total_processed >= max_rows:
            if verbose:
                print(f"Reached max_rows limit of {max_rows:,}. Stopping.")
            break

    total_time = time.time() - start_time
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024) if os.path.exists(output_path) else 0

    if verbose:
        print("=" * 70)
        print(" PREPROCESSING COMPLETE")
        print("=" * 70)
        print(f"Total Rows Processed: {total_processed:,}")
        print(f"Total Time Elapsed:   {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")
        print(f"Average Throughput:   {total_processed / total_time:,.0f} rows/second")
        print(f"Output File Size:     {file_size_mb:.2f} MB")
        print(f"Output Location:      {output_path}")
        print("=" * 70)

    return {
        "input_path": input_path,
        "output_path": output_path,
        "total_rows": total_processed,
        "total_seconds": total_time,
        "rows_per_second": total_processed / total_time if total_time > 0 else 0,
        "file_size_mb": file_size_mb,
    }


def clean_dataset(name: str, max_rows: Optional[int] = None, chunksize: int = 100_000) -> pd.DataFrame:
    """
    Simple function to clean and normalize any dataset chunk-wise and return a 5-row preview.
    
    Args:
        name: Name of dataset (e.g. 'train_source2', 'train_source3', 'test_source1', 
              'test_source2', 'test_source3') or direct file path.
        max_rows: Optional row limit for testing.
        chunksize: Rows per batch (default: 100,000 for low memory usage).
        
    Returns:
        pd.DataFrame: 5-row preview of the preprocessed dataset.
        
    Examples:
        clean_dataset('train_source2', max_rows=10) # Quick test
        clean_dataset('train_source2')              # Full dataset
    """
    base_name = os.path.splitext(os.path.basename(name))[0]
    split = "test" if "test" in base_name else "train"
    
    # 1. Resolve input path
    in_path = name if os.path.exists(name) else os.path.join("..", "..", "dataset", split, f"{base_name}.tsv")
    if not os.path.exists(in_path):
        in_path = os.path.join("dataset", split, f"{base_name}.tsv")
        
    # 2. Resolve output path
    out_path = os.path.join("..", "..", "data", "pre-processing", f"preprocessed_{base_name}.tsv")
    if not os.path.exists(os.path.dirname(out_path)):
        out_path = os.path.join("data", "pre-processing", f"preprocessed_{base_name}.tsv")
        
    # 3. Run chunk-wise preprocessing
    preprocess_tsv_chunkwise(in_path, out_path, chunksize=chunksize, max_rows=max_rows)
    
    # 4. Return preview DataFrame
    return pd.read_csv(out_path, sep="\t", nrows=5)


# ============================================================================
# 4. Command Line Entry Point
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chunk-wise entity normalization pipeline.")
    parser.add_argument(
        "--input", "-i",
        default="dataset/train/train_source1.tsv",
        help="Path to input raw TSV file."
    )
    parser.add_argument(
        "--output", "-o",
        default="data/pre-processing/preprocessed_train_sounce1.tsv",
        help="Path to destination cleaned TSV file."
    )
    parser.add_argument(
        "--chunksize", "-c",
        type=int,
        default=100_000,
        help="Chunk size (number of rows per batch)."
    )
    parser.add_argument(
        "--max-rows", "-m",
        type=int,
        default=None,
        help="Optional row limit for testing."
    )

    args = parser.parse_args()

    preprocess_tsv_chunkwise(
        input_path=args.input,
        output_path=args.output,
        chunksize=args.chunksize,
        max_rows=args.max_rows,
        verbose=True
    )
