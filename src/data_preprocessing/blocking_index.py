"""
Pre-Filter Layer: Blocking Index for Business Entity Resolution
==============================================================

This module provides the core filtering / candidate retrieval functions for
matching business entities across datasets (Source 1 -> Source 2 and Source 3).

Why is this needed?
-------------------
Comparing every Source 1 entity against every Source 2 and Source 3 entity
would require ~22.8 trillion comparisons (impossible in terms of time and storage).
The Pre-Filter Layer reduces this massive search space down to a few hundred
promising candidates per entity in milliseconds.

Key Architecture:
-----------------
1. Layer 0 (Country Partition):
   - Only entities in the exact same country can match (100% verified recall).
   - Opaque string matching (supports US, India, France, etc. automatically).

2. Layer 1 (Token Inverted Index - Primary):
   - Cleans and normalizes business names (lowercase, removes legal suffixes like 'Inc', 'LLC', 'Pvt Ltd').
   - Uses the "rarest token" strategy: selects candidates sharing the most unique / discriminative word.
   - Example: For "Prabhav Business Center", "prabhav" is very rare, while "business" and "center" are common.

3. Layer 1b (Character Trigram Fallback):
   - For entities with very few or no word matches (e.g. typos, concatenated names, or URLs like "landcruzhilliard.com"),
     this layer finds candidates sharing at least N character 3-grams.

Unified Train & Test Usage:
---------------------------
The exact same filtering logic is used for:
  - Training: Finding realistic "hard negatives" (entities that look similar but are NOT a match).
  - Test / Inference: Restricting the search space before running the Two-Tower neural network.
"""

import os
import sys
import re
import csv
import pickle
import random
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from typing import List, Set, Dict, Tuple, Optional, Any, Iterable


# Ensure UTF-8 output on Windows consoles if running standalone
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


# ============================================================================
# Default Stopwords and Legal Suffixes
# ============================================================================
# These common words do not help distinguish one business from another.
# Removing them keeps the index compact and prevents huge posting lists.
DEFAULT_STOP_TOKENS: Set[str] = {
    # Legal company suffixes (US, India, UK, etc.)
    'inc', 'llc', 'ltd', 'limited', 'corp', 'corporation', 'co', 'company',
    'pvt', 'private', 'llp', 'plc', 'group', 'enterprises', 'solutions',
    'services', 'associates', 'partners', 'international', 'industries',
    'technologies', 'consulting', 'holdings', 'foundation', 'trust',
    # Common English grammatical stopwords
    'the', 'of', 'and', 'a', 'an', 'in', 'for', 'on', 'at', 'to', 'by',
    # Overly generic domain tokens
    'india', 'com', 'org', 'net', 'www'
}


# ============================================================================
# Text Normalization & Tokenization Functions
# ============================================================================

def clean_text(text: Optional[str]) -> str:
    """
    Cleans raw text:
    - Converts null/None to empty string.
    - Strips leading/trailing whitespace.
    """
    if text is None:
        return ""
    return str(text).strip()


def normalize_country(country: Optional[str]) -> str:
    """
    Standardizes country string.
    If country is missing or empty, assigns '_UNKNOWN_'.
    """
    c = clean_text(country)
    return c if c else "_UNKNOWN_"


def normalize_tokens(
    name: Optional[str],
    stop_tokens: Optional[Set[str]] = None,
    min_token_length: int = 2
) -> List[str]:
    """
    Tokenizes and normalizes a business name into discriminative word tokens.

    Steps:
    1. Lowercase the name.
    2. Replace all punctuation and non-alphanumeric characters with spaces.
    3. Split into words on whitespace.
    4. Remove legal suffixes and common stopwords.
    5. Discard single-character tokens (length < min_token_length).

    Example:
        >>> normalize_tokens("Davis Family Office, Inc.")
        ['davis', 'family', 'office']

        >>> normalize_tokens("Prabhav Business Center Pvt Ltd")
        ['prabhav', 'business', 'center']
    """
    if not name:
        return []

    if stop_tokens is None:
        stop_tokens = DEFAULT_STOP_TOKENS

    # Lowercase and replace non-alphanumeric characters with spaces
    cleaned = re.sub(r'[^a-z0-9\s]', ' ', str(name).lower())

    # Tokenize on whitespace
    raw_tokens = cleaned.split()

    # Filter out stopwords and very short tokens
    filtered_tokens = [
        t for t in raw_tokens
        if len(t) >= min_token_length and t not in stop_tokens
    ]

    return filtered_tokens


def extract_char_trigrams(name: Optional[str]) -> Set[str]:
    """
    Extracts 3-character substrings (character trigrams) from a business name.

    Used as a fallback when word-token matching finds too few candidates
    (e.g., handles typos, transliterations, or website domains like 'landcruz.com').

    Steps:
    1. Lowercase and remove all non-alphanumeric characters (including spaces).
    2. Slide a 3-character window over the resulting string.

    Example:
        >>> extract_char_trigrams("Davis Office")
        {'dav', 'avi', 'vis', 'iso', 'sof', 'off', 'ffi', 'fic', 'ice'}
    """
    if not name:
        return set()

    # Lowercase and remove non-alphanumeric characters (no spaces)
    compact_str = re.sub(r'[^a-z0-9]', '', str(name).lower())

    if len(compact_str) < 3:
        return set()

    # Generate sliding 3-character slices
    return {compact_str[i:i+3] for i in range(len(compact_str) - 2)}


# ============================================================================
# Configuration Data Class
# ============================================================================

@dataclass
class BlockingConfig:
    """
    Configuration parameters for the Pre-Filter Layer (BlockingIndex).

    Attributes:
    -----------
    stop_tokens : Set[str]
        Set of legal suffixes and common words to exclude from name tokens.
    min_token_length : int
        Minimum length of tokens to include (default: 2 characters).
    top_k_rarest_tokens : int
        Number of rarest tokens to use when querying (default: 1, the single rarest token).
    enable_trigram_fallback : bool
        Whether to enable Layer 1b (character trigram fallback).
        Set to False in low-memory environments (< 4 GB RAM) for maximum efficiency.
    min_candidates : int
        Candidate count threshold below which trigram fallback triggers (default: 10).
    min_shared_trigrams : int
        Minimum number of shared trigrams required for fallback match (default: 2).
    max_trigram_candidates : int
        Safety cap on trigram candidates to avoid excessive memory use (default: 500).
    max_candidates_per_entity : Optional[int]
        Hard limit on total candidates returned per entity (default: 2000).
    neg_ratio : int
        Number of hard negative pairs to sample per positive pair (default: 5).
    random_seed : int
        Random seed for reproducible negative sampling (default: 42).
    """
    stop_tokens: Set[str] = field(default_factory=lambda: set(DEFAULT_STOP_TOKENS))
    min_token_length: int = 2
    top_k_rarest_tokens: int = 1
    enable_trigram_fallback: bool = False
    min_candidates: int = 10
    min_shared_trigrams: int = 2
    max_trigram_candidates: int = 500
    max_candidates_per_entity: Optional[int] = 2000
    neg_ratio: int = 5
    random_seed: int = 42


# ============================================================================
# Pre-Filter Layer: BlockingIndex Class
# ============================================================================

class BlockingIndex:
    """
    Inverted Index for high-speed candidate filtering in business entity resolution.

    Features:
    ---------
    - Country-partitioned inverted index:
      Country is an exact match partition (0 cross-country matches in training data).
    - Rarest-token candidate retrieval:
      Finds entities sharing the rarest (most discriminative) normalized token.
    - Character trigram fallback:
      Recovers candidates when token matching returns too few results.
    - Low-memory streaming build:
      Reads TSV files line-by-line without loading entire datasets into memory.
    """

    def __init__(self, config: Optional[BlockingConfig] = None):
        """
        Initializes the BlockingIndex with optional custom configuration.
        """
        self.config = config if config is not None else BlockingConfig()

        # Inverted index for normalized word tokens:
        # structure: { country -> { token -> set(entity_id) } }
        self.token_index: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))

        # Inverted index for character trigrams (optional fallback):
        # structure: { country -> { trigram -> set(entity_id) } }
        self.trigram_index: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))

        # Track total indexed entities and entity count per country
        self.total_entities: int = 0
        self.country_counts: Dict[str, int] = Counter()

    def add_entity(self, entity_id: str, name: str, country: str) -> None:
        """
        Adds a single business entity into the inverted index.

        Parameters:
        -----------
        entity_id : str
            Unique ID of the entity (e.g., 'S2-163963287' or 'S3-202863386').
        name : str
            Business name.
        country : str
            Country code / name (e.g., 'US', 'India').
        """
        # Clean country and intern strings to save RAM
        c = sys.intern(normalize_country(country))
        eid = sys.intern(clean_text(entity_id))

        self.total_entities += 1
        self.country_counts[c] += 1

        # 1. Normalize name and index word tokens
        tokens = normalize_tokens(
            name=name,
            stop_tokens=self.config.stop_tokens,
            min_token_length=self.config.min_token_length
        )

        country_tokens = self.token_index[c]
        for token in set(tokens):  # use set to avoid duplicate additions
            country_tokens[sys.intern(token)].add(eid)

        # 2. Extract and index character trigrams if fallback is enabled
        if self.config.enable_trigram_fallback:
            trigrams = extract_char_trigrams(name)
            country_trigrams = self.trigram_index[c]
            for tg in trigrams:
                country_trigrams[sys.intern(tg)].add(eid)

    def build_from_tsv(
        self,
        *tsv_paths: str,
        country_filter: Optional[str] = None,
        max_entities: Optional[int] = None,
        show_progress: bool = True
    ) -> 'BlockingIndex':
        """
        Builds the index by streaming one or more source TSV files.
        Memory-efficient: streams line-by-line using csv.reader,
        never loading the whole file into pandas DataFrame.

        Parameters:
        -----------
        *tsv_paths : str
            Paths to source TSV files (e.g., 'train_source2.tsv', 'train_source3.tsv').
        country_filter : Optional[str]
            If specified, only entities from this country are indexed (e.g. 'US' or 'India').
        max_entities : Optional[int]
            Optional cap on total entities to index (useful for fast local testing/demos).
        show_progress : bool
            Whether to print progress updates every 500,000 entities.

        Returns:
        --------
        self : BlockingIndex (for method chaining)
        """
        for tsv_path in tsv_paths:
            if not os.path.exists(tsv_path):
                print(f"[!] Warning: File not found: {tsv_path}")
                continue

            file_basename = os.path.basename(tsv_path)
            if show_progress:
                print(f"[*] Indexing entities from: {file_basename} ...")

            with open(tsv_path, "r", encoding="utf-8", errors="ignore") as f:
                # Use tab-separated reader with QUOTE_NONE to prevent quote errors
                reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)

                # Read header
                try:
                    header = next(reader)
                except StopIteration:
                    continue

                # Map column names
                header_lower = [h.strip().lower() for h in header]
                try:
                    id_idx = header_lower.index("entity_id")
                except ValueError:
                    id_idx = 0  # default first column

                try:
                    name_idx = header_lower.index("business_name")
                except ValueError:
                    name_idx = 1  # default second column

                try:
                    country_idx = header_lower.index("country")
                except ValueError:
                    country_idx = 3 if len(header) > 3 else -1

                line_count = 0
                for row in reader:
                    if not row:
                        continue

                    eid = row[id_idx] if len(row) > id_idx else ""
                    name = row[name_idx] if len(row) > name_idx else ""
                    country = row[country_idx] if (country_idx >= 0 and len(row) > country_idx) else ""

                    if country_filter and normalize_country(country) != normalize_country(country_filter):
                        continue

                    self.add_entity(eid, name, country)
                    line_count += 1

                    if show_progress and line_count % 500_000 == 0:
                        print(f"    Indexed {line_count:,} entities from {file_basename} (Total: {self.total_entities:,})")

                    if max_entities and self.total_entities >= max_entities:
                        if show_progress:
                            print(f"[!] Reached max_entities limit: {max_entities:,}")
                        return self

            if show_progress:
                print(f"[+] Finished {file_basename}: {line_count:,} entities processed.")

        if show_progress:
            print(f"[✓] Index build complete! Total entities indexed: {self.total_entities:,}")
        return self

    def get_token_candidates(self, name: str, country: str) -> Set[str]:
        """
        Retrieves candidates sharing the rarest token(s) with the query name.

        The rarest token has the smallest posting list (highest IDF) in that country,
        making it the most specific and discriminative word in the business name.

        Example:
            For "Prabhav Business Center":
            - "business": 150,000 matches
            - "center":    80,000 matches
            - "prabhav":       12 matches  <-- Rarest token! Returns only 12 candidates!
        """
        c = normalize_country(country)
        if c not in self.token_index:
            return set()

        tokens = normalize_tokens(
            name=name,
            stop_tokens=self.config.stop_tokens,
            min_token_length=self.config.min_token_length
        )
        if not tokens:
            return set()

        country_tokens = self.token_index[c]

        # Find posting list sizes for each token
        token_postings: List[Tuple[int, str, Set[str]]] = []
        for token in set(tokens):
            if token in country_tokens:
                posting = country_tokens[token]
                if posting:
                    token_postings.append((len(posting), token, posting))

        if not token_postings:
            return set()

        # Sort ascending by posting list length (smallest first = rarest)
        token_postings.sort(key=lambda x: x[0])

        # Return union of top K rarest tokens (default: k=1)
        k = max(1, self.config.top_k_rarest_tokens)
        candidates: Set[str] = set()
        for i in range(min(k, len(token_postings))):
            candidates.update(token_postings[i][2])

        return candidates

    def get_trigram_candidates(
        self,
        name: str,
        country: str,
        min_shared: Optional[int] = None
    ) -> Set[str]:
        """
        Retrieves candidates sharing at least `min_shared` character trigrams.

        This catches cases where word tokens don't match exactly due to:
        - Typos (e.g., 'Orelee' vs 'Oreleys')
        - URLs / merged words (e.g., 'landcruzhilliard.com' vs 'Land, Cruz and Hilliard')
        - Abbreviations or transliterations
        """
        c = normalize_country(country)
        if c not in self.trigram_index:
            return set()

        trigrams = extract_char_trigrams(name)
        if not trigrams:
            return set()

        if min_shared is None:
            min_shared = self.config.min_shared_trigrams

        country_trigrams = self.trigram_index[c]

        # Count trigram overlap per candidate entity
        overlap_counts: Dict[str, int] = defaultdict(int)
        for tg in trigrams:
            if tg in country_trigrams:
                for eid in country_trigrams[tg]:
                    overlap_counts[eid] += 1

        # Keep candidates with at least min_shared matching trigrams
        matching_candidates = [
            eid for eid, count in overlap_counts.items()
            if count >= min_shared
        ]

        # Apply safety cap if set
        if self.config.max_trigram_candidates and len(matching_candidates) > self.config.max_trigram_candidates:
            # Sort by highest overlap first
            matching_candidates.sort(key=lambda x: overlap_counts[x], reverse=True)
            return set(matching_candidates[:self.config.max_trigram_candidates])

        return set(matching_candidates)

    def get_candidates(self, name: str, country: str) -> Set[str]:
        """
        Unified candidate retrieval entry point.

        Identical function used during:
        1. Training: To generate hard negative candidates.
        2. Test/Inference: To retrieve candidates for the neural network.

        Steps:
        1. Layer 0: Query only within the same country partition.
        2. Layer 1: Query token inverted index (rarest token strategy).
        3. Layer 1b: If candidates < min_candidates and trigrams are enabled,
           fall back to character trigram matching.
        4. Cap results if max_candidates_per_entity is configured.
        """
        # Step 1: Token blocking (Primary)
        candidates = self.get_token_candidates(name, country)

        # Step 2: Trigram fallback (if enabled and candidates are scarce)
        if self.config.enable_trigram_fallback and len(candidates) < self.config.min_candidates:
            trigram_cands = self.get_trigram_candidates(
                name=name,
                country=country,
                min_shared=self.config.min_shared_trigrams
            )
            candidates = candidates | trigram_cands

        # Step 3: Optional candidate cap
        if self.config.max_candidates_per_entity and len(candidates) > self.config.max_candidates_per_entity:
            # Randomly sample down to cap
            candidates = set(random.sample(list(candidates), self.config.max_candidates_per_entity))

        return candidates

    def stats(self) -> Dict[str, Any]:
        """
        Returns summary statistics about the indexed entities and vocabulary.
        """
        total_tokens = sum(len(toks) for toks in self.token_index.values())
        total_trigrams = sum(len(tgs) for tgs in self.trigram_index.values())

        return {
            "total_entities": self.total_entities,
            "countries": dict(self.country_counts),
            "unique_tokens_by_country": {c: len(toks) for c, toks in self.token_index.items()},
            "total_unique_tokens": total_tokens,
            "trigram_index_enabled": self.config.enable_trigram_fallback,
            "total_unique_trigrams": total_trigrams
        }

    def save(self, filepath: str) -> None:
        """
        Serializes the BlockingIndex to disk using pickle.
        """
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[+] Saved BlockingIndex to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'BlockingIndex':
        """
        Loads a previously saved BlockingIndex from disk.
        """
        with open(filepath, "rb") as f:
            index = pickle.load(f)
        print(f"[+] Loaded BlockingIndex from {filepath} ({index.total_entities:,} entities)")
        return index


# ============================================================================
# Helper Functions: Ground Truth & Negative Sampling
# ============================================================================

def load_ground_truth(
    tsv_path: str,
    max_records: Optional[int] = None
) -> Tuple[Dict[str, Set[str]], Dict[str, Any]]:
    """
    Loads train_ground_truth.tsv into a mapping:
        { source1_entity_id: set(matched_entity_ids) }

    Handles commas, null values, and singletons (entities with no matches).
    Returns the mapping along with summary statistics.
    """
    if not os.path.exists(tsv_path):
        raise FileNotFoundError(f"Ground truth file not found: {tsv_path}")

    gt_map: Dict[str, Set[str]] = {}
    total_positives = 0
    singletons = 0
    with_matches = 0

    with open(tsv_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        header = next(reader, None)

        for i, row in enumerate(reader):
            if not row:
                continue

            s1_id = row[0].strip()
            raw_matches = row[1].strip() if len(row) > 1 else ""

            if raw_matches and raw_matches.lower() not in {"nan", "none", "null", ""}:
                # Split comma-separated IDs
                match_ids = {m.strip() for m in raw_matches.split(",") if m.strip()}
                gt_map[s1_id] = match_ids
                total_positives += len(match_ids)
                with_matches += 1
            else:
                gt_map[s1_id] = set()
                singletons += 1

            if max_records and (i + 1) >= max_records:
                break

    stats = {
        "total_s1_entities": len(gt_map),
        "entities_with_matches": with_matches,
        "singletons": singletons,
        "total_positive_pairs": total_positives
    }
    return gt_map, stats


def sample_hard_negatives(
    candidates: Set[str],
    positives: Set[str],
    num_to_sample: int,
    rng: Optional[random.Random] = None
) -> List[str]:
    """
    Samples 'num_to_sample' hard negatives from pre-filter candidates.
    A hard negative is an entity retrieved by blocking (shares country and name token)
    that is NOT in the true positive matches set.
    """
    if not candidates:
        return []

    # True negatives are candidates that are NOT in positives
    negative_candidates = candidates - positives
    if not negative_candidates:
        return []

    neg_list = list(negative_candidates)

    if len(neg_list) <= num_to_sample:
        return neg_list

    if rng is not None:
        return rng.sample(neg_list, num_to_sample)
    return random.sample(neg_list, num_to_sample)
