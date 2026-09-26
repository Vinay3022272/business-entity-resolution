# Pre-Filter Layer: Blocking Index

## Unified candidate retrieval for training label generation & test-time inference

---

## 1. Purpose

The Pre-Filter Layer is a **deterministic, rule-based candidate retrieval system** that sits upstream of the ML model in the entity resolution pipeline. It answers one question:

> **Given a Source 1 entity, which Source 2 and Source 3 entities are plausible matches worth scoring?**

This same code runs in two contexts:

| Context | Input | Output |
|---------|-------|--------|
| **Training** (label generation) | S1 entity + train S2/S3 | Candidate set → combined with GT to produce `(pair, label)` rows |
| **Test** (inference) | S1 entity + test S2/S3 | Candidate set → scored by two-tower model → final matches |

> [!IMPORTANT]
> The pre-filter must be **identical** in both contexts. Any filtering applied during training but not during test (or vice versa) creates a train/test distribution mismatch that degrades model performance.

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      PRE-FILTER LAYER                           │
│                                                                 │
│  Input: S1 entity (name, address, country)                      │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  LAYER 0: Country Partition                               │  │
│  │  ─────────────────────────────────────                    │  │
│  │  Rule:     exact match on country field                   │  │
│  │  Recall:   100.00% (0 cross-country matches in GT)        │  │
│  │  Purpose:  partition the entire search space              │  │
│  │  Cost:     O(1) lookup                                    │  │
│  └───────────────────────────┬───────────────────────────────┘  │
│                              │                                  │
│  ┌───────────────────────────▼───────────────────────────────┐  │
│  │  LAYER 1: Token Inverted Index (primary)                  │  │
│  │  ─────────────────────────────────────                    │  │
│  │  Rule:     retrieve entities sharing the rarest           │  │
│  │            normalized name token with the query            │  │
│  │  Recall:   ~85.5%                                         │  │
│  │  Median:   ~448 candidates per entity                     │  │
│  │  Cost:     O(1) index lookup + O(k) result iteration      │  │
│  └───────────────────────────┬───────────────────────────────┘  │
│                              │                                  │
│  ┌───────────────────────────▼───────────────────────────────┐  │
│  │  LAYER 1b: Character Trigram Fallback                     │  │
│  │  ─────────────────────────────────────                    │  │
│  │  Rule:     for entities with few/no token candidates,     │  │
│  │            retrieve entities sharing ≥ N char trigrams     │  │
│  │  Recall:   ~91.9% (catches typos, URLs, transliterations) │  │
│  │  Median:   ~3,391 candidates per entity                   │  │
│  │  Trigger:  when Layer 1 returns < min_candidates          │  │
│  │  Cost:     O(T) trigram lookups + O(k) counting            │  │
│  └───────────────────────────┬───────────────────────────────┘  │
│                              │                                  │
│  Output: set of candidate entity_ids (S2/S3)                   │
│  Typical size: 100 – 2,000 per S1 entity                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Layer Specifications

### Layer 0: Country Partition

| Property | Value |
|----------|-------|
| **Filter type** | Exact string match |
| **Field** | `country` |
| **Data structure** | Entities are grouped into per-country buckets at index build time |
| **Recall** | 100.00% — verified: 0 cross-country positive pairs in 7,638,364 GT pairs |
| **Reduction** | ~1.9× (but provides a perfect partition for downstream layers) |

**Rules:**
- Country is treated as an **opaque string label** — no hardcoding of `{"US", "India"}`
- Test set introduces `France` (not in training); the filter handles it automatically
- Entities with missing/null country are placed in a special `"_UNKNOWN_"` bucket and matched against all buckets

```python
# Conceptual implementation
country_buckets = defaultdict(list)
for entity in s2_entities + s3_entities:
    country_buckets[entity.country or "_UNKNOWN_"].append(entity)

# Query: only search within same country
candidates = country_buckets[query_country]
```

---

### Layer 1: Token Inverted Index (Primary Blocker)

| Property | Value |
|----------|-------|
| **Filter type** | Inverted index on normalized name tokens |
| **Recall** | ~85.5% |
| **Median candidates** | ~448 per S1 entity |
| **P95 candidates** | ~10,403 per S1 entity |

**Normalization pipeline:**

```
Input:  "Davis Family Office, Inc."
         │
         ▼  lowercase
        "davis family office, inc."
         │
         ▼  remove non-alphanumeric (keep spaces)
        "davis family office  inc "
         │
         ▼  tokenize on whitespace
        ["davis", "family", "office", "inc"]
         │
         ▼  remove stopwords + legal suffixes
        ["davis", "family", "office"]
         │
         ▼  remove tokens with len <= 1
        ["davis", "family", "office"]
```

**Stopword / legal suffix list:**

```python
STOP_TOKENS = {
    # Legal suffixes
    'inc', 'llc', 'ltd', 'limited', 'corp', 'corporation', 'co', 'company',
    'pvt', 'private', 'llp', 'plc', 'group', 'enterprises', 'solutions',
    'services', 'associates', 'partners', 'international', 'industries',
    'technologies', 'consulting', 'holdings', 'foundation', 'trust',
    # Common English stopwords
    'the', 'of', 'and', 'a', 'an', 'in', 'for', 'on', 'at', 'to', 'by',
    # Overly common domain words
    'india', 'com',
}
```

**Data structure:**

```
token_index = {
    "US": {
        "davis":   {S2-001, S2-047, S3-812, ...},    # posting list
        "family":  {S2-001, S2-193, S3-445, ...},
        "office":  {S2-047, S3-812, S3-999, ...},
        ...
    },
    "India": {
        "prabhav": {S2-102, S3-331},
        "business":{S2-102, S2-500, S3-331, ...},
        ...
    }
}
```

**Retrieval logic — rarest token strategy:**

```python
def get_token_candidates(name, country):
    tokens = normalize_tokens(name)
    if not tokens:
        return set()
    
    # Find the token with the SMALLEST posting list (highest IDF)
    # This is the most discriminative token
    posting_lists = []
    for token in set(tokens):
        posting = token_index[country].get(token, set())
        posting_lists.append((len(posting), posting))
    
    posting_lists.sort(key=lambda x: x[0])  # smallest first
    
    # Return the rarest token's posting list
    # Optionally: union of the 2 rarest for higher recall
    return set(posting_lists[0][1])
```

> [!NOTE]
> **Why "rarest token" and not "all tokens"?**
> Using ALL tokens (union of posting lists) gives median 12,207 candidates — too many.
> The rarest token alone gives median 448 — much tighter.
> The rarest token is by definition the most discriminative part of the name (e.g., `"prabhav"` in `"Prabhav Business Center"` vs the common `"business"` or `"center"`).

---

### Layer 1b: Character Trigram Fallback

| Property | Value |
|----------|-------|
| **Filter type** | Character trigram overlap count |
| **Recall** | ~91.9% (standalone) |
| **Trigger condition** | Layer 1 returns < `min_candidates` (default: 10) |
| **Min shared trigrams** | ≥ 2 (configurable) |

**Why this layer exists:**

Token blocking misses matches where names share no exact word tokens:

```
S1: "Land, Cruz and Hilliard Circle Corporation"
S2: "landcruzhilliard.com"                        ← URL-ified name

Shared tokens after normalization: NONE
Shared trigrams: "lan", "and", "cru", "ruz", "hil", "ill", "lla", "lar", "ard" → 9 shared ✓
```

**Trigram generation:**

```
Input:  "Davis Family Office"
         │
         ▼  lowercase + remove non-alphanumeric (NO spaces)
        "davisfamilyoffice"
         │
         ▼  extract character trigrams
        {"dav", "avi", "vis", "isf", "sfa", "fam", "ami", "mil",
         "ily", "lyo", "yof", "off", "ffi", "fic", "ice"}
```

**Data structure:**

```
trigram_index = {
    "US": {
        "dav": {S2-001, S2-047, S3-100, ...},
        "avi": {S2-001, S2-193, ...},
        ...
    },
    "India": { ... }
}
```

**Retrieval logic:**

```python
def get_trigram_candidates(name, country, min_shared=2):
    trigrams = char_trigrams(name)
    if not trigrams:
        return set()
    
    # Count how many trigrams each candidate shares with query
    overlap_counts = defaultdict(int)
    for tg in trigrams:
        for entity_id in trigram_index[country].get(tg, set()):
            overlap_counts[entity_id] += 1
    
    # Return entities sharing >= min_shared trigrams
    return {eid for eid, count in overlap_counts.items() 
            if count >= min_shared}
```

> [!WARNING]
> The trigram index is **larger in memory** than the token index (more keys, larger posting lists). It's used as a fallback, not the primary path. If memory is tight, this layer can be skipped at the cost of ~6% recall.

---

## 4. Combined Retrieval

```python
def get_candidates(name, country, min_candidates=10):
    """
    Main entry point. Returns the union of candidates from all layers.
    Identical function for training and inference.
    """
    # Layer 0: Country (implicit — indexes are per-country)
    
    # Layer 1: Token blocking (primary)
    candidates = get_token_candidates(name, country)
    
    # Layer 1b: Trigram fallback (if token blocking insufficient)
    if len(candidates) < min_candidates:
        trigram_cands = get_trigram_candidates(name, country, min_shared=2)
        candidates = candidates | trigram_cands
    
    return candidates
```

---

## 5. Usage Patterns

### 5A. Training — Labeled Pair Generation

```python
# ┌──────────────────────────────────────────────┐
# │  STEP 1: Build index on train S2 + S3        │
# └──────────────────────────────────────────────┘
index = BlockingIndex()
for entity in train_s2_entities:
    index.add_entity(entity.id, entity.name, entity.country)
for entity in train_s3_entities:
    index.add_entity(entity.id, entity.name, entity.country)

# ┌──────────────────────────────────────────────┐
# │  STEP 2: Generate labeled pairs              │
# └──────────────────────────────────────────────┘
ground_truth = load_ground_truth()  # {s1_id: set(matched_ids)}

labeled_pairs = []
for s1_entity in train_s1_entities:
    # Positive pairs (label=1) — from ground truth
    positives = ground_truth.get(s1_entity.id, set())
    for match_id in positives:
        labeled_pairs.append((s1_entity.id, match_id, 1))
    
    # Negative pairs (label=0) — from blocking candidates NOT in GT
    candidates = index.get_candidates(s1_entity.name, s1_entity.country)
    negatives = candidates - positives
    
    # Sample negatives at desired ratio
    sampled_neg = random.sample(negatives, min(len(negatives), NEG_RATIO * len(positives)))
    for neg_id in sampled_neg:
        labeled_pairs.append((s1_entity.id, neg_id, 0))

# ┌──────────────────────────────────────────────┐
# │  STEP 3: Save to labeledPairs.tsv            │
# └──────────────────────────────────────────────┘
save_tsv(labeled_pairs, "data/pre-processing/labeledPairs.tsv")
```

### 5B. Test — Candidate Retrieval for Inference

```python
# ┌──────────────────────────────────────────────┐
# │  STEP 1: Build index on test S2 + S3         │
# │  (SAME BlockingIndex class, SAME parameters) │
# └──────────────────────────────────────────────┘
test_index = BlockingIndex()
for entity in test_s2_entities:
    test_index.add_entity(entity.id, entity.name, entity.country)
for entity in test_s3_entities:
    test_index.add_entity(entity.id, entity.name, entity.country)

# ┌──────────────────────────────────────────────┐
# │  STEP 2: For each test S1, get candidates    │
# └──────────────────────────────────────────────┘
all_candidates = {}  # for candidate_pairs.tsv
all_matches = {}     # for matching_results.tsv

for s1_entity in test_s1_entities:
    candidates = test_index.get_candidates(s1_entity.name, s1_entity.country)
    all_candidates[s1_entity.id] = candidates
    
    # ┌──────────────────────────────────────────┐
    # │  STEP 3: Score with trained model         │
    # └──────────────────────────────────────────┘
    if candidates:
        scores = model.predict(s1_entity, candidates)
        matches = {c_id for c_id, score in scores if score > THRESHOLD}
    else:
        matches = set()
    
    all_matches[s1_entity.id] = matches

# ┌──────────────────────────────────────────────┐
# │  STEP 4: Write output files                  │
# └──────────────────────────────────────────────┘
save_tsv(all_candidates, "output/candidate_pairs.tsv")
save_tsv(all_matches, "output/matching_results.tsv")
```

---

## 6. Class Interface

```python
class BlockingIndex:
    """
    Pre-filter layer for entity resolution candidate retrieval.
    
    Used identically for:
    - Training: generating labeled pairs with hard negatives
    - Inference: defining the search space for the ML model
    
    Thread-safe for read operations after build.
    """
    
    def __init__(self, config: BlockingConfig = DEFAULT_CONFIG):
        """
        Parameters
        ----------
        config : BlockingConfig
            min_candidates : int = 10
                Minimum candidates from token blocking before
                trigram fallback triggers.
            min_shared_trigrams : int = 2
                Minimum trigram overlap for trigram fallback.
            stop_tokens : set[str]
                Tokens to exclude from indexing (legal suffixes,
                stopwords, overly common terms).
            enable_trigram_fallback : bool = True
                Whether to use the trigram fallback layer.
        """
    
    def build_from_tsv(self, *tsv_paths: str) -> 'BlockingIndex':
        """
        Build the index by streaming one or more source TSV files.
        Memory-efficient: reads line by line, does not load full
        DataFrame into memory.
        
        Parameters
        ----------
        tsv_paths : str
            Paths to source TSV files (e.g., train_source2.tsv,
            train_source3.tsv)
        
        Returns
        -------
        self (for chaining)
        """
    
    def add_entity(self, entity_id: str, name: str, country: str) -> None:
        """Add a single entity to the index."""
    
    def get_candidates(self, name: str, country: str) -> set[str]:
        """
        Retrieve candidate entity IDs for a query entity.
        
        This is THE main function. It is called identically
        during training (for negative sampling) and during
        inference (for candidate retrieval).
        
        Parameters
        ----------
        name : str
            Business name of the query S1 entity.
        country : str
            Country of the query S1 entity.
        
        Returns
        -------
        set[str]
            Set of candidate entity IDs (S2-xxx, S3-xxx).
        """
    
    def stats(self) -> dict:
        """
        Return index statistics:
        - total_entities: number of indexed entities
        - countries: list of country partitions
        - token_index_size: number of (country, token) entries
        - trigram_index_size: number of (country, trigram) entries
        - avg_posting_list_size: average entities per token
        """
    
    def save(self, path: str) -> None:
        """Serialize index to disk for reuse."""
    
    def load(self, path: str) -> 'BlockingIndex':
        """Load a previously saved index."""
```

---

## 7. Configuration

```python
@dataclass
class BlockingConfig:
    # Layer 1: Token blocking
    stop_tokens: set = field(default_factory=lambda: {
        'inc', 'llc', 'ltd', 'limited', 'corp', 'corporation', 'co',
        'company', 'pvt', 'private', 'llp', 'plc', 'group',
        'enterprises', 'solutions', 'services', 'associates',
        'partners', 'international', 'industries', 'technologies',
        'consulting', 'holdings', 'foundation', 'trust',
        'the', 'of', 'and', 'a', 'an', 'in', 'for', 'on', 'at',
        'to', 'by', 'india', 'com',
    })
    min_token_length: int = 2       # skip single-char tokens
    
    # Layer 1b: Trigram fallback
    enable_trigram_fallback: bool = True
    min_candidates: int = 10        # trigger threshold
    min_shared_trigrams: int = 2    # minimum overlap
    
    # Negative sampling (training only)
    neg_ratio: int = 5              # negatives per positive
    random_seed: int = 42
```

---

## 8. Performance & Memory Estimates

### Index Build Time

| Dataset | Entities | Est. Build Time | Notes |
|---------|----------|----------------|-------|
| Train S2+S3 | ~10.3M | ~30-60s | Single-pass streaming read |
| Test S2+S3 | ~10.0M | ~30-60s | Same |

### Memory Usage

| Component | Est. Memory | Notes |
|-----------|-------------|-------|
| Token index (all countries) | ~2-4 GB | ~800K unique tokens × posting lists |
| Trigram index (all countries) | ~3-6 GB | ~64K unique trigrams × larger posting lists |
| Entity country map | ~400 MB | ~10M entity_id → country string |
| **Total** | **~6-10 GB** | Fits comfortably in 16GB RAM |

### Query Time

| Operation | Time per S1 entity | Notes |
|-----------|-------------------|-------|
| Token candidate retrieval | ~0.1 ms | Single hash lookup + set copy |
| Trigram candidate retrieval | ~5-50 ms | Multiple lookups + counting |
| Total per S1 entity (avg) | ~1-5 ms | Mostly token path, trigram only when needed |
| Full train set (2.2M S1) | ~5-15 min | Candidate retrieval only |
| Full test set (1.7M S1) | ~4-12 min | Candidate retrieval only |

---

## 9. File Locations

```
business-entity-resolution/
├── src/
│   ├── data_preprocessing/
│   │   ├── blocking_index.py          ← BlockingIndex class
│   │   └── labelPairGenerator.ipynb   ← Uses BlockingIndex for training
│   └── models/
│       └── inference.py               ← Uses BlockingIndex for test
├── data/
│   └── pre-processing/
│       ├── labeledPairs.tsv           ← Output of label generation
│       └── blocking_index.pkl         ← Serialized index (optional)
└── output/
    ├── matching_results.tsv           ← Final scored matches
    └── candidate_pairs.tsv            ← All blocked candidates
```

---

## 10. Guarantees & Invariants

> [!CAUTION]
> These invariants MUST hold. Violating any of them creates a train/test mismatch.

| # | Invariant | Consequence if violated |
|---|-----------|----------------------|
| 1 | **Same normalization** function used in training and test | Model sees different token distributions → degraded accuracy |
| 2 | **Same stop_tokens** set used in training and test | Different tokens get indexed → different candidate sets |
| 3 | **Same min_shared_trigrams** threshold | Different recall/candidate-size tradeoff |
| 4 | **Same min_candidates** trigger threshold | Trigram fallback fires at different rates |
| 5 | `matching_results` ⊆ `candidate_pairs` | Validation script warns if a match wasn't a candidate |
| 6 | Every test S1 entity appears in output | Submission rejected otherwise |
| 7 | Singletons get empty `matched_entity_ids` | Scored as 1.0 when correct, 0.0 when wrong |

---

## 11. Edge Cases

| Edge Case | Handling |
|-----------|----------|
| S1 entity with empty/null `business_name` | Token blocking returns ∅. Trigram fallback triggers. If still ∅, entity is treated as singleton (no candidates). |
| S1 entity with only stopword tokens (e.g., "The Company Inc") | All tokens filtered → empty. Trigram fallback triggers on the raw concatenated name. |
| S2/S3 entity with Hindi/Telugu script name (e.g., "राम मार्केटिंग") | Non-ASCII is stripped by `[^a-z0-9]` regex → empty tokens. These entities are only findable via trigram matching on their address or by address-based blocking (future enhancement). |
| New country in test (France) | Country partition handles it automatically — creates a new bucket. No code change needed. |
| Entity with very common rarest token (e.g., "care" appears in 52K S1 entities) | Rarest token posting list may be large (~10K+). This is acceptable — the model handles the precision filtering. |
