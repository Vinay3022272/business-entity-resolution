"""
Filter Functions Module (Alias to blocking_index.py)
===================================================

This module exposes the pre-filter candidate retrieval functions and classes
defined in `blocking_index.py`. It can be imported as either:

    from filter_functions import BlockingIndex, BlockingConfig, normalize_tokens
or
    from blocking_index import BlockingIndex, BlockingConfig, normalize_tokens
"""

from blocking_index import (
    DEFAULT_STOP_TOKENS,
    clean_text,
    normalize_country,
    normalize_tokens,
    extract_char_trigrams,
    BlockingConfig,
    BlockingIndex,
    load_ground_truth,
    sample_hard_negatives,
)

__all__ = [
    "DEFAULT_STOP_TOKENS",
    "clean_text",
    "normalize_country",
    "normalize_tokens",
    "extract_char_trigrams",
    "BlockingConfig",
    "BlockingIndex",
    "load_ground_truth",
    "sample_hard_negatives",
]
