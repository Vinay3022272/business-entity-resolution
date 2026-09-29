# Entity Resolution: Two-Stage ANN + LightGBM Reranker Pipeline

A modular, production-ready Two-Stage Entity Resolution architecture designed for large-scale multi-source business record matching.

```text
Entity IDs (Query)
       ↓
Two-Tower Embeddings
       ↓
Stage 1: ANN Candidate Retrieval (Top-K)
       ↓
Align with Labelled Ground-Truth Pairs
       ↓
Create Reranker Dataset (0/1 labels + Query Partitioning)
       ↓
Stage 2: Pairwise Feature Engineering (Local text similarities, legal forms, address)
       ↓
LightGBM Ranker (LambdaRank)
       ↓
Final Ranked Matches (Thresholded)
```

---

## 📁 Package Structure (`src/Reranker/`)

```text
src/Reranker/
├── __init__.py               # Public API exports
├── config.py                 # Dataclasses: ANNConfig, FeatureConfig, RerankerTrainingConfig, InferenceConfig
├── data_placeholders.py      # Abstract interfaces for external DBs, vector stores, and Two-Tower model
├── ann_retriever.py          # Stage 1: Vectorized Top-K ANN candidate retrieval
├── feature_extractor.py      # Local pairwise similarity features (Unicode, Levenshtein, Jaro-Winkler, etc.)
├── dataset_builder.py        # Align with existing labels, query-level train/val split (zero leakage)
├── lgbm_ranker.py            # Stage 2: LightGBM LGBMRanker (lambdarank objective with query grouping)
├── pipeline.py               # End-to-end inference orchestrator (rerank_candidates, thresholding)
├── evaluation.py             # ANN Recall@K, Reranker MRR/NDCG, and Rank Migration analysis
└── example_pipeline.py       # End-to-end executable test and verification workflow
```

---

## 🚀 Quick Usage Example

```python
from src.Reranker import (
    ANNConfig,
    ANNRetriever,
    LightGBMReranker,
    RerankerInferencePipeline,
    create_reranker_dataset,
    train_val_split_by_query,
    evaluate_ann_retrieval,
    evaluate_reranker,
    evaluate_rank_migration
)

# 1. First-Stage ANN Retrieval
retriever = ANNRetriever(
    config=ANNConfig(top_k=10),
    candidate_ids=candidate_ids,
    candidate_embeddings=candidate_embeddings_matrix
)
ann_candidates = retriever.batch_retrieve(query_ids, query_embeddings, k=10)

# 2. Build Reranker Dataset from Existing Ground-Truth
dataset_df, diagnostics = create_reranker_dataset(
    ann_candidates=ann_candidates,
    ground_truth_pairs=ground_truth_df,
    entities_df=all_entities_metadata
)

# 3. Query-Partitioned Split (Prevents Data Leakage)
train_df, val_df = train_val_split_by_query(dataset_df, val_fraction=0.2)

# 4. Train LightGBM LambdaRank Reranker
ranker = LightGBMReranker()
ranker.fit(train_df, val_df=val_df)

# 5. Evaluate Metrics & Rank Migration
ann_metrics = evaluate_ann_retrieval(ann_candidates, ground_truth_df)
val_df["reranker_score"] = ranker.predict(val_df)
val_df["rank"] = val_df.groupby("query_id")["reranker_score"].rank(ascending=False).astype(int)

reranker_metrics = evaluate_reranker(val_df, ground_truth_df)
migration_df, migration_summary = evaluate_rank_migration(ann_candidates, val_df, ground_truth_df)

# 6. End-to-End Inference
pipeline = RerankerInferencePipeline(retriever=retriever, reranker=ranker)
results = pipeline.predict_entity(new_query_record, k_ann=20, top_k_final=5)
```

---

## 🔌 Connecting Real Data & Model (`data_placeholders.py`)

All external connections are isolated in `src/Reranker/data_placeholders.py` as clean functions:

1. `load_two_tower_model(model_path)`: Connect to PyTorch checkpoint (`best.pt`).
2. `load_entity_data(source_path)`: Connect to your database (PostgreSQL, BigQuery, Parquet/TSV).
3. `load_stored_embeddings(embeddings_path)`: Connect to `.npy` file or vector store.
4. `build_or_load_ann_index(embeddings, index_path)`: Connect to FAISS/HNSWLib index.
5. `generate_embedding(entity, model)`: Call Two-Tower model forward pass.
6. `load_ground_truth_pairs(labels_path)`: Connect to ground-truth labelled pairs.
