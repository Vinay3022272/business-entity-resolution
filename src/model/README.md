# Two-Tower model

This folder contains only the shared Character-CNN two-tower model and its training notebook.

Use `train.ipynb` on Kaggle. Set the four file paths in its first code cell:

```text
preprocessed_train_sounce1.tsv
preprocessed_train_sounce2.tsv
preprocessed_train_sounce3.tsv
labeledPairs.tsv
```

The model uses `name_norm`, `name_ascii`, `address_norm`, `country_norm`, and
`legal_form`. Source 1 and Source 2/3 use the same `EntityEncoder` weights.

Saved files are `best.pt`, `last.pt`, `tokenizer.json`, `legal_form_vocab.json`,
`config.json`, and `history.json`.
