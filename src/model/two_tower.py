"""A small, fully local shared two-tower model for business records."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset


TEXT_FIELDS = ("name_norm", "name_ascii", "address_norm", "country_norm")


def split_legal_forms(value: str) -> list[str]:
    """Accepts 'llc, private_limited', JSON lists, or 'missing'."""
    value = str(value or "").strip()
    if value.startswith("["):
        try:
            value = ",".join(json.loads(value))
        except (json.JSONDecodeError, TypeError):
            pass
    return [x.strip(" '\"") for x in re.split(r"[,;|]", value)
            if x.strip(" '\"") and x.strip(" '\"") != "missing"]


class CharTokenizer:
    """Unicode character vocabulary learned from local training records only."""
    PAD, UNK = 0, 1

    def __init__(self, vocab: dict[str, int]):
        self.vocab = vocab

    @classmethod
    def fit(cls, records: pd.DataFrame, max_chars: int = 12000) -> "CharTokenizer":
        counts = Counter()
        for field in TEXT_FIELDS:
            counts.update("".join(records[field].fillna("").astype(str)))
        vocab = {char: index + 2 for index, (char, _) in
                 enumerate(counts.most_common(max_chars))}
        return cls(vocab)

    def encode(self, values, length: int) -> torch.Tensor:
        output = np.zeros((len(values), length), dtype=np.int64)
        for row, value in enumerate(values):
            ids = [self.vocab.get(char, self.UNK) for char in str(value)[:length]]
            output[row, :len(ids)] = ids
        return torch.from_numpy(output)

    def save(self, path):
        Path(path).write_text(json.dumps(self.vocab, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))


class LegalFormVocab:
    PAD, UNK = 0, 1

    def __init__(self, vocab: dict[str, int]):
        self.vocab = vocab

    @classmethod
    def fit(cls, records: pd.DataFrame) -> "LegalFormVocab":
        values = sorted({form for value in records["legal_form"].fillna("")
                         for form in split_legal_forms(value)})
        return cls({value: index + 2 for index, value in enumerate(values)})

    def encode(self, values) -> torch.Tensor:
        # Output a padded bag. 0 means no supplied legal form.
        forms = [split_legal_forms(value) for value in values]
        width = max(1, max(map(len, forms), default=0))
        output = np.zeros((len(forms), width), dtype=np.int64)
        for row, values_for_row in enumerate(forms):
            if values_for_row:
                output[row, :len(values_for_row)] = [self.vocab.get(value, self.UNK) for value in values_for_row]
        return torch.from_numpy(output)

    def save(self, path):
        Path(path).write_text(json.dumps(self.vocab, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))


class CharCNN(nn.Module):
    def __init__(self, vocab_size: int, char_dim: int = 48, channels: int = 64, output_dim: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, char_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(char_dim, channels, kernel_size=kernel, padding=kernel // 2)
            for kernel in (3, 5, 7)
        ])
        self.projection = nn.Linear(channels * 3, output_dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        present = token_ids.ne(0)
        x = self.embedding(token_ids).transpose(1, 2)
        pooled = []
        for conv in self.convs:
            value = F.relu(conv(x)).masked_fill(~present.unsqueeze(1), -1e4)
            value = value.max(dim=-1).values
            pooled.append(torch.where(present.any(1, keepdim=True), value, torch.zeros_like(value)))
        return self.projection(torch.cat(pooled, dim=1))


class EntityEncoder(nn.Module):
    """The one shared encoder used by both Source 1 and Source 2/3 towers."""
    def __init__(self, char_vocab_size: int, legal_vocab_size: int,
                 char_dim: int = 48, channels: int = 64, field_dim: int = 128,
                 legal_dim: int = 16, output_dim: int = 128, dropout: float = 0.15):
        super().__init__()
        self.name_cnn = CharCNN(char_vocab_size, char_dim, channels, field_dim)
        self.address_cnn = CharCNN(char_vocab_size, char_dim, channels, field_dim)
        self.legal_embedding = nn.Embedding(legal_vocab_size, legal_dim, padding_idx=0)
        self.mlp = nn.Sequential(
            nn.Linear(field_dim * 4 + legal_dim, 256),
            nn.LayerNorm(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, output_dim),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        legal_ids = batch["legal_form"]
        legal_vectors = self.legal_embedding(legal_ids)
        legal_mask = legal_ids.ne(0).unsqueeze(-1)
        legal = (legal_vectors * legal_mask).sum(1) / legal_mask.sum(1).clamp_min(1)
        features = torch.cat([
            self.name_cnn(batch["name_norm"]),
            self.name_cnn(batch["name_ascii"]),
            self.address_cnn(batch["address_norm"]),
            self.address_cnn(batch["country_norm"]),
            legal,
        ], dim=1)
        return F.normalize(self.mlp(features), p=2, dim=1)


def multi_positive_infonce_loss(query_embeddings, candidate_embeddings,
                                query_ids, candidate_owner_ids, temperature: float = 0.07):
    """Every candidate with the same owning S1 ID is a positive for that query."""
    logits = (query_embeddings @ candidate_embeddings.T) / temperature
    positive = query_ids[:, None].eq(candidate_owner_ids[None, :])
    if not positive.any(dim=1).all():
        raise ValueError("Every query needs at least one positive candidate in its batch.")
    positive_score = torch.logsumexp(logits.masked_fill(~positive, -torch.inf), dim=1)
    all_score = torch.logsumexp(logits, dim=1)
    return (all_score - positive_score).mean()


def cosine_triplet_loss(query_embedding, positive_embedding, negative_embedding, margin: float = 0.2):
    """Embeddings are normalized, so dot product is cosine similarity."""
    positive_similarity = (query_embedding * positive_embedding).sum(1)
    negative_similarity = (query_embedding * negative_embedding).sum(1)
    return F.relu(margin + negative_similarity - positive_similarity).mean()


def records_to_batch(records: pd.DataFrame, tokenizer: CharTokenizer,
                     legal_vocab: LegalFormVocab, device, name_length: int = 112,
                     address_length: int = 224, country_length: int = 32):
    return {
        "name_norm": tokenizer.encode(records["name_norm"], name_length).to(device),
        "name_ascii": tokenizer.encode(records["name_ascii"], name_length).to(device),
        "address_norm": tokenizer.encode(records["address_norm"], address_length).to(device),
        "country_norm": tokenizer.encode(records["country_norm"], country_length).to(device),
        "legal_form": legal_vocab.encode(records["legal_form"]).to(device),
    }


class PositivePairDataset(Dataset):
    """Positive pairs plus one locally supplied label=0 negative per query when available."""
    def __init__(self, positives: pd.DataFrame, records: pd.DataFrame, negatives: pd.DataFrame):
        self.left = records.loc[positives.source1_entity_id].reset_index(drop=True)
        self.positive = records.loc[positives.target_entity_id].reset_index(drop=True)
        negative_lookup = defaultdict(list)
        for row in negatives.itertuples(index=False):
            negative_lookup[row.source1_entity_id].append(row.target_entity_id)
        self.negative_ids = [values[0] if values else None for values in
                             (negative_lookup[q] for q in positives.source1_entity_id)]
        self.records = records

    def __len__(self):
        return len(self.left)

    def __getitem__(self, index):
        negative = self.records.loc[self.negative_ids[index]] if self.negative_ids[index] else None
        return self.left.iloc[index], self.positive.iloc[index], negative


def collate_pairs(items):
    left, positive, negative = zip(*items)
    negative_indices = [index for index, row in enumerate(negative) if row is not None]
    return (pd.DataFrame(left), pd.DataFrame(positive),
            pd.DataFrame([row for row in negative if row is not None]), negative_indices)


def load_needed_records(path, wanted_ids: set[str]) -> pd.DataFrame:
    """Reads only IDs referenced by labels, so source TSVs need not fit in RAM."""
    selected = []
    for chunk in pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, chunksize=100_000):
        selected.append(chunk[chunk.entity_id.isin(wanted_ids)])
    return pd.concat(selected, ignore_index=True) if selected else pd.DataFrame()


def load_labeled_training_data(source1_path, source2_path, source3_path, labels_path,
                               max_label_rows: int | None = None):
    """Load labelled rows. Set max_label_rows for a quick small experiment."""
    labels = pd.read_csv(labels_path, sep="\t", dtype=str, keep_default_na=False,
                         nrows=max_label_rows)
    expected = {"source1_entity_id", "target_entity_id", "label"}
    if not expected.issubset(labels.columns):
        raise ValueError(f"Labels must contain: {expected}")
    labels.label = labels.label.astype(float).astype(int)
    query_ids = set(labels.source1_entity_id)
    target_ids = set(labels.target_entity_id)
    s1 = load_needed_records(source1_path, query_ids)
    s2 = load_needed_records(source2_path, target_ids)
    s3 = load_needed_records(source3_path, target_ids)
    records = pd.concat([s1, s2, s3], ignore_index=True).drop_duplicates("entity_id").set_index("entity_id", drop=False)
    labels = labels[labels.source1_entity_id.isin(records.index) & labels.target_entity_id.isin(records.index)].copy()
    return records, labels


@torch.inference_mode()
def generate_embeddings(model, tokenizer, legal_vocab, records: pd.DataFrame, device, batch_size: int = 512):
    output = []
    model.eval()
    for start in range(0, len(records), batch_size):
        batch = records_to_batch(records.iloc[start:start + batch_size], tokenizer, legal_vocab, device)
        output.append(model(batch).cpu())
    return torch.cat(output).numpy() if output else np.empty((0, 128), dtype=np.float32)
