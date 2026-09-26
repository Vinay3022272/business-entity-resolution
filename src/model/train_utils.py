"""Simple training helpers for the shared two-tower model."""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from tqdm.auto import tqdm

from .two_tower import (CharTokenizer, EntityEncoder, LegalFormVocab, PositivePairDataset,
                        collate_pairs, cosine_triplet_loss, multi_positive_infonce_loss,
                        records_to_batch)


def split_by_source1(positive_labels, validation_fraction: float = 0.2, seed: int = 42):
    ids = np.array(sorted(positive_labels.source1_entity_id.unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    validation_ids = set(ids[:max(1, int(len(ids) * validation_fraction))])
    return (positive_labels[~positive_labels.source1_entity_id.isin(validation_ids)].copy(),
            positive_labels[positive_labels.source1_entity_id.isin(validation_ids)].copy())


def train_model(records, labels, output_dir, epochs: int = 5, batch_size: int = 128,
                learning_rate: float = 1e-3, device=None):
    """Train from label=1 pairs; label=0 records provide same-query triplet negatives."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    positives = labels[labels.label == 1].copy()
    negatives = labels[labels.label == 0].copy()
    if positives.empty:
        raise ValueError("At least one label=1 pair is required.")
    train_pos, validation_pos = split_by_source1(positives)
    train_qids = set(train_pos.source1_entity_id)
    train_neg = negatives[negatives.source1_entity_id.isin(train_qids)]
    validation_neg = negatives[~negatives.source1_entity_id.isin(train_qids)]
    train_record_ids = list(set(train_pos.source1_entity_id) | set(train_pos.target_entity_id))
    train_records = records.loc[train_record_ids]
    tokenizer, legal_vocab = CharTokenizer.fit(train_records), LegalFormVocab.fit(train_records)
    model = EntityEncoder(len(tokenizer.vocab) + 2, len(legal_vocab.vocab) + 2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    loaders = {
        "train": DataLoader(PositivePairDataset(train_pos, records, train_neg), batch_size=batch_size,
                            shuffle=True, collate_fn=collate_pairs),
        "validation": DataLoader(PositivePairDataset(validation_pos, records, validation_neg), batch_size=batch_size,
                                 shuffle=False, collate_fn=collate_pairs),
    }

    print(f"\n[INFO] Starting Training on Device: {device}")
    print(f"[INFO] Train Batches: {len(loaders['train']):,} | Validation Batches: {len(loaders['validation']):,}")
    print(f"[INFO] Total Epochs: {epochs} | Batch Size: {batch_size} | Learning Rate: {learning_rate}\n")

    history, best_loss = [], float("inf")
    best_state_dict = None
    best_epoch = 1

    for epoch in range(1, epochs + 1):
        values = {}
        for stage, loader in loaders.items():
            training = stage == "train"
            model.train(training)
            losses = []
            desc = f"Epoch [{epoch:02d}/{epochs:02d}] {stage.capitalize()}"
            pbar = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
            for left, positive, negative, negative_indices in pbar:
                if training:
                    optimizer.zero_grad(set_to_none=True)
                with torch.set_grad_enabled(training), torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    q = model(records_to_batch(left, tokenizer, legal_vocab, device))
                    p = model(records_to_batch(positive, tokenizer, legal_vocab, device))
                    owner = torch.as_tensor(left.entity_id.astype("category").cat.codes.to_numpy(copy=True), device=device)
                    # Candidate ownership is the corresponding S1 ID for every positive row.
                    candidate_owner = owner
                    loss = multi_positive_infonce_loss(q, p, owner, candidate_owner)
                    if not negative.empty:
                        n = model(records_to_batch(negative, tokenizer, legal_vocab, device))
                        selected = torch.as_tensor(negative_indices, device=device)
                        loss = loss + 0.25 * cosine_triplet_loss(q[selected], p[selected], n)
                if training:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer); scaler.update()
                batch_loss = float(loss.detach().cpu())
                losses.append(batch_loss)
                pbar.set_postfix({"batch_loss": f"{batch_loss:.4f}", "avg_loss": f"{np.mean(losses):.4f}"})
            values[f"{stage}_loss"] = float(np.mean(losses))
        values["epoch"] = epoch
        history.append(values)
        
        if values["validation_loss"] < best_loss:
            best_loss = values["validation_loss"]
            best_epoch = epoch
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_marker = "[BEST SO FAR]"
        else:
            best_marker = ""
            
        print(f"Epoch [{epoch:02d}/{epochs:02d}]  |  "
              f"Train Loss: {values['train_loss']:.4f}  |  "
              f"Val Loss: {values['validation_loss']:.4f}  {best_marker}")
        
    print(f"\n[INFO] Complete training finished. Saving final weights to: {output_dir}")
    if best_state_dict is not None:
        torch.save({"model": best_state_dict, "epoch": best_epoch, "best_loss": best_loss}, output_dir / "best.pt")
    torch.save({"model": model.state_dict(), "epoch": epochs, "history": history}, output_dir / "last.pt")
    tokenizer.save(output_dir / "tokenizer.json")
    legal_vocab.save(output_dir / "legal_form_vocab.json")
    (output_dir / "config.json").write_text(json.dumps({"output_dim": 128}, indent=2), encoding="utf-8")
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print("[INFO] Model weights and vocabularies saved successfully.")
    return model, tokenizer, legal_vocab, history
