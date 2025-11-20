"""Simple training harness for the TransformerDisfluencyTagger.

This script is intentionally small and configurable. It expects the
CSV data in the repository format (word, tag, timing...) and uses the
existing TransformerDisfluencyTagger class to provide the model/tokenizer.

Usage (example):
  python -m deep_disfluency.tagger.train_transformer \
    --train-file data/disfluency_detection/switchboard/swbd_disf_train_1_data_timings.csv \
    --outdir experiments/021/epoch_test --epochs 3 --batch-size 16

Note: This script performs a small-scale fine-tune of the classifier
head only by default to keep runs quick. For full training, increase
epochs and consider using a GPU device.
"""
import argparse
import os
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
import torch.nn.functional as F

from deep_disfluency.tagger.transformer_tagger import TransformerDisfluencyTagger


class CsvDisfDataset(Dataset):
    """Very small CSV loader that groups tokens by sentence id.
    The CSV format is expected to have at least: sentence_id, word, tag
    If timings/others exist they are ignored by this trainer.
    """
    def __init__(self, path):
        self.examples = []
        cur_id = None
        words = []
        tags = []
        with open(path, 'r', encoding='utf-8') as fh:
            header = fh.readline().strip().split(',')
            # find columns
            try:
                sid_idx = header.index('sentence_id')
            except ValueError:
                sid_idx = 0
            try:
                word_idx = header.index('word')
            except ValueError:
                word_idx = 1
            try:
                tag_idx = header.index('tag')
            except ValueError:
                tag_idx = 2
            for line in fh:
                if not line.strip():
                    continue
                parts = line.strip().split(',')
                sid = parts[sid_idx]
                w = parts[word_idx]
                t = parts[tag_idx]
                if cur_id is None:
                    cur_id = sid
                if sid != cur_id:
                    self.examples.append((words, tags))
                    words = []
                    tags = []
                    cur_id = sid
                words.append(w)
                tags.append(t)
        if words:
            self.examples.append((words, tags))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate_batch(batch):
    # identity collate; trainer will handle tokenization
    return batch


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    model = TransformerDisfluencyTagger(device=device)
    model.model.train()

    dataset = CsvDisfDataset(args.train_file)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)

    # Optimizer only for classifier head
    optimizer = AdamW(model.model.parameters(), lr=args.lr)

    os.makedirs(args.outdir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        for batch in loader:
            # batch is list of (words, tags)
            optimizer.zero_grad()
            batch_losses = []
            for words, tags in batch:
                inputs = model.tokenizer(words, is_split_into_words=True, return_tensors='pt', padding=True, truncation=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model.model(**inputs)
                logits = outputs.logits  # (batch=1, seq_len, num_labels)
                # map word-level labels to token-level using word_ids
                word_ids = inputs['input_ids']  # placeholder; we will use tokenizer mapping
                # simple alignment using tokenizer.word_ids()
                word_id_map = model.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
                # fallback: compute cross-entropy over first token positions per word
                # This is a small-scale trainer — use simplistic alignment.
                # Create targets per token: expand label of first token of each word, ignore others using -100
                token_word_ids = model.tokenizer(words, is_split_into_words=True, return_offsets_mapping=False, add_special_tokens=True)
                # Use model.prepare_labels helper if exists, else fallback
                try:
                    target_ids = model._encode_tags(tags, token_word_ids, device)
                except Exception:
                    # fallback: map every token to the first tag
                    num_tokens = logits.size(1)
                    target_ids = torch.full((1, num_tokens), -100, dtype=torch.long, device=device)
                    # set inner tokens to first tag index
                    first_tag_idx = model.TAG_TO_ID.get(tags[0], 0)
                    target_ids[0, 1: min(num_tokens-1, 1+len(tags))] = first_tag_idx

                loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(logits.view(-1, logits.size(-1)), target_ids.view(-1))
                loss.backward()
                batch_losses.append(loss.item())
            optimizer.step()
            total_loss += sum(batch_losses)

        avg_loss = total_loss / max(len(loader), 1)
        print(f"Epoch {epoch}/{args.epochs} avg_loss={avg_loss:.4f}")
        # checkpoint
        ckpt_path = Path(args.outdir) / f"epoch_{epoch}.pt"
        model.save_model(str(ckpt_path))

    # write config
    with open(Path(args.outdir) / 'train_config.json', 'w') as fh:
        json.dump({'epochs': args.epochs, 'batch_size': args.batch_size, 'lr': args.lr}, fh)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', required=True)
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
