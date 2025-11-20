"""Prediction script for the TransformerDisfluencyTagger.

Loads a saved transformer checkpoint (or the base model), runs
inference on a CSV corpus (sentence grouped by sentence_id) and
writes incremental-style outputs to a target file.

Usage example:
  python -m deep_disfluency.tagger.predict_transformer \
    --model-checkpoint experiments/021/epoch_1.pt \
    --input data/disfluency_detection/switchboard/swbd_disf_heldout_partial_data_timings.csv \
    --output experiments/021/epoch_1/swbd_disf_heldout_partial_data_output_increco.tex
"""
import argparse
import os
from pathlib import Path
from deep_disfluency.tagger.transformer_tagger import TransformerDisfluencyTagger


def read_corpus_csv(path):
    examples = []
    cur_id = None
    words = []
    tags = []
    with open(path, 'r', encoding='utf-8') as fh:
        header = fh.readline().strip().split(',')
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
                examples.append((words, tags))
                words = []
                tags = []
                cur_id = sid
            words.append(w)
            tags.append(t)
    if words:
        examples.append((words, tags))
    return examples


def write_increco_output(out_path, utterance_id, words, pred_tags):
    # Very small increco-like writer: write per-word lines with tags
    with open(out_path, 'a', encoding='utf-8') as fh:
        fh.write(f"% Utterance: {utterance_id}\n")
        for i, (w, t) in enumerate(zip(words, pred_tags), start=1):
            fh.write(f"{utterance_id}:{i}\t{w}\t{t}\n")
        fh.write("\n")


def predict(args):
    device = 'cpu'
    if args.cuda:
        device = 'cuda'
    tagger = TransformerDisfluencyTagger(device=device)
    if args.model_checkpoint:
        tagger.load_model(args.model_checkpoint)

    data = read_corpus_csv(args.input)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    # clear output
    Path(args.output).write_text('')

    for uid, (words, gold) in enumerate(data):
        tagger.reset()
        pred = []
        for w in words:
            p = tagger.tag_new_word(w)
            # tag_new_word may return one or a list; coerce to string
            if isinstance(p, list):
                pred.append(p[-1])
            else:
                pred.append(p)
        write_increco_output(args.output, uid, words, pred)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-checkpoint', default=None)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--cuda', action='store_true')
    args = parser.parse_args()
    predict(args)


if __name__ == '__main__':
    main()
