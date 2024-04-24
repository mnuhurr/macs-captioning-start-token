from pathlib import Path
import csv
import pandas as pd

from common import read_yaml, init_log
from evaluation import evaluate_dicts
from utils import get_captions


def get_predictions(fn, max_length=None):
    captions = {}
    with Path(fn).open('rt') as f:
        reader = csv.DictReader(f)
        for row in reader:
            capt = row['caption']

            if max_length is not None:
                capt = capt[:max_length]

            captions[row['filename']] = capt

    return captions


def main(config_fn='settings.yaml'):
    cfg = read_yaml(config_fn)

    # collect references
    ref = {}
    for fn in cfg.get('caption_files', []):
        fn = Path(fn)
        ref[fn.stem] = get_captions(fn)

    predictions_dir = Path(cfg.get('predictions_dir', 'predictions'))
    results_dir = Path(cfg.get('results_dir', 'results'))
    results_dir.mkdir(exist_ok=True, parents=True)

    # collect predictions
    pred = {}
    for fn in predictions_dir.glob('predictions-*.csv'):
        _, city, *_ = fn.stem.split('-')
        ds = fn.stem[13 + len(city):]

        if ds not in ref:
            print(f'warning: reference data for {ds} not found')
            continue

        if ds not in pred:
            pred[ds] = {}

        #capts = get_captions(fn)
        #capts = {key: captlist[0] for key, captlist in capts.items()}
        capts = get_predictions(fn, max_length=128)

        pred[ds].update(capts)

    # evaluate
    metric_ids = ['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'METEOR', 'ROUGE_L', 'CIDEr', 'SPICE', 'SPIDEr']
    results = []

    for pred_ds in sorted(pred):
        for ref_ds in sorted(ref):
            metrics, per_file_metrics = evaluate_dicts(pred[pred_ds], ref[ref_ds])

            row = [pred_ds, ref_ds] + [metrics[m] for m in metric_ids]
            results.append(row)

    df = pd.DataFrame(results, columns=['predictions', 'references'] + metric_ids)
    df.to_csv(results_dir / 'pairwise.csv', index=False)



if __name__ == '__main__':
    main()
