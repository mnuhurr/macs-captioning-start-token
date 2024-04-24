
from pathlib import Path
from tempfile import TemporaryDirectory

from coco_caption.pycocotools.coco import COCO
from coco_caption.pycocoevalcap.eval import COCOEvalCap

# for cider
from coco_caption.pycocoevalcap.cider.cider import Cider
from coco_caption.pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

from common import write_json

from typing import Dict, List


def reformat_for_coco(predictions: Dict[str, str], references: Dict[str, List[str]]):
    pred = []
    ref = {
        'info': {'description': 'reference captions'},
        'licenses': [{'id': 1}, {'id': 2}, {'id': 3}],
        'type': 'captions',
        'audio samples': [],
        'annotations': [],
    }

    fn_ids = [fn for fn in predictions if fn in references]

    caption_id = 0
    for audio_id, fn in enumerate(fn_ids):
        pred.append({
            'audio_id': audio_id,
            'caption': predictions[fn],
        })

        ref['audio samples'].append({'id': audio_id})

        for caption in references[fn]:
            ref['annotations'].append({
                'audio_id': audio_id,
                'id': caption_id,
                'caption': caption,
            })
            caption_id += 1

    return pred, ref


def evaluate_files(pred_fn, ref_fn):
    coco = COCO(str(ref_fn))
    cocoRes = coco.loadRes(str(pred_fn))

    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['audio_id'] = cocoRes.getAudioIds()
    cocoEval.evaluate()
    metrics = dict(
        (m, s) for m, s in cocoEval.eval.items()
    )
    return metrics, cocoEval.audioToEval


def evaluate_dicts(predictions: Dict[str, str], references: Dict[str, List[str]]):
    pred, ref = reformat_for_coco(predictions, references)

    with TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        pred_fn = tmp_dir / 'pred.json'
        ref_fn = tmp_dir / 'ref.json'
        # pred_fn = 'pred.json'
        # ref_fn = 'ref.json'
        write_json(pred_fn, pred)
        write_json(ref_fn, ref)

        metrics, per_file_metrics = evaluate_files(pred_fn, ref_fn)

    return metrics, per_file_metrics


def evaluate_cider_files(pred_fn, ref_fn):
    coco = COCO(str(ref_fn))
    cocoRes = coco.loadRes(str(pred_fn))

    gts = {}
    res = {}
    for audio_id in coco.getAudioIds():
        gts[audio_id] = coco.audioToAnns[audio_id]
        res[audio_id] = cocoRes.audioToAnns[audio_id]
    tokenizer = PTBTokenizer()
    gts = tokenizer.tokenize(gts)
    res = tokenizer.tokenize(res)

    cider = Cider()
    score, scores = cider.compute_score(gts, res)
    return score, scores


def evaluate_cider_dicts(predictions: Dict[str, str], references: Dict[str, List[str]]):
    pred, ref = reformat_for_coco(predictions, references)

    with TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        pred_fn = tmp_dir / 'pred.json'
        ref_fn = tmp_dir / 'ref.json'
        # pred_fn = 'pred.json'
        # ref_fn = 'ref.json'
        write_json(pred_fn, pred)
        write_json(ref_fn, ref)

        metrics, per_file_metrics = evaluate_cider_files(pred_fn, ref_fn)

    return metrics, per_file_metrics
