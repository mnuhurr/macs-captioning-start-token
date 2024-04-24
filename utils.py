from pathlib import Path
import csv
import math
import string

import torch
import torch.nn.functional as F

from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from tokenizers.normalizers import BertNormalizer

from typing import Dict, List, Tuple, Union


def save_captions(filename: Union[str, Path], rows: List[Tuple[str, str]]):
    with Path(filename).open('wt') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'caption'])
        writer.writerows(rows)


def get_captions(filename: Union[str, Path]) -> Dict[str, List[str]]:
    captions = {}
    with Path(filename).open('rt') as f:
        reader = csv.reader(f)
        next(reader)

        for row in reader:
            fn = Path(row[0]).stem
            captions[fn] = list(filter(lambda s: len(s) > 0, row[1:]))

    return captions


def clean_caption(caption: str) -> str:
    #translator = str.maketrans("", "", string.punctuation)
    translator = str.maketrans(string.punctuation, " " * len(string.punctuation))

    capt = caption.lower().translate(translator)

    # convert several spaces into single
    if capt.find('  ') >= 0:
        capt = ' '.join(capt.split())

    return capt


def clean_references(captions: Dict[str, List[str]]):
    for fn in captions:
        captions[fn] = list(map(clean_caption, captions[fn]))


def clean_predictions(captions: Dict[str, str]):
    for fn in captions:
        captions[fn] = clean_caption(captions[fn])


def make_splits(file_ids: List[str]) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    city_splits = {}
    scene_splits = {}

    for fn in file_ids:
        scene, city, *_ = fn.split('-')
        if scene not in scene_splits:
            scene_splits[scene] = [fn]
        else:
            scene_splits[scene].append(fn)

        if city not in city_splits:
            city_splits[city] = [fn]
        else:
            city_splits[city].append(fn)

    return city_splits, scene_splits


def load_tokenizer(tokenizer_path, max_length=None):

    vocab_fn = str(Path(tokenizer_path) / 'tokenizer-vocab.json')
    merges_fn = str(Path(tokenizer_path) / 'tokenizer-merges.txt')

    tokenizer = ByteLevelBPETokenizer(vocab_fn, merges_fn)
    tokenizer._tokenizer.post_processor = BertProcessing(
        ('</s>', tokenizer.token_to_id('</s>')),
        ('<s_0>', tokenizer.token_to_id('<s_0>')),
    )

    tokenizer._tokenizer.normalizer = BertNormalizer(strip_accents=False, lowercase=True)

    if max_length is not None:
        tokenizer.enable_truncation(max_length=max_length)

    return tokenizer


def step_lr(step, warmup_steps=4000):
    # learning rate from the original attention paper modified in such a way that
    # this function peaks at 1, tune learning rate with optimizer
    arg1 = torch.tensor(1 / math.sqrt(step)) if step > 0 else torch.tensor(float('inf'))
    arg2 = torch.tensor(step * warmup_steps**-1.5)

    return math.sqrt(warmup_steps) * torch.minimum(arg1, arg2)


def masked_loss(y_pred, y_true, mask):
    loss = F.cross_entropy(y_pred.permute(0, 2, 1), y_true, reduction='none')
    mask = (mask == 0).to(loss.dtype)
    return torch.sum(loss * mask) / torch.sum(mask).to(loss.dtype)


def mask_tokens(tokens, mask_token_id, p, n_special_tokens=7):
    r = torch.rand(size=tokens.shape, device=tokens.device)

    ind = (r < p) & (tokens >= n_special_tokens)

    masked = tokens.clone()
    masked[ind] = mask_token_id
    return masked
