from pathlib import Path
from functools import reduce
import tokenizers
from tokenizers.normalizers import BertNormalizer

from common import read_yaml, init_log
from utils import get_captions


def main(config_fn='settings.yaml'):
    cfg = read_yaml(config_fn)
    logger = init_log('fit-tokenizer', level=cfg.get('log_level', 'info'))

    captions = []

    caption_files = cfg.get('caption_files', [])

    # use a different start token for each data file
    n_start_tokens = len(caption_files)

    for fn in caption_files:
        captions.extend(reduce(lambda x, y: x + y, get_captions(fn).values(), []))

    logger.info(f'using {len(captions)} captions in total')

    vocab_size = cfg.get('vocab_size', 10000)
    min_frequency = cfg.get('min_frequency', 2)
    logger.info(f'vocab_size={vocab_size}, min_frequency={min_frequency}, n_start_tokens={n_start_tokens}')

    start_tokens = [f'<s_{k}>' for k in range(n_start_tokens)]
    special_tokens = start_tokens + ['</s>', '<pad>', '<unk>', '<mask>']

    tokenizer = tokenizers.ByteLevelBPETokenizer()
    tokenizer._tokenizer.normalizer = BertNormalizer(strip_accents=False, lowercase=True)
    tokenizer.train_from_iterator(
        captions,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens)

    tokenizer_dir = Path(cfg.get('tokenizer_dir', '.'))
    tokenizer_dir.mkdir(exist_ok=True, parents=True)
    tokenizer.save_model(str(tokenizer_dir), 'tokenizer')


if __name__ == '__main__':
    main()
