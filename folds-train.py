from pathlib import Path
from tqdm import tqdm

import torch
import pandas as pd
from sklearn.model_selection import train_test_split

from common import read_yaml, init_log
from dataset import EmbeddingCaptionDataset, collate_fn
from models import AudioCaptioner
from models.utils import model_size
from trainer import Trainer
from utils import load_tokenizer
from utils import get_captions

from typing import Dict, List


def city_splits(filenames: List[Path]) -> Dict[str, List[Path]]:
    split = {}

    for fn in filenames:
        _, city, *_ = fn.stem.split('-')
        if city not in split:
            split[city] = [fn]
        else:
            split[city].append(fn)

    return split


def combine_captions(captions: List[Dict[str, List[str]]]) -> Dict[str, List[str]]:
    combined = {}

    for capt_dict in captions:
        for fn in capt_dict:
            if fn not in combined:
                combined[fn] = capt_dict[fn]
            else:
                combined[fn].extend(capt_dict[fn])

    return combined


def main(config_fn='settings.yaml'):
    cfg = read_yaml(config_fn)
    logger = init_log('train')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer_dir = cfg.get('tokenizer_dir', 'tokenizer')
    tokenizer = load_tokenizer(tokenizer_dir)

    vocab_size = tokenizer.get_vocab_size()

    d_model = cfg.get('d_model')
    n_enc_layers = cfg.get('n_enc_layers', 2)
    n_enc_heads = cfg.get('n_enc_heads', 8)
    n_dec_layers = cfg.get('n_dec_layers', 2)
    n_dec_heads = cfg.get('n_dec_heads', 8)
    dropout = cfg.get('dropout', 0.2)
    p_mask_embedding = cfg.get('p_mask_embedding', 0.5)
    p_mask_tokens = cfg.get('p_mask_tokens', 0.5)

    batch_size = cfg.get('batch_size', 64)
    num_workers = cfg.get('num_dataloader_workers', 0)

    learning_rate = cfg.get('learning_rate', 1e-4)
    weight_decay = cfg.get('weight_decay', 1e-2)
    warmup_epochs = cfg.get('warmup_epochs', 5)
    n_epochs = cfg.get('n_epochs', 100)
    log_interval = cfg.get('log_interval')
    max_patience = cfg.get('patience')

    validation_size = cfg.get('validation_size', 0.1)

    embedding_pooling_factor = cfg.get('embedding_pooling_factor')
    single_token = cfg.get('single_start_token', False)

    ckpt_dir = Path(cfg.get('checkpoint_dir', 'checkpoints'))
    save_every = cfg.get('save_every')

    models_dir = Path(cfg.get('models_dir', 'folds-models'))
    models_dir.mkdir(exist_ok=True, parents=True)

    captions = []
    fn_keys = set()
    n_tot = 0
    caption_files = cfg.get('caption_files', [])
    for caption_fn in caption_files:
        capts = get_captions(caption_fn)
        captions.append(capts)
        n_capt = sum(map(lambda x: len(x), capts.values()))
        #logger.info(f'read {n_capt} captions for {len(capts)} files from {caption_fn}')
        fn_keys.update(capts.keys())
        n_tot += n_capt

    if single_token:
        # for baseline:
        captions = combine_captions(captions)

        # check
        n_files = len(captions)
        n_captions = sum(map(len, captions.values()))
        logger.info(f'using single start token for {n_files} files, {n_captions} captions')
        captions = [captions]
    else:
        logger.info(f'using {len(caption_files)} start tokens for {len(fn_keys)} files, {n_tot} captions')

    embedding_fns = sorted(Path(cfg.get('embedding_dir', 'embeddings')).glob('*.pt'))
    embedding_fns = [fn for fn in embedding_fns if fn.stem in fn_keys]

    predictions_dir = Path(cfg.get('predictions_dir', 'predictions'))
    predictions_dir.mkdir(exist_ok=True, parents=True)
    dataset_names = [Path(fn).stem for fn in caption_files]

    # split embedding filenames according to the city
    folds = city_splits(embedding_fns)

    # the actual testing filenames:
    folds_from = cfg.get('folds_from')
    splitting_keys = get_captions(folds_from).keys()
    test_keys = {city: [fn for fn in splitting_keys if fn.split('-')[1] == city] for city in folds.keys()}

    for city in folds:
        test_fns = folds[city]
        train_fns = [fn for fn in embedding_fns if fn not in test_fns]
        logger.info(f'starting fold {city}')

        # divide the files not in test split for training and validation
        train_fns, val_fns = train_test_split(train_fns, test_size=validation_size, random_state=303)

        train_ds = EmbeddingCaptionDataset(
            embedding_filenames=train_fns,
            captions=captions,
            tokenizer=tokenizer,
            embedding_pooling_factor=embedding_pooling_factor)

        val_ds = EmbeddingCaptionDataset(
            embedding_filenames=val_fns,
            captions=captions,
            tokenizer=tokenizer,
            embedding_pooling_factor=embedding_pooling_factor)

        # read the embedding dim from the dataset output
        x, _ = next(iter(train_ds))
        d_embedding = x.size(-1)

        train_loader = torch.utils.data.DataLoader(
            dataset=train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn)

        val_loader = torch.utils.data.DataLoader(
            dataset=val_ds,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn)

        model = AudioCaptioner(
            d_embedding=d_embedding,
            vocab_size=vocab_size,
            d_model=d_model,
            n_enc_layers=n_enc_layers,
            n_enc_heads=n_enc_heads,
            n_dec_layers=n_dec_layers,
            n_dec_heads=n_dec_heads,
            dropout=dropout,
            p_mask_embedding=p_mask_embedding,
            p_mask_tokens=p_mask_tokens)

        logger.info(f'model size {model_size(model)/1e6:.1f}M')

        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        model_path = models_dir / f'model-{city}.pt'
        trainer = Trainer(
            device=device,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            warmup_epochs=warmup_epochs,
            checkpoint_dir=ckpt_dir,
            save_every=save_every,
            patience=max_patience,
            best_model_path=model_path,
            log_interval=log_interval,
            logger=logger)

        logger.info(f'start training for {n_epochs} epochs')
        trainer.train(n_epochs)

        # load the best model
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        # generate captions and move on
        predictions = {ds: [] for ds in dataset_names}
        test_fns = [fn for fn in test_fns if fn.stem in test_keys[city]]
        for fn in tqdm(sorted(test_fns)):
            x = torch.load(fn, map_location='cpu')
            if embedding_pooling_factor is not None:
                x = torch.nn.functional.avg_pool1d(x.t(), embedding_pooling_factor).t()

            for k, ds in enumerate(dataset_names):
                if single_token:
                    start_token = tokenizer.token_to_id('<s_0>')
                else:
                    start_token = tokenizer.token_to_id(f'<s_{k}>')
                end_token = tokenizer.token_to_id('</s>')
                tokens = model.generate(x.to(device), start_token=start_token, end_token=end_token)
                tokens = tokens.squeeze()[1:-1]
                caption = tokenizer.decode(tokens.cpu().numpy())
                predictions[ds].append((fn.stem, caption))

        for ds in predictions:
            df = pd.DataFrame(predictions[ds], columns=['filename', 'caption']).set_index('filename')
            df.to_csv(predictions_dir / f'predictions-{city}-{ds}.csv')


if __name__ == '__main__':
    main()
