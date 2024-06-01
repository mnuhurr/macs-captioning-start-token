from pathlib import Path

import torch
from sklearn.model_selection import train_test_split

from common import read_yaml, init_log
from dataset import EmbeddingCaptionDataset, collate_fn
from models import AudioCaptioner
from models.utils import model_size
from trainer import Trainer
from utils import load_tokenizer
from utils import get_captions


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

    ckpt_dir = Path(cfg.get('checkpoint_dir', 'checkpoints'))
    save_every = cfg.get('save_every')

    captions = []
    fn_keys = set()
    n_tot = 0
    for caption_fn in cfg.get('caption_files', []):
        capts = get_captions(caption_fn)
        captions.append(capts)
        n_capt = sum(map(lambda x: len(x), capts.values()))
        #logger.info(f'read {n_capt} captions for {len(capts)} files from {caption_fn}')
        fn_keys.update(capts.keys())
        n_tot += n_capt

    logger.info(f'found {n_tot} captions for {len(fn_keys)} files')

    embedding_fns = sorted(Path(cfg.get('embedding_dir', 'embeddings')).glob('*.pt'))
    embedding_fns = [fn for fn in embedding_fns if fn.stem in fn_keys]

    train_fns, val_fns = train_test_split(embedding_fns, test_size=validation_size, random_state=303)

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

    trainer = Trainer(
        device=device,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        warmup_epochs=warmup_epochs,
        checkpoint_dir=ckpt_dir,
        patience=max_patience,
        save_every=save_every,
        best_model_path=cfg.get('model_path'),
        log_interval=log_interval,
        logger=logger)

    logger.info(f'start training for {n_epochs} epochs')
    trainer.train(n_epochs)


if __name__ == '__main__':
    main()
