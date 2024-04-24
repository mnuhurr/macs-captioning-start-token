
from pathlib import Path
import time
import logging

import torch
import pandas as pd

from models import AudioCaptioner
from utils import masked_loss, step_lr

from typing import Optional


class Trainer:
    def __init__(self,
                 model: AudioCaptioner,
                 optimizer: torch.optim.Optimizer,
                 train_loader: torch.utils.data.DataLoader,
                 val_loader: torch.utils.data.DataLoader,
                 device: Optional[torch.device] = None,
                 log_interval: Optional[int] = None,
                 warmup_epochs: int = 5,
                 patience: Optional[int] = None,
                 best_model_path: Optional[str | Path] = None,
                 final_model_path: Optional[str | Path] = None,
                 checkpoint_dir: Optional[str | Path] = None,
                 save_every: Optional[int] = None,
                 logger: Optional[logging.Logger] = None):

        self.device = device
        self.model = model.to(device)
        self.optimizer = optimizer

        self.train_loader = train_loader
        self.val_loader = val_loader

        warmup_steps = warmup_epochs * len(self.train_loader)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                           lambda step: step_lr(step, warmup_steps=warmup_steps))

        self.log_interval = log_interval
        self.logger = logger

        self.scaler = torch.cuda.amp.GradScaler()

        self.history = []

        self.max_patience = patience

        self.best_model_path = Path(best_model_path) if best_model_path is not None else None
        self.final_model_path = Path(final_model_path) if final_model_path is not None else None

        self.checkpoint_dir = checkpoint_dir
        self.save_every = save_every

    def _log(self, msg: str):
        if self.logger is not None:
            self.logger.info(msg)

    def _train_batch(self, x: torch.Tensor, x_mask: torch.Tensor, y: torch.Tensor, y_mask: torch.Tensor) -> float:
        x = x.to(self.device)
        x_mask = x_mask.to(self.device)
        y = y.to(self.device)
        y_mask = y_mask.to(self.device)

        inp = y[:, :-1]
        tar = y[:, 1:]
        y_mask = y_mask[:, :-1]

        with torch.cuda.amp.autocast():
            y_pred = self.model(embeddings=x, tokens=inp, embedding_mask=x_mask, token_mask=y_mask)
            loss = masked_loss(y_pred, tar, mask=y_mask)

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()

        return loss.item()

    def _train_epoch(self) -> float:
        self.model.train()
        train_loss = 0.0
        batch_t0 = time.time()

        for batch, (x, x_mask, y, y_mask) in enumerate(self.train_loader):
            batch_loss = self._train_batch(x, x_mask, y, y_mask)
            train_loss += batch_loss

            if self.log_interval is not None and batch % self.log_interval == 0:
                t_batch = int(1000 * (time.time() - batch_t0) / self.log_interval)
                print(f'batch {batch:4d}/{len(self.train_loader)} - {t_batch} ms/batch - training loss {batch_loss:.4f}')
                batch_t0 = time.time()

        return train_loss / len(self.train_loader)

    @torch.inference_mode()
    def _val_epoch(self) -> float:
        self.model.eval()
        val_loss = 0.0
        for x, xm, y, ym in self.val_loader:
            x = x.to(self.device)
            xm = xm.to(self.device)
            y = y.to(self.device)
            ym = ym.to(self.device)

            inp = y[:, :-1]
            tar = y[:, 1:]
            ym = ym[:, :-1]

            y_pred = self.model(embeddings=x, tokens=inp, embedding_mask=xm, token_mask=ym)
            loss = masked_loss(y_pred, tar, mask=ym)
            val_loss += loss

        return val_loss / len(self.val_loader)

    def train(self, n_epochs: int):
        patience = self.max_patience
        best_loss = float('inf')

        for epoch in range(n_epochs):
            train_loss = self._train_epoch()
            val_loss = self._val_epoch()

            self.history.append((train_loss, val_loss))

            self._log(f'epoch {epoch + 1} - training loss {train_loss:.4f} - validation loss {val_loss:.4f}')

            if val_loss < best_loss:
                best_loss = val_loss
                patience = self.max_patience

                if self.best_model_path is not None:
                    self.best_model_path.parent.mkdir(exist_ok=True, parents=True)
                    torch.save(self.model.state_dict(), self.best_model_path)

            elif patience is not None:
                patience -= 1
                if patience <= 0:
                    self._log('results not improving, stopping...')
                    break

            if self.checkpoint_dir is not None and self.save_every is not None and (epoch + 1) % self.save_every == 0:
                self.save_ckpt(Path(self.checkpoint_dir) / f'ckpt-{epoch + 1:03d}.pt')

        if self.final_model_path is not None:
            self.final_model_path.parent.mkdir(exist_ok=True, parents=True)
            torch.save(self.model.state_dict(), self.final_model_path)

    def save_ckpt(self, ckpt_path: str | Path):
        ckpt_path = Path(ckpt_path)
        ckpt_path.parent.mkdir(exist_ok=True, parents=True)

        data = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'history': self.history,
        }
        torch.save(data, ckpt_path)

    def load_ckpt(self, ckpt_path: str | Path):
        data = torch.load(ckpt_path)
        self.model.load_state_dict(data['model'])
        self.optimizer.load_state_dict(data['optimizer'])
        self.scheduler.load_state_dict(data['scheduler'])
        self.history = data['history']

    def make_history_csv(self, filename: str | Path):
        df = pd.DataFrame(self.history, columns=['train_loss', 'val_loss'])
        df.to_csv(filename)
