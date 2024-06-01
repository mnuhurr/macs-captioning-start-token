from pathlib import Path
import torch

from tokenizers.implementations import ByteLevelBPETokenizer

from typing import Dict, List, Optional, Tuple


def collate_embeddings(embeddings: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size = len(embeddings)
    emb_dim = embeddings[0].size(1)

    maxlen = max(map(lambda x: x.size(0), embeddings))
    x = torch.zeros(batch_size, maxlen, emb_dim, dtype=embeddings[0].dtype)
    x_mask = torch.ones(batch_size, maxlen, dtype=bool)

    for k, item in enumerate(embeddings):
        x[k, :item.size(0)] = item
        x_mask[k, :item.size(0)] = False

    return x, x_mask


def collate_tokens(tokens: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size = len(tokens)
    maxlen = max(map(len, tokens))
    x = torch.zeros(batch_size, maxlen, dtype=tokens[0].dtype)
    x_mask = torch.ones(batch_size, maxlen, dtype=bool)

    for k, item in enumerate(tokens):
        x[k, :len(item)] = item
        x_mask[k, :len(item)] = False

    return x, x_mask


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    embs, tokens = zip(*batch)

    x, x_mask = collate_embeddings(embs)
    y, y_mask = collate_tokens(tokens)

    return x, x_mask, y, y_mask


class EmbeddingCaptionDataset(torch.utils.data.Dataset):
    def __init__(self,
                 embedding_filenames: List[str | Path],
                 captions: List[Dict[str, List[str]]],
                 tokenizer: ByteLevelBPETokenizer,
                 embedding_pooling_factor: Optional[int] = None):
        """
        embedding_filenames: list of paths to the serialized audio features
        captions: list of dicts containing captions. different dicts in the list denote different datasets,
                  and each dataset has it's own start token to make a distinction
        tokenizer: tokenizer to use converting the captions
        embedding_pooling_factor: downsampling factor for the time axis
        """
        self.embeddings = {}
        self.pairs = []
        self.embedding_pooling_factor = embedding_pooling_factor

        start_tokens = [tokenizer.token_to_id(f'<s_{k}>') for k in range(len(captions))]
        for fn in embedding_filenames:
            fn_key = Path(fn).stem

            self.embeddings[fn_key] = self._load_embedding(fn)

            for ds_ind, capts in enumerate(captions):
                if fn_key not in capts:
                    continue

                for caption in capts[fn_key]:
                    tokens = tokenizer.encode(caption).ids
                    tokens[0] = start_tokens[ds_ind]
                    tokens = torch.tensor(tokens, dtype=torch.int64)
                    self.pairs.append((fn_key, tokens))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, torch.Tensor]:
        fn_key, tokens = self.pairs[item]
        embeddings = self.embeddings[fn_key]
        return embeddings, tokens

    def _load_embedding(self, filename):
        x = torch.load(filename, map_location='cpu')
        # x: (t, 768) for beats
        if self.embedding_pooling_factor is not None:
            # pool over time so do the transposing trick
            x = torch.nn.functional.avg_pool1d(x.t(), self.embedding_pooling_factor).t()

        return x
