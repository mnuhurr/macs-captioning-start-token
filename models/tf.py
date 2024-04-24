import math
import torch

from typing import Optional


@torch.jit.script
def mask_tokens(x: torch.Tensor, mask_token: torch.Tensor, p: float) -> torch.Tensor:
    batch_size, seq_len, _ = x.shape

    mask_pos = torch.rand(batch_size, seq_len, device=x.device) < p
    x[mask_pos] = mask_token.to(x.dtype)

    return x


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_sequence_length: int = 1024):
        super().__init__()

        position = torch.arange(max_sequence_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_sequence_length, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x


class EmbeddingEncoder(torch.nn.Module):
    def __init__(self,
                 d_embedding: int,
                 d_model: int,
                 n_encoder_layers: int,
                 n_encoder_heads: int,
                 dropout: float = 0.1,
                 p_embedding_mask: float = 0.0,
                 max_tokens: int = 1024):
        super().__init__()

        self.embedding_projection = torch.nn.Sequential(
            torch.nn.LayerNorm(d_embedding),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(d_embedding, d_model))

        self.positional_encoding = PositionalEncoding(d_model, max_tokens)

        d_ff = 4 * d_model
        enc_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            dim_feedforward=d_ff,
            nhead=n_encoder_heads,
            dropout=dropout,
            batch_first=True,
            norm_first=True)

        self.embedding_encoder = torch.nn.TransformerEncoder(
            enc_layer,
            num_layers=n_encoder_layers,
            enable_nested_tensor=False)

        self.p_mask = p_embedding_mask
        self.register_parameter('mask_token', torch.nn.Parameter(1 / math.sqrt(d_model) * torch.randn(d_model)))

    def forward(self, x: torch.Tensor, x_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.embedding_projection(x)

        if self.training and self.p_mask > 0:
            x = mask_tokens(x, mask_token=self.mask_token, p=self.p_mask)

        x = self.positional_encoding(x)
        x = self.embedding_encoder(x, src_key_padding_mask=x_mask)
        return x


class TokenDecoder(torch.nn.Module):
    def __init__(self,
                 vocab_size: int,
                 d_model: int,
                 n_decoder_layers: int,
                 n_decoder_heads: int,
                 dropout: float = 0.1,
                 p_token_mask: float = 0.0,
                 max_tokens: int = 1024):
        super().__init__()

        self.token_embedding = torch.nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_tokens)

        d_ff = 4 * d_model
        dec_layer = torch.nn.TransformerDecoderLayer(
            d_model=d_model,
            dim_feedforward=d_ff,
            nhead=n_decoder_heads,
            dropout=dropout,
            batch_first=True,
            norm_first=True)

        self.token_decoder = torch.nn.TransformerDecoder(dec_layer, num_layers=n_decoder_layers)

        self.ln = torch.nn.LayerNorm(d_model)

        self.p_mask = p_token_mask
        self.register_parameter('mask_token', torch.nn.Parameter(1 / math.sqrt(d_model) * torch.randn(d_model)))

        # causal mask
        mask = torch.empty(max_tokens, max_tokens).fill_(-float('inf')).triu(1)
        self.register_buffer('mask', mask.to(torch.bool), persistent=False)

    def forward(self, x: torch.Tensor, x_enc: torch.Tensor,
                x_mask: Optional[torch.Tensor] = None, x_enc_mask: Optional[torch.Tensor] = None) -> torch.Tensor:

        seq_len = x.size(-1)
        x = self.token_embedding(x)
        if self.training and self.p_mask > 0:
            x = mask_tokens(x, self.mask_token, p=self.p_mask)

        x = self.positional_encoding(x)
        x = self.token_decoder(tgt=x,
                               memory=x_enc,
                               tgt_mask=self.mask[:seq_len, :seq_len],
                               tgt_key_padding_mask=x_mask,
                               tgt_is_causal=True,
                               memory_key_padding_mask=x_enc_mask)
        x = self.ln(x)
        x = x @ self.token_embedding.weight.T
        return x


class AudioCaptioner(torch.nn.Module):
    def __init__(self,
                 d_embedding: int,
                 vocab_size: int,
                 d_model: int,
                 n_enc_layers: int,
                 n_enc_heads: int,
                 n_dec_layers: int,
                 n_dec_heads: int,
                 dropout: float = 0.1,
                 p_mask_embedding: float = 0.0,
                 p_mask_tokens: float = 0.0,
                 max_tokens: int = 1024):
        super().__init__()

        self.encoder = EmbeddingEncoder(
            d_embedding=d_embedding,
            d_model=d_model,
            n_encoder_layers=n_enc_layers,
            n_encoder_heads=n_enc_heads,
            p_embedding_mask=p_mask_embedding,
            dropout=dropout)

        self.decoder = TokenDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            n_decoder_layers=n_dec_layers,
            n_decoder_heads=n_dec_heads,
            p_token_mask=p_mask_tokens,
            dropout=dropout)

    def forward(self,
                embeddings: torch.Tensor,
                tokens: torch.Tensor,
                embedding_mask: Optional[torch.Tensor] = None,
                token_mask: Optional[torch.Tensor] = None) -> torch.Tensor:

        enc_out = self.encoder(x=embeddings, x_mask=embedding_mask)
        logits = self.decoder(x=tokens, x_enc=enc_out, x_mask=token_mask, x_enc_mask=embedding_mask)

        return logits

    @torch.inference_mode()
    def generate(self, embeddings: torch.Tensor, start_token: int = 0, end_token: int = 1, max_tokens: int = 128) -> torch.Tensor:
        """ greedy search """
        dev = embeddings.device
        if embeddings.dim() == 2:
            embeddings = embeddings.unsqueeze(0)

        enc_out = self.encoder(embeddings)

        new_token = -1
        tokens = torch.tensor([start_token], dtype=torch.int64).to(dev)
        end_token = torch.tensor(end_token, dtype=torch.int64).to(dev)
        k = 1

        while new_token != end_token:
            if k < max_tokens:
                logits = self.decoder(x=tokens.unsqueeze(0), x_enc=enc_out)
                new_token = torch.argmax(logits[0, -1, :])
            else:
                new_token = end_token

            tokens = torch.cat([tokens, new_token.unsqueeze(0)])
            k = tokens.size(0)

        return tokens
