"""Set-attention latent VAE used by LLapDiff."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class _SetTransformer(nn.Module):
    """Stack of TransformerEncoderLayer blocks (batch_first=True)."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        num_layers: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [B, N, D], key_padding_mask: [B, N] where True = padded
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=key_padding_mask)
        return x


class LatentVAE(nn.Module):
    """
    Set-attention VAE.

    Expected inputs:
      - x_tok: [B, T, N, input_dim] where input_dim=2*C is [values*obs, obs]
      - entity_pad: [B, N] (bool), True for padded/non-existent entities

    Outputs:
      - x_hat: [B, T, N, output_dim]
      - mu: [B, T, C]
      - logvar: [B, T, C]
    """

    def __init__(
        self,
        seq_len: int,
        latent_dim: int,
        latent_channel: int,
        enc_layers: int = 2,
        enc_heads: int = 4,
        enc_ff: int = 256,
        dec_layers: int = 2,
        dec_heads: int = 4,
        dec_ff: int = 256,
        input_dim: int = 2,
        output_dim: int = 1,
        dropout: float = 0.1,
        num_entities: Optional[int] = None,
        entity_conditioned: bool = False,
    ) -> None:
        super().__init__()
        self.seq_len = int(seq_len)
        self.latent_channel = int(latent_channel)
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        if self.output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {self.output_dim}")
        if self.input_dim < 2 * self.output_dim:
            raise ValueError(
                f"input_dim={self.input_dim} is too small for output_dim={self.output_dim}; "
                f"expected at least {2 * self.output_dim} token channels."
            )
        self.entity_conditioned = bool(entity_conditioned)
        self.num_entities = None if num_entities is None else int(num_entities)
        if self.entity_conditioned and (self.num_entities is None or self.num_entities <= 0):
            raise ValueError("num_entities must be provided when entity_conditioned=True")

        self.in_proj = nn.Linear(self.input_dim, latent_dim)
        self.encoder = _SetTransformer(latent_dim, enc_heads, enc_ff, enc_layers, dropout=dropout)

        self.mu_head = nn.Linear(latent_dim, self.latent_channel)
        self.logvar_head = nn.Linear(latent_dim, self.latent_channel)

        self.z_proj = nn.Linear(self.latent_channel, latent_dim)
        if self.entity_conditioned:
            self.entity_emb = nn.Embedding(self.num_entities, latent_dim)
            nn.init.normal_(self.entity_emb.weight, std=0.02)
        else:
            self.entity_emb = None
        self.decoder = _SetTransformer(latent_dim, dec_heads, dec_ff, dec_layers, dropout=dropout)
        self.out_proj = nn.Linear(latent_dim, self.output_dim)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def _coerce_layout(self, x_tok: torch.Tensor, entity_pad: Optional[torch.Tensor]) -> torch.Tensor:
        """Ensure x_tok is [B, T, N, D]."""
        if x_tok.ndim != 4:
            raise ValueError(f"x_tok must be rank-4 [B,T,N,D], got shape={tuple(x_tok.shape)}")

        if entity_pad is not None and entity_pad.ndim == 2:
            entity_count = entity_pad.shape[1]
            if x_tok.shape[1] == entity_count and x_tok.shape[2] != entity_count:
                return x_tok.permute(0, 2, 1, 3).contiguous()

        if x_tok.shape[1] != self.seq_len and x_tok.shape[2] == self.seq_len:
            return x_tok.permute(0, 2, 1, 3).contiguous()

        return x_tok

    def _prepare_entity_pad(
        self,
        x_tok: torch.Tensor,
        entity_pad: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, int, int, int, int]:
        x_tok = self._coerce_layout(x_tok, entity_pad)
        B, T, N, D = x_tok.shape
        if D != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got D={D}")

        if entity_pad is None:
            entity_pad = torch.zeros(B, N, dtype=torch.bool, device=x_tok.device)
        else:
            entity_pad = entity_pad.to(dtype=torch.bool, device=x_tok.device)
        if entity_pad.shape != (B, N):
            raise ValueError(f"entity_pad must have shape [B,N]=({B},{N}), got {tuple(entity_pad.shape)}")
        if self.entity_conditioned and self.num_entities is not None and N > self.num_entities:
            raise ValueError(f"N={N} exceeds configured num_entities={self.num_entities}")
        return x_tok, entity_pad, B, T, N, D

    def encode_mu(
        self,
        x_tok: torch.Tensor,
        entity_pad: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_tok, entity_pad, B, T, N, D = self._prepare_entity_pad(x_tok, entity_pad)
        x_bt = x_tok.reshape(B * T, N, D)
        pad_bt = entity_pad.unsqueeze(1).expand(B, T, N).reshape(B * T, N)

        h = self.in_proj(x_bt)
        h = self.encoder(h, key_padding_mask=pad_bt)

        obs_channels = x_bt[..., self.output_dim : self.output_dim * 2].float()
        if obs_channels.numel() == 0:
            raise ValueError(
                f"x_tok input_dim={D} does not contain observation channels for output_dim={self.output_dim}"
            )
        obs = obs_channels.amax(dim=-1)
        w = obs.masked_fill(pad_bt, 0.0)
        valid_bt = w.sum(dim=1) > 0
        denom = w.sum(dim=1, keepdim=True).clamp(min=1.0)
        h_pool = (h * w.unsqueeze(-1)).sum(dim=1) / denom

        mu_bt = self.mu_head(h_pool)
        logvar_bt = self.logvar_head(h_pool).clamp(min=-10.0, max=10.0)
        mu_bt = torch.where(valid_bt.unsqueeze(-1), mu_bt, torch.zeros_like(mu_bt))
        logvar_bt = torch.where(valid_bt.unsqueeze(-1), logvar_bt, torch.zeros_like(logvar_bt))
        return mu_bt.reshape(B, T, self.latent_channel), logvar_bt.reshape(B, T, self.latent_channel)

    def decode_mu(self, mu: torch.Tensor, entity_pad: torch.Tensor) -> torch.Tensor:
        """Decode latent means into observation-space trajectories."""
        if mu.dim() != 3:
            raise ValueError(f"mu must be [B,T,C], got {tuple(mu.shape)}")
        B, T, Z = mu.shape
        if Z != self.latent_channel:
            raise ValueError(f"Expected latent_channel={self.latent_channel}, got {Z}")
        entity_pad = entity_pad.to(dtype=torch.bool, device=mu.device)
        if entity_pad.dim() != 2 or entity_pad.shape[0] != B:
            raise ValueError(f"entity_pad must be [B,N], got {tuple(entity_pad.shape)}")
        N = entity_pad.shape[1]
        z = mu.reshape(B * T, self.latent_channel)
        return self._decode_latent(z, entity_pad, B, T, N)

    def _decode_latent(
        self,
        z: torch.Tensor,
        entity_pad: torch.Tensor,
        B: int,
        T: int,
        N: int,
    ) -> torch.Tensor:
        pad_bt = entity_pad.unsqueeze(1).expand(B, T, N).reshape(B * T, N)
        dec = self.z_proj(z).unsqueeze(1).expand(-1, N, -1)
        if self.entity_emb is not None:
            ids = torch.arange(N, device=z.device)
            dec = dec + self.entity_emb(ids).unsqueeze(0).to(dtype=dec.dtype)
        dec = self.decoder(dec, key_padding_mask=pad_bt)
        x_hat_bt = self.out_proj(dec)
        return x_hat_bt.reshape(B, T, N, self.output_dim)

    def forward(self, x_tok: torch.Tensor, entity_pad: Optional[torch.Tensor] = None):
        x_tok, entity_pad, B, T, N, _ = self._prepare_entity_pad(x_tok, entity_pad)
        mu, logvar = self.encode_mu(x_tok, entity_pad)
        z = self.reparameterize(mu, logvar).reshape(B * T, self.latent_channel)
        x_hat = self._decode_latent(z, entity_pad, B, T, N)
        return x_hat, mu, logvar
