import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from llapdiffusion.models.time_utils import relative_time_offsets


class TVHead(nn.Module):
    """Single-hidden-layer MLP that projects per-step features to a scalar signal.

    Used for lightweight proxy channels described in Appendix F.2 of the paper.
    """

    def __init__(self, feat_dim: int, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., feat_dim] -> [...]
        return self.net(x).squeeze(-1)


class Time2Vec(nn.Module):
    """Time2Vec encoding for scalar timestamps.

    We follow the common Time2Vec form:
        [w0 * t + b0, sin(w1 * t + b1), ..., sin(w_{d-1} * t + b_{d-1})]
    This matches the paper's use of a lightweight timestamp encoder (Appendix F.2).
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim < 1:
            raise ValueError(f"Time2Vec dim must be >= 1, got {dim}")
        self.dim = int(dim)
        self.w = nn.Parameter(torch.randn(self.dim))
        self.b = nn.Parameter(torch.zeros(self.dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Encode timestamps.

        Args:
            t: tensor with last dimension either absent or 1 (scalar timestamps), shape [...], [..., 1]

        Returns:
            Encoded timestamps with shape [..., dim]
        """
        if t.dim() > 0 and t.size(-1) == 1:
            t = t.squeeze(-1)
        # Broadcast: [..., dim]
        wt = t.unsqueeze(-1) * self.w + self.b
        if self.dim == 1:
            return wt
        out = torch.empty_like(wt)
        out[..., 0] = wt[..., 0]
        out[..., 1:] = torch.sin(wt[..., 1:])
        return out


class ContinuousRoPESelfAttention(nn.Module):
    """Multi-head self-attention with continuous-time RoPE applied to Q/K."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        *,
        dropout: float = 0.0,
        rope_base: float = 10000.0,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.head_dim = self.d_model // self.n_heads
        self.rope_dim = self.head_dim - (self.head_dim % 2)
        if self.rope_dim <= 0:
            raise ValueError(f"head_dim={self.head_dim} leaves no even RoPE dimensions")

        inv_freq = 1.0 / (
            float(rope_base) ** (torch.arange(0, self.rope_dim, 2, dtype=torch.float32) / self.rope_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.qkv = nn.Linear(self.d_model, 3 * self.d_model)
        self.out_proj = nn.Linear(self.d_model, self.d_model)
        self.attn_dropout = nn.Dropout(dropout)

    def _apply_rope_one(self, x: torch.Tensor, rel_t: torch.Tensor) -> torch.Tensor:
        # x: [B, heads, K, head_dim], rel_t: [B, K]
        x_rope = x[..., : self.rope_dim]
        x_pass = x[..., self.rope_dim :]

        angles = rel_t[:, None, :, None].to(dtype=x.dtype, device=x.device) * self.inv_freq.to(dtype=x.dtype)[
            None, None, None, :
        ]
        cos = torch.cos(angles)
        sin = torch.sin(angles)

        x_pair = x_rope.reshape(*x_rope.shape[:-1], self.rope_dim // 2, 2)
        x_even = x_pair[..., 0]
        x_odd = x_pair[..., 1]
        x_rot = torch.stack((x_even * cos - x_odd * sin, x_even * sin + x_odd * cos), dim=-1).flatten(-2)
        if x_pass.numel() == 0:
            return x_rot
        return torch.cat((x_rot, x_pass), dim=-1)

    def forward(self, x: torch.Tensor, rel_t: torch.Tensor) -> torch.Tensor:
        if rel_t.dim() == 3 and rel_t.size(-1) == 1:
            rel_t = rel_t.squeeze(-1)
        if rel_t.dim() != 2 or rel_t.shape != x.shape[:2]:
            raise ValueError(f"rel_t must be [B,K]={tuple(x.shape[:2])}, got {tuple(rel_t.shape)}")

        B, K, _ = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, K, 3, self.n_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = self._apply_rope_one(q, rel_t)
        k = self._apply_rope_one(k, rel_t)

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, K, self.d_model)
        return self.out_proj(out)


class ContinuousRoPEEncoderLayer(nn.Module):
    """Transformer encoder layer whose temporal self-attention uses continuous-time RoPE."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        *,
        dim_feedforward: int,
        dropout: float,
        rope_base: float = 10000.0,
    ) -> None:
        super().__init__()
        self.self_attn = ContinuousRoPESelfAttention(
            d_model,
            n_heads,
            dropout=dropout,
            rope_base=rope_base,
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, rel_t: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.dropout1(self.self_attn(x, rel_t)))
        ff = self.linear2(self.dropout(F.gelu(self.linear1(x))))
        return self.norm2(x + self.dropout2(ff))


class ContinuousRoPEEncoder(nn.Module):
    """Stack of continuous-time RoPE Transformer encoder layers."""

    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int,
        dim_feedforward: int,
        num_layers: int,
        dropout: float,
        rope_base: float = 10000.0,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                ContinuousRoPEEncoderLayer(
                    d_model,
                    n_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    rope_base=rope_base,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor, rel_t: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, rel_t)
        return x


class LaplaceAE(nn.Module):
    """History summarizer (LaplaceAE) used to condition the diffusion model.

    Paper alignment (Section 5.1, Appendix F.2):
      - Port token: soft patching via Conv1D along time.
      - Dynamics tokens: two proxy scalars from (i) raw inputs and (ii) finite differences.
      - Temporal token: Time2Vec encoding of per-series (relative) timestamps, plus missingness signal.
      - Temporal encoder: Transformer encoder over time.
      - Pooling: learnable query pooling to produce summary tokens E_ti.
    """

    def __init__(
        self,
        num_entities: int,
        feat_dim: int,
        window_size: int,
        *,
        mix_dim: int = 64,
        tv_hidden: int = 32,
        out_len: int = 16,
        context_dim: int = 256,
        enc_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
        patch_kernel: int = 3,
        time2vec_dim: int = 9,
        irreg_pooling: str = "none",
        irreg_hidden: int = 32,
        irreg_residual_scale: float = 0.1,
        t_token_mode: str = "none",
        t_token_scale: float = 0.1,
        pos_encoding: str = "learned_abs",
        rope_base: float = 10000.0,
        channel_balanced_x_loss: bool = False,
    ) -> None:
        super().__init__()

        if patch_kernel % 2 == 0:
            raise ValueError(f"patch_kernel must be odd to maintain sequence length, got {patch_kernel}")
        pos_encoding = str(pos_encoding).strip().lower()
        valid_pos_encodings = {"learned_abs", "continuous_rope", "learned_plus_continuous_rope"}
        if pos_encoding not in valid_pos_encodings:
            raise ValueError(f"Unknown pos_encoding={pos_encoding!r}; expected one of {sorted(valid_pos_encodings)}")

        self.N = int(num_entities)
        self.D = int(feat_dim)
        self.window_size = int(window_size)
        self.Hc = int(context_dim)
        self.S = int(out_len)
        self.mix_dim = int(mix_dim)
        self.time2vec_dim = int(time2vec_dim)
        self.irreg_pooling = str(irreg_pooling)
        self.irreg_residual_scale = float(irreg_residual_scale)
        self.t_token_mode = str(t_token_mode)
        self.t_token_scale = float(t_token_scale)
        self.pos_encoding = pos_encoding
        self.channel_balanced_x_loss = bool(channel_balanced_x_loss)
        self.use_learned_pos = self.pos_encoding in {"learned_abs", "learned_plus_continuous_rope"}
        self.use_rope = self.pos_encoding in {"continuous_rope", "learned_plus_continuous_rope"}
        self.rope_time_scale = float(max(self.window_size - 1, 1))

        # 1) Port token: soft patching / feature mixing over time
        padding = (patch_kernel - 1) // 2
        self.input_mixer = nn.Sequential(
            nn.Conv1d(self.D, self.mix_dim, kernel_size=patch_kernel, stride=1, padding=padding),
        )
        self.mixer_norm = nn.LayerNorm(self.mix_dim)
        self.mixer_act = nn.GELU()
        if self.t_token_mode != "none":
            self.t_input_mixer = nn.Conv1d(
                self.D, self.mix_dim, kernel_size=patch_kernel, stride=1, padding=padding
            )
            self.t_mixer_norm = nn.LayerNorm(self.mix_dim)
            self.t_mixer_act = nn.GELU()
            self.t_pool_bias_proj = nn.Linear(self.mix_dim, self.S)
            nn.init.normal_(self.t_pool_bias_proj.weight, mean=0.0, std=1e-2)
            nn.init.zeros_(self.t_pool_bias_proj.bias)
        else:
            self.t_input_mixer = None
            self.t_mixer_norm = None
            self.t_mixer_act = None
            self.t_pool_bias_proj = None

        # 2) Dynamics tokens (proxy signals)
        self.v_head = TVHead(self.D, tv_hidden)        # V_sig from raw input
        self.t_head = TVHead(self.D, tv_hidden)        # T_sig from finite differences

        # 3) Temporal token (Time2Vec timestamp encoding + missingness)
        self.time2vec = Time2Vec(self.time2vec_dim)
        self.irreg_input_dim = 2 + self.time2vec_dim

        if self.irreg_pooling != "none":
            self.irreg_proj = nn.Sequential(
                nn.LayerNorm(self.irreg_input_dim),
                nn.Linear(self.irreg_input_dim, int(irreg_hidden)),
                nn.GELU(),
                nn.Linear(int(irreg_hidden), self.mix_dim),
            )
            self.pool_bias_proj = nn.Sequential(
                nn.LayerNorm(self.irreg_input_dim),
                nn.Linear(self.irreg_input_dim, int(irreg_hidden)),
                nn.GELU(),
                nn.Linear(int(irreg_hidden), self.S),
            )
            nn.init.normal_(self.irreg_proj[-1].weight, mean=0.0, std=1e-2)
            nn.init.zeros_(self.irreg_proj[-1].bias)
            nn.init.normal_(self.pool_bias_proj[-1].weight, mean=0.0, std=1e-2)
            nn.init.zeros_(self.pool_bias_proj[-1].bias)
        else:
            self.irreg_proj = None
            self.pool_bias_proj = None

        # Encoder input dimension:
        #   mix features + {v_sig, t_sig} + missingness_scalar + time2vec(t)
        base_dim = self.mix_dim + 2 + 1 + self.time2vec_dim
        self.encoder_dim = base_dim

        # Learnable positional embeddings (over time dimension) for the baseline path.
        if self.use_learned_pos:
            self.pos_embedding = nn.Parameter(torch.randn(1, self.window_size, self.encoder_dim) * 0.02)

        # Ensure encoder_dim divisible by n_heads (pad if necessary)
        if self.encoder_dim % n_heads != 0:
            new_dim = ((self.encoder_dim // n_heads) + 1) * n_heads
            self.input_pad = nn.Linear(self.encoder_dim, new_dim)
            self.encoder_dim = new_dim
            if self.use_learned_pos:
                self.pos_embedding = nn.Parameter(torch.randn(1, self.window_size, self.encoder_dim) * 0.02)
        else:
            self.input_pad = nn.Identity()

        if self.use_rope:
            self.history_encoder = ContinuousRoPEEncoder(
                d_model=self.encoder_dim,
                n_heads=n_heads,
                dim_feedforward=self.encoder_dim * 4,
                num_layers=enc_layers,
                dropout=dropout,
                rope_base=rope_base,
            )
        else:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.encoder_dim,
                nhead=n_heads,
                dim_feedforward=self.encoder_dim * 4,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.history_encoder = nn.TransformerEncoder(encoder_layer, num_layers=enc_layers)

        # Project temporal encoder output to context token dimension
        self.token_proj = nn.Linear(self.encoder_dim, self.Hc)

        # Learnable queries for pooling (S summary tokens)
        self.queries = nn.Parameter(torch.randn(self.S, self.Hc) / math.sqrt(self.Hc))
        self.norm = nn.LayerNorm(self.Hc)

        # 5) Decoders used during summarizer pretraining
        self.decoder_net = nn.Sequential(
            nn.Linear(self.Hc, self.Hc * 2),
            nn.GELU(),
            nn.Linear(self.Hc * 2, self.window_size * self.N * self.D),
        )
        self.v_decoder = nn.Linear(self.Hc, self.window_size * self.N)
        self.t_decoder = nn.Linear(self.Hc, self.window_size * self.N)
        self.dt_decoder = nn.Linear(self.Hc, self.window_size * self.N)
        self.obs_decoder = nn.Linear(self.Hc, self.window_size * self.N)

    @staticmethod
    def _masked_mse(
        pred: torch.Tensor,
        target: torch.Tensor,
        mask_bn: torch.Tensor,
        *,
        obs_mask: Optional[torch.Tensor] = None,
        channel_balanced: bool = False,
    ) -> torch.Tensor:
        """MSE loss only on valid entities.

        pred/target: [B, K, N, D] or [B, K, N]
        mask_bn: [B, N] boolean mask
        obs_mask: optional observed-entry mask matching pred/target.
        """
        mask_bn = torch.as_tensor(mask_bn, device=pred.device, dtype=torch.bool)

        if pred.ndim == 4:
            m_bool = mask_bn[:, None, :, None].expand_as(pred)
            if obs_mask is not None:
                obs = torch.as_tensor(obs_mask, device=pred.device, dtype=torch.bool)
                if obs.ndim == 3:
                    obs = obs.unsqueeze(-1)
                if obs.ndim != 4:
                    raise ValueError(f"obs_mask must have 3 or 4 dims for 4D loss, got {tuple(obs.shape)}")
                if obs.shape[-1] == 1 and pred.shape[-1] != 1:
                    obs = obs.expand_as(pred)
                if obs.shape != pred.shape:
                    raise ValueError(f"obs_mask shape {tuple(obs.shape)} does not match prediction {tuple(pred.shape)}")
                m_bool = m_bool & obs
        else:
            m_bool = mask_bn[:, None, :].expand_as(pred)
            if obs_mask is not None:
                obs = torch.as_tensor(obs_mask, device=pred.device, dtype=torch.bool)
                if obs.ndim == 4 and obs.shape[-1] == 1:
                    obs = obs.squeeze(-1)
                if obs.shape != pred.shape:
                    raise ValueError(f"obs_mask shape {tuple(obs.shape)} does not match prediction {tuple(pred.shape)}")
                m_bool = m_bool & obs

        m = m_bool.to(dtype=pred.dtype)
        se = (pred - target).pow(2) * m
        if channel_balanced and pred.ndim == 4:
            denom_ch = m.sum(dim=(0, 1, 2))
            valid = denom_ch > 0
            if valid.any():
                per_channel = se.sum(dim=(0, 1, 2))[valid] / denom_ch[valid].clamp_min(1.0)
                return per_channel.mean()
        return se.sum() / m.sum().clamp_min(1.0)

    @staticmethod
    def _canon_dt(dt: Optional[torch.Tensor], *, B: int, K: int, N: int, device, dtype) -> torch.Tensor:
        """Canonicalize dt to shape [B, K, N] (float)."""
        if dt is None:
            dt_bk = torch.arange(K, device=device, dtype=dtype).view(1, K).expand(B, K)
            return dt_bk.unsqueeze(-1).expand(B, K, N)

        dt = torch.as_tensor(dt, device=device, dtype=dtype)
        if dt.dim() == 4 and dt.size(-1) == 1:
            dt = dt.squeeze(-1)

        if dt.dim() == 2:
            # [B, K]
            if dt.size(0) != B or dt.size(1) != K:
                raise ValueError(f"dt 2D must be [B,K]={B,K}, got {tuple(dt.shape)}")
            return dt.unsqueeze(-1).expand(B, K, N)

        if dt.dim() != 3:
            raise ValueError(f"dt must have 2 or 3 dims (or 4 with trailing 1). Got {tuple(dt.shape)}")

        # Common dataset layouts:
        #   [B, N, K] -> permute
        #   [B, K, N] -> keep
        if dt.size(0) != B:
            raise ValueError(f"dt batch mismatch: expected B={B}, got {dt.size(0)}")
        if dt.size(1) == N and dt.size(2) == K:
            return dt.permute(0, 2, 1).contiguous()
        if dt.size(1) == K and dt.size(2) == N:
            return dt.contiguous()

        # Handle degenerate last dim 1: [B, K, 1]
        if dt.size(1) == K and dt.size(2) == 1:
            return dt.expand(B, K, N)

        raise ValueError(f"Unrecognized dt layout: expected [B,K,N] or [B,N,K], got {tuple(dt.shape)}")

    @staticmethod
    def _canon_obs_mask(obs_mask: Optional[torch.Tensor], *, x: torch.Tensor, B: int, K: int, N: int, D: int, device) -> torch.Tensor:
        """Canonicalize observation mask to [B, K, N, D] (bool).

        If obs_mask is None, fall back to all-ones (treat everything as observed).
        """
        if obs_mask is None:
            return torch.ones((B, K, N, D), device=device, dtype=torch.bool)

        m = torch.as_tensor(obs_mask, device=device)
        if m.dim() == 4:
            # [B,K,N,D] or [B,N,K,D]
            if m.size(0) != B:
                raise ValueError(f"obs_mask batch mismatch: expected B={B}, got {m.size(0)}")
            if m.size(3) not in (1, D):
                raise ValueError(f"obs_mask feature dim must be 1 or D={D}, got {m.size(3)}")
            if m.size(1) == K and m.size(2) == N:
                m = m.contiguous()
            elif m.size(1) == N and m.size(2) == K:
                m = m.permute(0, 2, 1, 3).contiguous()
            else:
                raise ValueError(f"obs_mask 4D must be [B,K,N,D] or [B,N,K,D], got {tuple(m.shape)}")
            if m.size(3) == 1:
                m = m.expand(B, K, N, D)
        elif m.dim() == 3:
            # [B,K,N] or [B,N,K] -> expand over D
            if m.size(0) != B:
                raise ValueError(f"obs_mask batch mismatch: expected B={B}, got {m.size(0)}")
            if m.size(1) == K and m.size(2) == N:
                m = m.unsqueeze(-1).expand(B, K, N, D)
            elif m.size(1) == N and m.size(2) == K:
                m = m.permute(0, 2, 1).unsqueeze(-1).expand(B, K, N, D)
            else:
                raise ValueError(f"obs_mask 3D must be [B,K,N] or [B,N,K], got {tuple(m.shape)}")
        else:
            raise ValueError(f"obs_mask must have 3 or 4 dims, got {tuple(m.shape)}")

        return m.to(dtype=torch.bool)

    @staticmethod
    def _normalize_rel_t(rel_t: torch.Tensor) -> torch.Tensor:
        """Scale relative time to [0, 1] per series for stable reconstruction."""

        scale = rel_t.amax(dim=1, keepdim=True).clamp_min(1.0)
        return rel_t / scale

    @staticmethod
    def _relative_time_from_dt(dt_bkn: torch.Tensor) -> torch.Tensor:
        """Return shared window-local time offsets."""

        return relative_time_offsets(dt_bkn, time_dim=1)

    def forward(
        self,
        x: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
        dt: Optional[torch.Tensor] = None,
        ctx_diff: Optional[torch.Tensor] = None,
        obs_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        if ctx_diff is None:
            raise ValueError("ctx_diff required")

        B, K, N, D = x.shape
        if N != self.N or D != self.D:
            raise ValueError(f"Shape mismatch. Expected (..., {self.N}, {self.D}), got {tuple(x.shape)}")
        if K != self.window_size:
            raise ValueError(f"Expected window_size={self.window_size}, got K={K}")

        device = x.device
        dtype = x.dtype
        if pad_mask is None:
            entity_mask = torch.ones((B, N), device=device, dtype=torch.bool)
        else:
            entity_mask = torch.as_tensor(pad_mask, device=device, dtype=torch.bool)
            if entity_mask.shape != (B, N):
                raise ValueError(f"pad_mask must be a valid-entity mask with shape [B,N]={B,N}, got {tuple(entity_mask.shape)}")
        entity_weight = entity_mask.to(dtype=dtype)
        entity_denom = entity_weight.sum(dim=1).clamp_min(1.0)
        entity_weight_b = entity_weight[:, :, None, None]
        x = x * entity_mask[:, None, :, None].to(dtype=dtype)
        ctx_diff = ctx_diff * entity_mask[:, None, :, None].to(dtype=dtype)

        # ---------------------------------------------------------
        # 1) Port token: soft patching (Conv1d over time)
        # ---------------------------------------------------------
        # [B, K, N, D] -> [B, N, K, D] -> [B*N, K, D]
        x_flat_in = x.permute(0, 2, 1, 3).reshape(B * N, K, D)
        # Conv1d expects [B*N, D, K]
        x_perm = x_flat_in.permute(0, 2, 1)
        x_conv = self.input_mixer(x_perm)             # [B*N, mix_dim, K]
        x_mixed = x_conv.permute(0, 2, 1)             # [B*N, K, mix_dim]
        x_mixed = self.mixer_act(self.mixer_norm(x_mixed))
        t_pool_bias = None
        if self.t_input_mixer is not None:
            t_flat_in = ctx_diff.permute(0, 2, 1, 3).reshape(B * N, K, D)
            t_perm = t_flat_in.permute(0, 2, 1)
            t_conv = self.t_input_mixer(t_perm)
            t_mixed = t_conv.permute(0, 2, 1)
            t_mixed = self.t_mixer_act(self.t_mixer_norm(t_mixed))
            if self.t_token_mode in {"residual", "both"}:
                x_mixed = x_mixed + self.t_token_scale * t_mixed
            if self.t_token_mode in {"bias", "both"}:
                t_pool_bias_bn = self.t_pool_bias_proj(t_mixed).view(B, N, K, self.S)
                t_pool_bias = (t_pool_bias_bn * entity_weight_b).sum(dim=1) / entity_denom[:, None, None]

        # ---------------------------------------------------------
        # 2) Dynamics proxies: V_sig and T_sig
        # ---------------------------------------------------------
        v_sig = self.v_head(x)                          # [B, K, N]
        t_sig = self.t_head(ctx_diff)                   # [B, K, N]
        v_flat = v_sig.permute(0, 2, 1).reshape(B * N, K, 1)
        t_flat = t_sig.permute(0, 2, 1).reshape(B * N, K, 1)

        # ---------------------------------------------------------
        # 3) Temporal token: Time2Vec(relative time) + missingness scalar
        # ---------------------------------------------------------
        dt_bkn = self._canon_dt(dt, B=B, K=K, N=N, device=device, dtype=torch.float32)  # [B,K,N]
        rel_t = relative_time_offsets(dt_bkn, time_dim=1)
        rel_t_unit = self._normalize_rel_t(rel_t)
        # Use normalized relative time in Time2Vec to keep timestamp features
        # numerically stable across datasets with very different raw time scales.
        rel_t_unit_flat = rel_t_unit.permute(0, 2, 1).reshape(B * N, K, 1)

        obs_bknd = self._canon_obs_mask(obs_mask, x=x, B=B, K=K, N=N, D=D, device=device)  # [B,K,N,D]
        obs_bknd = obs_bknd & entity_mask[:, None, :, None]
        obs_frac = obs_bknd.to(dtype=torch.float32).mean(dim=-1)  # [B,K,N] in [0,1]
        obs_flat = obs_frac.permute(0, 2, 1).reshape(B * N, K, 1)

        t2v = self.time2vec(rel_t_unit_flat)            # [B*N, K, time2vec_dim]
        t2v = t2v.to(dtype=dtype)
        irreg_input = torch.cat([t_flat.to(dtype), obs_flat.to(dtype), t2v], dim=-1)
        pool_bias = None
        if self.irreg_proj is not None:
            irreg_feat = self.irreg_proj(irreg_input)
            x_mixed = x_mixed + self.irreg_residual_scale * irreg_feat
            pool_bias_bn = self.pool_bias_proj(irreg_input).view(B, N, K, self.S)
            pool_bias = (pool_bias_bn * entity_weight_b).sum(dim=1) / entity_denom[:, None, None]

        # ---------------------------------------------------------
        # 4) Fuse + positional embedding + temporal encoder
        # ---------------------------------------------------------
        fused = torch.cat([x_mixed, v_flat.to(dtype), t_flat.to(dtype), obs_flat.to(dtype), t2v], dim=-1)
        fused = self.input_pad(fused)                   # maybe pad to multiple of heads
        if self.use_learned_pos:
            fused = fused + self.pos_embedding          # [B*N, K, encoder_dim]

        if self.use_rope:
            rope_t = rel_t_unit_flat.squeeze(-1).to(dtype=fused.dtype) * self.rope_time_scale
            encoded_hist = self.history_encoder(fused, rope_t)  # [B*N, K, encoder_dim]
        else:
            encoded_hist = self.history_encoder(fused)  # [B*N, K, encoder_dim]

        # ---------------------------------------------------------
        # 5) Aggregate across entities and pool to S summary tokens
        # ---------------------------------------------------------
        encoded_bn = encoded_hist.view(B, N, K, -1)      # [B, N, K, Dim]
        encoded_mean = (encoded_bn * entity_weight[:, :, None, None]).sum(dim=1) / entity_denom[:, None, None]
        tokens = self.token_proj(encoded_mean)           # [B, K, Hc]

        norm_tokens = self.norm(tokens)
        norm_queries = F.layer_norm(self.queries, (self.Hc,))

        att = torch.matmul(norm_tokens, norm_queries.t()) / math.sqrt(self.Hc)  # [B,K,S]
        if t_pool_bias is not None:
            att = att + t_pool_bias.to(dtype=att.dtype)
        if pool_bias is not None:
            att = att + pool_bias.to(dtype=att.dtype)
        att = att.softmax(dim=1)                         # softmax over time
        context = torch.matmul(att.transpose(1, 2), tokens)  # [B,S,Hc]

        # ---------------------------------------------------------
        # 6) Decoding heads (for summarizer pretraining)
        # ---------------------------------------------------------
        # Decode per-time/per-entity signals (B,N,K,*) from context tokens
        # A small linear head is sufficient here; keep consistent with existing behavior by pooling:
        ctx_mean = context.mean(dim=1)                   # [B, Hc]

        x_hat = self.decoder_net(ctx_mean).view(B, K, N, D)
        v_hat = self.v_decoder(ctx_mean).view(B, K, N)
        t_hat = self.t_decoder(ctx_mean).view(B, K, N)
        dt_hat = self.dt_decoder(ctx_mean).view(B, K, N)
        obs_hat = torch.sigmoid(self.obs_decoder(ctx_mean).view(B, K, N))

        aux = {
            "x": x,
            "x_hat": x_hat,
            "obs_mask": obs_bknd,
            "v_sig": v_sig,
            "v_hat": v_hat,
            "t_sig": t_sig,
            "t_hat": t_hat,
            "rel_t_unit": rel_t_unit,
            "dt_hat": dt_hat,
            "obs_frac": obs_frac,
            "obs_hat": obs_hat,
            "rel_t": rel_t,
        }

        return context, aux

    def recon_loss(
        self,
        aux: Dict[str, torch.Tensor],
        entity_mask: torch.Tensor,
        weights: Tuple[float, ...] = (1.0, 0.1, 0.1, 0.0, 0.0),
    ) -> torch.Tensor:
        """Weighted reconstruction loss used during summarizer pretraining."""
        if len(weights) == 3:
            w_x, w_v, w_t = weights
            w_dt = 0.0
            w_obs = 0.0
        elif len(weights) == 5:
            w_x, w_v, w_t, w_dt, w_obs = weights
        else:
            raise ValueError(f"Expected 3 or 5 summarizer loss weights, got {weights!r}")

        loss_x = self._masked_mse(
            aux["x_hat"],
            aux["x"],
            entity_mask,
            obs_mask=aux.get("obs_mask"),
            channel_balanced=self.channel_balanced_x_loss,
        )
        loss_v = self._masked_mse(aux["v_hat"], aux["v_sig"], entity_mask)
        loss_t = self._masked_mse(aux["t_hat"], aux["t_sig"], entity_mask)
        loss_dt = self._masked_mse(aux["dt_hat"], aux["rel_t_unit"], entity_mask)
        loss_obs = self._masked_mse(aux["obs_hat"], aux["obs_frac"], entity_mask)
        return (
            w_x * loss_x
            + w_v * loss_v
            + w_t * loss_t
            + w_dt * loss_dt
            + w_obs * loss_obs
        )
