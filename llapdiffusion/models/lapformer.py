from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from llapdiffusion.models.laptrans import LaplaceTransformEncoder, LaplacePseudoInverse


def _init_small_out_proj(layer: nn.Linear, *, std: float = 1e-2) -> None:
    """Keep residual branches near-identity while preserving gradient flow at step 1."""
    nn.init.normal_(layer.weight, mean=0.0, std=std)
    nn.init.zeros_(layer.bias)


def _scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    attn_bias: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    training: bool = False,
) -> torch.Tensor:
    dropout_p = float(dropout_p) if training else 0.0
    if hasattr(F, "scaled_dot_product_attention"):
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_bias,
            dropout_p=dropout_p,
        )

    attn = torch.matmul(q, k.transpose(-2, -1)) / (k.shape[-1] ** 0.5)
    if attn_bias is not None:
        attn = attn + attn_bias
    attn = torch.softmax(attn, dim=-1)
    attn = F.dropout(attn, p=dropout_p, training=training)
    return torch.matmul(attn, v)


class AdaLayerNorm(nn.Module):
    """LayerNorm with feature-wise affine parameters conditioned on a vector."""

    def __init__(self, normalized_shape: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape, elementwise_affine=False)
        self.to_ss = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 2 * normalized_shape),
        )
        nn.init.zeros_(self.to_ss[-1].weight)
        nn.init.zeros_(self.to_ss[-1].bias)

    def forward(self, x: torch.Tensor, c: Optional[torch.Tensor]) -> torch.Tensor:
        h = self.norm(x)
        if c is None:
            scale = torch.zeros(h.size(0), h.size(-1), device=h.device, dtype=h.dtype)
            bias = torch.zeros_like(scale)
        else:
            scale, bias = self.to_ss(c).chunk(2, dim=-1)
        return (1 + scale).unsqueeze(1) * h + bias.unsqueeze(1)


class SummaryAttentionPool(nn.Module):
    """Learned pooling over summary tokens using a timestep-conditioned query."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.token_norm = nn.LayerNorm(hidden_dim)
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        _init_small_out_proj(self.out_proj)

    def forward(self, summary_tokens: torch.Tensor, query_vec: torch.Tensor) -> torch.Tensor:
        if summary_tokens.dim() != 3:
            raise ValueError(f"summary_tokens must be [B,S,H], got {tuple(summary_tokens.shape)}")
        if query_vec.dim() != 2 or query_vec.shape[0] != summary_tokens.shape[0]:
            raise ValueError(
                "query_vec must be [B,H] and match the batch dimension of summary_tokens"
            )

        h = self.token_norm(summary_tokens)
        q = self.query_proj(query_vec).unsqueeze(1)  # [B,1,H]
        k = self.key_proj(h)
        v = self.value_proj(h)
        attn = torch.matmul(q, k.transpose(-2, -1)) / (k.shape[-1] ** 0.5)
        attn = torch.softmax(attn, dim=-1)
        pooled = torch.matmul(attn, v).squeeze(1)
        return self.out_proj(pooled)


class TransformerBlock(nn.Module):
    """Self-attention block (over modal tokens) with AdaLayerNorm conditioning."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        cond_dim: Optional[int] = None,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        cond_width = int(cond_dim or hidden_dim)

        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = AdaLayerNorm(hidden_dim, cond_width)
        self.norm2 = AdaLayerNorm(hidden_dim, cond_width)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, int(hidden_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(hidden_dim * mlp_ratio), hidden_dim),
            nn.Dropout(dropout),
        )
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        cond_vec: Optional[torch.Tensor] = None,
        attn_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L, H = x.shape
        h = self.norm1(x, cond_vec)
        qkv = (
            self.qkv(h)
            .reshape(B, L, 3, self.num_heads, H // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B,heads,L,dh]
        out = _scaled_dot_product_attention(
            q,
            k,
            v,
            attn_bias=attn_bias,
            dropout_p=self.attn_dropout.p,
            training=self.training,
        ).transpose(1, 2).reshape(B, L, H)
        x = x + self.resid_dropout(self.proj(out))
        x = x + self.mlp(self.norm2(x, cond_vec))
        return x


class CrossAttnBlock(nn.Module):
    """Cross-attention block: modal tokens attend to summary tokens."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        cond_dim: Optional[int] = None,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.num_heads = num_heads
        cond_width = int(cond_dim or hidden_dim)
        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.kv = nn.Linear(hidden_dim, hidden_dim * 2)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm_q = AdaLayerNorm(hidden_dim, cond_width)
        self.norm_kv = AdaLayerNorm(hidden_dim, cond_width)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x_q: torch.Tensor,
        x_kv: torch.Tensor,
        cond_vec: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, Lq, H = x_q.shape
        Lkv = x_kv.shape[1]
        xq = self.norm_q(x_q, cond_vec)
        xkv = self.norm_kv(x_kv, cond_vec)

        q = self.q(xq).reshape(B, Lq, self.num_heads, H // self.num_heads).transpose(1, 2)
        kv = self.kv(xkv).reshape(B, Lkv, 2, self.num_heads, H // self.num_heads)
        k = kv[:, :, 0].transpose(1, 2)
        v = kv[:, :, 1].transpose(1, 2)

        out = _scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_dropout.p,
            training=self.training,
        ).transpose(1, 2).reshape(B, Lq, H)
        return x_q + self.resid_dropout(self.proj(out))


class ModalSandwichBlock(nn.Module):
    """One modal-token refinement block (operates on theta in R^{2K x D})."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int,
        k: int,
        cond_dim: Optional[int] = None,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        cross_first: bool = True,
    ) -> None:
        super().__init__()
        self.k = int(k)
        self.cross_first = bool(cross_first)

        # theta token embedding/projection
        self.coef2hid = nn.Linear(input_dim, hidden_dim)
        self.hid2coef = nn.Linear(hidden_dim, input_dim)
        _init_small_out_proj(self.hid2coef)

        self.mode_pos = nn.Parameter(torch.zeros(1, 2 * self.k, hidden_dim))
        nn.init.normal_(self.mode_pos, mean=0.0, std=0.02)

        self.self_blk = TransformerBlock(
            hidden_dim,
            num_heads,
            cond_dim=cond_dim,
            mlp_ratio=4.0,
            dropout=dropout,
            attn_dropout=attn_dropout,
        )
        self.cross_blk = CrossAttnBlock(
            hidden_dim,
            num_heads,
            cond_dim=cond_dim,
            dropout=dropout,
            attn_dropout=attn_dropout,
        )

    def forward(
        self,
        theta: torch.Tensor,              # [B,2K,D]
        t_vec: torch.Tensor,              # [B,H]
        summary_kv_H: Optional[torch.Tensor] = None,  # [B,S,H]
        adaln_cond: Optional[torch.Tensor] = None,    # [B,H] or [B,2H]
        attn_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if theta.dim() != 3 or theta.shape[1] != 2 * self.k:
            raise ValueError(f"theta must be [B, {2*self.k}, D]")

        h = self.coef2hid(theta) + self.mode_pos + t_vec.unsqueeze(1)
        cond_vec = t_vec if adaln_cond is None else adaln_cond

        def apply_cross(x: torch.Tensor) -> torch.Tensor:
            if summary_kv_H is None:
                return x
            return self.cross_blk(x, summary_kv_H, cond_vec=cond_vec)

        if self.cross_first:
            h = apply_cross(h)
            h = self.self_blk(h, cond_vec=cond_vec, attn_bias=attn_bias)
        else:
            h = self.self_blk(h, cond_vec=cond_vec, attn_bias=attn_bias)
            h = apply_cross(h)

        return theta + self.hid2coef(h)


class LapFormer(nn.Module):
    """Modal-token LapFormer.

    Flow:
        1) Analyze x_time -> theta (modal residues) and effective poles (rho, omega).
        2) Refine theta via a stack of ModalSandwichBlocks (self_blk kept).
        3) Synthesize hat{z}_0 in time domain with residual refinement.

    Analysis always uses the learned time cross-attention path (the only supported mode).
    There is no lap_mode/use_time_attn switch in the public forward API.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        laplace_k: int = 8,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        cross_first: bool = True,
        use_mlp_residual: bool = True,
        self_conditioning: bool = False,
        summary_pool_mode: str = "mean",
        pole_pool_use_raw_summary: bool = False,
        block_summary_adaln: bool = False,
        analysis_summary_qk: bool = False,
        analysis_qk_use_raw_summary: bool = False,
        rho_conditioning_mode: str = "raw",
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.self_conditioning = bool(self_conditioning)
        self.k = int(laplace_k)
        pool_mode = str(summary_pool_mode).strip().lower()
        if pool_mode not in {"mean", "attn"}:
            raise ValueError(f"Unknown summary_pool_mode '{summary_pool_mode}'. Use 'mean' or 'attn'.")
        self.summary_pool_mode = pool_mode
        self.pole_pool_use_raw_summary = bool(pole_pool_use_raw_summary)
        self.block_summary_adaln = bool(block_summary_adaln)
        self.analysis_summary_qk = bool(analysis_summary_qk)
        self.analysis_qk_use_raw_summary = bool(analysis_qk_use_raw_summary)
        self.rho_conditioning_mode = str(rho_conditioning_mode).strip().lower()

        # Modal analysis/synthesis
        self.analysis = LaplaceTransformEncoder(
            k=self.k,
            feat_dim=input_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            cond_dim=2 * hidden_dim,
            attn_cond_dim=(hidden_dim if self.analysis_summary_qk else None),
            rho_conditioning_mode=self.rho_conditioning_mode,
            attn_dropout=attn_dropout,
        )
        self.synthesis = LaplacePseudoInverse(
            self.analysis,
            hidden_dim=hidden_dim,
            use_mlp_residual=use_mlp_residual,
        )

        # Optional self-conditioning in modal space: project sc_feat -> theta_sc
        self.sc_gate = nn.Parameter(torch.tensor(1.0)) if self.self_conditioning else None
        self.summary_pool = (
            SummaryAttentionPool(hidden_dim)
            if self.summary_pool_mode == "attn"
            else None
        )
        block_cond_dim = (2 * hidden_dim) if self.block_summary_adaln else hidden_dim

        self.blocks = nn.ModuleList(
            [
                ModalSandwichBlock(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    k=self.k,
                    cond_dim=block_cond_dim,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                    cross_first=cross_first,
                )
                for _ in range(int(num_layers))
            ]
        )

        self.head_norm = nn.LayerNorm(input_dim)
        self.head_proj = nn.Linear(input_dim, input_dim)
        nn.init.zeros_(self.head_proj.weight)
        nn.init.zeros_(self.head_proj.bias)
        self.output_skip_scale = nn.Parameter(torch.tensor(0.1))

    def _select_summary_tokens(
        self,
        *,
        cond_summary: Optional[torch.Tensor] = None,
        cond_summary_raw: Optional[torch.Tensor] = None,
        use_raw: bool = False,
    ) -> Optional[torch.Tensor]:
        if use_raw and cond_summary_raw is not None:
            return cond_summary_raw
        if cond_summary is not None:
            return cond_summary
        return cond_summary_raw

    def pool_summary_tokens(
        self,
        t_vec: torch.Tensor,
        *,
        cond_summary: Optional[torch.Tensor] = None,
        cond_summary_raw: Optional[torch.Tensor] = None,
        use_raw: bool = False,
    ) -> torch.Tensor:
        summary_tokens = self._select_summary_tokens(
            cond_summary=cond_summary,
            cond_summary_raw=cond_summary_raw,
            use_raw=use_raw,
        )
        if summary_tokens is None:
            return torch.zeros_like(t_vec)

        if self.summary_pool is not None:
            return self.summary_pool(summary_tokens, t_vec)
        return summary_tokens.mean(dim=1)

    def make_pole_cond(
        self,
        t_vec: torch.Tensor,
        *,
        cond_summary: Optional[torch.Tensor] = None,
        cond_summary_raw: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        summary_pool = self.pool_summary_tokens(
            t_vec,
            cond_summary=cond_summary,
            cond_summary_raw=cond_summary_raw,
            use_raw=self.pole_pool_use_raw_summary,
        )
        return torch.cat([t_vec, summary_pool], dim=-1)

    def make_analysis_attn_cond(
        self,
        t_vec: torch.Tensor,
        *,
        cond_summary: Optional[torch.Tensor] = None,
        cond_summary_raw: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if not self.analysis_summary_qk:
            return None
        summary_tokens = self._select_summary_tokens(
            cond_summary=cond_summary,
            cond_summary_raw=cond_summary_raw,
            use_raw=self.analysis_qk_use_raw_summary,
        )
        if summary_tokens is None:
            return None
        if self.summary_pool is not None:
            return self.summary_pool(summary_tokens, t_vec)
        return summary_tokens.mean(dim=1)

    def make_block_cond(
        self,
        t_vec: torch.Tensor,
        *,
        cond_summary: Optional[torch.Tensor] = None,
        cond_summary_raw: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not self.block_summary_adaln:
            return t_vec
        summary_pool = self.pool_summary_tokens(
            t_vec,
            cond_summary=cond_summary,
            cond_summary_raw=cond_summary_raw,
            use_raw=self.pole_pool_use_raw_summary,
        )
        return torch.cat([t_vec, summary_pool], dim=-1)

    def forward(
        self,
        x_tokens: torch.Tensor,                 # [B,T,D]
        t_vec: torch.Tensor,                    # [B,H]
        cond_summary: Optional[torch.Tensor] = None,  # [B,S,H]
        cond_summary_raw: Optional[torch.Tensor] = None,  # [B,S,H]
        sc_feat: Optional[torch.Tensor] = None,       # [B,T,D]
        dt: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, _ = x_tokens.shape
        if t_vec.dim() != 2 or t_vec.shape[0] != B or t_vec.shape[1] != self.hidden_dim:
            raise ValueError(
                f"t_vec must be [B, hidden_dim]={B, self.hidden_dim}; got {tuple(t_vec.shape)}"
            )
        if cond_summary is not None:
            if cond_summary.dim() != 3 or cond_summary.shape[0] != B or cond_summary.shape[2] != self.hidden_dim:
                raise ValueError(
                    f"cond_summary must be [B,S,hidden_dim]={B,'S',self.hidden_dim}; got {tuple(cond_summary.shape)}"
                )
        if cond_summary_raw is not None:
            if (
                cond_summary_raw.dim() != 3
                or cond_summary_raw.shape[0] != B
                or cond_summary_raw.shape[2] != self.hidden_dim
            ):
                raise ValueError(
                    f"cond_summary_raw must be [B,S,hidden_dim]={B,'S',self.hidden_dim}; got {tuple(cond_summary_raw.shape)}"
                )

        # Conditioning vector used for pole prediction
        cond_vec = self.make_pole_cond(
            t_vec,
            cond_summary=cond_summary,
            cond_summary_raw=cond_summary_raw,
        )
        block_cond = self.make_block_cond(
            t_vec,
            cond_summary=cond_summary,
            cond_summary_raw=cond_summary_raw,
        )
        analysis_attn_cond = self.make_analysis_attn_cond(
            t_vec,
            cond_summary=cond_summary,
            cond_summary_raw=cond_summary_raw,
        )

        # Compute poles once per forward (reused for x and optional self-conditioning)
        rho, omega = self.analysis.effective_poles(B, x_tokens.dtype, x_tokens.device, cond=cond_vec)

        # Modal analysis: x_time -> theta
        theta, _, _, _ = self.analysis(
            x_tokens,
            dt=dt,
            t=t,
            cond=cond_vec,
            attn_cond=analysis_attn_cond,
            poles=(rho, omega),
            return_t_rel=False,
        )

        # Optional modal self-conditioning (project sc_feat onto SAME poles)
        if self.self_conditioning and sc_feat is not None:
            theta_sc, _, _, _ = self.analysis(
                sc_feat,
                dt=dt,
                t=t,
                cond=cond_vec,
                attn_cond=analysis_attn_cond,
                poles=(rho, omega),
                return_t_rel=False,
            )
            gate = torch.tanh(self.sc_gate)  # bounded scalar
            theta = theta + gate * theta_sc

        # Modal-token refinement stack
        for blk in self.blocks:
            theta = blk(
                theta,
                t_vec=t_vec,
                summary_kv_H=cond_summary,
                adaln_cond=block_cond,
                attn_bias=None,
            )

        # Synthesis (parallel over all queried timestamps)
        y_time = self.synthesis(theta, rho=rho, omega=omega, dt=dt, t=t, target_T=T)
        return self.output_skip_scale * y_time + self.head_proj(self.head_norm(y_time))
