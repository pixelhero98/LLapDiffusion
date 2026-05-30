import pytest
import torch

from llapdiffusion.models import lapformer
from llapdiffusion.models.lapformer import CrossAttnBlock, TransformerBlock
from llapdiffusion.trainers.train_val_llapdiff import _resolve_final_test_eval_mode


def _manual_attention(q, k, v, attn_bias=None):
    attn = torch.matmul(q, k.transpose(-2, -1)) / (k.shape[-1] ** 0.5)
    if attn_bias is not None:
        attn = attn + attn_bias
    attn = torch.softmax(attn, dim=-1)
    return torch.matmul(attn, v)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("run", "run"),
        ("eval", "run"),
        (True, "run"),
        ("skip", "skip"),
        ("off", "skip"),
        (False, "skip"),
        ("defer", "defer"),
    ],
)
def test_resolve_final_test_eval_mode(value, expected):
    assert _resolve_final_test_eval_mode(value) == expected


def test_resolve_final_test_eval_mode_rejects_unknown():
    with pytest.raises(ValueError, match="FINAL_TEST_EVAL"):
        _resolve_final_test_eval_mode("sometimes")


def test_scaled_dot_product_attention_matches_manual_attention_without_dropout():
    torch.manual_seed(7)
    q = torch.randn(2, 3, 5, 4)
    k = torch.randn(2, 3, 6, 4)
    v = torch.randn(2, 3, 6, 4)
    attn_bias = torch.randn(2, 3, 5, 6) * 0.1

    actual = lapformer._scaled_dot_product_attention(
        q,
        k,
        v,
        attn_bias=attn_bias,
        dropout_p=0.0,
        training=False,
    )
    expected = _manual_attention(q, k, v, attn_bias=attn_bias)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_lapformer_attention_blocks_preserve_shapes():
    torch.manual_seed(11)
    self_block = TransformerBlock(hidden_dim=16, num_heads=4, dropout=0.0, attn_dropout=0.0)
    cross_block = CrossAttnBlock(hidden_dim=16, num_heads=4, dropout=0.0, attn_dropout=0.0)
    self_block.eval()
    cross_block.eval()

    x = torch.randn(2, 7, 16)
    summary = torch.randn(2, 5, 16)
    cond = torch.randn(2, 16)

    assert self_block(x, cond_vec=cond).shape == x.shape
    assert cross_block(x, summary, cond_vec=cond).shape == x.shape
