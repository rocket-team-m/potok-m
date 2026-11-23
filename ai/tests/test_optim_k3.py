# SPDX-License-Identifier: GPL-3.0-or-later
"""Gates for the two evidence-backed Kimi K3 optimizer refinements.

Evidence: MUON01 report, arXiv:2607.24653 §2.5 (per-head Q/K/V Muon) and
§3.3 (K2's QK-Clip, arXiv:2507.20534 §2.1 Alg. 1). No NS coefficient is
invented here; the existing ones are used unchanged.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from potok_ai.optim import K3Config, MuonPlus, muon_update, per_head_muon_update, qk_clip_

HEADS = 2
HEAD_DIM = 4
IN_DIM = 6
TAU = 100.0


class _Attn(nn.Module):
    """Minimal dense attention projections with standard names/layout."""

    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(IN_DIM, HEADS * HEAD_DIM, bias=False)
        self.k_proj = nn.Linear(IN_DIM, HEADS * HEAD_DIM, bias=False)
        self.v_proj = nn.Linear(IN_DIM, HEADS * HEAD_DIM, bias=False)


def _weight_logit_scale(q: torch.Tensor, k: torch.Tensor, heads: int) -> torch.Tensor:
    """Independent read of the same weight-space bound used by qk_clip_."""
    head_dim = q.shape[0] // heads
    qh = q.reshape(heads, head_dim, -1).float()
    kh = k.reshape(heads, head_dim, -1).float()
    gram = torch.bmm(qh.transpose(1, 2), kh)
    return torch.linalg.svdvals(gram).amax(dim=-1) / head_dim**0.5


def test_per_head_update_is_independent_head_orthogonalization() -> None:
    torch.manual_seed(0)
    momentum = torch.randn(HEADS * HEAD_DIM, IN_DIM)
    got = per_head_muon_update(momentum, HEADS, steps=3)
    expected = torch.cat(
        [muon_update(momentum[i * HEAD_DIM : (i + 1) * HEAD_DIM], 3) for i in range(HEADS)],
        dim=0,
    )
    assert torch.allclose(got, expected, atol=1e-6)
    whole = muon_update(momentum, 3)
    assert (got - whole).abs().max() > 1e-3  # the split actually changed the update


def test_per_head_split_hits_q_k_and_v() -> None:
    torch.manual_seed(1)
    attn = _Attn()
    lr = 0.01
    opt = MuonPlus(attn.named_parameters(), lr=lr, ns_steps=3, k3=K3Config(num_heads=HEADS))
    grads = {name: torch.randn_like(p) for name, p in attn.named_parameters()}
    before = {name: p.detach().clone() for name, p in attn.named_parameters()}
    for name, p in attn.named_parameters():
        p.grad = grads[name]
    opt.step()
    for name, p in attn.named_parameters():
        momentum = 0.05 * grads[name]  # first step: EMA buffer starts at zero
        expected = before[name] - lr * per_head_muon_update(momentum, HEADS, 3)
        assert torch.allclose(p, expected, atol=1e-5), name
        whole = before[name] - lr * muon_update(momentum, 3)
        assert not torch.allclose(p, whole, atol=1e-5), name  # not the unsplit update


def test_qk_clip_caps_each_head_at_tau() -> None:
    torch.manual_seed(2)
    q, k = torch.randn(HEADS * HEAD_DIM, IN_DIM) * 5, torch.randn(HEADS * HEAD_DIM, IN_DIM) * 5
    gamma = qk_clip_(q, k, HEADS, tau=TAU)
    assert (gamma <= 1).all() and (gamma < 1).any()
    assert (_weight_logit_scale(q, k, HEADS) <= TAU * (1 + 1e-4)).all()


def test_qk_clip_is_noop_when_below_tau() -> None:
    torch.manual_seed(3)
    q, k = torch.randn(HEADS * HEAD_DIM, IN_DIM) * 0.1, torch.randn(HEADS * HEAD_DIM, IN_DIM) * 0.1
    q0, k0 = q.clone(), k.clone()
    gamma = qk_clip_(q, k, HEADS, tau=TAU)
    assert torch.all(gamma == 1)
    assert torch.equal(q, q0) and torch.equal(k, k0)


def test_step_qk_clip_changes_qk_but_not_v() -> None:
    torch.manual_seed(4)
    plain = _Attn()
    torch.manual_seed(4)  # identical init so only the clip can cause a difference
    clipped = _Attn()
    with torch.no_grad():  # push logit scale past tau so the clip actually engages
        for model in (plain, clipped):
            model.q_proj.weight.mul_(20)
            model.k_proj.weight.mul_(20)
    grads = {name: torch.randn_like(p) for name, p in plain.named_parameters()}
    opt_plain = MuonPlus(
        plain.named_parameters(), lr=0.01, ns_steps=3, k3=K3Config(num_heads=HEADS)
    )
    opt_clipped = MuonPlus(
        clipped.named_parameters(),
        lr=0.01,
        ns_steps=3,
        k3=K3Config(num_heads=HEADS, qk_clip=True, qk_clip_tau=5.0),
    )
    for model, opt in ((plain, opt_plain), (clipped, opt_clipped)):
        for name, p in model.named_parameters():
            p.grad = grads[name].clone()
        opt.step()
    assert not torch.allclose(plain.q_proj.weight, clipped.q_proj.weight)
    assert not torch.allclose(plain.k_proj.weight, clipped.k_proj.weight)
    assert torch.allclose(plain.v_proj.weight, clipped.v_proj.weight)
    assert (
        _weight_logit_scale(clipped.q_proj.weight, clipped.k_proj.weight, HEADS) <= 5.0001
    ).all()


def test_k3_config_defaults_are_off_and_validate_inputs() -> None:
    cfg = K3Config()
    assert cfg.num_heads == 0 and cfg.qk_clip is False and cfg.qk_clip_tau == TAU
    assert cfg.qkv_suffixes == ("q_proj", "k_proj", "v_proj")
    with pytest.raises(ValueError, match="tau"):
        K3Config(qk_clip_tau=0)
    with pytest.raises(ValueError, match="num_heads"):
        K3Config(qk_clip=True)  # clip needs heads
    with pytest.raises(ValueError, match="named"):
        MuonPlus(_Attn().parameters(), k3=K3Config(num_heads=HEADS, qk_clip=True))
