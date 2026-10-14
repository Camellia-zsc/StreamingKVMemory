"""Fidelity metrics: RAM, AORE, logit KL/JS.

All attention here is computed in **eager** mode with ``output_attentions`` so
we get true softmax probabilities. This path is for FIDELITY PROFILING only;
it must never be used for wall-clock measurements. The deployment path keeps
its original attention implementation.
"""

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------- #
# small stats helpers
# --------------------------------------------------------------------- #
def _percentile(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    t = torch.tensor(sorted(values), dtype=torch.float64)
    return float(torch.quantile(t, q / 100.0).item())


def summarize(values: Sequence[float]) -> Dict[str, Optional[float]]:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return {
            "mean": None,
            "median": None,
            "p10": None,
            "p90": None,
            "p95": None,
            "min": None,
            "max": None,
            "n": 0,
        }
    return {
        "mean": sum(vals) / len(vals),
        "median": _percentile(vals, 50.0),
        "p10": _percentile(vals, 10.0),
        "p90": _percentile(vals, 90.0),
        "p95": _percentile(vals, 95.0),
        "min": min(vals),
        "max": max(vals),
        "n": len(vals),
    }


# --------------------------------------------------------------------- #
# RAM
# --------------------------------------------------------------------- #
def ram_per_head(
    attn_probs: torch.Tensor,
    kept_mask: torch.Tensor,
    norm_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Retained Attention Mass per attention head (region-normalised).

    ``attn_probs`` [num_heads, kv_len] (fp32, from the FULL-KV distribution).
    ``kept_mask``  [kv_len] bool over the same key axis.
    ``norm_mask``  [kv_len] bool delimiting the evaluated KV region
                   (defaults to the whole sequence). The denominator is the
                   attention mass falling inside the region, so transient
                   text KV never leaks into the RAM denominator.
    Returns [num_heads].
    """

    probs = attn_probs.float()
    kept = kept_mask.to(probs.device).bool().view(1, -1)
    kept_mass = (probs * kept).sum(dim=-1)
    if norm_mask is None:
        denom = probs.sum(dim=-1)
    else:
        region = norm_mask.to(probs.device).bool().view(1, -1)
        denom = (probs * region).sum(dim=-1)
    # Some heads assign a positive but extremely small mass to video tokens
    # (e.g. 1e-28). An absolute epsilon would turn even complete retention
    # into RAM ~= 0. Normalise every positive mass by its actual denominator;
    # zero-mass heads are undefined and excluded from metric aggregation.
    safe_denom = torch.where(denom > 0, denom, torch.ones_like(denom))
    ratio = kept_mass / safe_denom
    return torch.where(denom > 0, ratio, torch.full_like(ratio, float("nan")))


# --------------------------------------------------------------------- #
# AORE
# --------------------------------------------------------------------- #
def aore(o_comp: torch.Tensor, o_full: torch.Tensor, eps: float = 1e-8) -> float:
    """Attention Output Relative Error, computed in float32."""

    oc = o_comp.float()
    of = o_full.float()
    num = torch.linalg.norm((oc - of).reshape(-1))
    den = torch.linalg.norm(of.reshape(-1)) + eps
    return float((num / den).item())


def region_context(
    attn_probs: torch.Tensor,
    value_states: torch.Tensor,
    region_start: int,
    region_end: int,
    num_query_heads: int,
) -> torch.Tensor:
    """Per-head attention context ``A @ V`` restricted to a KV sub-region.

    ``attn_probs``   [num_heads, kv_len] fp32, the anchor row.
    ``value_states`` [num_kv_heads, kv_len, head_dim].
    Region slicing is ``[region_start, region_end)`` on the key axis.
    Returns [num_heads, head_dim] in fp32.
    """

    probs = attn_probs.float()
    v = value_states.float()
    num_heads = probs.shape[0]
    num_kv_heads = v.shape[0]
    if num_kv_heads == 0:
        return torch.zeros(
            (num_heads, v.shape[-1]), dtype=torch.float32, device=v.device
        )
    group = max(1, num_heads // num_kv_heads)

    probs_region = probs[:, region_start:region_end]  # [H, R]
    v_region = v[:, region_start:region_end, :]       # [KVH, R, D]
    out = torch.zeros(
        (num_heads, v_region.shape[-1]),
        dtype=torch.float32,
        device=v.device,
    )
    for h in range(num_heads):
        kvh = min(h // group, num_kv_heads - 1)
        out[h] = torch.matmul(probs_region[h], v_region[kvh])
    return out


# --------------------------------------------------------------------- #
# Logit KL / JS
# --------------------------------------------------------------------- #
def kl_divergence(logits_full: torch.Tensor, logits_comp: torch.Tensor) -> float:
    """Forward KL( full || comp ), fp32."""

    zf = logits_full.float().reshape(-1)
    zc = logits_comp.float().reshape(-1)
    logp_full = F.log_softmax(zf, dim=-1)
    logp_comp = F.log_softmax(zc, dim=-1)
    logp_full, logp_comp = _align(logp_full, logp_comp)
    p_full = logp_full.exp()
    kl = (p_full * (logp_full - logp_comp)).sum()
    return float(kl.item())


def js_divergence(logits_full: torch.Tensor, logits_comp: torch.Tensor) -> float:
    """Jensen-Shannon divergence (base e), fp32."""

    zf = logits_full.float().reshape(-1)
    zc = logits_comp.float().reshape(-1)
    logp_full = F.log_softmax(zf, dim=-1)
    logp_comp = F.log_softmax(zc, dim=-1)
    logp_full, logp_comp = _align(logp_full, logp_comp)
    log_m = torch.logaddexp(logp_full, logp_comp) - torch.log(
        torch.tensor(2.0, device=logp_full.device, dtype=logp_full.dtype)
    )
    p_full = logp_full.exp()
    p_comp = logp_comp.exp()
    js = 0.5 * (
        (p_full * (logp_full - log_m)).sum()
        + (p_comp * (logp_comp - log_m)).sum()
    )
    return float(js.item())


def _align(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if a.shape == b.shape:
        return a, b
    n = min(a.numel(), b.numel())
    return a.reshape(-1)[:n], b.reshape(-1)[:n]


# --------------------------------------------------------------------- #
# Eager attention capture at the anchor
# --------------------------------------------------------------------- #
@dataclass
class LayerAttention:
    layer_idx: int
    attn_row: torch.Tensor       # [num_heads, kv_len] fp32
    value_states: torch.Tensor   # [num_kv_heads, kv_len, head_dim] fp32
    num_heads: int
    num_kv_heads: int


@torch.inference_mode()
def capture_anchor_attention(
    qa_model,
    input_ids: torch.Tensor,
    per_layer_offsets,
    past_key_values,
    num_layers: Optional[int] = None,
) -> Tuple[List[LayerAttention], torch.Tensor]:
    """Re-run the query prefill in eager mode over ``past_key_values``.

    ``input_ids`` [1, q_len] are the *exact* query prefill ids. The forward is
    identical to the deployment prefill except (a) eager attention and (b)
    ``past_key_values`` passed explicitly. Because of the causal mask, the last
    position of this forward equals the deployment anchor.

    Returns (per-layer attention rows for the LAST query token + value states,
    anchor logits).
    """

    device = qa_model.device
    lm = qa_model.language_model
    prev_impl = getattr(lm.config, "_attn_implementation", None)
    lm.config._attn_implementation = "eager"
    try:
        ids = input_ids.to(device)
        q_len = int(ids.shape[1])
        batch = int(ids.shape[0])

        offsets = list(per_layer_offsets)
        n_layers = num_layers or len(offsets)

        # Per-layer M-RoPE positions (SDPA HERMES injects these via hooks).
        qa_model._layer_position_ids.clear()
        for layer_idx in range(n_layers):
            offset = offsets[layer_idx] if layer_idx < len(offsets) else offsets[0]
            qa_model._layer_position_ids[layer_idx] = torch.arange(
                int(offset), int(offset) + q_len, device=device, dtype=torch.float32
            ).view(1, 1, q_len).expand(3, batch, q_len).clone()

        position_ids = torch.arange(
            int(offsets[0]), int(offsets[0]) + q_len, device=device, dtype=torch.float32
        ).view(1, 1, q_len).expand(3, batch, q_len).clone()

        out = lm(
            input_ids=ids,
            use_cache=False,
            past_key_values=past_key_values,
            output_attentions=True,
            position_ids=position_ids,
        )
        logits = qa_model.lm_head(out.last_hidden_state)  # [1,q_len,V]
        attentions = out.attentions

        # value states come from the cache (post-prefill snapshot)
        layers: List[LayerAttention] = []
        n = num_layers or len(attentions)
        for layer_idx in range(n):
            attn = attentions[layer_idx]
            # [B, H, q_len, kv_len] -> [H, kv_len] for the LAST query token
            row = attn[0, :, -1, :].float()
            v = past_key_values[layer_idx][1]
            v = v[0].float()  # [num_kv_heads, kv_len, head_dim]
            layers.append(
                LayerAttention(
                    layer_idx=layer_idx,
                    attn_row=row,
                    value_states=v,
                    num_heads=int(attn.shape[1]),
                    num_kv_heads=int(v.shape[0]),
                )
            )
        return layers, logits[:, -1, :].float()  # [1, V] anchor logits
    finally:
        try:
            qa_model._layer_position_ids.clear()
        except AttributeError:
            pass
        if prev_impl is not None:
            lm.config._attn_implementation = prev_impl
        else:  # pragma: no cover - defensive
            try:
                del lm.config._attn_implementation
            except AttributeError:
                pass
