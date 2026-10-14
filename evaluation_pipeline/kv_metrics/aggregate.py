"""Paired metric computation and summary aggregation."""

import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch

from .config import KVMetricConfig
from .fidelity import (
    aore,
    js_divergence,
    kl_divergence,
    ram_per_head,
    region_context,
    summarize,
)
from .hooks import AnchorArtifact
from .temporal import temporal_ram


def compute_paired_metrics(
    compressed: AnchorArtifact,
    full: AnchorArtifact,
    config: KVMetricConfig,
) -> dict:
    """Compute RAM / AORE / JS / KL / Temporal RAM for one paired event."""

    eps = config.eps
    record = {
        "event_id": compressed.event_id,
        "video_id": compressed.video_id,
        "sample_id": compressed.sample_id or compressed.video_id,
        "query_id": compressed.query_id,
        "query_time": compressed.query_time,
        "stream_time": compressed.stream_time,
        "compaction_index": compressed.compaction_index,
        "compaction_trigger_count_at_query": compressed.compaction_index,
        "effective_compaction_count_at_query": compressed.effective_compaction_count,
        "last_compaction_wallclock": compressed.last_compaction_wallclock,
        "full_kv_bytes": full.kv_bytes,
        "compressed_kv_bytes": compressed.kv_bytes,
        "full_kv_tokens": full.kv_tokens,
        "kept_kv_tokens": compressed.kv_tokens,
        "num_attention_heads": full.num_heads,
        "num_key_value_heads": full.num_kv_heads,
    }
    record["compression_ratio"] = (
        full.kv_bytes / compressed.kv_bytes
        if compressed.kv_bytes
        else None
    )
    record["retained_size"] = (
        compressed.kv_bytes / full.kv_bytes if full.kv_bytes else None
    )

    # ---------------- logits: JS / KL ----------------
    if compressed.logits is not None and full.logits is not None:
        record["js"] = js_divergence(full.logits, compressed.logits)
        record["kl"] = kl_divergence(full.logits, compressed.logits)
    else:
        record["js"] = None
        record["kl"] = None

    # ---------------- RAM / AORE ----------------
    ram_per_layer = {}
    aore_per_layer = {}
    ram_all_heads: List[float] = []
    aore_all_heads: List[float] = []
    ram_all_layers_mean: List[float] = []
    aore_all_layers_mean: List[float] = []
    per_head_rows = []

    has_attention = bool(full.layers) and bool(compressed.layers)
    if has_attention:
        for layer_idx in sorted(full.layers.keys()):
            if layer_idx not in compressed.layers:
                continue
            full_att = full.layers[layer_idx]
            comp_att = compressed.layers[layer_idx]
            full_region = full.visual_regions.get(layer_idx)
            comp_region = compressed.visual_regions.get(layer_idx)
            if full_region is None or comp_region is None:
                continue

            retained = compressed.retained_uids.get(layer_idx, set())
            full_probs = full_att.attn_row  # [H, kv_len_full]
            kv_len_full = int(full_probs.shape[1])
            mask = torch.zeros(kv_len_full, dtype=torch.bool, device=full_probs.device)
            region_mask = torch.zeros(
                kv_len_full, dtype=torch.bool, device=full_probs.device
            )
            for offset, uid in enumerate(full_region.uids):
                pos = full_region.start + offset
                if pos >= kv_len_full:
                    continue
                region_mask[pos] = True
                if uid.uid in retained:
                    mask[pos] = True

            # Region-normalised RAM so that "no compression" -> RAM == 1.
            ram_h = ram_per_head(full_probs, mask, region_mask).cpu()  # [H]
            ram_raw_h = ram_per_head(full_probs, mask).cpu()  # unnormalised
            ram_per_layer[layer_idx] = summarize(ram_h.tolist())
            ram_per_layer[layer_idx]["undefined_heads"] = int((~torch.isfinite(ram_h)).sum().item())
            ram_per_layer[layer_idx]["raw_mean"] = float(ram_raw_h.mean().item())
            ram_all_heads.extend(ram_h.tolist())
            ram_all_layers_mean.append(float(ram_h.nanmean().item()))

            # AORE on the video-region attention contribution
            o_full = region_context(
                full_probs,
                full_att.value_states,
                full_region.start,
                min(full_region.end, int(full_att.value_states.shape[1])),
                full_att.num_heads,
            )
            o_comp = region_context(
                comp_att.attn_row,
                comp_att.value_states,
                comp_region.start,
                min(comp_region.end, int(comp_att.value_states.shape[1])),
                comp_att.num_heads,
            )
            # per-head AORE
            n_heads = min(o_full.shape[0], o_comp.shape[0])
            head_aore = []
            for h in range(n_heads):
                head_aore.append(aore(o_comp[h], o_full[h], eps))
            aore_per_layer[layer_idx] = summarize(head_aore)
            aore_all_heads.extend(head_aore)
            aore_all_layers_mean.append(
                sum(head_aore) / len(head_aore) if head_aore else None
            )

            if config.save_per_head:
                per_head_rows.append(
                    {
                        "event_id": compressed.event_id,
                        "layer_id": layer_idx,
                        "ram_per_head": [float(v) if torch.isfinite(v) else None for v in ram_h],
                        "aore_per_head": head_aore,
                    }
                )

        record["ram_mean"] = summarize(ram_all_layers_mean)["mean"]
        record["ram_valid_heads"] = int(torch.isfinite(torch.tensor(ram_all_heads)).sum().item())
        record["ram_undefined_heads"] = len(ram_all_heads) - record["ram_valid_heads"]
        record["ram_median"] = summarize(ram_all_heads)["median"]
        record["ram_p10"] = summarize(ram_all_heads)["p10"]
        record["ram_per_layer"] = ram_per_layer
        record["aore_mean"] = summarize(aore_all_layers_mean)["mean"]
        record["aore_median"] = summarize(aore_all_heads)["median"]
        record["aore_p90"] = summarize(aore_all_heads)["p90"]
        record["aore_p95"] = summarize(aore_all_heads)["p95"]
        record["aore_per_layer"] = aore_per_layer
        record["aore_level"] = "per_head_pre_o_proj"
        if config.save_per_head:
            record["ram_per_head"] = [v if torch.isfinite(torch.tensor(v)) else None for v in ram_all_heads]
            record["aore_per_head"] = aore_all_heads
            record["per_head_rows"] = per_head_rows

        # ---------------- Temporal RAM ----------------
        temporal = _temporal_from_pair(compressed, full, config)
        record.update(temporal)
    else:
        record["ram_mean"] = None
        record["aore_mean"] = None
        record["aore_level"] = "unavailable"
        record["temporal_ram"] = None

    return record


def _temporal_from_pair(
    compressed: AnchorArtifact,
    full: AnchorArtifact,
    config: KVMetricConfig,
) -> dict:
    """Temporal RAM aggregated over layers using per-layer UID timestamps."""

    query_time = full.query_time
    if query_time is None:
        return {
            "temporal_ram": None,
            "temporal_full_attention_mass": None,
            "temporal_num_full_tokens": None,
            "temporal_num_kept_tokens": None,
            "temporal_ram_supported": False,
        }

    n_bins = len(config.temporal_bins) - 1
    from .temporal import bin_index, bin_label

    mass_sum = [0.0] * n_bins
    kept_sum = [0.0] * n_bins
    n_full = [0] * n_bins
    n_kept = [0] * n_bins

    for layer_idx in sorted(full.layers.keys()):
        full_att = full.layers[layer_idx]
        region = full.visual_regions.get(layer_idx)
        if region is None:
            continue
        retained = compressed.retained_uids.get(layer_idx, set())
        probs = full_att.attn_row  # [H, kv_len]
        # average over heads to get a per-position mass
        mass = probs.mean(dim=0).float().cpu()  # [kv_len]
        for offset, uid in enumerate(region.uids):
            pos = region.start + offset
            if pos >= mass.numel():
                continue
            ts = uid.frame_timestamp_sec
            if ts is None:
                continue
            idx = bin_index(float(query_time) - float(ts), config.temporal_bins)
            if idx < 0:
                continue
            m = float(mass[pos].item())
            mass_sum[idx] += m
            n_full[idx] += 1
            if uid.uid in retained:
                kept_sum[idx] += m
                n_kept[idx] += 1

    temporal = {}
    full_map = {}
    n_full_map = {}
    n_kept_map = {}
    for idx in range(n_bins):
        label = bin_label(config.temporal_bins, idx)
        temporal[label] = (
            kept_sum[idx] / mass_sum[idx]
            if mass_sum[idx] > 0
            else None
        )
        full_map[label] = mass_sum[idx]
        n_full_map[label] = n_full[idx]
        n_kept_map[label] = n_kept[idx]
    return {
        "temporal_ram": temporal,
        "temporal_full_attention_mass": full_map,
        "temporal_num_full_tokens": n_full_map,
        "temporal_num_kept_tokens": n_kept_map,
        "temporal_ram_supported": True,
        # No reliable evidence timestamp in current benchmarks:
        "temporal_aore_supported": False,
        "temporal_js_supported": False,
    }


class MetricAggregator:
    """Writes per-event jsonl and a compact summary.json / summary.csv."""

    def __init__(self, config: KVMetricConfig):
        self.config = config
        self.root = Path(config.output_dir) / config.run_name
        self.records: List[dict] = []
        self.probe_records: List[dict] = []

    def paths(self) -> dict:
        return {
            "root": self.root,
            "compressed": self.root / "compressed",
            "full": self.root / "full",
            "paired": self.root / "paired",
            "summary": self.root / "summary.json",
            "summary_csv": self.root / "summary.csv",
            "config": self.root / "config.json",
        }

    def _ensure_dirs(self) -> dict:
        paths = self.paths()
        for key in ("compressed", "full", "paired"):
            paths[key].mkdir(parents=True, exist_ok=True)
        self.root.mkdir(parents=True, exist_ok=True)
        return paths

    def add_paired(self, record: dict) -> None:
        self.records.append(record)

    def add_probe(self, record: dict) -> None:
        self.probe_records.append(record)

    def write_probe_sidecars(self, probes, kind: str) -> None:
        """Write ``raw/detailed_trajectory/<sample_id>/compaction_probes.jsonl``.

        The probe *input* sidecar only records metadata + the raw per-layer
        summary; heavy tensors stay in memory and are paired immediately.
        """

        if not probes:
            return
        base = self.root / "detailed_trajectory_raw" / kind
        by_sample: Dict[str, List[dict]] = {}
        for probe in probes:
            record = {
                "sample_id": probe.sample_id,
                "video_id": probe.video_id,
                "query_id": probe.query_id,
                "query_time": probe.query_time,
                "probe_k": probe.probe_k,
                "stream_time": probe.stream_time,
                "kv_tokens": probe.kv_tokens,
                "kv_bytes": probe.kv_bytes,
                "compaction_index": probe.compaction_index,
                "effective_compaction_count": probe.effective_compaction_count,
                "num_layers": probe.num_layers,
                "num_heads": probe.num_heads,
                "num_kv_heads": probe.num_kv_heads,
                "query_input_ids": probe.query_input_ids,
                "probe_query_hash": probe.probe_query_hash,
            }
            by_sample.setdefault(str(probe.video_id), []).append(record)
        for sample_id, records in by_sample.items():
            out = base / _safe(sample_id) / "compaction_probes.jsonl"
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w", encoding="utf-8") as fh:
                for record in records:
                    fh.write(json.dumps(_jsonable(record), ensure_ascii=False) + "\n")

    def write_pass_sidecar(self, events, kind: str) -> None:
        """Write per-video compressed/ or full/ sidecars (no heavy tensors)."""

        paths = self._ensure_dirs()
        by_video: Dict[str, List[dict]] = {}
        for artifact in events:
            record = {
                "event_id": artifact.event_id,
                "video_id": artifact.video_id,
                "sample_id": artifact.sample_id or artifact.video_id,
                "query_id": artifact.query_id,
                "query_time": artifact.query_time,
                "stream_time": artifact.stream_time,
                "compaction_index": artifact.compaction_index,
                "kv_bytes": artifact.kv_bytes,
                "kv_tokens": artifact.kv_tokens,
                "num_layers": artifact.num_layers,
                "num_heads": artifact.num_heads,
                "num_kv_heads": artifact.num_kv_heads,
                "summary_token_counts": artifact.summary_parent_counts,
                "last_compaction_wallclock": artifact.last_compaction_wallclock,
            }
            by_video.setdefault(str(artifact.video_id), []).append(record)
        for video_id, records in by_video.items():
            out = paths[kind] / f"{_safe(video_id)}.jsonl"
            with open(out, "w", encoding="utf-8") as fh:
                for record in records:
                    fh.write(json.dumps(_jsonable(record), ensure_ascii=False) + "\n")

    def write(self, task_by_event: Optional[Dict[str, dict]] = None) -> dict:
        paths = self._ensure_dirs()
        with open(paths["config"], "w", encoding="utf-8") as fh:
            json.dump(self.config.as_dict(), fh, indent=2, ensure_ascii=False)

        # group paired records by video for sidecar jsonl
        by_video: Dict[str, List[dict]] = {}
        for record in self.records:
            by_video.setdefault(str(record.get("video_id")), []).append(record)
        for video_id, records in by_video.items():
            out = paths["paired"] / f"{_safe(video_id)}.jsonl"
            with open(out, "w", encoding="utf-8") as fh:
                for record in records:
                    fh.write(json.dumps(_jsonable(record), ensure_ascii=False) + "\n")

        probes_by_pair: Dict[tuple, List[dict]] = {}
        for record in self.probe_records:
            key = (str(record.get("sample_id")), str(record.get("query_id")))
            probes_by_pair.setdefault(key, []).append(record)
        for (sample_id, query_id), records in probes_by_pair.items():
            out = (
                self.root / "detailed_trajectory" / _safe(sample_id)
                / f"probe_metrics_{_safe(query_id)}.jsonl"
            )
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w", encoding="utf-8") as fh:
                for record in records:
                    fh.write(json.dumps(_jsonable(record), ensure_ascii=False) + "\n")

        summary = self._summary(task_by_event or {})
        with open(paths["summary"], "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, ensure_ascii=False)
        _write_csv(paths["summary_csv"], summary)
        return summary

    def write_compactions(self, records: List[dict]) -> None:
        if not records:
            return
        paths = self._ensure_dirs()
        out = self.root / "compactions.jsonl"
        with open(out, "w", encoding="utf-8") as fh:
            for record in records:
                fh.write(json.dumps(_jsonable(record), ensure_ascii=False) + "\n")

    def write_runtime_breakdown(self, stats, summary) -> None:
        """Write tables/runtime_breakdown.csv + runtime_summary.json."""

        if not stats:
            return
        tables = self.root / "tables"
        tables.mkdir(parents=True, exist_ok=True)
        fields = [
            "component",
            "total_sec",
            "fraction_of_model_compute",
            "call_count",
            "mean_sec",
            "p50_sec",
            "p95_sec",
            "stream_time_overhead_ratio",
        ]
        with open(tables / "runtime_breakdown.csv", "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            for row in stats:
                writer.writerow({k: row.get(k) for k in fields})
        if summary:
            (self.root / "runtime_summary.json").write_text(
                json.dumps(_jsonable(summary), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

    def _summary(self, task_by_event: Dict[str, dict]) -> dict:
        ram = [r["ram_mean"] for r in self.records if r.get("ram_mean") is not None]
        aore_vals = [r["aore_mean"] for r in self.records if r.get("aore_mean") is not None]
        js = [r["js"] for r in self.records if r.get("js") is not None]
        kl = [r["kl"] for r in self.records if r.get("kl") is not None]
        cr = [
            r["compression_ratio"]
            for r in self.records
            if r.get("compression_ratio") is not None
        ]
        walls = [
            r["last_compaction_wallclock"]
            for r in self.records
            if r.get("last_compaction_wallclock") is not None
        ]

        temporal_keys = set()
        for r in self.records:
            if isinstance(r.get("temporal_ram"), dict):
                temporal_keys.update(r["temporal_ram"].keys())
        temporal_mean = {}
        for key in sorted(temporal_keys):
            values = [
                r["temporal_ram"][key]
                for r in self.records
                if isinstance(r.get("temporal_ram"), dict)
                and r["temporal_ram"].get(key) is not None
            ]
            temporal_mean[key] = sum(values) / len(values) if values else None

        summary = {
            "config": self.config.as_dict(),
            "num_events": len(self.records),
            "ram_mean": _mean(ram),
            "ram_median": _median(ram),
            "ram_p10": _quantile(ram, 0.10),
            "aore_mean": _mean(aore_vals),
            "aore_median": _median(aore_vals),
            "aore_p90": _quantile(aore_vals, 0.90),
            "aore_p95": _quantile(aore_vals, 0.95),
            "js_mean": _mean(js),
            "js_median": _median(js),
            "js_p90": _quantile(js, 0.90),
            "kl_mean": _mean(kl),
            "kl_median": _median(kl),
            "kl_p90": _quantile(kl, 0.90),
            "compression_ratio_mean": _mean(cr),
            "compression_ratio_max": max(cr) if cr else None,
            "last_compaction_wallclock_mean": _mean(walls),
            "temporal_ram_mean": temporal_mean,
            "temporal_aore_supported": False,
            "temporal_js_supported": False,
        }
        # task correlation (reuse existing evaluator results, never recompute)
        if task_by_event:
            summary["task"] = _task_correlation(self.records, task_by_event)
        return summary


def _task_correlation(records, task_by_event: dict) -> dict:
    rows = []
    for record in records:
        task = task_by_event.get(record["event_id"])
        if not task:
            continue
        rows.append(
            {
                "event_id": record["event_id"],
                "compression_ratio": record.get("compression_ratio"),
                "ram_mean": record.get("ram_mean"),
                "aore_mean": record.get("aore_mean"),
                "js": record.get("js"),
                **task,
            }
        )
    return {"num_events": len(rows), "per_event": rows}


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    if isinstance(obj, float) and obj != obj:
        return None
    return obj


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(name))


def _mean(values: Sequence[float]) -> Optional[float]:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def _median(values: Sequence[float]) -> Optional[float]:
    return _quantile(values, 0.5)


def _quantile(values: Sequence[float], q: float) -> Optional[float]:
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    t = torch.tensor(values, dtype=torch.float64)
    return float(torch.quantile(t, q).item())


def _write_csv(path: Path, summary: dict) -> None:
    flat = {}
    for key, value in summary.items():
        if isinstance(value, dict):
            for sub_key, sub_val in value.items():
                if isinstance(sub_val, (int, float, str)) or sub_val is None:
                    flat[f"{key}.{sub_key}"] = sub_val
        elif isinstance(value, (int, float, str)) or value is None:
            flat[key] = value
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["metric", "value"])
        for key, value in flat.items():
            writer.writerow([key, value])
