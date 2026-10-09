"""Loads existing kv_metrics sidecar outputs.

The analyzer never recomputes RAM/AORE/JS/KL: it only reads
``paired/*.jsonl`` and ``compactions.jsonl`` produced by the runner.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


def read_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def read_json(path: Path, default=None):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def video_id_from_event(event_id: str) -> str:
    return str(event_id).split("|", 1)[0]


@dataclass
class RunData:
    name: str
    directory: Path
    config: dict = field(default_factory=dict)
    summary: dict = field(default_factory=dict)
    events: List[dict] = field(default_factory=list)
    compactions: List[dict] = field(default_factory=list)

    @property
    def is_full(self) -> bool:
        memory = self.config.get("memory", {}) or {}
        return self.config.get("mode") == "full" or memory.get("method") == "full_cache"

    def effective_compression(self) -> bool:
        return len(self.compactions) > 0

    def sample_ids(self) -> List[str]:
        return sorted({str(e.get("sample_id")) for e in self.events})


def load_run(run_dir: Path, experiment_name: Optional[str] = None) -> RunData:
    run_dir = Path(run_dir)
    name = experiment_name or run_dir.name
    config = read_json(run_dir / "config.json", {}) or {}
    summary = read_json(run_dir / "summary.json", {}) or {}

    events: List[dict] = []
    paired_dir = run_dir / "paired"
    if paired_dir.is_dir():
        for path in sorted(paired_dir.glob("*.jsonl")):
            for record in read_jsonl(path):
                # event_id is authoritative for the video instance
                record["video_id"] = video_id_from_event(record.get("event_id", ""))
                if not record.get("sample_id"):
                    record["sample_id"] = record["video_id"]
                record["experiment"] = name
                events.append(record)

    compactions = read_jsonl(run_dir / "compactions.jsonl")
    for record in compactions:
        record.setdefault("video_id", video_id_from_event(record.get("event_id", "")))
        record.setdefault("sample_id", record.get("video_id"))
        record["experiment"] = name

    return RunData(
        name=name,
        directory=run_dir,
        config=config,
        summary=summary,
        events=events,
        compactions=compactions,
    )


@dataclass
class ExperimentData:
    directory: Path
    name: str
    runs: Dict[str, RunData] = field(default_factory=dict)
    selected_samples: List[dict] = field(default_factory=list)
    selection_meta: dict = field(default_factory=dict)

    def all_events(self) -> List[dict]:
        out = []
        for run in self.runs.values():
            out.extend(run.events)
        return out

    def all_compactions(self) -> List[dict]:
        out = []
        for run in self.runs.values():
            out.extend(run.compactions)
        return out

    def ordered_runs(self) -> List[RunData]:
        return [self.runs[k] for k in sorted(self.runs)]


def load_experiment(experiment_dir: Path) -> ExperimentData:
    experiment_dir = Path(experiment_dir)
    selection = read_json(experiment_dir / "selected_samples.json", []) or []
    if isinstance(selection, dict):
        selected = selection.get("samples", [])
        selection_meta = selection.get("meta", {})
    else:
        selected = selection
        selection_meta = {}
    runs: Dict[str, RunData] = {}
    raw_dir = experiment_dir / "raw"
    if raw_dir.is_dir():
        for child in sorted(raw_dir.iterdir()):
            if not child.is_dir():
                continue
            runs[child.name] = load_run(child, experiment_name=child.name)
    return ExperimentData(
        directory=experiment_dir,
        name=experiment_dir.name,
        runs=runs,
        selected_samples=selected,
        selection_meta=selection_meta,
    )


# --------------------------------------------------------------------- #
# derived helpers
# --------------------------------------------------------------------- #
def first_compaction_time(compactions: List[dict], sample_id: str) -> Optional[float]:
    times = [
        c.get("stream_time")
        for c in compactions
        if str(c.get("sample_id")) == str(sample_id)
        and c.get("stream_time") is not None
    ]
    return min(times) if times else None


def compactions_by_sample(compactions: List[dict]) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = {}
    for c in compactions:
        out.setdefault(str(c.get("sample_id")), []).append(c)
    for values in out.values():
        values.sort(key=lambda c: (c.get("stream_time") or 0.0, c.get("compaction_index") or 0))
    return out


def events_by_sample(events: List[dict]) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = {}
    for e in events:
        out.setdefault(str(e.get("sample_id")), []).append(e)
    for values in out.values():
        values.sort(key=lambda e: float(e.get("query_time") or 0.0))
    return out
