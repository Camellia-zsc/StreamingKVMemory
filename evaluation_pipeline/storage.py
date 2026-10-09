import csv
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List


def append_jsonl(path: Path, record: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl_recover(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    records = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            # An interrupted write can leave a partial final line.
            if line_number == len(lines):
                break
            raise ValueError(f"Invalid JSONL at {path}:{line_number}")
    return records


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def write_summary_csv(path: Path, metrics: Dict[str, Dict]) -> None:
    rows = []
    for benchmark, values in metrics.items():
        for subset, value in flatten_metrics(values):
            rows.append((benchmark, subset, value))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["benchmark", "subset", "metric", "value"])
        for benchmark, subset, value in rows:
            if isinstance(value, dict):
                for metric, metric_value in value.items():
                    writer.writerow([benchmark, subset, metric, metric_value])
            else:
                writer.writerow([benchmark, subset, "value", value])


def flatten_metrics(values: Dict, prefix: str = "") -> Iterable:
    for key, value in values.items():
        label = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            yield from flatten_metrics(value, label)
        else:
            yield label, value


def safe_float(value):
    return round(float(value), 6) if value is not None else None
