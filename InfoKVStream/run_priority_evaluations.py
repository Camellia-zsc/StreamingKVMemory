"""Finish RT20 diagnostics/figures, then resume the existing seven-budget OVO run."""
from __future__ import annotations

import fcntl
import json
from pathlib import Path
import os
import subprocess
import sys

from InfoKVStream.run_ovo import ROOT, atomic_json, now


def main():
    rt = ROOT / "outputs/InfoKVStream/streamingbench_rt20_20261013"
    ovo = ROOT / "outputs/InfoKVStream/ovo_20261013"
    rt.mkdir(parents=True, exist_ok=True)
    lock = (rt / "priority_queue.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("Evaluation queue already running; see priority_queue.json.")
        return
    commands = [
        ("streamingbench_rt", [sys.executable, "-u", "-m", "InfoKVStream.run_streamingbench_rt", "all", "--output", str(rt)]),
        ("plots", [sys.executable, "-u", "-m", "InfoKVStream.analyze_streamingbench_rt", "--root", str(rt)]),
        ("ovobench_seven_budgets", [sys.executable, "-u", "-m", "InfoKVStream.run_ovo", "all", "--output", str(ovo)]),
    ]
    for stage, command in commands:
        if stage != "streamingbench_rt":
            if json.loads((rt / "status.json").read_text())["status"] != "complete":
                raise RuntimeError("RT diagnostics are not complete; OVO remains paused")
        atomic_json(rt / "priority_queue.json", {"phase": stage, "status": "running", "pid": os.getpid(),
                    "command": command, "updated_at": now()})
        target = ovo if stage == "ovobench_seven_budgets" else rt
        (target / "logs").mkdir(exist_ok=True)
        with (target / "logs" / ("resumed_after_rt.log" if target == ovo else f"queue_{stage}.log")).open("a") as log:
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            atomic_json(rt / "priority_child.json", {"pid": child.pid, "phase": stage, "started_at": now()})
            rc = child.wait()
        if rc:
            atomic_json(rt / "priority_queue.json", {"phase": stage, "status": "failed", "exit_code": rc, "updated_at": now()})
            raise RuntimeError(f"Evaluation stage {stage} failed")
    atomic_json(rt / "priority_queue.json", {"status": "complete", "updated_at": now()})


if __name__ == "__main__":
    main()
