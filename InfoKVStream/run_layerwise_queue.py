"""Durable RT20 layerwise queue, with source contracts and per-sample resume."""
from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

from InfoKVStream.run_ovo import ROOT, atomic_json, now
from InfoKVStream.run_docstyle_diagnostics import DEFAULT_ROOT, BUDGETS, prepare


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=DEFAULT_ROOT)
    p.add_argument('--gpu', default='1')
    p.add_argument('--wait-pid', type=int)
    args = p.parse_args()
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / 'layerwise_queue.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    prepare(root)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, OMP_NUM_THREADS='4',
               OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTHONPATH=str(ROOT))
    if args.wait_pid:
        while (Path('/proc') / str(args.wait_pid)).exists():
            atomic_json(root / 'layerwise_queue_status.json',
                        {'status': 'waiting', 'wait_pid': args.wait_pid,
                         'pid': os.getpid(), 'updated_at': now()})
            time.sleep(10)
    commands = [('trajectory_hermes', ['trajectory', '--method', 'hermes']),
                ('trajectory_infokv_stream', ['trajectory', '--method', 'infokv_stream']),
                ('trajectory_full_cache', ['trajectory', '--method', 'full_cache'])]
    commands += [(f'stress_{b}_infokv_stream', ['stress', '--method', 'infokv_stream',
                                             '--budget', str(b)]) for b in BUDGETS]
    logs = root / 'logs'
    logs.mkdir(exist_ok=True)
    try:
        # Complete the three-way latency comparison early; it does not require
        # fidelity trajectories and should not include diagnostic probes.
        with (logs / 'compression_timing.log').open('a') as log:
            command = [sys.executable, '-u', '-m', 'InfoKVStream.run_compression_timing']
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                     stderr=subprocess.STDOUT)
            atomic_json(root / 'layerwise_queue_status.json',
                        {'status': 'running', 'phase': 'compression_timing', 'pid': os.getpid(),
                         'worker_pid': child.pid, 'updated_at': now()})
            if child.wait():
                raise RuntimeError(f'Compression timing failed; inspect {logs}/compression_timing.log')
        for label, arguments in commands:
            command = [sys.executable, '-u', '-m', 'InfoKVStream.run_docstyle_diagnostics',
                       *arguments, '--output', str(root)]
            with (logs / f'{label}_queue.log').open('a') as log:
                child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                         stderr=subprocess.STDOUT)
                atomic_json(root / 'layerwise_queue_status.json',
                            {'status': 'running', 'phase': label, 'pid': os.getpid(),
                             'worker_pid': child.pid, 'command': command, 'updated_at': now()})
                code = child.wait()
                if code:
                    raise RuntimeError(f'{label} exited {code}; inspect {logs}/{label}_queue.log')
            subprocess.run([sys.executable, '-m', 'InfoKVStream.analyze_docstyle_diagnostics',
                            '--root', str(root)], cwd=ROOT, env=env, check=True)
        atomic_json(root / 'layerwise_queue_status.json',
                    {'status': 'complete', 'pid': os.getpid(), 'updated_at': now()})
    except BaseException as e:
        atomic_json(root / 'layerwise_queue_status.json',
                    {'status': 'failed', 'error': str(e), 'pid': os.getpid(), 'updated_at': now()})
        raise


if __name__ == '__main__':
    main()
