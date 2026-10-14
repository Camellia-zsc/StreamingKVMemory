"""Pause the existing OVO queue, run RT comparison diagnostics, then resume OVO."""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from InfoKVStream.run_ovo import ROOT, atomic_json, now
from InfoKVStream.run_docstyle_diagnostics import DEFAULT_ROOT, prepare, BUDGETS

OVO = ROOT / 'outputs/InfoKVStream/ovo_20261013'


def command_line(pid):
    try:
        return (Path('/proc') / str(pid) / 'cmdline').read_bytes().replace(b'\0', b' ').decode()
    except (FileNotFoundError, ProcessLookupError):
        return ''


def pause_ovo(output):
    state = json.loads((OVO/'status.json').read_text())
    if state['status'] == 'complete':
        return False
    supervisor = state.get('pid')
    worker = json.loads((OVO/'worker.json').read_text()).get('pid')
    if not supervisor or 'InfoKVStream.run_ovo all' not in command_line(supervisor):
        if state['status'] == 'running':
            raise RuntimeError('Cannot identify the active OVO supervisor; leave it untouched')
        return True
    # Stop supervision before ending the worker so no next job can start.
    os.kill(supervisor, signal.SIGSTOP)
    stat = (Path('/proc')/str(supervisor)/'stat').read_text().split()
    parent = int(stat[3])
    priority_parent = parent if 'InfoKVStream.run_priority_evaluations' in command_line(parent) else None
    if priority_parent:
        os.kill(priority_parent, signal.SIGSTOP)
    before = {str(p.relative_to(OVO)):json.loads(p.read_text()) for p in (OVO/'full').glob('*/*/progress.json')}
    try:
        if worker and 'InfoKVStream.run_ovo worker' in command_line(worker):
            os.kill(worker, signal.SIGINT)
            deadline = time.monotonic()+30
            while command_line(worker) and time.monotonic()<deadline:
                time.sleep(.5)
            if command_line(worker):
                os.kill(worker, signal.SIGTERM)
        # A partial current sample has no commit file; run_ovo resumes it safely.
        for pid in [supervisor, priority_parent]:
            if pid and command_line(pid):
                os.kill(pid, signal.SIGTERM)
                os.kill(pid, signal.SIGCONT)
        deadline = time.monotonic()+10
        while command_line(supervisor) and time.monotonic()<deadline:
            time.sleep(.2)
        if command_line(supervisor):
            raise RuntimeError('OVO supervisor did not exit; cannot acquire GPU for diagnostics')
    except BaseException:
        for pid in [supervisor, priority_parent]:
            if pid and command_line(pid):
                os.kill(pid, signal.SIGCONT)
        raise
    atomic_json(output/'ovo_pause_audit.json',{'paused_at':now(),'previous_status':state,
                'supervisor_pid':supervisor,'worker_pid':worker,'priority_parent_pid':priority_parent,
                'committed_progress':before,'reason':'user requested same-variable document-style diagnostics'})
    atomic_json(OVO/'status.json',{'status':'paused','reason':'StreamingBench RT document-style comparison',
                'diagnostic_output':str(output),'previous_job':state.get('job'),'updated_at':now()})
    return True


def resume_ovo(output):
    # Use the frozen output manifest and normal resume logic, preserving all
    # successfully committed questions and their compression event histories.
    manifest=json.loads((OVO/'manifest.json').read_text())
    gpu=manifest.get('cuda_visible_devices')
    if gpu is None:
        raise RuntimeError('OVO manifest has no CUDA_VISIBLE_DEVICES; cannot reproduce GPU placement')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',
             OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONPATH=str(ROOT))
    command=[manifest['python'],'-u','-m','InfoKVStream.run_ovo','all','--output',str(OVO)]
    with (OVO/'logs/resumed_after_docstyle.log').open('a') as log:
        child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    atomic_json(output/'ovo_resume.json',{'pid':child.pid,'command':command,
                'cuda_visible_devices':str(gpu),'resumed_at':now()})
    return child.pid


def main():
    output=DEFAULT_ROOT
    output.mkdir(parents=True,exist_ok=True)
    lock=(output/'queue.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    prepare(output)
    (output/'logs').mkdir(exist_ok=True)
    paused=False
    resume_pid=None
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
             MKL_NUM_THREADS='4',PYTHONPATH=str(ROOT))
    try:
        paused=pause_ovo(output)
        commands=[]
        # Validate a complete paired fixed-query video before the rest.
        for method in ['hermes','infokv_stream','full_cache']:
            commands.append((f'trajectory_smoke_{method}',[
                sys.executable,'-u','-m','InfoKVStream.run_docstyle_diagnostics','trajectory',
                '--method',method,'--limit','1','--output',str(output)]))
        for method in ['hermes','infokv_stream','full_cache']:
            commands.append((f'trajectory_{method}',[
                sys.executable,'-u','-m','InfoKVStream.run_docstyle_diagnostics','trajectory',
                '--method',method,'--output',str(output)]))
        for budget in BUDGETS:
            for method in ['hermes','infokv_stream']:
                commands.append((f'budget_{budget}_{method}',[
                    sys.executable,'-u','-m','InfoKVStream.run_docstyle_diagnostics','stress',
                    '--method',method,'--budget',str(budget),'--output',str(output)]))
        for label,command in commands:
            atomic_json(output/'status.json',{'status':'running','phase':label,'pid':os.getpid(),
                        'command':command,'updated_at':now()})
            with (output/'logs'/f'{label}.log').open('a') as log:
                child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                atomic_json(output/'worker.json',{'pid':child.pid,'phase':label,'started_at':now()})
                code=child.wait()
            if code:
                raise RuntimeError(f'{label} failed (exit={code}); inspect {output}/logs/{label}.log')
            subprocess.run([sys.executable,'-m','InfoKVStream.analyze_docstyle_diagnostics','--root',str(output)],
                           cwd=ROOT,env=env,check=True)
        atomic_json(output/'status.json',{'status':'complete','updated_at':now()})
    except BaseException as exc:
        atomic_json(output/'status.json',{'status':'failed','error':str(exc),'updated_at':now()})
        raise
    finally:
        if paused:
            resume_pid=resume_ovo(output)
            print(f'OVO resumed, supervisor PID={resume_pid}',flush=True)


if __name__=='__main__':
    main()
