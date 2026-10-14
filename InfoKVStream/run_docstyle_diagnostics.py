"""Controlled RT20 repeated-compaction probes and budget stress experiments.

The input protocol and sample set come from the completed RT20 manifest. Only
the memory method and (in the budget experiment) the shared budget change.
Immediate probes use the exact prompt IDs saved by the official Full answer;
each probe clones the live cache, and probe artifacts are stored on CPU/disk,
so repeated probes do not accumulate CUDA snapshots.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

from InfoKVStream.run_ovo import ROOT, atomic_json, memory_config, now, fingerprints

RT_ROOT = ROOT / 'outputs/InfoKVStream/streamingbench_rt20_20261013'
DEFAULT_ROOT = ROOT / 'outputs/InfoKVStream/rt_docstyle_layerwise_20261014_fa2'
BUDGETS = [6000, 2000, 1000, 500, 200, 100, 50]


def prepare(output):
    source = json.loads((RT_ROOT / 'manifest.json').read_text())
    contract = {
        'reference_root': str(RT_ROOT), 'reference_contract': source['contract_id'],
        'base_config': source['base_config'], 'sample_ids': source['sample_ids'],
        'video_ids': source['video_ids'],
        'q_star_ids': [r['q_star_id'] for r in source['selected_samples']['samples']],
        'budgets': BUDGETS, 'methods': ['hermes', 'infokv_stream'],
        'source_sha256': fingerprints(),
        'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'control': 'same Qwen weights, official prompt/free decode, frames, FPS, chunks, '
                   'max_side, question set, full_prefix and common Full reference',
        'trajectory': 'q_star fixed per video; k=0 immediately before first eviction, '
                      'then every effective eviction; Full sampled at identical times',
        'budget_stress': 'same RT20/100 queries at all budgets; not the historical MMDuet2 RT28 set',
    }
    contract_id = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'manifest.json'
    if path.exists():
        previous = json.loads(path.read_text())
        if previous['contract_id'] != contract_id:
            raise RuntimeError('Diagnostic source/config changed; preserve this run and use a new output')
        return previous
    result = {**contract, 'contract_id': contract_id, 'created_at': now()}
    atomic_json(path, result)
    for relative in contract['source_sha256']:
        dest = output / 'source' / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((ROOT / relative).read_bytes())
    (output / 'source/InfoKVStream/run_docstyle_diagnostics.py').write_bytes(Path(__file__).read_bytes())
    # Reuse only the identical Full/HERMES controls. InfoKV is rerun with
    # layerwise retention, including at the otherwise reusable 2K budget.
    for method in ['full_cache', 'hermes']:
        target = output / 'budget_stress' / 'budget_2000' / method
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(RT_ROOT / method, target_is_directory=True)
    return result


def config_for(manifest, method, budget):
    config = copy.deepcopy(manifest['base_config'])
    config['memory'] = (memory_config(budget, layerwise=True) if method == 'infokv_stream' else
                        {'method': method, 'compression': {'enabled': method != 'full_cache',
                                                          'budget': budget}})
    return config


def trajectory(args, manifest):
    import torch
    from evaluation_pipeline.adapters import StreamingBenchAdapter
    from evaluation_pipeline.evaluate import make_runtime_config
    from evaluation_pipeline.runtime import ModelRuntime
    from evaluation_pipeline.kv_metrics.config import KVMetricConfig
    from evaluation_pipeline.kv_metrics.hooks import KVInstrumentation
    from evaluation_pipeline.kv_metrics.paired_runner import _load_chunk_module, _wrap_runtime_events, _official_correct
    from evaluation_pipeline.kv_metrics.aggregate import compute_paired_metrics

    torch.set_num_threads(4)
    # Save actual frame IDs/times, visual-token deltas and keep identities to
    # audit control variables. This is a read-only observer in both arms.
    os.environ['HERMES_DUMP_KEEP_PROVENANCE'] = '1'
    method = args.method
    qstars = list(manifest['q_star_ids'])
    # A short q_star first provides a real GPU check before larger videos.
    qstars.sort(key=lambda sid: float(json.loads((RT_ROOT / 'full_cache/samples' / f'{sid}.json').read_text())['record']['question_timestamp']))
    if args.limit:
        qstars = qstars[:args.limit]
    directory = args.output / 'trajectory' / method
    pending = [sid for sid in qstars if not (directory / 'samples' / f'{sid}.json').exists()]
    if not pending:
        return
    config = config_for(manifest, method, 2000)
    adapter = StreamingBenchAdapter(config['benchmarks']['streamingbench'])
    samples = {s.sample_id: s for s in adapter.load_samples()}
    runtime = ModelRuntime(make_runtime_config(config)).load()
    metric_config = KVMetricConfig(enable_kv_metrics=True, attention_backend='eager',
                                   profile_wallclock=True, summary_token_mode='identity_only')

    class ImmediateProbes(KVInstrumentation):
        def _snapshot_for_probe(self, k):
            if not self._probing_active():
                return None
            # Hooks used by the formal answer must not consume diagnostic logits.
            armed = self._option_armed
            sidecar_armed = getattr(self.model, '_sidecar_armed', False)
            self._option_armed = False
            self.model._sidecar_armed = False
            lengths_before = self._lengths()
            offsets_before = list(self.qa_model._get_next_global_offset_per_layer())
            try:
                artifact = self._run_probe(k, 'diagnostic_probe', self._clone_cache())
            finally:
                self._option_armed = armed
                self.model._sidecar_armed = sidecar_armed
            if self._lengths() != lengths_before or list(self.qa_model._get_next_global_offset_per_layer()) != offsets_before:
                raise RuntimeError('Diagnostic probe mutated live cache lengths/positions')
            if artifact is None:
                raise RuntimeError('Immediate probe failed')
            artifact.video_id = artifact.event_id.split('|')[0]
            for layer in artifact.layers.values():
                layer.value_states = layer.value_states.to(torch.bfloat16)
            self.probe_consumer(artifact)
            return {'k': int(k), 'stream_time': artifact.stream_time}

        def _capture_anchor(self, snapshot, input_ids, offsets, input_text):
            # The exact official prompt must equal the one used for every probe.
            if list(input_ids) != self.probe_query['input_ids']:
                raise RuntimeError('Fixed probe prompt differs from official question prompt')

        def _finalize_probes(self):
            self._probe_snapshots = []

    instrumentation = ImmediateProbes(metric_config)
    instrumentation.install(runtime.model, _load_chunk_module(runtime.backend))
    captured, set_targets = _wrap_runtime_events(runtime, instrumentation, True)
    atomic_json(directory / 'config.json', config)
    for sid in pending:
        runtime.reset()
        instrumentation.reset_collected()
        set_targets({})
        baseline = torch.load(RT_ROOT / 'full_artifacts' / f'{sid}.pt', map_location='cpu', weights_only=False)
        expected_ids = list(baseline.query_input_ids)
        del baseline
        record = json.loads((RT_ROOT / 'full_cache/samples' / f'{sid}.json').read_text())['record']
        instrumentation.set_probe_query(sid, sid, record['question_timestamp'], input_ids=expected_ids)
        probe_rows = []
        artifacts = directory / 'probes' / sid
        artifacts.mkdir(parents=True, exist_ok=True)
        if method == 'full_cache':
            saved = {m: json.loads((args.output / 'trajectory' / m / 'samples' / f'{sid}.json').read_text())
                     for m in ['hermes', 'infokv_stream']}
            hplan = saved['hermes']['probe_time_plan']
            if hplan != saved['infokv_stream']['probe_time_plan']:
                raise RuntimeError('Two methods have different probe times; cannot share Full')
            instrumentation.set_probe_time_plan(hplan)

            def consume(full):
                for arm in ['hermes', 'infokv_stream']:
                    path = args.output / 'trajectory' / arm / 'probes' / sid / f'k_{full.probe_k:03d}.pt'
                    comp = torch.load(path, map_location='cpu', weights_only=False)
                    if comp.query_input_ids != full.query_input_ids or abs(comp.stream_time - full.stream_time) > 1e-8:
                        raise RuntimeError('Trajectory prompt/time pairing mismatch')
                    metrics = compute_paired_metrics(comp, full, metric_config)
                    metrics.update(method=arm, probe_k=comp.probe_k, video_id=sid.rsplit('_', 2)[-2],
                                   query_hash=comp.probe_query_hash, query_hash_full=full.probe_query_hash)
                    probe_rows.append(metrics)
                    print(f'[probe] {sid} {arm} k={comp.probe_k} RAM={metrics["ram_mean"]:.5f}', flush=True)
            instrumentation.probe_consumer = consume
        else:
            def consume(artifact):
                path = artifacts / f'k_{artifact.probe_k:03d}.pt'
                torch.save(artifact, path.with_suffix('.tmp'))
                path.with_suffix('.tmp').replace(path)
                probe_rows.append({'probe_k': artifact.probe_k, 'stream_time': artifact.stream_time,
                                   'query_hash': artifact.probe_query_hash})
                print(f'[probe] {method} {sid} k={artifact.probe_k}', flush=True)
            instrumentation.probe_consumer = consume
        produced = adapter.run_sample(runtime, samples[sid])
        if len(produced) != 1:
            raise RuntimeError('Expected one q_star answer')
        actual = produced[0]
        reference_path = RT_ROOT / method / 'samples' / f'{sid}.json'
        reference = json.loads(reference_path.read_text())['record'] if reference_path.exists() else None
        if method != 'infokv_stream' and reference is not None and actual['prediction'] != reference['prediction']:
            raise RuntimeError(f'Read-only probes changed task prediction for {method}/{sid}')
        plan = ([[r['probe_k'], r['stream_time']] for r in probe_rows] if method != 'full_cache' else hplan)
        if not plan or plan[0][0] != 0:
            raise RuntimeError('No k=0 trajectory probe')
        bundle = {'record': {**actual, 'correct': _official_correct(adapter, actual)},
                  'method': method, 'sample_id': sid, 'contract_id': manifest['contract_id'],
                  'probe_time_plan': plan, 'metrics': probe_rows if method == 'full_cache' else [],
                  'physical_compactions': instrumentation.collected_compactions,
                  'infokv_events': ([e.as_dict() for e in runtime.model.infokv_events()]
                                    if method == 'infokv_stream' and hasattr(runtime.model, 'infokv_events') else []),
                  'chunks': instrumentation.collected_chunks,
                  'prediction_matches_uninstrumented_run': (reference is None or actual['prediction'] == reference['prediction']),
                  'prediction_matches_previous_global_infokv': (None if method != 'infokv_stream' or reference is None else actual['prediction'] == reference['prediction']),
                  'finished_at': now()}
        atomic_json(directory / 'samples' / f'{sid}.json', bundle)
        atomic_json(args.output / 'progress.json', {'phase': 'trajectory', 'method': method,
                    'completed': len(list((directory / 'samples').glob('*.json'))),
                    'expected': len(manifest['q_star_ids']), 'last_sample': sid, 'updated_at': now()})
        print(f'[completed] trajectory/{method}/{sid}, probes={len(plan)}', flush=True)
        if method == 'full_cache':
            # Full-relative scalar/per-layer metrics are the commit record.
            # Discard temporary large value-state artifacts after both pairings.
            for arm in ['hermes', 'infokv_stream']:
                for path in (args.output / 'trajectory' / arm / 'probes' / sid).glob('*.pt'):
                    path.unlink()


def stress(args, manifest):
    import torch
    from evaluation_pipeline.adapters import StreamingBenchAdapter
    from evaluation_pipeline.evaluate import make_runtime_config
    from evaluation_pipeline.runtime import ModelRuntime
    from evaluation_pipeline.memory import resolve_memory_config
    from evaluation_pipeline.kv_metrics.config import KVMetricConfig
    from evaluation_pipeline.kv_metrics.hooks import KVInstrumentation
    from evaluation_pipeline.kv_metrics.paired_runner import _load_chunk_module, _wrap_runtime_events, _official_correct
    from evaluation_pipeline.kv_metrics.aggregate import compute_paired_metrics
    from InfoKVStream.run_streamingbench_rt import Timing

    torch.set_num_threads(4)
    method = args.method
    config = config_for(manifest, method, args.budget)
    adapter = StreamingBenchAdapter(config['benchmarks']['streamingbench'])
    samples = {s.sample_id: s for s in adapter.load_samples()}
    directory = args.output / 'budget_stress' / f'budget_{args.budget}' / method
    ids = manifest['sample_ids'][:args.limit] if args.limit else manifest['sample_ids']
    pending = [sid for sid in ids if not (directory / 'samples' / f'{sid}.json').exists()]
    if not pending:
        return
    runtime = ModelRuntime(make_runtime_config(config)).load()
    timing = Timing(runtime.model)
    metric_config = KVMetricConfig(enable_kv_metrics=True, attention_backend='eager',
                                   capture_option_logits=True, profile_wallclock=True)
    instrumentation = KVInstrumentation(metric_config)
    instrumentation.install(runtime.model, _load_chunk_module(runtime.backend))
    captured, set_targets = _wrap_runtime_events(runtime, instrumentation, True)
    atomic_json(directory / 'config.json', config)
    for sid in pending:
        runtime.reset(); instrumentation.reset_collected(); set_targets({}); timing.reset()
        produced = adapter.run_sample(runtime, samples[sid])
        if len(produced) != 1 or len(instrumentation.collected_events) != 1:
            raise RuntimeError('Expected one answer/anchor per actual query')
        row, artifact = produced[0], instrumentation.collected_events[0]
        full = torch.load(RT_ROOT / 'full_artifacts' / f'{sid}.pt', map_location='cpu', weights_only=False)
        if artifact.query_input_ids != full.query_input_ids or artifact.event_id != full.event_id:
            raise RuntimeError('Budget stress changed official query prompt/identity')
        reference = json.loads((RT_ROOT / 'full_cache/samples' / f'{sid}.json').read_text())['record']
        if timing.frames != reference['runtime']['frames']:
            raise RuntimeError('Budget stress changed sampled frames')
        instrumentation.tracker.assert_sync(runtime.model.qa_model.kv_cache)
        if any(n - timing.prefix > args.budget for n in timing.lengths):
            raise RuntimeError('Query-time visual budget exceeded')
        events = [e.as_dict() for e in runtime.model.infokv_events()] if method == 'infokv_stream' else []
        obs = captured.get(artifact.event_id, {})
        metrics = compute_paired_metrics(artifact, full, metric_config)
        row.update(correct=_official_correct(adapter, row), method=method, budget=args.budget,
                   contract_id=manifest['contract_id'], status='success', finished_at=now(),
                   option_token_logits=obs.get('option_token_logits'), option_token_ids=obs.get('option_token_ids'),
                   fidelity=metrics, runtime={
                       'compression_trigger_count': len(events) if events else artifact.compaction_index,
                       'effective_compaction_count': artifact.effective_compaction_count,
                       'frames': timing.frames, 'stream_processing_sec': timing.stream_sec,
                       'qa_sec': timing.qa_sec, 'protected_prefix_tokens': timing.prefix,
                       'cache_lengths_at_query': timing.lengths, 'kv_bytes_at_query': artifact.kv_bytes})
        atomic_json(directory / 'samples' / f'{sid}.json', {'record': row,
                    'physical_compactions': instrumentation.collected_compactions, 'infokv_events': events})
        completed = len(list((directory / 'samples').glob('*.json')))
        atomic_json(args.output / 'progress.json', {'phase': 'budget_stress', 'method': method,
                    'budget': args.budget, 'completed': completed, 'expected': len(manifest['sample_ids']),
                    'last_sample': sid, 'updated_at': now()})
        print(f'[completed] budget/{args.budget}/{method} {completed}/100 {sid}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'trajectory', 'stress'])
    parser.add_argument('--output', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--method', choices=['hermes', 'infokv_stream', 'full_cache'], default='hermes')
    parser.add_argument('--budget', type=int, choices=BUDGETS, default=2000)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    os.chdir(ROOT)
    manifest = prepare(args.output)
    if args.command == 'trajectory': trajectory(args, manifest)
    elif args.command == 'stress': stress(args, manifest)
    else: print(json.dumps({'output': str(args.output), 'contract_id': manifest['contract_id']}, indent=2))


if __name__ == '__main__':
    main()
