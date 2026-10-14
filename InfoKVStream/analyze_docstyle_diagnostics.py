"""Controlled HERMES/InfoKV figures corresponding to document figures 1–11."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from statistics import mean, median

import numpy as np

from InfoKVStream.run_docstyle_diagnostics import DEFAULT_ROOT, RT_ROOT, BUDGETS
from InfoKVStream.analyze_streamingbench_rt import decision_values
from InfoKVStream.plot_doc_style_comparison import save, write_csv, COLOR, LABEL, METHODS


def analyze(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10})
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text())
    out, figures = root / 'analysis', root / 'figures'
    out.mkdir(exist_ok=True); figures.mkdir(exist_ok=True)
    trajectories = []
    controls = []
    for sid in manifest['q_star_ids']:
        path = root / 'trajectory/full_cache/samples' / f'{sid}.json'
        if not path.exists():
            continue
        full = json.loads(path.read_text())
        comp = {m: json.loads((root / 'trajectory' / m / 'samples' / f'{sid}.json').read_text()) for m in METHODS}
        signatures = []
        for bundle in [full, *(comp[m] for m in METHODS)]:
            signatures.append([(c['frame_indices'], c['frame_times'], c['visual_tokens']) for c in bundle['chunks']])
        same_chunks = signatures[0] == signatures[1] == signatures[2]
        if not same_chunks:
            raise RuntimeError(f'Trajectory input frames/token deltas changed: {sid}')
        hashes = {r['query_hash'] for r in full['metrics']} | {r['query_hash_full'] for r in full['metrics']}
        if len(hashes) != 1:
            raise RuntimeError(f'Fixed query is not constant across trajectory: {sid}')
        controls.append({'sample_id': sid, 'same_frames_times_token_deltas': same_chunks,
                         'fixed_query_hash': next(iter(hashes)),
                         'same_probe_times': comp['hermes']['probe_time_plan'] == comp['infokv_stream']['probe_time_plan'],
                         'baseline_predictions_unchanged': all(
                             b['prediction_matches_uninstrumented_run'] for b in [full, comp['hermes']]
                         ),
                         'layerwise_infokv_matches_previous_global': comp['infokv_stream'].get(
                             'prediction_matches_previous_global_infokv')})
        trajectories.extend(full['metrics'])
    flat = [{k:r.get(k) for k in ['method','sample_id','video_id','probe_k','stream_time','ram_mean',
                                 'aore_mean','aore_median','js','kl','compression_ratio','query_hash','query_hash_full']}
            for r in trajectories]
    if flat:
        write_csv(flat, out / 'trajectory_per_probe.csv')
    summary = []
    for method in METHODS:
        arm = [r for r in trajectories if r['method'] == method]
        for field in ['ram_mean','aore_mean','aore_median','js','kl','compression_ratio']:
            by_k = defaultdict(list)
            for r in arm:
                if r.get(field) is not None:
                    by_k[r['probe_k']].append(r[field])
            for k, vals in sorted(by_k.items()):
                summary.append({'method':method,'metric':field,'k':k,'N_video':len(vals),
                                'mean':mean(vals),'median':median(vals),'min':min(vals),'max':max(vals)})
    if summary:
        write_csv(summary, out / 'trajectory_by_k.csv')
        spec = [('ram_mean','RAM',False), ('aore_mean','AORE (mean per head)',True),
                ('js','JS divergence',False), ('kl','Forward KL',False),
                ('aore_median','AORE (median per head)',False), ('compression_ratio','Compression ratio',False)]
        fig, axes = plt.subplots(2,3,figsize=(13,7.5))
        for j,(field,label,symlog) in enumerate(spec):
            single, ax = plt.subplots(figsize=(6.7,4.6))
            for target in [ax,axes.flat[j]]:
                for method in METHODS:
                    points=[r for r in summary if r['method']==method and r['metric']==field]
                    x=[r['k'] for r in points]
                    target.plot(x,[r['median'] for r in points],marker='o',markersize=2.5,
                                color=COLOR[method],label=LABEL[method])
                    target.fill_between(x,[r['min'] for r in points],[r['max'] for r in points],
                                        alpha=.12,color=COLOR[method])
                target.set_xlabel('Effective compaction count (k)');target.set_ylabel(label)
                target.grid(True,alpha=.25);target.legend(frameon=False)
                if symlog:target.set_yscale('symlog',linthresh=.01)
            ax.set_title(f'{label} vs effective compaction count\nFixed q_star; {len(controls)}/20 matched videos')
            single.tight_layout(); save(single,figures,f'fig_{j+1:02d}_trajectory_{field}')
        fig.suptitle(f'Fixed-query repeated-compaction trajectories | {len(controls)}/20 videos | 2K\n'
                     'Median across videos; shaded min–max; support decreases at high k')
        fig.tight_layout();save(fig,figures,'trajectory_overview')
        fig,ax=plt.subplots(figsize=(7,3))
        for method in METHODS:
            p=[r for r in summary if r['method']==method and r['metric']=='ram_mean']
            ax.step([r['k'] for r in p],[r['N_video'] for r in p],where='mid',label=LABEL[method],color=COLOR[method])
        ax.set_xlabel('Effective compaction count (k)');ax.set_ylabel('Supported videos');ax.legend();ax.grid(alpha=.25)
        fig.tight_layout();save(fig,figures,'trajectory_support')

    # Actual-query DeltaP/DeltaM by exact k, matching document figures 5/6.
    baseline = {sid:json.loads((RT_ROOT/'full_cache/samples'/f'{sid}.json').read_text())['record']
                for sid in manifest['sample_ids']}
    decisions = []
    for method in METHODS:
        for sid in manifest['sample_ids']:
            new_path = root/'budget_stress'/'budget_2000'/method/'samples'/f'{sid}.json'
            old_path = RT_ROOT/method/'samples'/f'{sid}.json'
            if method == 'infokv_stream' and not new_path.exists():
                # The revised arm must never silently fall back to global
                # InfoKV. Partial new runs remain explicitly partial.
                continue
            rec=json.loads((new_path if new_path.exists() else old_path).read_text())['record']
            cv,fv=decision_values(rec),decision_values(baseline[sid])
            decisions.append({'method':method,'sample_id':sid,'video_id':sid.rsplit('_',2)[-2],
                              'k':rec['runtime']['effective_compaction_count'],
                              'delta_p_gold':cv['p_gold']-fv['p_gold'],
                              'delta_margin':cv['margin']-fv['margin']})
    write_csv(decisions,out/'actual_query_decision_shifts.csv')
    for number,field,label in [(5,'delta_p_gold',r'$\Delta P_{gold}$'),(6,'delta_margin',r'$\Delta M_z$')]:
        fig,ax=plt.subplots(figsize=(7,4.5))
        for method in METHODS:
            groups=defaultdict(list)
            for r in decisions:
                if r['method']==method:groups[r['k']].append(r[field])
            for stat,style in [(mean,'-'),(median,'--')]:
                ax.plot(sorted(groups),[stat(groups[k]) for k in sorted(groups)],color=COLOR[method],
                        linestyle=style,marker='o',markersize=3,label=f'{LABEL[method]} {stat.__name__}')
        ax.axhline(0,color='gray',linestyle=':',linewidth=1)
        ax.set_xlabel('Effective compaction count before actual query');ax.set_ylabel(label)
        ax.set_title(f'{label} vs effective compaction count | same 100 actual queries')
        ax.legend(frameon=False,ncol=2);ax.grid(alpha=.25)
        fig.tight_layout();save(fig,figures,f'fig_{number:02d}_{field}_vs_k')

    stress_rows, stress_summary = [], []
    expected=set(manifest['sample_ids'])
    for budget in BUDGETS:
        arm={}
        for method in METHODS:
            records={p.stem:json.loads(p.read_text())['record'] for p in
                     (root/'budget_stress'/f'budget_{budget}'/method/'samples').glob('*.json')}
            arm[method]=records
        common=expected.intersection(*(set(a) for a in arm.values()))
        if not common:continue
        for method in METHODS:
            values=[]
            transitions={t:0 for t in ['CC','CW','WC','WW']}
            for sid in sorted(common):
                rec,ref=arm[method][sid],baseline[sid]
                for key in ['ground_truth','question_timestamp','task_type']:
                    if rec[key]!=ref[key]:raise RuntimeError(f'Budget pairing changed {key}: {sid}')
                cv,fv=decision_values(rec),decision_values(ref)
                row={'budget':budget,'method':method,'sample_id':sid,'video_id':sid.rsplit('_',2)[-2],
                     'correct':int(rec['correct']),'full_correct':int(ref['correct']),
                     'k':rec['runtime']['effective_compaction_count'],
                     'delta_p_gold':cv['p_gold']-fv['p_gold'],
                     'delta_margin':cv['margin']-fv['margin'],
                     'boundary_crossing':int(fv['margin']>0 and cv['margin']<0)}
                row.update({key: rec.get('fidelity', {}).get(key)
                            for key in ['ram_mean', 'aore_mean', 'aore_median', 'js', 'kl',
                                        'compression_ratio']})
                row.update(trigger_count=rec['runtime'].get('compression_trigger_count'),
                           stream_processing_sec=rec['runtime'].get('stream_processing_sec'),
                           qa_sec=rec['runtime'].get('qa_sec'),
                           kv_bytes_at_query=rec['runtime'].get('kv_bytes_at_query'))
                values.append(row);stress_rows.append(row)
                transitions[('C' if row['full_correct'] else 'W')+('C' if row['correct'] else 'W')]+=1
            entry={'budget':budget,'method':method,'N':len(common),'expected':len(expected),
                   'complete':common==expected,'accuracy':mean(r['correct'] for r in values),
                   'full_accuracy':mean(r['full_correct'] for r in values),**transitions,
                   'mean_k':mean(r['k'] for r in values),'boundary_crossings':sum(r['boundary_crossing'] for r in values)}
            for key in ['ram_mean', 'aore_mean', 'aore_median', 'js', 'kl', 'compression_ratio',
                        'stream_processing_sec', 'qa_sec', 'kv_bytes_at_query']:
                vals = [r[key] for r in values if r.get(key) is not None]
                entry[f'mean_{key}'] = mean(vals) if vals else None
            entry['total_effective_compactions'] = sum(r['k'] for r in values)
            entry['total_triggers'] = sum(r['trigger_count'] or 0 for r in values)
            for key in ['delta_p_gold','delta_margin']:
                videos=defaultdict(list)
                for r in values:videos[r['video_id']].append(r[key])
                vm=[mean(v) for v in videos.values()]
                entry[f'video_mean_{key}']=mean(vm);entry[f'video_median_{key}']=median(vm)
            stress_summary.append(entry)
    if stress_rows:
        write_csv(stress_rows,out/'budget_stress_per_query.csv');write_csv(stress_summary,out/'budget_stress_summary.csv')
    completed_budgets=[b for b in BUDGETS if sum(r['budget']==b and r['complete'] for r in stress_summary)==2]
    usable=[r for r in stress_summary if r['budget'] in completed_budgets]
    if usable:
        fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
        for ax, key, label in zip(axes.flat,
                                 ['ram_mean', 'aore_mean', 'js', 'kl', 'aore_median', 'compression_ratio'],
                                 ['RAM', 'AORE (mean per head)', 'JS', 'KL', 'AORE (median per head)', 'Compression ratio']):
            for method in METHODS:
                points = sorted([r for r in usable if r['method'] == method], key=lambda r:r['budget'])
                ax.plot([r['budget'] for r in points], [r.get(f'mean_{key}') for r in points],
                        marker='o', color=COLOR[method], label=LABEL[method])
            ax.set_xscale('log')
            ax.set_xticks(sorted(completed_budgets), [str(b) for b in sorted(completed_budgets)])
            ax.set_xlabel('Visual KV budget per layer'); ax.set_ylabel(label)
            ax.grid(alpha=.25); ax.legend(frameon=False)
            if key == 'aore_mean':
                ax.set_yscale('symlog', linthresh=.01)
        fig.suptitle('Fidelity at official query anchors | same RT20 / 100 queries')
        fig.tight_layout(); save(fig, figures, 'budget_fidelity_overview')
        for number,field,label in [(7,'accuracy','Task accuracy'),(9,'mean_k','Mean effective compaction count')]:
            fig,ax=plt.subplots(figsize=(7,4.5))
            for method in METHODS:
                p=sorted([r for r in usable if r['method']==method],key=lambda r:r['budget'])
                ax.plot([r['budget'] for r in p],[r[field] for r in p],marker='o',color=COLOR[method],label=LABEL[method])
            if field=='accuracy':ax.axhline(mean(int(r['correct']) for r in baseline.values()),linestyle='--',color='gray',label='Full-KV')
            ax.set_xlabel('Visual KV budget per layer');ax.set_ylabel(label)
            ax.set_xscale('log'); ax.set_xticks(sorted(completed_budgets), [str(b) for b in sorted(completed_budgets)])
            ax.set_title(f'{label} vs KV budget | {len(completed_budgets)}/{len(BUDGETS)} budgets complete')
            ax.grid(alpha=.25);ax.legend(frameon=False);fig.tight_layout()
            save(fig,figures,f'fig_{number:02d}_{field}_vs_budget')
        fig,ax=plt.subplots(figsize=(9,4.8));bs=sorted(completed_budgets);x=np.arange(len(bs))
        for index,method in enumerate(METHODS):
            bottom=np.zeros(len(bs))
            vals={r['budget']:r for r in usable if r['method']==method}
            for transition,color in [('CC','#2878b5'),('CW','#e67e22'),('WC','#55a868'),('WW','#c44e52')]:
                height=np.array([vals[b][transition] for b in bs])
                ax.bar(x+(index-.5)*.38,height,.36,bottom=bottom,color=color,label=transition if index==0 else None,
                       hatch='' if index==0 else '//',alpha=.9)
                bottom+=height
        ax.set_xticks(x,[str(b)+'\nHERMES / InfoKV' for b in bs]);ax.set_ylabel('Query count')
        ax.set_title('Full → compressed transitions vs budget');ax.legend(ncol=4,frameon=False)
        fig.tight_layout();save(fig,figures,'fig_08_transitions_vs_budget')
        for number,field,label in [(10,'delta_p_gold',r'$\Delta P_{gold}$'),(11,'delta_margin',r'$\Delta M_z$')]:
            fig,ax=plt.subplots(figsize=(7,4.5))
            for method in METHODS:
                p=sorted([r for r in usable if r['method']==method],key=lambda r:r['budget'])
                for stat,style in [('mean','-'),('median','--')]:
                    ax.plot([r['budget'] for r in p],[r[f'video_{stat}_{field}'] for r in p],
                            color=COLOR[method],linestyle=style,marker='o',label=f'{LABEL[method]} video {stat}')
            ax.axhline(0,color='gray',linestyle=':');ax.set_xlabel('Visual KV budget per layer');ax.set_ylabel(label)
            ax.set_xscale('log'); ax.set_xticks(sorted(completed_budgets), [str(b) for b in sorted(completed_budgets)])
            ax.set_title(f'{label} vs budget | same RT20/100 queries');ax.grid(alpha=.25);ax.legend(frameon=False,ncol=2)
            fig.tight_layout();save(fig,figures,f'fig_{number:02d}_{field}_vs_budget')
    audit={'trajectory_videos':len(controls),'trajectory_expected':20,'checks':controls,
           'completed_budgets':completed_budgets,'expected_budgets':BUDGETS,
           'experiment_contract':manifest['contract_id'],
           'analyzer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out/'control_variables_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    lines=['# 文档口径的 HERMES / InfoKV 对比','',f'固定问题压缩轨迹：{len(controls)}/20 个视频完成。',
           f'预算扫描：{completed_budgets} / {BUDGETS} 已完成，两种方法每档均为同一 100 题。','',
           '采用同一 Qwen2.5-VL-7B、官方 prompt 和自由解码、0.5 FPS、2 帧/chunk、max_side=336、full_prefix。',
           '文档的历史数值来自 MMDuet2，预算扫描是 28 视频。本次沿用已约定的 20 视频/100 题并重测两种方法，历史数值不混入曲线。',
           '固定 q_star 和所有 probe 的 token IDs 完全相同；Full 与两种方法在相同视频时间比较。探测使用克隆缓存和 use_cache=False；实际答案必须等于未开启轨迹探测的已有结果。',
           '图 1–4：跨视频中位数和 min–max 范围；高 k 的支持数见 trajectory_support。AORE 原始每头均值采用 symlog，另给每头中位数图避免极小 Full 输出范数掩盖典型行为。',
           '图 5–6：真实问题相对同一 Full 的 ΔP_gold / ΔM_z，按 exact k 统计 mean/median。这两个图包含 100 个实际问题，并不使用未来问题 probe 计算任务决策指标。',
           '图 7–11：仅纳入两种方法均完成 100 题的预算；部分结果不与全量结果连线。ΔP/ΔM 先在视频内聚合 5 题，再跨 20 视频算 mean/median。',
           '图 12–17（OVO）：需要原先七档全量队列完成后再画；不能把不同覆盖数的未完成预算当成完整曲线。','',
           '所有图保存 PNG 和 PDF；绘图数据及控制变量审计保存在 analysis/。','']
    (out/'report_zh.md').write_text('\n'.join(lines))
    print(json.dumps({k:v for k,v in audit.items() if k!='checks'},ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    analyze(p.parse_args().root)
