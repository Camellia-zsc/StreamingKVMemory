"""Document-style OVO comparison plots; only completed matched budgets enter.

Margin diagnostics use a common measured subset across methods and budgets.
The InfoKV OVO task runner saved max logits per option rather than raw variant
logits, so its exact logsumexp-based DeltaP cannot be reconstructed from these
files. We explicitly leave document figure 17 pending instead of substituting
a differently defined probability shift.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from InfoKVStream.analyze_ovo import analyze, finite, REPO_ROOT, BUDGETS
from InfoKVStream.plot_doc_style_comparison import save, write_csv, COLOR, LABEL, METHODS


def plot(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(root)
    data=analyze(root,'full',REPO_ROOT/'outputs/qwen25vl_full_eval')
    output=root/'figures/doc_style_comparison'
    output.mkdir(parents=True,exist_ok=True)
    report=[]
    exported=[]
    for group,tasks in [('backward',['EPM','ASI','HLD']),('realtime',['OCR','ACR','ATR','STU','FPD','OJR'])]:
        completed=[int(r['budget']) for r in data['summary'] if r['group']==group and
                   r['category']=='ALL' and r['status']=='complete']
        pairs={b:[r for r in data['paired'] if r['group']==group and r['budget']==b and
                  r['hermes_pair_status']=='paired' and r['full_pair_status']=='paired'] for b in completed}
        common=set.intersection(*[{r['sample_id'] for r in rows} for rows in pairs.values()]) if pairs else set()
        report.append(f'{group}: {len(completed)}/7 个预算完成；跨预算 Full/HERMES/InfoKV 共同成功样本 {len(common)}。')
        if not common:continue
        summaries=[]
        for b in sorted(completed):
            subset=[r for r in pairs[b] if r['sample_id'] in common]
            for method in METHODS:
                prefix='infokv' if method=='infokv_stream' else 'hermes'
                for task in ['ALL',*tasks]:
                    rows=[r for r in subset if task=='ALL' or r['task_type']==task]
                    if not rows:continue
                    trans={t:0 for t in ['CC','CW','WC','WW']}
                    for r in rows:
                        trans[('C' if r['full_correct'] else 'W')+('C' if r[f'{prefix}_correct'] else 'W')]+=1
                    full_correct=sum(r['full_correct'] for r in rows)
                    summaries.append({'group':group,'budget':b,'method':method,'task':task,'N':len(rows),
                                      'accuracy':mean(r[f'{prefix}_correct'] for r in rows),
                                      'full_accuracy':full_correct/len(rows),**trans,
                                      'cw_conditional_rate':trans['CW']/full_correct if full_correct else None,
                                      'cw_sample_rate':trans['CW']/len(rows)})
        exported.extend(summaries)
        # Corresponds to document figures 12 and 13.
        for num,key,ylabel in [(12,'accuracy','Compressed accuracy'),(13,'cw_conditional_rate','P(compressed wrong | Full correct)')]:
            fig,ax=plt.subplots(figsize=(8,4.8))
            palette=plt.get_cmap('tab10')
            for j,task in enumerate(tasks):
                for method in METHODS:
                    p=[r for r in summaries if r['method']==method and r['task']==task and r[key] is not None]
                    ax.plot([r['budget'] for r in p],[r[key] for r in p],marker='o',markersize=4,
                            color=palette(j),linestyle='-' if method=='hermes' else '--',label=f'{task} {LABEL[method]}')
            ax.set_xscale('log');ax.set_xlabel('Visual KV budget per layer');ax.set_ylabel(ylabel)
            ax.set_title(f'OVO {group}: by-task comparison | matched N={len(common)}\n{len(completed)}/7 budgets completed')
            ax.grid(alpha=.25);ax.legend(frameon=False,fontsize=8,ncol=2)
            fig.tight_layout();save(fig,output,f'{group}_fig_{num:02d}_{key}_by_task')
        for num,key,ylabel in [(14,'accuracy','Accuracy'),(15,'CW','Correct → wrong count')]:
            fig,ax=plt.subplots(figsize=(8,4.8))
            for method in METHODS:
                p=[r for r in summaries if r['method']==method and r['task']=='ALL']
                ax.plot([r['budget'] for r in p],[r[key] for r in p],marker='o',color=COLOR[method],label=LABEL[method])
            if key=='accuracy':ax.axhline(p[0]['full_accuracy'],color='gray',linestyle='--',label='Full-KV')
            ax.set_xscale('log');ax.set_xlabel('Visual KV budget per layer');ax.set_ylabel(ylabel)
            ax.set_title(f'OVO {group} | same matched N={len(common)}');ax.grid(alpha=.25);ax.legend(frameon=False)
            fig.tight_layout();save(fig,output,f'{group}_fig_{num:02d}_{key}_vs_budget')
        # Exact max-logit margin, available at diagnostic IDs in the baseline.
        ids=[]
        for b in completed:
            ids.append({r['sample_id'] for r in pairs[b] if all(finite(r.get(k)) is not None for k in
                         ['infokv_margin','hermes_margin','full_margin'])})
        diagnostic=set.intersection(*ids) if ids else set()
        if diagnostic:
            fig,ax=plt.subplots(figsize=(8,4.8))
            for method in METHODS:
                prefix='infokv' if method=='infokv_stream' else 'hermes'
                vals=[]
                for b in sorted(completed):
                    rows=[r for r in pairs[b] if r['sample_id'] in diagnostic]
                    vals.append(mean(r[f'{prefix}_margin']-r['full_margin'] for r in rows))
                ax.plot(sorted(completed),vals,marker='o',color=COLOR[method],label=LABEL[method])
            ax.axhline(0,color='gray',linestyle=':');ax.set_xscale('log')
            ax.set_xlabel('Visual KV budget per layer');ax.set_ylabel(r'Mean $\Delta M_z$')
            ax.set_title(f'OVO {group}: margin shift | common measured subset N={len(diagnostic)}')
            ax.grid(alpha=.25);ax.legend(frameon=False);fig.tight_layout()
            save(fig,output,f'{group}_fig_16_delta_margin_vs_budget')
        report.append(f'{group} 的图 16 使用共同 margin 诊断子集 N={len(diagnostic)}，不能称为全量 ΔM。')
    if exported:write_csv(exported,output/'paired_summary.csv')
    report += ['','图 12–16 在预算全量完成后生成；图中明确标注共同样本分母。',
               '图 17 的文档 ΔP_gold 定义使用合法首 token 变体的 logsumexp。当前 OVO InfoKV 全量记录仅保存每个选项的最大 logit，无法精确重建该定义；需要另补 raw-token 诊断，不能直接替换成 max-softmax。',
               'StreamingBench RT 已保存完整 token 变体 logits，故其 ΔP / ΔM 均按文档定义绘制。']
    (output/'README_zh.md').write_text('\n'.join(report)+'\n')
    print(json.dumps({'output':str(output),'plots':len(list(output.glob('*.png')))},ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=REPO_ROOT/'outputs/InfoKVStream/ovo_20261013')
    plot(p.parse_args().root)
