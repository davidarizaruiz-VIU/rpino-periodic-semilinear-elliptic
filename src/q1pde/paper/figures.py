from __future__ import annotations
import matplotlib.pyplot as plt
import pandas as pd
from q1pde.paths import figures_dir, evaluation_dir, aux_dir


def _try_read(path):
    return pd.read_csv(path) if path.exists() else None


def make_figures(config: dict) -> None:
    fd = figures_dir(config)
    rpino_samples = _try_read(evaluation_dir(config, 'rpino') / 'sample_metrics.csv')
    fno_samples = _try_read(evaluation_dir(config, 'fno') / 'sample_metrics.csv')
    contraction = _try_read(evaluation_dir(config, 'rpino') / 'contraction_metrics_test.csv')
    if contraction is None:
        contraction = _try_read(evaluation_dir(config, 'rpino') / 'contraction_metrics.csv')
    stability = _try_read(evaluation_dir(config, 'rpino') / 'stability_metrics_test.csv')
    if stability is None:
        stability = _try_read(evaluation_dir(config, 'rpino') / 'stability_metrics.csv')
    ablation = _try_read(aux_dir(config, 'ablations') / 'ablation_summary.csv')
    sweep = _try_read(aux_dir(config, 'ablations') / 'iteration_sweep.csv')

    if rpino_samples is not None:
        plt.figure(figsize=(6, 4))
        plt.hist(rpino_samples['rel_l2_backbone'], bins=20, alpha=0.7, label='Backbone')
        plt.hist(rpino_samples['rel_l2_rpino'], bins=20, alpha=0.7, label='RP-INO')
        if fno_samples is not None and 'rel_l2_fno' in fno_samples:
            plt.hist(fno_samples['rel_l2_fno'], bins=20, alpha=0.7, label='FNO')
        plt.xlabel('Relative L2 error')
        plt.ylabel('Count')
        plt.legend()
        plt.tight_layout()
        plt.savefig(fd / 'error_histogram_comparison.png', dpi=200)
        plt.close()

        plt.figure(figsize=(5.5, 5.5))
        plt.loglog(rpino_samples['rel_l2_backbone'] + 1e-14, rpino_samples['rel_l2_rpino'] + 1e-14, 'o')
        lo = min(rpino_samples['rel_l2_backbone'].min(), rpino_samples['rel_l2_rpino'].min())
        hi = max(rpino_samples['rel_l2_backbone'].max(), rpino_samples['rel_l2_rpino'].max())
        plt.loglog([lo, hi], [lo, hi], '--')
        plt.xlabel('Backbone relative L2 error')
        plt.ylabel('RP-INO relative L2 error')
        plt.tight_layout()
        plt.savefig(fd / 'backbone_vs_rpino_scatter.png', dpi=200)
        plt.close()

    if contraction is not None:
        plt.figure(figsize=(6, 4))
        for sample in sorted(contraction['sample'].unique())[:3]:
            subset = contraction[contraction['sample'] == sample]
            plt.semilogy(subset['iter'], subset['increment'], marker='o', label=f'sample {sample}')
        plt.xlabel('Fixed-point iteration')
        plt.ylabel('Increment norm')
        plt.legend()
        plt.tight_layout()
        plt.savefig(fd / 'contraction_curves.png', dpi=200)
        plt.close()

    if stability is not None:
        plt.figure(figsize=(6, 4))
        plt.scatter(stability['input_perturbation'], stability['output_change'])
        plt.xlabel('Input perturbation')
        plt.ylabel('Output change')
        plt.tight_layout()
        plt.savefig(fd / 'stability_scatter.png', dpi=200)
        plt.close()

    if ablation is not None:
        plt.figure(figsize=(6, 4))
        plt.bar(ablation['variant'], ablation['mean_rel_l2'])
        plt.ylabel('Mean relative L2 error')
        plt.xticks(rotation=15)
        plt.tight_layout()
        plt.savefig(fd / 'ablation_barplot.png', dpi=200)
        plt.close()

    if sweep is not None:
        plt.figure(figsize=(6, 4))
        plt.plot(sweep['eval_iters'], sweep['mean_rel_l2'], marker='o')
        plt.xlabel('Evaluation fixed-point iterations')
        plt.ylabel('Mean relative L2 error')
        plt.tight_layout()
        plt.savefig(fd / 'iteration_sweep.png', dpi=200)
        plt.close()

    print(f'Figures saved to {fd}')
