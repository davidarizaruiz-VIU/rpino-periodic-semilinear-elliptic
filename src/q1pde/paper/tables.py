from __future__ import annotations
import json
import pandas as pd
from q1pde.paths import tables_dir, evaluation_dir, aux_dir


def _read_json(path):
    if not path.exists():
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def make_tables(config: dict) -> None:
    td = tables_dir(config)
    records = []
    for name in ('rpino', 'fno'):
        summary = _read_json(evaluation_dir(config, name) / 'summary.json')
        if summary is not None:
            summary = {'model': name, **summary}
            records.append(summary)
    main = pd.DataFrame(records)
    if not main.empty:
        main.to_csv(td / 'main_results.csv', index=False)
        with open(td / 'main_results.tex', 'w', encoding='utf-8') as f:
            f.write(main.to_latex(index=False, float_format=lambda x: f'{x:.4e}' if isinstance(x, float) else str(x)))

    ablation = aux_dir(config, 'ablations') / 'ablation_summary.csv'
    if ablation.exists():
        abl = pd.read_csv(ablation)
        abl.to_csv(td / 'ablation_results.csv', index=False)
        with open(td / 'ablation_results.tex', 'w', encoding='utf-8') as f:
            f.write(abl.to_latex(index=False, float_format=lambda x: f'{x:.4e}' if isinstance(x, float) else str(x)))

    sweep = aux_dir(config, 'ablations') / 'iteration_sweep.csv'
    if sweep.exists():
        sw = pd.read_csv(sweep)
        sw.to_csv(td / 'iteration_sweep.csv', index=False)
        with open(td / 'iteration_sweep.tex', 'w', encoding='utf-8') as f:
            f.write(sw.to_latex(index=False, float_format=lambda x: f'{x:.4e}' if isinstance(x, float) else str(x)))

    print(f'Tables saved to {td}')
