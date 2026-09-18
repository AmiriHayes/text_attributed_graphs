#!/usr/bin/env python3
"""Step 3: full 6-proxy x 5-dataset correlation battery on corrected N8
data. History is N/A (no N8 exists there -- has_secondary_id=false)."""
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import pandas as pd
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

RAGAS_FILES = {
    'arxiv': 'ragas_results_arxiv_n8_corrected.csv',
    'amazon': 'ragas_results_amazon_n8_corrected.csv',
    'electronics': 'ragas_results_electronics_corrected.csv',
    'toys': 'ragas_results_toys_corrected.csv',
}
PROXY_FILES = {ds: f'proxy_list1_{ds}_n8_corrected.csv' for ds in RAGAS_FILES}

PROXIES = ['pagerank_std', 'link_pred_auc', 'modularity', 'n_communities',
           'mean_degree', 'largest_cc_fraction']


def load_dataset(ds):
    ragas = pd.read_csv(REPO_ROOT / f'data/graphrag/{RAGAS_FILES[ds]}')
    ragas = ragas[ragas['text_idx'] != 'T12e']  # main correlation set excludes T12e (reported separately)
    target = ragas.groupby('system_name').agg(
        mean_ragas_composite=('composite_score', 'mean'),
        train_mean=('train_mean', 'first'),
        edge_idx=('edge_idx', 'first'), text_idx=('text_idx', 'first'),
    ).reset_index()
    target['variant'] = 'N8_' + target['edge_idx'] + '_' + target['text_idx']

    proxies = pd.read_csv(REPO_ROOT / f'data/graphrag/{PROXY_FILES[ds]}')
    proxies['variant'] = 'N8_' + proxies['variant'].str.split('_', n=1).str[1]
    merged = target.merge(proxies, on='variant', how='left')
    merged['edge_idx'] = merged['variant'].str.split('_').str[1]
    return merged


def corr(df, xcol, ycol):
    sub = df[[xcol, ycol]].dropna()
    n = len(sub)
    if n < 3 or sub[xcol].nunique() < 2:
        return float('nan'), float('nan'), n
    r, p = spearmanr(sub[xcol], sub[ycol])
    return r, p, n


if __name__ == '__main__':
    datasets = {}
    for ds in RAGAS_FILES:
        datasets[ds] = load_dataset(ds)
        print(f'{ds}: n={len(datasets[ds])} N8 variants (T12e excluded, main correlation set)')

    all_rows = []
    for proxy in PROXIES:
        for ds, df in datasets.items():
            no_e11b = df[df['edge_idx'] != 'E11b']
            rf, pf, nf = corr(df, proxy, 'mean_ragas_composite')
            r0, p0, n0 = corr(no_e11b, proxy, 'mean_ragas_composite')
            rg, pg, ng = corr(no_e11b, proxy, 'train_mean')
            ragas_independent = bool((not pd.isna(r0)) and (not pd.isna(rg)) and
                                      abs(r0) > 0.3 and p0 < 0.1 and abs(rg) < 0.3)
            all_rows.append({
                'proxy': proxy, 'dataset': ds,
                'rho_full': rf, 'p_full': pf, 'n_full': nf,
                'rho_no_e11b': r0, 'p_no_e11b': p0, 'n_no_e11b': n0,
                'rho_vs_rawgnn': rg, 'p_vs_rawgnn': pg, 'n_rawgnn': ng,
                'ragas_independent': ragas_independent,
            })
    # History row, all proxies -- explicit N/A, not blank/omitted
    for proxy in PROXIES:
        all_rows.append({'proxy': proxy, 'dataset': 'history',
                          'rho_full': None, 'p_full': None, 'n_full': 0,
                          'rho_no_e11b': None, 'p_no_e11b': None, 'n_no_e11b': 0,
                          'rho_vs_rawgnn': None, 'p_vs_rawgnn': None, 'n_rawgnn': 0,
                          'ragas_independent': False, 'note': 'N/A -- no N8 exists for History'})

    full_df = pd.DataFrame(all_rows)
    out = REPO_ROOT / 'data/graphrag/proxy_correlation_corrected_all_datasets.csv'
    full_df.to_csv(out, index=False)
    print(f'\nsaved -> {out}')

    order = ['arxiv', 'amazon', 'history', 'electronics', 'toys']
    print(f'\n{"="*90}\nPRIMARY TABLE: pagerank_std and link_pred_auc, all 5 datasets\n{"="*90}')
    print(f'{"dataset":<14}{"pagerank_std rho":>18}{"p":>10}{"n":>5}   {"link_pred_auc rho":>20}{"p":>10}{"n":>5}')
    for ds in order:
        if ds == 'history':
            print(f'{"History":<14}{"N/A (no N8)":>18}')
            continue
        pr = full_df[(full_df['proxy'] == 'pagerank_std') & (full_df['dataset'] == ds)].iloc[0]
        lp = full_df[(full_df['proxy'] == 'link_pred_auc') & (full_df['dataset'] == ds)].iloc[0]
        print(f'{ds:<14}{pr["rho_no_e11b"]:>18.3f}{pr["p_no_e11b"]:>10.4f}{pr["n_no_e11b"]:>5}   '
              f'{lp["rho_no_e11b"]:>20.3f}{lp["p_no_e11b"]:>10.4f}{lp["n_no_e11b"]:>5}')

    print(f'\n{"="*90}\nFULL 6-PROXY x 5-DATASET TABLE (rho_no_e11b / p_no_e11b / ragas_independent)\n{"="*90}')
    for proxy in PROXIES:
        print(f'\n--- {proxy} ---')
        for ds in order:
            if ds == 'history':
                print(f'  {ds:<14} N/A (no N8 exists)')
                continue
            r = full_df[(full_df['proxy'] == proxy) & (full_df['dataset'] == ds)].iloc[0]
            print(f'  {ds:<14} rho={r["rho_no_e11b"]:+.3f}  p={r["p_no_e11b"]:.4f}  n={r["n_no_e11b"]}  '
                  f'vs_rawgnn={r["rho_vs_rawgnn"]:+.3f}  independent={r["ragas_independent"]}')
