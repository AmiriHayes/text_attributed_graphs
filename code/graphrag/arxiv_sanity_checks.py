#!/usr/bin/env python3
"""
Three purely computational sanity checks on the ArXiv GraphRAG results.
No API calls. Reads ragas_results_arxiv.csv + arxiv_selected_configs.csv,
writes arxiv_sanity_checks.csv.
"""
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

df = pd.read_csv(REPO_ROOT / 'data/graphrag/ragas_results_arxiv.csv')
configs = pd.read_csv(REPO_ROOT / 'data/graphrag/arxiv_selected_configs.csv')

# 5 config systems only (drop kg_baseline — not part of the quintile design)
tag_df = df[df.system_type.isin(['community_based', 'text_based'])].copy()

per_system = tag_df.groupby(['system_name', 'band', 'system_type']).agg(
    mean_composite=('composite_score', 'mean'),
    std_composite=('composite_score', 'std'),
).reset_index()
per_system = per_system.merge(configs[['band', 'train_mean']], on='band')
per_system['cv'] = per_system['std_composite'] / per_system['mean_composite']

print('=' * 22, 'CHECK 1 — EXACT PERMUTATION TEST', '=' * 22)

results_check1 = {}
for stype in ['text_based', 'community_based']:
    sub = per_system[per_system.system_type == stype].sort_values('train_mean', ascending=False)
    observed_rho, observed_p_scipy = spearmanr(sub['train_mean'], sub['mean_composite'])

    # Fixed reference = actual observed RAGAS rank order (train_mean-descending row order)
    ragas_ranks = list(sub['mean_composite'].rank(ascending=False).astype(int))

    all_rhos = []
    for perm in permutations([1, 2, 3, 4, 5]):
        rho, _ = spearmanr(list(perm), ragas_ranks)
        all_rhos.append(rho)
    all_rhos = np.array(all_rhos)

    frac_ge_090 = np.mean(all_rhos >= 0.90 - 1e-9)
    frac_eq_100 = np.mean(all_rhos >= 1.0 - 1e-9)
    p_exact = np.mean(all_rhos >= observed_rho - 1e-9)

    # 10,000-shuffle Monte Carlo cross-check
    rng = np.random.RandomState(42)
    mc_rhos = []
    base = np.array([1, 2, 3, 4, 5])
    for _ in range(10000):
        shuffled = rng.permutation(base)
        rho, _ = spearmanr(shuffled, ragas_ranks)
        mc_rhos.append(rho)
    mc_rhos = np.array(mc_rhos)
    p_mc = np.mean(mc_rhos >= observed_rho - 1e-9)

    print(f'\n--- {stype} ---')
    print(f'  observed rho={observed_rho:.3f}  (scipy asymptotic p={observed_p_scipy:.4f})')
    print(f'  exact permutation distribution over all 120 orderings:')
    print(f'    fraction with rho >= 0.90 : {frac_ge_090:.4f}  ({int(round(frac_ge_090*120))}/120)')
    print(f'    fraction with rho >= 1.00 : {frac_eq_100:.4f}  ({int(round(frac_eq_100*120))}/120)')
    print(f'    exact one-tailed p (rho >= observed) : {p_exact:.4f}  ({int(round(p_exact*120))}/120)')
    print(f'  Monte Carlo (10,000 shuffles) cross-check: p = {p_mc:.4f}')
    agreement_str = 'yes' if abs(p_exact - p_mc) < 0.01 else 'DIVERGENT'
    print(f'  exact vs MC agreement: {agreement_str}')

    results_check1[stype] = {'observed_rho': observed_rho, 'p_exact': p_exact, 'p_mc': p_mc,
                              'frac_ge_090': frac_ge_090, 'frac_eq_100': frac_eq_100}

print('\n\n' + '=' * 22, 'CHECK 2 — CROSS-SYSTEM-TYPE CORRELATION', '=' * 22)

comm = per_system[per_system.system_type == 'community_based'].set_index('band')['mean_composite']
text = per_system[per_system.system_type == 'text_based'].set_index('band')['mean_composite']
cross_rho, cross_p = spearmanr(comm.reindex(configs['band']), text.reindex(configs['band']))
print(f'\nSpearman rho(community mean_composite, text mean_composite) across 5 quintiles: '
      f'rho={cross_rho:+.3f}  p={cross_p:.4f}')
print(f'-> {"consistent across retrieval paradigms" if cross_rho > 0.8 else "NOT strongly consistent"} (threshold 0.8)')

# per-question agreement: does each (config,question) deviate from its OWN system-type's
# overall mean in the same direction for both system types?
comm_q = tag_df[tag_df.system_type == 'community_based'][['band', 'question_id', 'composite_score']].rename(
    columns={'composite_score': 'community_score'})
text_q = tag_df[tag_df.system_type == 'text_based'][['band', 'question_id', 'composite_score']].rename(
    columns={'composite_score': 'text_score'})
paired = comm_q.merge(text_q, on=['band', 'question_id'])

comm_overall_mean = tag_df[tag_df.system_type == 'community_based']['composite_score'].mean()
text_overall_mean = tag_df[tag_df.system_type == 'text_based']['composite_score'].mean()

paired['comm_dir'] = np.sign(paired['community_score'] - comm_overall_mean)
paired['text_dir'] = np.sign(paired['text_score'] - text_overall_mean)
agree = (paired['comm_dir'] == paired['text_dir']).mean()
print(f'\nPer-(config,question) direction agreement (above/below own system-type overall mean): '
      f'{agree:.3f}  ({int(agree*len(paired))}/{len(paired)})')
print(f'  community overall mean={comm_overall_mean:.3f}  text overall mean={text_overall_mean:.3f}')

print('\n\n' + '=' * 22, 'CHECK 3 — SCORE VARIANCE AS SECONDARY METRIC', '=' * 22)

print('\n--- per-system table (sorted by train_mean desc) ---')
tbl = per_system[['system_name', 'system_type', 'band', 'train_mean', 'mean_composite', 'std_composite', 'cv']]
tbl = tbl.sort_values('train_mean', ascending=False)
print(tbl.to_string(index=False))

print('\n--- rho(train_mean, std_composite) per system type ---')
variance_rows = []
for stype in ['text_based', 'community_based']:
    sub = per_system[per_system.system_type == stype]
    rho, p = spearmanr(sub['train_mean'], sub['std_composite'])
    interp = 'supports hypothesis (higher train_mean -> lower std)' if rho < 0 else 'does NOT support hypothesis'
    print(f'  {stype:<18} rho={rho:+.3f}  p={p:.4f}  -> {interp}')
    variance_rows.append({'system_type': stype, 'rho_train_std': rho, 'p_train_std': p})

print('\n--- variance by question subtype (across all 10 TAG systems) ---')
subtype_var = tag_df.groupby('question_subtype')['composite_score'].agg(['mean', 'std']).reset_index()
print(subtype_var.to_string(index=False))
highest_var_subtype = subtype_var.loc[subtype_var['std'].idxmax(), 'question_subtype']
print(f'\nhighest-variance subtype: {highest_var_subtype}  '
      f'-> {"confirms edge_multihop is hardest/most discriminating" if highest_var_subtype == "edge_multihop" else "does NOT confirm edge_multihop hypothesis"}')

# ---- save summary CSV ----
out = per_system.copy()
out['permutation_p_text'] = results_check1['text_based']['p_exact']
out['permutation_p_community'] = results_check1['community_based']['p_exact']
out['cross_system_rho'] = cross_rho
out = out[['system_name', 'system_type', 'band', 'train_mean', 'mean_composite', 'std_composite',
           'cv', 'permutation_p_text', 'permutation_p_community', 'cross_system_rho']]
out.to_csv(REPO_ROOT / 'data/graphrag/arxiv_sanity_checks.csv', index=False)
print(f'\n\nsaved -> data/graphrag/arxiv_sanity_checks.csv')
