# TACO-Graph reading list (judged from abstracts, 2026-09-22)

Caveat: ranked from ABSTRACTS only, not full texts. Similarity = SPECTER v1 cosine
to the v7 abstract; note how weakly it correlates with actual priority below.

## TIER 1 — read before you submit (4)
1. AutoGrable: What Is a Good Graph for a Table?  arXiv:2608.11431 (2026, 0 cites, sim 0.856)
   Same problem, competing cheap-scoring method. No trained GNN needed.
2. RDB2G-Bench  arXiv:2506.01360 (NeurIPS 2025 D&B, 8 cites, sim 0.861)
   Same core artifact (construction->performance table) + same motivating claim.
   Code: github.com/chlehdwon/RDB2G-Bench
3. Relatron  arXiv:2602.22552 (2026, 2 cites, sim 0.804)  <- AutoG's own authors
   "Validation accuracy is an unreliable guide" = direct threat to proxy scoring.
4. Is Fixing Schema Graphs Necessary? (FROG)  arXiv:2605.21475 (2026, 0 cites, sim 0.815)
   Explicitly attacks fixed construction; makes table roles learnable.

## TIER 2 — read for methodology defence (4)
5. ZAPS: Zero-Cost Active Proxy Search  arXiv:2609.14184 (2026, sim 0.781)
6. RoSE: Automatic Relation Decomposition  arXiv:2405.18581 (2024, sim 0.911)
7. 4DBInfer  arXiv:2404.18209 (NeurIPS 2024 D&B, 27 cites, sim 0.877)
8. Graph ML Meets Multi-Table Relational Data  KDD 2024, DOI 10.1145/3637528.3671471

## TIER 3 — skim (6)
9.  GNN4TDL survey  arXiv:2401.02143 (ACM CSUR)
10. TEG-DB  arXiv:2406.10310 (NeurIPS 2024 D&B)
11. Multi-Scale Heterogeneous TAG datasets  arXiv:2412.08937 (WWW)
12. LAGA (When LLM Agents Meet Graph Optimization)  sim 0.897
13. TAGLAS  arXiv:2406.14683
14. GL-Agent / LLM4GNAS  (LLM agents configuring graph learning)

## TIER 4 — cite only, do not read
Griffin, GraphMaster, TabGSL, G2T-FM, GAugLLM, HiCom, OpenRTAG, GAGA, AutoGL, CAAFE
