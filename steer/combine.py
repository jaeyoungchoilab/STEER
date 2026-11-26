"""steer combine: STEER, the combination of ES and ERC (evolutionary rate covariation), for every protein pair.

  steer combine --es es_out --erc erc_dir --out steer_out
  steer combine --es es_out --erc erc_pairs.tsv --alias names.tsv --out steer_out

ES and ERC are each ranked over the pairs that have a value (ties: average rank) and turned into normal scores,
z = PHI^-1(rank / (n + 1)); STEER = (z_ES + z_ERC) / sqrt(k), where k is the number of scores the pair has. A pair
involving a protein that ERC does not score keeps its ES normal score, so every protein of the ES matrix stays in.
Nothing is fitted. STEER is a ranking within this protein set: values from different sets are not comparable.

ERC input (--erc), either
  a folder with ERC_genes.txt, ERC_r.f32 and ERC_n.f32 (the ERC2.0 export: gene names, and G x G float32 matrices,
      column-major, of the correlation r and the number of branches n), or
  a TSV with a header and the columns gene_a, gene_b and r with n (branches), or ft (FtERC) directly.
ERC is used as FtERC = atanh(r) x sqrt(n - 3), r clipped to +-0.9999 (as ERC2.0). ERC gene names are matched to
proteins.txt, ignoring case; --alias (TSV: name in proteins.txt, then other names) matches other names.

Outputs in --out: proteins.txt, steer.npy (n x n, NaN diagonal; float32, or float16 with --float16), info.json.
Memory: roughly 6-10 GB for 14,000 proteins (it grows with the number of protein pairs).
"""
import json
import os
import shutil
import sys

import numpy as np


def normal_scores(v):
    from scipy.stats import norm, rankdata
    return norm.ppf(rankdata(v, method="average") / (len(v) + 1.0)).astype(np.float32)


def name_map(names, alias):
    """upper-case name or alias -> index in names."""
    idx = {g.upper(): i for i, g in enumerate(names)}
    if alias:
        for line in open(alias):
            x = line.rstrip("\n").split("\t")
            if x and x[0].upper() in idx:
                for other in x[1:]:
                    if other:
                        idx.setdefault(other.upper(), idx[x[0].upper()])
    return idx


def read_erc(path, names, alias, clip=0.9999):
    """FtERC over the proteins (n x n float32, NaN where ERC has no value) and the proteins ERC scores."""
    idx = name_map(names, alias)
    n = len(names)
    F = np.full((n, n), np.nan, np.float32)
    if os.path.isdir(path):
        eg = open(os.path.join(path, "ERC_genes.txt")).read().split()
        G = len(eg)
        pos = [(idx[g.upper()], k) for k, g in enumerate(eg) if g.upper() in idx]
        if not pos:
            sys.exit("no ERC gene matches a protein name (use --alias)")
        mine, theirs = np.array([p for p, _ in pos]), np.array([k for _, k in pos])
        R = np.memmap(os.path.join(path, "ERC_r.f32"), dtype=np.float32, mode="r", shape=(G, G), order="F")
        N = np.memmap(os.path.join(path, "ERC_n.f32"), dtype=np.float32, mode="r", shape=(G, G), order="F")
        probe = np.asarray(N[:min(G, 300), :min(G, 300)]).ravel()
        probe = probe[np.isfinite(probe) & (probe != 0)]
        if probe.size and np.abs(probe).max() < 1e-30:          # counts written by R as 4-byte integers
            N = np.memmap(os.path.join(path, "ERC_n.f32"), dtype=np.int32, mode="r", shape=(G, G), order="F")
        r = np.asarray(R[np.ix_(theirs, theirs)], np.float32)
        nb = np.asarray(N[np.ix_(theirs, theirs)], np.float32)
        F[np.ix_(mine, mine)] = np.arctanh(np.clip(r, -clip, clip)) * np.sqrt(np.maximum(nb - 3, 1))
    else:
        import pandas as pd
        T = pd.read_csv(path, sep="\t")
        a = T["gene_a"].astype(str).str.upper().map(idx)
        b = T["gene_b"].astype(str).str.upper().map(idx)
        ok = a.notna() & b.notna()
        if not ok.any():
            sys.exit("no ERC pair matches protein names (use --alias)")
        if "ft" in T:
            v = T["ft"].to_numpy(np.float32)
        else:
            v = np.arctanh(np.clip(T["r"].to_numpy(np.float64), -clip, clip)) * \
                np.sqrt(np.maximum(T["n"].to_numpy(np.float64) - 3, 1))
        i, j, v = a[ok].astype(int).to_numpy(), b[ok].astype(int).to_numpy(), np.asarray(v)[ok.to_numpy()]
        F[i, j] = F[j, i] = v.astype(np.float32)
        mine = np.unique(np.r_[i, j])
    np.fill_diagonal(F, np.nan)
    have = np.zeros(n, bool)
    have[mine] = True
    return F, have


def steer_matrix(E, F):
    """STEER from ES and FtERC matrices over the same proteins (n x n, NaN diagonal)."""
    n = len(E)
    gi, gj = np.triu_indices(n, 1)
    sc = {"ES": np.asarray(E[gi, gj], np.float32), "ERC": np.asarray(F[gi, gj], np.float32)}
    k = np.zeros(len(gi), np.int8)
    total = np.zeros(len(gi), np.float32)
    for v in sc.values():
        ok = np.isfinite(v)
        total[ok] += normal_scores(v[ok])
        k += ok
    S = np.full((n, n), np.nan, np.float32)
    S[gi, gj] = S[gj, gi] = np.where(k > 0, total / np.sqrt(np.maximum(k, 1)), np.nan)
    return S, {int(c): int((k == c).sum()) for c in (0, 1, 2)}


def add_args(ap):
    ap.add_argument("--es", required=True, help="a `steer es` output folder (proteins.txt, es.npy)")
    ap.add_argument("--erc", required=True, help="ERC folder (ERC_genes.txt, ERC_r.f32, ERC_n.f32) or pair TSV")
    ap.add_argument("--alias", default="", help="TSV: name in proteins.txt, then other names (to match ERC genes)")
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--float16", action="store_true", help="write steer.npy as float16")


def run(a):
    names = open(os.path.join(a.es, "proteins.txt")).read().split()
    E = np.load(os.path.join(a.es, "es.npy"), mmap_mode="r")
    F, have = read_erc(a.erc, names, a.alias)
    print(f"{len(names)} proteins; ERC scores {int(have.sum())} of them", flush=True)
    S, classes = steer_matrix(E, F)
    os.makedirs(a.out, exist_ok=True)
    shutil.copy(os.path.join(a.es, "proteins.txt"), os.path.join(a.out, "proteins.txt"))
    np.save(os.path.join(a.out, "steer.npy"), S.astype(np.float16 if a.float16 else np.float32))
    json.dump(dict(method="Stouffer combination of van der Waerden normal scores of ES and FtERC", es=a.es,
                   erc=a.erc, proteins=len(names), proteins_with_erc=int(have.sum()),
                   pairs_by_number_of_scores=classes, dtype="float16" if a.float16 else "float32"),
              open(os.path.join(a.out, "info.json"), "w"), indent=1)
    print(f"wrote {a.out}/steer.npy (pairs with both scores: {classes[2]:,}; ES only: {classes[1]:,})")
