"""steer analyze: simple analyses of a matrix written by `steer es` (es.npy) or `steer combine` (steer.npy).

  steer analyze partners DIR PROTEIN [--top 20]            a protein's most similar proteins
  steer analyze pair DIR PROTEIN_A PROTEIN_B               the score of one pair and how it ranks
  steer analyze group DIR --proteins A,B,C,... --out grp   the scores among a set of proteins: heatmap and group test
  steer analyze rank DIR --known A,B,C [--candidates X,Y]  rank proteins by mean score to a set of known proteins

DIR holds proteins.txt and es.npy or steer.npy (--matrix chooses when both are there). Proteins are given by their
names in proteins.txt (case is ignored), as a comma-separated list or @file (one name per line). --alias FILE (tab
separated: name in the matrix, name to show such as a gene symbol, then any other names such as database IDs) lets
the other names be used and shows the second column.

'top %' is the share of all protein pairs with a score at least as high, estimated from 2,000,000 random pairs
(seed 1): 0.1 means the pair is among the top 0.1% of the matrix.
Group test: the median score over the set's pairs, against random sets of the same size drawn uniformly and drawn gene
by gene from the same decile of median score with all proteins (so that proteins scoring high with everything do not make a
set look coherent); one-sided p = (random sets with a median at least as high + 1) / (draws + 1).
"""
import argparse
import os
import sys
import warnings

import numpy as np

from .similarity import random_pair_values, top_percent


class ES:
    """A matrix folder: names, the matrix (memory-mapped) and the random-pair background."""

    def __init__(self, folder, alias="", matrix=""):
        self.dir = folder
        self.names = open(os.path.join(folder, "proteins.txt")).read().split()
        if not matrix:
            have = [m for m in ("es", "steer") if os.path.exists(os.path.join(folder, m + ".npy"))]
            if len(have) != 1:
                sys.exit(f"{folder}: " + ("no es.npy or steer.npy" if not have else "both es.npy and steer.npy; "
                                                                                   "choose with --matrix"))
            matrix = have[0]
        self.kind = matrix
        self.S = "ES" if matrix == "es" else "STEER"
        self.M = np.load(os.path.join(folder, matrix + ".npy"), mmap_mode="r")
        self.n = len(self.names)
        self.index = {g.upper(): i for i, g in enumerate(self.names)}
        self.label = list(self.names)
        if alias:
            for line in open(alias):
                x = line.rstrip("\n").split("\t")
                if len(x) >= 2 and x[0].upper() in self.index and x[1]:
                    i = self.index[x[0].upper()]
                    self.label[i] = x[1]
                    for other in x[1:]:
                        if other:
                            self.index.setdefault(other.upper(), i)
        self._bg = None

    @property
    def background(self):
        if self._bg is None:
            self._bg = random_pair_values(self.M)
        return self._bg

    def row(self, i):
        return np.asarray(self.M[i], np.float64)

    def find(self, items):
        """Indices of the named proteins (in the order given, duplicates dropped) and the names not found."""
        if len(items) == 1 and items[0].startswith("@"):
            items = [l.strip() for l in open(items[0][1:]) if l.strip()]
        found, missing = [], []
        for x in items:
            for t in x.split(","):
                t = t.strip()
                if not t:
                    continue
                i = self.index.get(t.upper())
                if i is None:
                    missing.append(t)
                elif i not in found:
                    found.append(i)
        return found, missing

    def median_es(self):
        """Each protein's median score with all other proteins (cached in median_<matrix>.npy)."""
        path = os.path.join(self.dir, f"median_{self.kind}.npy")
        if os.path.exists(path):
            return np.load(path)
        med = np.empty(self.n)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for r0 in range(0, self.n, 1000):
                med[r0:r0 + 1000] = np.nanmedian(np.asarray(self.M[r0:r0 + 1000], np.float32), axis=1)
        np.save(path, med)
        return med


def write_table(rows, header, out):
    lines = ["\t".join(header)] + ["\t".join(str(v) for v in r) for r in rows]
    if out:
        open(out, "w").write("\n".join(lines) + "\n")
        print(f"wrote {out}")
    w = [max(len(str(v)) for v in col) for col in zip(*([header] + [list(map(str, r)) for r in rows]))]
    for r in [header] + rows[:50]:
        print("  ".join(str(v).ljust(k) for v, k in zip(r, w)))
    if len(rows) > 50:
        print(f"... {len(rows) - 50} more rows" + (f" in {out}" if out else ""))


def fmt(v, d=4):
    return "" if v is None or not np.isfinite(v) else f"{v:.{d}g}"


def cmd_partners(E, a):
    idx, miss = E.find([a.protein])
    if not idx:
        sys.exit(f"not found: {a.protein}")
    i = idx[0]
    r = E.row(i)
    r[i] = -np.inf
    r = np.where(np.isfinite(r), r, -np.inf)
    best = np.argsort(-r, kind="stable")[:a.top]
    tp = top_percent(r[best], E.background)
    rows = [(k, E.label[j], fmt(r[j]), fmt(t, 3)) for k, (j, t) in enumerate(zip(best, tp), 1)]
    print(f"{E.label[i]}: top {len(rows)} of {E.n - 1} proteins by {E.S}")
    write_table(rows, ["rank", "partner", E.S, "top_%"], a.out)


def cmd_pair(E, a):
    idx, miss = E.find([a.protein_a, a.protein_b])
    if len(idx) < 2:
        sys.exit(f"not found: {', '.join(miss) or 'the same protein twice'}")
    i, j = idx
    v = float(E.M[i, j])
    ri, rj = E.row(i), E.row(j)
    rank_i = int(np.sum(ri[np.isfinite(ri)] > v)) + 1
    rank_j = int(np.sum(rj[np.isfinite(rj)] > v)) + 1
    print(f"{E.S}({E.label[i]}, {E.label[j]}) = {v:.4f}")
    print(f"  top {top_percent([v], E.background)[0]:.3g}% of all protein pairs")
    print(f"  {E.label[j]} is #{rank_i} of {E.n - 1} partners of {E.label[i]}; "
          f"{E.label[i]} is #{rank_j} of {E.n - 1} partners of {E.label[j]}")


def group_test(E, idx, draws, seed=1):
    rng = np.random.default_rng(seed)
    m = len(idx)
    tri = np.triu_indices(m, 1)

    def stat(s):
        s = np.sort(s)
        v = np.asarray(E.M[np.ix_(s, s)], np.float64)[tri]
        v = v[np.isfinite(v)]
        return np.median(v) if len(v) else np.nan

    obs = stat(np.array(idx))
    uni = np.array([stat(rng.choice(E.n, m, replace=False)) for _ in range(draws)])
    med = E.median_es()
    ok = np.isfinite(med)
    edges = np.nanquantile(med[ok], np.linspace(0, 1, 11))
    dec = np.clip(np.searchsorted(edges, med, side="right") - 1, 0, 9)
    pools = [np.flatnonzero(ok & (dec == b)) for b in range(10)]

    def matched():
        chosen = []
        for g in idx:
            pool = pools[dec[g]] if ok[g] else np.flatnonzero(ok)
            while True:
                c = int(pool[rng.integers(len(pool))])
                if c not in chosen:
                    chosen.append(c)
                    break
        return np.array(chosen)

    mat = np.array([stat(matched()) for _ in range(draws)])
    uni, mat = uni[np.isfinite(uni)], mat[np.isfinite(mat)]
    return dict(proteins=m, pairs=len(tri[0]), median=obs, median_top_percent=top_percent([obs], E.background)[0],
                random_uniform_median=float(np.median(uni)), p_uniform=(np.sum(uni >= obs) + 1) / (len(uni) + 1),
                random_matched_median=float(np.median(mat)), p_matched=(np.sum(mat >= obs) + 1) / (len(mat) + 1),
                draws=draws)


def cmd_group(E, a):
    idx, miss = E.find(a.proteins)
    if miss:
        print(f"not found: {', '.join(miss)}")
    if len(idx) < 3:
        sys.exit("need at least 3 proteins in the matrix")
    S = np.asarray(E.M[np.ix_(idx, idx)], np.float64)
    from scipy.cluster.hierarchy import leaves_list, linkage       # order: average linkage on (highest - score)
    from scipy.spatial.distance import squareform
    D = np.nanmax(S) - np.nan_to_num(S, nan=np.nanmin(S))
    np.fill_diagonal(D, 0)
    order = list(leaves_list(linkage(squareform(np.clip(D, 0, None), checks=False), method="average")))
    labels = [E.label[idx[k]] for k in order]
    S = S[np.ix_(order, order)]
    out = a.out or "group"
    rows = []
    for x in range(len(order)):
        for y in range(x + 1, len(order)):
            v = S[x, y]
            rows.append((labels[x], labels[y], fmt(v), fmt(top_percent([v], E.background)[0], 3)))
    rows.sort(key=lambda r: -float(r[2]) if r[2] else 0)
    write_table(rows, ["protein_a", "protein_b", E.S, "top_%"], out + "_pairs.tsv")
    res = group_test(E, idx, a.draws)
    with open(out + "_test.tsv", "w") as f:
        f.write("\t".join(res) + "\n" + "\t".join(fmt(v, 6) if isinstance(v, float) else str(v)
                                                  for v in res.values()) + "\n")
    print(f"group test: median {E.S} {res['median']:.3f} (top {res['median_top_percent']:.3g}% of pairs); random "
          f"sets: uniform {res['random_uniform_median']:.3f} (p = {res['p_uniform']:.2g}), matched on median {E.S} "
          f"{res['random_matched_median']:.3f} (p = {res['p_matched']:.2g}); {a.draws:,} draws each")
    print(f"wrote {out}_test.tsv")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    k = len(labels)
    size = max(3.5, 0.28 * k + 1.8)
    fig, ax = plt.subplots(figsize=(size + 0.8, size))
    hi = float(np.quantile(E.background, 0.999))
    im = ax.imshow(S, cmap="Reds", vmin=float(np.median(E.background)), vmax=hi)
    ax.set_xticks(range(k))
    ax.set_yticks(range(k))
    fs = max(4, min(8, 200 / k))
    ax.set_xticklabels(labels, rotation=90, fontsize=fs)
    ax.set_yticklabels(labels, fontsize=fs)
    if k <= 25:
        for x in range(k):
            for y in range(k):
                if x != y and np.isfinite(S[x, y]):
                    ax.text(y, x, f"{S[x, y]:.2f}", ha="center", va="center", fontsize=fs * 0.8,
                            color="white" if S[x, y] > (hi + np.median(E.background)) / 2 else "black")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(f"{E.S} (white: matrix median; red: top 0.1% of pairs)", fontsize=7)
    cb.ax.tick_params(labelsize=7)
    ax.set_title(f"median {E.S} {res['median']:.2f}; p = {res['p_matched']:.2g} against matched random sets",
                 fontsize=8)
    fig.tight_layout()
    fig.savefig(out + "_heatmap.png", dpi=200)
    fig.savefig(out + "_heatmap.pdf")
    print(f"wrote {out}_heatmap.png, {out}_heatmap.pdf")


def cmd_rank(E, a):
    known, miss = E.find(a.known)
    if miss:
        print(f"not found: {', '.join(miss)}")
    if len(known) < 2:
        sys.exit("need at least 2 known proteins in the matrix")
    F = np.column_stack([E.row(t) for t in known])
    for c, t in enumerate(known):
        F[t, c] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        score = np.nanmean(F, axis=1)
    nval = np.isfinite(F).sum(axis=1)
    ok = np.isfinite(score) & (nval >= max(2, len(known) // 2))
    ok[known] = False
    ref = np.sort(score[ok])
    if a.candidates:
        cand, cmiss = E.find(a.candidates)
        if cmiss:
            print(f"candidates not found: {', '.join(cmiss)}")
    else:
        cand = list(np.flatnonzero(ok))
    cand = sorted(cand, key=lambda c: -score[c] if ok[c] else np.inf)
    rows = []
    for k, c in enumerate(cand, 1):
        p = (len(ref) - np.searchsorted(ref, score[c], side="left") + 1) / (len(ref) + 1) if ok[c] else np.nan
        rows.append((k, E.label[c], fmt(score[c] if ok[c] else np.nan), fmt(p, 3),
                     "known protein" if c in known else ""))
    print(f"{len(known)} known proteins; score = mean {E.S} with them; p = share of the {len(ref):,} scored proteins "
          "with a score at least as high")
    shown = rows if a.candidates else rows[:a.show]
    write_table(shown, ["rank", "protein", f"mean_{E.S}", "p", "note"], None)
    if a.out:
        with open(a.out, "w") as f:
            f.write("\n".join([f"rank\tprotein\tmean_{E.S}\tp\tnote"] + ["\t".join(map(str, r)) for r in rows]) + "\n")
        print(f"wrote {a.out}")

def add_args(ap):
    sub = ap.add_subparsers(dest="analysis", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("es_dir", metavar="DIR", help="folder with proteins.txt and es.npy or steer.npy")
    common.add_argument("--matrix", choices=["es", "steer"], default="", help="which matrix, if both are in DIR")
    common.add_argument("--alias", default="", help="TSV: name in the matrix, name to show, other names")
    p = sub.add_parser("partners", parents=[common], help="a protein's most similar proteins")
    p.add_argument("protein")
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--out", default="", help="also write the table to this TSV")
    p = sub.add_parser("pair", parents=[common], help="the score of one pair and how it ranks")
    p.add_argument("protein_a")
    p.add_argument("protein_b")
    p = sub.add_parser("group", parents=[common], help="scores among a set of proteins: heatmap and group test")
    p.add_argument("--proteins", nargs="+", required=True, help="comma-separated names or @file")
    p.add_argument("--draws", type=int, default=10000, help="random sets for each null (default 10,000)")
    p.add_argument("--out", default="group", help="output prefix (_pairs.tsv, _test.tsv, _heatmap.png/.pdf)")
    p = sub.add_parser("rank", parents=[common], help="rank proteins by mean score to known proteins")
    p.add_argument("--known", nargs="+", required=True, help="comma-separated names or @file")
    p.add_argument("--candidates", nargs="+", default=None, help="proteins to rank (default: every protein)")
    p.add_argument("--show", type=int, default=30, help="rows printed (all are written with --out)")
    p.add_argument("--out", default="", help="write the full ranking to this TSV")


def run(a):
    E = ES(a.es_dir, a.alias, a.matrix)
    {"partners": cmd_partners, "pair": cmd_pair, "group": cmd_group, "rank": cmd_rank}[a.analysis](E, a)
