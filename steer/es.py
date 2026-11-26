"""steer es: an ES (embedding similarity) matrix from a FASTA file of protein sequences.

  steer es proteins.fa --out es_out                    # embed with ESM-2 650M, then the ES matrix
  steer es --embeddings es_out/embeddings.npz --out es_small --center proteome/embeddings.npz

Input: one protein per FASTA record; the record's name (first word of the header) names the protein in every output,
so give one protein per gene (for example the longest isoform) and unique names. Gaps and stop characters are
removed; letters outside the amino acid alphabet become X. Proteins shorter than --min-length are left out (listed in
info.json).

Outputs in --out:
  proteins.txt      protein names, in matrix order
  embeddings.npz    names, mean (n x D float32 mean embeddings), length; reusable with --embeddings or --center
  es.npy            n x n ES matrix (float32, or float16 with --float16; NaN on the diagonal); numpy.load(...,
                    mmap_mode="r") reads rows without loading the whole matrix
  es.tsv            the matrix as a table with row and column names (with --tsv; only sensible for small sets)
  info.json         model, settings, centring, proteins left out
Embedding is the slow step (use a GPU for a whole proteome; a few hundred proteins run on a CPU). It saves
embeddings.partial.npz as it goes, so an interrupted run started again with the same command continues.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

from .embed import MODEL, clean
from .similarity import es_matrix


def read_fasta(path):
    """[(name, sequence)] from a FASTA file (optionally gzipped); name = first word of the header."""
    import gzip
    recs, name, buf = [], None, []
    with (gzip.open(path, "rt") if path.endswith(".gz") else open(path)) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if name is not None:
                    recs.append((name, "".join(buf)))
                name, buf = (line[1:].split() or [""])[0], []
            elif line:
                buf.append(line)
    if name is not None:
        recs.append((name, "".join(buf)))
    return recs


def load_embeddings(path):
    z = np.load(path, allow_pickle=False)
    return [str(s) for s in z["names"]], z["mean"].astype(np.float32), z


def embed_fasta(path, out, model_name, device, max_tokens, min_length, log):
    from .embed import load_model, mean_embeddings
    recs = read_fasta(path)
    names = [n for n, _ in recs]
    dup = sorted({n for n in names if names.count(n) > 1}) if len(set(names)) < len(names) else []
    if dup:
        sys.exit(f"duplicate protein names in {path}: {', '.join(dup[:10])}")
    seqs = {n: clean(s) for n, s in recs}
    short = [n for n in names if len(seqs[n]) < min_length]
    keep = [n for n in names if len(seqs[n]) >= min_length]
    log(f"{len(names)} proteins in {path}; {len(short)} shorter than {min_length} residues left out")
    partial = os.path.join(out, "embeddings.partial.npz")
    done = {}
    if os.path.exists(partial):
        pn, pm, z = load_embeddings(partial)
        if str(z["model"]) == model_name:
            done = dict(zip(pn, pm))
            log(f"continuing: {len(done)} proteins already embedded")
    todo = sorted([n for n in keep if n not in done], key=lambda n: len(seqs[n]))
    if todo:
        tok, model, device = load_model(model_name, device)
        log(f"model {model_name} on {device}")
        step = 500
        for c0 in range(0, len(todo), step):
            part = todo[c0:c0 + step]
            for n, v in zip(part, mean_embeddings([seqs[n] for n in part], tok, model, device, max_tokens=max_tokens)):
                done[n] = v
            np.savez(partial + ".tmp.npz", names=np.array(list(done)), mean=np.array(list(done.values())),
                     model=model_name)
            os.replace(partial + ".tmp.npz", partial)
            log(f"embedded {min(c0 + step, len(todo))}/{len(todo)}")
    V = np.array([done[n] for n in keep], np.float32)
    np.savez_compressed(os.path.join(out, "embeddings.npz"), names=np.array(keep), mean=V,
                        length=np.array([len(seqs[n]) for n in keep]), model=model_name)
    if os.path.exists(partial):
        os.remove(partial)
    return keep, V, short


def add_args(ap):
    ap.add_argument("fasta", nargs="?", help="protein sequences (FASTA, .gz allowed)")
    ap.add_argument("--embeddings", default="", help="start from an embeddings.npz instead of a FASTA file")
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--model", default=MODEL, help=f"ESM-2 model id or local folder (default {MODEL}, as the paper)")
    ap.add_argument("--device", default=None, help="cuda, cuda:1, cpu (default: a GPU if there is one)")
    ap.add_argument("--max-tokens", type=int, default=12000, help="tokens per batch (lower it if the GPU runs out "
                                                                  "of memory)")
    ap.add_argument("--min-length", type=int, default=30, help="shorter proteins are left out (1 keeps all)")
    ap.add_argument("--center", default="set", help="'set' (default: the mean embedding of these proteins), 'none',"
                                                     " or an embeddings.npz whose mean is used (e.g. a whole proteome)")
    ap.add_argument("--float16", action="store_true", help="write es.npy as float16 (half the size)")
    ap.add_argument("--tsv", action="store_true", help="also write the matrix as a named table (small sets only)")


def run(a):
    if bool(a.fasta) == bool(a.embeddings):
        sys.exit("give a FASTA file or --embeddings, not both")
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:7.0f} s] {m}", flush=True)

    if a.embeddings:
        names, V, z = load_embeddings(a.embeddings)
        model_name, short = str(z["model"]) if "model" in z else "unknown", []
        if os.path.abspath(a.embeddings) != os.path.abspath(os.path.join(a.out, "embeddings.npz")):
            np.savez_compressed(os.path.join(a.out, "embeddings.npz"), **{k: z[k] for k in z.files})
    else:
        model_name = a.model
        names, V, short = embed_fasta(a.fasta, a.out, a.model, a.device, a.max_tokens, a.min_length, log)
    if len(names) < 2:
        sys.exit("fewer than 2 proteins")

    if a.center == "set":
        center, how = None, f"mean embedding of the {len(names)} proteins"
    elif a.center == "none":
        center, how = False, "none"
    else:
        cn, cv, cz = load_embeddings(a.center)
        if "model" in cz and str(cz["model"]) != model_name:
            sys.exit(f"--center embeddings come from {cz['model']}, these from {model_name}")
        center, how = cv.astype(np.float64).mean(0), f"mean embedding of the {len(cn)} proteins in {a.center}"

    with open(os.path.join(a.out, "proteins.txt"), "w") as f:
        f.write("\n".join(names) + "\n")
    M = es_matrix(V, center=center, out=os.path.join(a.out, "es.npy"),
                  dtype=np.float16 if a.float16 else np.float32)
    log(f"wrote es.npy ({len(names)} x {len(names)})")
    if a.tsv:
        import pandas as pd
        pd.DataFrame(np.asarray(M, np.float32), index=names, columns=names).to_csv(
            os.path.join(a.out, "es.tsv"), sep="\t", float_format="%.4f", na_rep="NA")
    info = dict(model=model_name, layer="last", representation="mean over residues", proteins=len(names),
                dimension=int(V.shape[1]), center=how, dtype="float16" if a.float16 else "float32",
                min_length=a.min_length, left_out_short=short, input=a.fasta or a.embeddings)
    json.dump(info, open(os.path.join(a.out, "info.json"), "w"), indent=1)
    log("done")
