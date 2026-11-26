"""ESM-2 protein embeddings with Hugging Face transformers.

Each protein is run through ESM-2 and its last-layer vectors (after the final layer norm; <cls> and <eos> dropped)
are averaged over residues, giving one vector per protein (1,280 numbers for the 650M model). Proteins longer than
WINDOW residues are embedded in overlapping windows (step STEP) and stitched back: each residue's vector is a
weighted mean over the windows that hold it, the weight rising towards the window's centre (a tent) except at the
protein's real ends. On a GPU the forward pass runs in float16 (autocast).
"""
import numpy as np
import torch

MODEL = "facebook/esm2_t33_650M_UR50D"
WINDOW, STEP = 1022, 511
STANDARD = set("ACDEFGHIKLMNPQRSTVWYXBZUO")


def load_model(name=MODEL, device=None):
    """Tokenizer, model (eval mode) and device; name is a Hugging Face model id or a local folder."""
    from transformers import AutoTokenizer, EsmModel
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(name)
    model = EsmModel.from_pretrained(name, add_pooling_layer=False).to(device).eval()
    return tok, model, device


def clean(seq):
    """A protein without gaps, stops and whitespace; letters outside the amino acid alphabet become X."""
    return "".join(c if c in STANDARD else "X" for c in seq.upper() if c not in "-.* \t\r\n")


def windows(L, win=WINDOW, step=STEP):
    if L <= win:
        return [(0, L)]
    starts = list(range(0, L - win + 1, step))
    if starts[-1] + win < L:
        starts.append(L - win)
    return [(s, s + win) for s in starts]


def tent(s, e, L):
    n = e - s
    pos = np.arange(n)
    w = np.minimum(pos + 1, n - pos).astype(np.float32)
    if s == 0:
        w[: n // 2] = w.max()          # the protein start is a real end, not a cut
    if e == L:
        w[n // 2:] = w.max()
    return w


@torch.no_grad()
def embed_residues(seqs, tok, model, device, max_tokens=12000):
    """Per-residue last-layer vectors (L x D float32) for each sequence in seqs (cleaned, non-empty)."""
    chunks = [((k, s), q[s:e]) for k, q in enumerate(seqs) for s, e in windows(len(q))]
    order = sorted(range(len(chunks)), key=lambda i: len(chunks[i][1]))
    out, i = {}, 0
    while i < len(order):
        j = i
        while j < len(order) and (j - i + 1) * (len(chunks[order[j]][1]) + 2) <= max_tokens:
            j += 1
        j = max(j, i + 1)
        batch = [chunks[k] for k in order[i:j]]
        enc = tok([s for _, s in batch], return_tensors="pt", padding=True, add_special_tokens=True)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.autocast("cuda", dtype=torch.float16, enabled=str(device).startswith("cuda")):
            h = model(**enc).last_hidden_state.float().cpu().numpy()
        for b, (key, s) in enumerate(batch):
            out[key] = h[b, 1:1 + len(s)].copy()
        i = j
    tables = []
    for k, q in enumerate(seqs):
        L = len(q)
        acc, wsum = None, np.zeros(L, np.float32)
        for s, e in windows(L):
            w = tent(s, e, L)
            v = out[(k, s)]
            acc = np.zeros((L, v.shape[1]), np.float32) if acc is None else acc
            acc[s:e] += w[:, None] * v
            wsum[s:e] += w
        tables.append(acc / wsum[:, None])
    return tables


def mean_embeddings(seqs, tok=None, model=None, device=None, model_name=MODEL, max_tokens=12000, chunk=200,
                    progress=None):
    """Mean embedding (n x D float32) of each protein sequence. Sequences are cleaned first and must not be empty.
    progress(done, total) is called after every chunk of proteins."""
    if model is None:
        tok, model, device = load_model(model_name, device)
    seqs = [clean(s) for s in seqs]
    if any(not s for s in seqs):
        raise ValueError("empty protein sequence")
    rows = []
    for c0 in range(0, len(seqs), chunk):
        rows += [t.mean(0) for t in embed_residues(seqs[c0:c0 + chunk], tok, model, device, max_tokens)]
        if progress:
            progress(min(c0 + chunk, len(seqs)), len(seqs))
    return np.array(rows, np.float32)
