"""STEER: embedding similarity (ES) of proteins from ESM-2, and its combination with ERC (STEER).

    from steer import mean_embeddings, es_matrix
    V = mean_embeddings(["MKT...", "MSE..."])     # n x 1280 mean ESM-2 embeddings
    M = es_matrix(V)                              # n x n ES (cosine of centred embeddings), NaN diagonal

The command-line tool `steer` (steer/cli.py) does this from a FASTA file (`steer es`), combines ES with an ERC
matrix (`steer combine`) and runs simple analyses (`steer analyze`).
"""
from .similarity import es_matrix, es_pair, top_partners, unit_rows  # noqa: F401

__version__ = "1.0.0"


def mean_embeddings(*args, **kwargs):
    """See steer.embed.mean_embeddings (imported on first use, so that torch is needed only for embedding)."""
    from .embed import mean_embeddings as f
    return f(*args, **kwargs)
