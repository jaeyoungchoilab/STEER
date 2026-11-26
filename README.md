# STEER: protein embedding similarity (ES)

Make an ES matrix from your protein sequences: for every pair of proteins, the cosine between their mean ESM-2
embeddings, centred on the protein set. High ES points to proteins that may work together.

## Install

```
git clone <this repository> && cd STEER
pip install .
```

## Run

```
steer es proteome.fa --out es_out
```

- `proteome.fa`: one protein per gene (for example the longest isoform); the first word of each header is the name.
- `es_out/es.npy`: the centred ES matrix (proteins × proteins), in the order of `es_out/proteins.txt`.
- Use a GPU for a whole proteome. The ESM-2 weights download on first use.

## Additional options

| Option | What it does |
|---|---|
| `--float16` | store the matrix at half the size |
| `--tsv` | also write the matrix as a table with protein names (small sets only) |
| `--center FILE.npz` | centre on another set's embeddings, e.g. score a few proteins on a proteome's scale |
| `--embeddings FILE.npz` | rebuild the matrix from saved embeddings (`es_out/embeddings.npz`), no GPU needed |
| `--min-length N` | leave out proteins shorter than N residues (default 30) |
| `--model NAME` | another ESM-2 model (default `facebook/esm2_t33_650M_UR50D`, as in the paper) |
| `--device`, `--max-tokens` | choose the GPU; lower `--max-tokens` (default 12000) if it runs out of memory |

Example, scoring a few proteins against a proteome that was embedded first:

```
steer es my_proteins.fa --out es_mine --center es_out/embeddings.npz
```

## Simple analyses

`steer analyze` answers the common questions from a finished matrix folder (no GPU needed):

```
steer analyze partners es_out PROTEIN --top 20                  # a protein's most similar proteins
steer analyze pair es_out PROTEIN_A PROTEIN_B                   # the score of one pair and how it ranks
steer analyze group es_out --proteins A,B,C,D --out mygroup     # heatmap of a set, and is its score elevated?
steer analyze rank es_out --known A,B,C --out ranked.tsv        # rank all proteins by mean score to known ones
```

- **partners**: each partner's score and top % (share of all protein pairs scoring at least as high).
- **pair**: the pair's score, its top %, and where each protein ranks among the other's partners.
- **group**: a clustered heatmap (`_heatmap.png`), every pair's score (`_pairs.tsv`) and a group test (`_test.tsv`):
  the set's median score against random sets of the same size, drawn uniformly and drawn from proteins with a similar
  median score to everything (10,000 draws each).
- **rank**: candidates for joining a set of known proteins, ranked by mean score to them (`--candidates` to rank only
  some), with p = share of proteins scoring at least as high.

Lists can be given as `A,B,C` or `@file` (one name per line). `--alias names.tsv` (tab separated: name in
`proteins.txt`, name to show, then any other names such as database IDs) lets you use and show other names.

## STEER: ES combined with ERC

If you have an ERC (evolutionary rate covariation) matrix for the same genes, combine the two:

```
steer combine --es es_out --erc erc_dir --out steer_out
steer analyze partners steer_out PROTEIN
```

`erc_dir` holds `ERC_genes.txt`, `ERC_r.f32` and `ERC_n.f32` (the ERC2.0 export format); a TSV with columns `gene_a`,
`gene_b`, `r`, `n` (branches) works too. ERC gene names must match `proteins.txt` (or be matched with `--alias`).
ES and ERC are each converted to normal scores and combined by Stouffer's method; proteins without ERC keep their ES,
so every protein stays in. ERC itself has to come from elsewhere: ERC2.0 (Little et al. 2025) published matrices for
mammals and yeast, and our fly matrix is in the data below.

## Example: the fly matrices from the paper

The D. melanogaster ES, ERC and STEER matrices are on Zenodo:
[zenodo.org/records/24559462](https://zenodo.org/records/24559462) (DOI 10.5281/zenodo.24559462). Download and unpack
them, then try:

```
wget -O fly_steer_data.zip "https://zenodo.org/records/24559462/files/fly_steer_data.zip?download=1"
unzip fly_steer_data.zip && cd fly_steer_data
steer analyze partners es del --alias fly_aliases.tsv                     # Deadlock's most similar proteins
steer analyze pair es del uno --alias fly_aliases.tsv                     # Deadlock and uno
steer analyze group steer --proteins piwi,aub,AGO3,zuc,armi,vas,krimp,tej --alias fly_aliases.tsv --out pirna
steer combine --es es --erc erc --out my_steer                            # recomputes steer/ from es/ and erc/
steer es proteins.fa --out my_es                                          # recomputes es/ from the sequences (GPU)
```

`fly_aliases.tsv` lets you use gene symbols, FlyBase IDs or annotation IDs.

The analysis code for the paper will be added to this repository.

## Citation

[Manuscript citation.] Please also cite ESM-2: Lin et al. (2023) Science 379:1123–1130.
