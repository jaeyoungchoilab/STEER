"""steer: protein embedding similarity (ES) and its combination with ERC (STEER).

  steer es proteome.fa --out es_out                     the ES matrix from protein sequences (ESM-2)
  steer combine --es es_out --erc erc_dir --out steer   STEER: ES combined with an ERC matrix you supply
  steer analyze partners es_out PROTEIN                 partners, pairs, group tests and rankings from a matrix

`steer <command> --help` describes each command.
"""
import argparse

from . import analyze, combine, es


def main(argv=None):
    ap = argparse.ArgumentParser(prog="steer", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    for name, mod, short in (("es", es, "ES matrix from protein sequences"),
                             ("combine", combine, "STEER from an ES matrix and an ERC matrix"),
                             ("analyze", analyze, "analyses of an ES or STEER matrix")):
        p = sub.add_parser(name, help=short, description=mod.__doc__,
                           formatter_class=argparse.RawDescriptionHelpFormatter)
        mod.add_args(p)
        p.set_defaults(run=mod.run)
    a = ap.parse_args(argv)
    a.run(a)


if __name__ == "__main__":
    main()
