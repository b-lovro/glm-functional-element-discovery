#!/usr/bin/env python
"""Positive-control smoke test for the TF-MoDISco + report toolchain.

This does NOT use RiNALMo data. It builds a SYNTHETIC dataset with a planted,
known motif (AP-1, TGACTCA) so that TF-MoDISco is guaranteed to discover at
least one pattern. The purpose is to validate that the whole pipeline runs:

    modisco motifs  ->  modisco report (with JASPAR TOMTOM matching)

Use it to confirm the report renders and TOMTOM matches to JASPAR work. The
resulting motifs are synthetic and carry NO biological meaning for the promoter
project.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

MOTIF = "TGACTCA"  # AP-1 / TRE consensus, present in JASPAR (FOS/JUN family)
NUC = np.array(list("ACGT"))
IDX = {c: i for i, c in enumerate(NUC)}


def build(n: int, L: int, seed: int, embed_frac: float):
    rng = np.random.RandomState(seed)
    m = len(MOTIF)
    one_hot = np.zeros((n, 4, L), dtype=np.float32)
    contrib = np.zeros((n, 4, L), dtype=np.float32)

    for i in range(n):
        seq = rng.randint(0, 4, size=L)
        # small signed background noise (gives TF-MoDISco a real negative side)
        track = rng.normal(0.0, 0.1, size=L).astype(np.float32)
        if rng.rand() < embed_frac:
            pos = rng.randint(0, L - m)
            for j, ch in enumerate(MOTIF):
                seq[pos + j] = IDX[ch]
            track[pos : pos + m] += rng.uniform(2.0, 3.0)  # strong positive block
        for p in range(L):
            one_hot[i, seq[p], p] = 1.0
        contrib[i] = one_hot[i] * track[None, :]  # project onto observed base
    return one_hot, contrib


def export_arr0(path, arr):
    np.savez_compressed(path, arr.astype("float32"))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", default="outputs/tfmodisco_runs/positive_control")
    p.add_argument("--n", type=int, default=800)
    p.add_argument("--length", type=int, default=100)
    p.add_argument("--embed-frac", type=float, default=0.6)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    one_hot, contrib = build(args.n, args.length, args.seed, args.embed_frac)
    print(f"synthetic: one_hot {one_hot.shape}, contrib range "
          f"{contrib.min():.2f}..{contrib.max():.2f}, frac<0 {(contrib<0).mean():.3f}")
    export_arr0(os.path.join(args.out_dir, "ohe.npz"), one_hot)
    export_arr0(os.path.join(args.out_dir, "contrib.npz"), contrib)
    print(f"wrote ohe.npz / contrib.npz to {args.out_dir} (planted motif: {MOTIF})")


if __name__ == "__main__":
    main()
