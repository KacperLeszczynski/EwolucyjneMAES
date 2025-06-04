from __future__ import annotations

import argparse
import functools
from pathlib import Path
from itertools import product

import numpy as np

from bbob_runner import make_cma_es_factory, make_ipop_maes_factory, _run_single_instance



def main() -> None:
    parser = argparse.ArgumentParser(description="Run IPOP-MA-ES vs. CMA-ES on BBOB.")
    parser.add_argument("--dims", nargs="+", type=int, default=[10],
                        help="Lista wymiarów, np. 5 10 20")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0],
                        help="Lista seedów – np. 0 1 2 ... 9")
    parser.add_argument("--budget-mult", type=int, default=100,
                        help="F-evals per dimension")
    args = parser.parse_args()

    for dim, seed in product(args.dims, args.seeds):
        ipop_factory = make_ipop_maes_factory(dim)
        cma_factory = make_cma_es_factory(dim)

        for factory in (ipop_factory, cma_factory):
            results = _run_single_instance(
                factory,
                dim=dim,
                budget_multiplier=args.budget_mult,
                seed=seed,
                output_folder=f"bbob_results/d{dim}/seed{seed}"
            )
            label = next(iter(results)).split("_")[0]
            factory_name = "ipop" if factory == ipop_factory else "cma"
            out_file = Path(f"npz/{factory_name}_{label}_d{dim}_seed{seed}.npz")
            out_file.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out_file, **results)
            print(f"[d={dim} seed={seed}] saved → {out_file}")


if __name__ == "__main__":
    main()