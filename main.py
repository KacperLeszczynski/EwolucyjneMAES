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
        # ipop_factory = None
        ipop_35_factory = make_ipop_maes_factory(dim, stall=35, factor=2)
        ipop_20_factory = make_ipop_maes_factory(dim, stall=20, factor=2)

        for factory in (ipop_factory, cma_factory, ipop_35_factory, ipop_20_factory):
        # for factory in (ipop_35_factory, ipop_20_factory):
            results = _run_single_instance(
                factory,
                dim=dim,
                budget_multiplier=args.budget_mult,
                seed=seed,
                output_folder=f"bbob_results/d{dim}/seed{seed}"
            )
            label = next(iter(results)).split("_")[0]
            if factory == ipop_factory:
                factory_name = "ipop"
            elif factory == ipop_20_factory:
                factory_name = "ipop_20"
            elif factory == ipop_35_factory:
                factory_name = "ipop_35"
            else:
                factory_name = "cma"

            out_file = Path(f"npz/{factory_name}_{label}_d{dim}_seed{seed}.npz")
            out_file.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out_file, **results)
            print(f"[d={dim} seed={seed}] saved → {out_file}")


if __name__ == "__main__":
    main()