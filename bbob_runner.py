from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Tuple, Type

import numpy as np
import cocoex

from ipop_maes import IPOPMAES, IPOPConfig
from maes import MAESConfig
from cma_es_wrapper import run_cma_es



SuiteFun = Callable[[np.ndarray], float]


def _run_single_instance(
    factory: Callable[[SuiteFun, int, int, int | None],
                      Tuple[np.ndarray, float]],
    *,
    dim: int,
    budget_multiplier: int,
    seed: int | None,
    suite_name: str = "bbob",
    output_folder: str = "bbob_results",
) -> Dict[str, np.ndarray]:

    full_path = Path(f"exdata/{output_folder}").resolve()
    full_path.mkdir(parents=True, exist_ok=True)

    suite = cocoex.Suite(suite_name, "", f"dimensions:{dim}")
    observer = cocoex.Observer(suite_name, f"result_folder:{output_folder}")

    results: Dict[str, np.ndarray] = {}
    for problem in suite:
        problem.observe_with(observer)
        _, best_f = factory(problem, dim, budget_multiplier, seed)
        results[str(problem.id)] = np.asarray(best_f)

    del observer, suite
    return results


def make_ipop_maes_factory(dim: int, stall: int = 50, factor: int = 2):
    def factory(problem, dim_, budget_multiplier, seed):
        cfg = MAESConfig(dim=dim)
        x0 = np.zeros(dim)                      # start z (0,…,0)
        IpopConfig = IPOPConfig(stall_generations=stall, ipop_factor=factor)
        ipop = IPOPMAES(problem, x0, cfg, IpopConfig)
        return ipop.run(budget_multiplier * dim)   # -> (best_x, best_f)
    return factory



def make_cma_es_factory(dim: int):
    def factory(problem, dim_, budget_multiplier, seed):
        x0 = np.zeros(dim)
        return run_cma_es(problem, x0, sigma0=0.3,
                          max_evals=budget_multiplier * dim,
                          seed=seed)
    return factory