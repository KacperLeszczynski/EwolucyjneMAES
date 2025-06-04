from __future__ import annotations

from typing import Callable, Tuple

import numpy as np
import cma


def run_cma_es(f: Callable[[np.ndarray], float], x0: np.ndarray, sigma0: float, max_evals: int, seed: int | None = None) -> Tuple[np.ndarray, float]:
    opts = {
        "seed": seed,
        "verb_log": 0,
        "maxfevals": max_evals,
        "verb_disp": 0,
    }
    es = cma.CMAEvolutionStrategy(x0.tolist(), sigma0, opts)
    while not es.stop():
        xs = es.ask()
        es.tell(xs, [f(np.asarray(x)) for x in xs])
    return np.asarray(es.best.get()[0]), es.best.f
