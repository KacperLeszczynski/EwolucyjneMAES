from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np

from maes import MAES, MAESConfig


@dataclass
class IPOPConfig:
    stall_generations: int = 50
    ipop_factor: int = 2
    max_restarts: int | None = None


class IPOPMAES:

    def __init__(
        self,
        f: Callable[[np.ndarray], float],
        x0: np.ndarray,
        cfg: MAESConfig,
        ipop: IPOPConfig = IPOPConfig(),
    ) -> None:
        self.f = f
        self.x0 = x0.copy()
        self.base_cfg = cfg
        self.ipop = ipop
        self.restart_counter = 0
        self._build_inner()

        self.global_best_y = np.inf
        self.global_best_x = x0.copy()

    def _build_inner(self) -> None:
        new_cfg = MAESConfig(
            dim=self.base_cfg.dim,
            sigma0=self.base_cfg.sigma0,
            popsize=(self.base_cfg.popsize or 4 + int(3 * np.log(self.base_cfg.dim))) * (self.ipop.ipop_factor**self.restart_counter),
            seed=None if self.base_cfg.seed is None else self.base_cfg.seed + self.restart_counter,
        )
        self.es = MAES(self.f, self.x0, new_cfg)
        self._stall_counter = 0
        self._prev_best = np.inf

    def run(self, max_evals: int) -> Tuple[np.ndarray, float]:
        evals = 0
        while evals < max_evals:
            x, y = self.es.ask()
            fitness = np.array([self.f(xi) for xi in x])
            evals += len(fitness)
            self.es.tell(x, y, fitness)

            if self.es.best_y < self.global_best_y:
                self.global_best_y = self.es.best_y
                self.global_best_x = self.es.best_x.copy()

            # Check stagnation (relative improvement < 1e‑12)
            if self._prev_best - self.es.best_y < 1e-12:
                self._stall_counter += 1
            else:
                self._stall_counter = 0
            self._prev_best = self.es.best_y

            # Restart if stagnated
            if self._stall_counter >= self.ipop.stall_generations:
                self.restart_counter += 1
                if self.ipop.max_restarts is not None and self.restart_counter > self.ipop.max_restarts:
                    break  # no more restarts allowed
                self._build_inner()

        return self.global_best_x, self.global_best_y
