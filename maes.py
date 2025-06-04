from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np


@dataclass
class MAESConfig:
    dim: int
    sigma0: float = 0.3
    popsize: int | None = None
    seed: int | None = None


class MAES:

    def __init__(self, f: Callable[[np.ndarray], float], x0: np.ndarray, cfg: MAESConfig):
        self.f = f
        self.dim = cfg.dim
        self.rng = np.random.default_rng(cfg.seed)

        self.popsize = cfg.popsize or 4 + int(3 * np.log(self.dim))
        self.mu = self.popsize // 2
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights /= np.sum(self.weights)
        self.mueff = 1.0 / np.sum(self.weights ** 2)

        self.sigma = cfg.sigma0
        self.c_sigma = (self.mueff + 2) / (self.dim + self.mueff + 5)
        self.d_sigma = 1 + 2 * max(0.0, np.sqrt((self.mueff - 1) / (self.dim + 1)) - 1) + self.c_sigma

        self.p_sigma = np.zeros(self.dim)
        self.p_c = np.zeros(self.dim)
        self.c_c = (4 + self.mueff / self.dim) / (self.dim + 4 + 2 * self.mueff / self.dim)

        self.c_1 = 2 / ((self.dim + 1.3) ** 2 + self.mueff)

        self.M = np.eye(self.dim)

        self.mean = x0.astype(float)
        self.generation = 0
        self.best_y = np.inf
        self.best_x = x0.copy()

    def ask(self) -> Tuple[np.ndarray, np.ndarray]:
        z = self.rng.standard_normal((self.popsize, self.dim))
        y = z @ self.M.T  # implicit covariance
        x = self.mean + self.sigma * y
        return x, y

    def tell(self, x: np.ndarray, y: np.ndarray, fitness: np.ndarray) -> None:
        idx = np.argsort(fitness)
        x_sel = x[idx[: self.mu]]
        y_sel = y[idx[: self.mu]]

        if fitness[idx[0]] < self.best_y:
            self.best_y = fitness[idx[0]]
            self.best_x = x_sel[0].copy()

        old_mean = self.mean.copy()
        self.mean = np.sum(x_sel * self.weights[:, None], axis=0)

        y_w = np.sum(y_sel * self.weights[:, None], axis=0)
        self.p_sigma = (1 - self.c_sigma) * self.p_sigma + np.sqrt(self.c_sigma * (2 - self.c_sigma) * self.mueff) * y_w

        norm_p_sigma = np.linalg.norm(self.p_sigma)
        expected_norm = np.sqrt(self.dim) * (1 - 1 / (4 * self.dim) + 1 / (21 * self.dim**2))
        self.sigma *= np.exp((self.c_sigma / self.d_sigma) * (norm_p_sigma / expected_norm - 1))

        h_sigma = 1.0 if norm_p_sigma < (1.4 + 2 / (self.dim + 1)) * expected_norm else 0.0
        self.p_c = (1 - self.c_c) * self.p_c + h_sigma * np.sqrt(self.c_c * (2 - self.c_c) * self.mueff) * (self.mean - old_mean) / self.sigma

        u = self.p_c
        v = y_sel[0]
        self.M = (1 - self.c_1) * self.M + self.c_1 * np.outer(u, v)

        self.generation += 1

    def step(self) -> Tuple[np.ndarray, float]:
        x, y = self.ask()
        fit = np.array([self.f(xi) for xi in x])
        self.tell(x, y, fit)
        return self.best_x, self.best_y

    def run(self, max_evals: int) -> Tuple[np.ndarray, float]:
        evals = 0
        while evals < max_evals:
            x, y = self.ask()
            fit = np.array([self.f(xi) for xi in x])
            evals += len(fit)
            self.tell(x, y, fit)
        return self.best_x, self.best_y