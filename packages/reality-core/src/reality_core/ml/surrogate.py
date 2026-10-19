from collections.abc import Sequence

import numpy as np
from scipy.interpolate import RBFInterpolator

from reality_core.discovery.oracle import DesignPoint, Observation
from reality_core.optimization.random_search import SearchBounds


class RBFSurrogate:
    """Cheap surrogate model for predicting physical simulation outcomes."""

    def __init__(self) -> None:
        self._model: RBFInterpolator | None = None

    def fit(self, observations: Sequence[Observation]) -> None:
        if len(observations) < 3:
            raise ValueError(
                "At least 3 observations are required to fit the surrogate."
            )

        designs = np.array(
            [
                [observation.design.alpha, observation.design.sigma]
                for observation in observations
            ],
            dtype=float,
        )

        spreads = np.array(
            [observation.spread for observation in observations],
            dtype=float,
        )

        self._model = RBFInterpolator(
            designs,
            spreads,
            kernel="linear",
            degree=0,
        )

    def predict_spread(self, design: DesignPoint) -> float:
        if self._model is None:
            raise RuntimeError("Surrogate must be fit before prediction.")

        point = np.array(
            [[design.alpha, design.sigma]],
            dtype=float,
        )

        prediction = self._model(point)

        return float(prediction[0])

    def predict_objective(
        self,
        design: DesignPoint,
        *,
        target_spread: float,
    ) -> float:
        predicted_spread = self.predict_spread(design)

        return -(predicted_spread - target_spread) ** 2

    def propose(
        self,
        bounds: SearchBounds,
        *,
        target_spread: float,
        grid_size: int = 25,
        excluded: Sequence[DesignPoint] = (),
    ) -> DesignPoint:
        """Propose the unexplored design predicted closest to the target."""

        if self._model is None:
            raise RuntimeError("Surrogate must be fit before proposing a design.")

        if grid_size < 2:
            raise ValueError("grid_size must be at least 2.")

        alpha_values = np.linspace(
            bounds.alpha_min,
            bounds.alpha_max,
            grid_size,
        )
        sigma_values = np.linspace(
            bounds.sigma_min,
            bounds.sigma_max,
            grid_size,
        )

        candidates = np.array(
            [
                [alpha, sigma]
                for alpha in alpha_values
                for sigma in sigma_values
            ],
            dtype=float,
        )

        predicted_spreads = self._model(candidates)

        predicted_objectives = -(
            predicted_spreads - target_spread
        ) ** 2

        ranked_indices = np.argsort(predicted_objectives)[::-1]

        excluded_points = {
            (design.alpha, design.sigma)
            for design in excluded
        }

        for index in ranked_indices:
            alpha, sigma = candidates[index]

            candidate = DesignPoint(
                alpha=float(alpha),
                sigma=float(sigma),
            )

            if (candidate.alpha, candidate.sigma) not in excluded_points:
                return candidate

        raise RuntimeError("No unexplored candidate remains in the search grid.")
