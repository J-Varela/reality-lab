from collections.abc import Sequence

import numpy as np

from reality_core.discovery.oracle import DesignPoint, Observation
from reality_core.ml.surrogate import RBFSurrogate
from reality_core.optimization.random_search import SearchBounds


class RBFEnsembleSurrogate:
    """Ensemble of RBF surrogates for estimating prediction uncertainty."""

    def __init__(
        self,
        *,
        max_models: int = 8,
    ) -> None:
        if max_models < 2:
            raise ValueError("max_models must be at least 2.")

        self.max_models = max_models
        self._models: list[RBFSurrogate] = []

    def fit(self, observations: Sequence[Observation]) -> None:
        if len(observations) < 4:
            raise ValueError(
                "At least 4 observations are required to fit the ensemble."
            )

        self._models = []

        num_models = min(self.max_models, len(observations))

        omitted_indices = np.linspace(
            0,
            len(observations) - 1,
            num=num_models,
            dtype=int,
        )

        for omitted_index in omitted_indices:
            subset = [
                observation
                for index, observation in enumerate(observations)
                if index != omitted_index
            ]

            model = RBFSurrogate()
            model.fit(subset)
            self._models.append(model)

    def predict_spread(
        self,
        design: DesignPoint,
    ) -> tuple[float, float]:
        """Return mean predicted spread and model disagreement."""

        if not self._models:
            raise RuntimeError("Ensemble must be fit before prediction.")

        predictions = np.array(
            [
                model.predict_spread(design)
                for model in self._models
            ],
            dtype=float,
        )

        return (
            float(np.mean(predictions)),
            float(np.std(predictions)),
        )

    def propose(
        self,
        bounds: SearchBounds,
        *,
        target_spread: float,
        exploration_weight: float = 1.0,
        grid_size: int = 25,
        excluded: Sequence[DesignPoint] = (),
    ) -> DesignPoint:
        """Balance predicted target fit against model uncertainty."""

        if not self._models:
            raise RuntimeError("Ensemble must be fit before proposing a design.")

        if exploration_weight < 0.0:
            raise ValueError("exploration_weight cannot be negative.")

        if grid_size < 2:
            raise ValueError("grid_size must be at least 2.")

        excluded_points = {
            (design.alpha, design.sigma)
            for design in excluded
        }

        best_design: DesignPoint | None = None
        best_score = -np.inf

        for alpha in np.linspace(
            bounds.alpha_min,
            bounds.alpha_max,
            grid_size,
        ):
            for sigma in np.linspace(
                bounds.sigma_min,
                bounds.sigma_max,
                grid_size,
            ):
                design = DesignPoint(
                    alpha=float(alpha),
                    sigma=float(sigma),
                )

                if (design.alpha, design.sigma) in excluded_points:
                    continue

                predicted_spread, uncertainty = self.predict_spread(design)

                target_error = abs(
                    predicted_spread - target_spread
                )

                score = (
                    -target_error
                    + exploration_weight * uncertainty
                )

                if score > best_score:
                    best_score = score
                    best_design = design

        if best_design is None:
            raise RuntimeError("No unexplored candidate remains in the search grid.")

        return best_design
