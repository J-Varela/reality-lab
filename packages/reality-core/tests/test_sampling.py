from reality_core.optimization.random_search import SearchBounds
from reality_core.optimization.sampling import latin_hypercube_designs


def make_bounds() -> SearchBounds:
    return SearchBounds(
        alpha_min=0.01,
        alpha_max=0.10,
        sigma_min=0.04,
        sigma_max=0.17,
    )


def test_latin_hypercube_returns_requested_number_of_designs() -> None:
    designs = latin_hypercube_designs(
        make_bounds(),
        num_candidates=4,
        seed=72,
    )

    assert len(designs) == 4


def test_latin_hypercube_designs_stay_within_bounds() -> None:
    bounds = make_bounds()

    designs = latin_hypercube_designs(
        bounds,
        num_candidates=4,
        seed=72,
    )

    for design in designs:
        assert bounds.alpha_min <= design.alpha <= bounds.alpha_max
        assert bounds.sigma_min <= design.sigma <= bounds.sigma_max


def test_latin_hypercube_covers_each_sigma_stratum() -> None:
    bounds = make_bounds()
    num_candidates = 4

    designs = latin_hypercube_designs(
        bounds,
        num_candidates=num_candidates,
        seed=72,
    )

    sigma_bins = set()

    for design in designs:
        normalized = (
            (design.sigma - bounds.sigma_min)
            / (bounds.sigma_max - bounds.sigma_min)
        )

        bin_index = min(
            int(normalized * num_candidates),
            num_candidates - 1,
        )

        sigma_bins.add(bin_index)

    assert sigma_bins == set(range(num_candidates))


def test_latin_hypercube_is_reproducible() -> None:
    bounds = make_bounds()

    first = latin_hypercube_designs(
        bounds,
        num_candidates=4,
        seed=72,
    )

    second = latin_hypercube_designs(
        bounds,
        num_candidates=4,
        seed=72,
    )

    assert first == second
