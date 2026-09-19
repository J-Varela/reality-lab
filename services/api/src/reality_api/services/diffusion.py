from reality_core.numerics.grid import Grid2D
from reality_core.simulation.diffusion import (
    DiffusionConfig,
    explicit_stability_limit,
    simulate_diffusion,
)
from reality_core.simulation.initial_conditions import gaussian_field

from reality_api.schemas.diffusion import (
    DiffusionSimulationRequest,
    DiffusionSimulationResponse,
    SimulationDiagnostics,
)


def run_diffusion_simulation(
    request: DiffusionSimulationRequest,
) -> DiffusionSimulationResponse:
    grid = Grid2D(
        nx=request.grid.nx,
        ny=request.grid.ny,
        length_x=request.grid.length_x,
        length_y=request.grid.length_y,
    )

    initial = gaussian_field(
        grid,
        center_x=request.initial_condition.center_x,
        center_y=request.initial_condition.center_y,
        sigma=request.initial_condition.sigma,
        amplitude=request.initial_condition.amplitude,
    )

    stability_limit = explicit_stability_limit(
        grid,
        alpha=request.alpha,
    )

    dt = request.dt if request.dt is not None else stability_limit * 0.9

    result = simulate_diffusion(
        initial,
        grid,
        DiffusionConfig(
            alpha=request.alpha,
            dt=dt,
            steps=request.steps,
        ),
    )

    return DiffusionSimulationResponse(
        nx=grid.nx,
        ny=grid.ny,
        length_x=grid.length_x,
        length_y=grid.length_y,
        alpha=request.alpha,
        initial_state=initial.tolist(),
        final_state=result.final_state.tolist(),
        diagnostics=SimulationDiagnostics(
            dt=result.dt,
            steps=result.steps,
            simulated_time=result.simulated_time,
            stability_limit=result.stability_limit,
            initial_min=float(initial.min()),
            initial_max=float(initial.max()),
            final_min=float(result.final_state.min()),
            final_max=float(result.final_state.max()),
        ),
    )
