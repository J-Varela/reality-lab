from pydantic import BaseModel, Field


class GridRequest(BaseModel):
    nx: int = Field(default=51, ge=3, le=501)
    ny: int = Field(default=51, ge=3, le=501)

    length_x: float = Field(default=1.0, gt=0.0)
    length_y: float = Field(default=1.0, gt=0.0)


class GaussianInitialConditionRequest(BaseModel):
    center_x: float = 0.5
    center_y: float = 0.5

    sigma: float = Field(default=0.05, gt=0.0)
    amplitude: float = Field(default=1.0, ge=0.0)


class DiffusionSimulationRequest(BaseModel):
    grid: GridRequest = Field(default_factory=GridRequest)

    initial_condition: GaussianInitialConditionRequest = Field(
        default_factory=GaussianInitialConditionRequest
    )

    alpha: float = Field(default=0.01, gt=0.0)

    dt: float | None = Field(
        default=None,
        gt=0.0,
        description=(
            "Explicit timestep. If omitted, Reality Lab uses "
            "90% of the numerical stability limit."
        ),
    )

    steps: int = Field(default=100, ge=1, le=100_000)


class SimulationDiagnostics(BaseModel):
    dt: float
    steps: int
    simulated_time: float
    stability_limit: float

    initial_min: float
    initial_max: float

    final_min: float
    final_max: float


class DiffusionSimulationResponse(BaseModel):
    nx: int
    ny: int

    length_x: float
    length_y: float

    alpha: float

    initial_state: list[list[float]]
    final_state: list[list[float]]

    diagnostics: SimulationDiagnostics
