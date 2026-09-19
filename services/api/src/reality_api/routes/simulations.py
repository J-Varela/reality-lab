from fastapi import APIRouter, HTTPException

from reality_api.schemas.diffusion import (
    DiffusionSimulationRequest,
    DiffusionSimulationResponse,
)
from reality_api.services.diffusion import run_diffusion_simulation

router = APIRouter(
    prefix="/simulations",
    tags=["simulations"],
)


@router.post(
    "/diffusion",
    response_model=DiffusionSimulationResponse,
)
def simulate_diffusion(
    request: DiffusionSimulationRequest,
) -> DiffusionSimulationResponse:
    try:
        return run_diffusion_simulation(request)
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc
