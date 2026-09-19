from fastapi import FastAPI

from reality_api.routes.simulations import router as simulations_router

app = FastAPI(
    title="Hermes Reality Lab API",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "hermes-reality-lab",
    }


app.include_router(simulations_router)
