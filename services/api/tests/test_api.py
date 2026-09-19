from fastapi.testclient import TestClient
from reality_api.main import app

client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "hermes-reality-lab",
    }


def test_diffusion_simulation() -> None:
    response = client.post(
        "/simulations/diffusion",
        json={
            "grid": {
                "nx": 21,
                "ny": 21,
                "length_x": 1.0,
                "length_y": 1.0,
            },
            "initial_condition": {
                "center_x": 0.5,
                "center_y": 0.5,
                "sigma": 0.05,
                "amplitude": 1.0,
            },
            "alpha": 0.01,
            "steps": 20,
        },
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["nx"] == 21
    assert payload["ny"] == 21
    assert len(payload["initial_state"]) == 21
    assert len(payload["final_state"]) == 21

    diagnostics = payload["diagnostics"]

    assert diagnostics["initial_max"] > diagnostics["final_max"]
    assert diagnostics["simulated_time"] > 0.0


def test_unstable_time_step_returns_422() -> None:
    response = client.post(
        "/simulations/diffusion",
        json={
            "grid": {
                "nx": 21,
                "ny": 21,
            },
            "alpha": 0.01,
            "dt": 10.0,
            "steps": 1,
        },
    )

    assert response.status_code == 422
    assert "stability" in response.json()["detail"]
