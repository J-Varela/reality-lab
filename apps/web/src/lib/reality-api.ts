export type GridRequest = {
  nx: number;
  ny: number;
  length_x: number;
  length_y: number;
};

export type GaussianInitialCondition = {
  center_x: number;
  center_y: number;
  sigma: number;
  amplitude: number;
};

export type DiffusionSimulationRequest = {
  grid: GridRequest;
  initial_condition: GaussianInitialCondition;
  alpha: number;
  steps: number;
};

export type SimulationDiagnostics = {
  dt: number;
  steps: number;
  simulated_time: number;
  stability_limit: number;
  initial_min: number;
  initial_max: number;
  final_min: number;
  final_max: number;
};

export type DiffusionSimulationResponse = {
  nx: number;
  ny: number;
  length_x: number;
  length_y: number;
  alpha: number;
  initial_state: number[][];
  final_state: number[][];
  diagnostics: SimulationDiagnostics;
};

export async function runDiffusionSimulation(
  request: DiffusionSimulationRequest,
): Promise<DiffusionSimulationResponse> {
  const response = await fetch("/api/simulations/diffusion", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const payload = await response.json().catch(() => null);

    throw new Error(
      payload?.detail ??
        payload?.error ??
        `Simulation failed with status ${response.status}.`,
    );
  }

  return response.json();
}