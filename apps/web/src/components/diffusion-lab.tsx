"use client";

import { FormEvent, useState } from "react";

import { FieldCanvas } from "@/components/field-canvas";
import {
  DiffusionSimulationResponse,
  runDiffusionSimulation,
} from "@/lib/reality-api";

const defaultRequest = {
  grid: {
    nx: 51,
    ny: 51,
    length_x: 1,
    length_y: 1,
  },
  initial_condition: {
    center_x: 0.5,
    center_y: 0.5,
    sigma: 0.05,
    amplitude: 1,
  },
  alpha: 0.01,
  steps: 100,
};

export function DiffusionLab() {
  const [alpha, setAlpha] = useState(
    defaultRequest.alpha,
  );

  const [sigma, setSigma] = useState(
    defaultRequest.initial_condition.sigma,
  );

  const [centerX, setCenterX] = useState(
    defaultRequest.initial_condition.center_x,
  );

  const [centerY, setCenterY] = useState(
    defaultRequest.initial_condition.center_y,
  );

  const [steps, setSteps] = useState(
    defaultRequest.steps,
  );

  const [gridSize, setGridSize] = useState(
    defaultRequest.grid.nx,
  );

  const [result, setResult] =
    useState<DiffusionSimulationResponse | null>(
      null,
    );

  const [error, setError] = useState<string | null>(
    null,
  );

  const [running, setRunning] = useState(false);

  async function handleSubmit(
    event: FormEvent<HTMLFormElement>,
  ) {
    event.preventDefault();

    setRunning(true);
    setError(null);

    try {
      const response = await runDiffusionSimulation({
        grid: {
          nx: gridSize,
          ny: gridSize,
          length_x: 1,
          length_y: 1,
        },
        initial_condition: {
          center_x: centerX,
          center_y: centerY,
          sigma,
          amplitude: 1,
        },
        alpha,
        steps,
      });

      setResult(response);
    } catch (caughtError) {
      setError(
        caughtError instanceof Error
          ? caughtError.message
          : "Simulation failed.",
      );
    } finally {
      setRunning(false);
    }
  }

  return (
    <main className="lab-shell">
      <header className="lab-header">
        <div>
          <p className="eyebrow">
            HERMES REALITY LAB
          </p>

          <h1>
            Diffusion Laboratory
          </h1>

          <p className="subtitle">
            Interactive 2D finite-difference simulation
          </p>
        </div>

        <div className="status-badge">
          <span className="status-dot" />
          Python scientific core
        </div>
      </header>

      <div className="workspace">
        <aside className="controls-panel">
          <form onSubmit={handleSubmit}>
            <div className="panel-heading">
              <div>
                <p className="section-kicker">
                  Experiment
                </p>
                <h2>Parameters</h2>
              </div>
            </div>

            <Control
              label="Diffusivity α"
              value={alpha}
              min={0.001}
              max={0.05}
              step={0.001}
              onChange={setAlpha}
            />

            <Control
              label="Hotspot σ"
              value={sigma}
              min={0.01}
              max={0.2}
              step={0.005}
              onChange={setSigma}
            />

            <Control
              label="Center x"
              value={centerX}
              min={0}
              max={1}
              step={0.01}
              onChange={setCenterX}
            />

            <Control
              label="Center y"
              value={centerY}
              min={0}
              max={1}
              step={0.01}
              onChange={setCenterY}
            />

            <Control
              label="Steps"
              value={steps}
              min={1}
              max={1000}
              step={1}
              onChange={setSteps}
            />

            <label className="control">
              <div className="control-row">
                <span>Grid</span>
                <strong>
                  {gridSize} × {gridSize}
                </strong>
              </div>

              <select
                value={gridSize}
                onChange={(event) =>
                  setGridSize(
                    Number(event.target.value),
                  )
                }
              >
                <option value={21}>
                  21 × 21
                </option>
                <option value={51}>
                  51 × 51
                </option>
                <option value={101}>
                  101 × 101
                </option>
                <option value={201}>
                  201 × 201
                </option>
              </select>
            </label>

            <button
              type="submit"
              disabled={running}
              className="run-button"
            >
              {running
                ? "Solving…"
                : "Run simulation"}
            </button>

            {error ? (
              <p className="error-message">
                {error}
              </p>
            ) : null}
          </form>
        </aside>

        <section className="visualization-panel">
          {result ? (
            <>
              <div className="field-grid">
                <FieldCanvas
                  title="Initial field"
                  field={result.initial_state}
                  maxValue={
                    result.diagnostics.initial_max
                  }
                />

                <FieldCanvas
                  title="Evolved field"
                  field={result.final_state}
                  maxValue={
                    result.diagnostics.initial_max
                  }
                />
              </div>

              <Diagnostics result={result} />
            </>
          ) : (
            <div className="empty-state">
              <div className="empty-orbit">
                <div className="empty-core" />
              </div>

              <h2>
                Ready to simulate
              </h2>

              <p>
                Choose experiment parameters and send
                them to the Python scientific core.
              </p>
            </div>
          )}
        </section>
      </div>
    </main>
  );
}

type ControlProps = {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
};

function Control({
  label,
  value,
  min,
  max,
  step,
  onChange,
}: ControlProps) {
  return (
    <label className="control">
      <div className="control-row">
        <span>{label}</span>
        <strong>{value}</strong>
      </div>

      <input
        type="range"
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={(event) =>
          onChange(Number(event.target.value))
        }
      />
    </label>
  );
}

function Diagnostics({
  result,
}: {
  result: DiffusionSimulationResponse;
}) {
  const diagnostics = result.diagnostics;

  return (
    <section className="diagnostics">
      <div>
        <p className="section-kicker">
          Numerical diagnostics
        </p>
        <h2>Solver state</h2>
      </div>

      <div className="diagnostic-grid">
        <Metric
          label="Δt"
          value={diagnostics.dt.toExponential(3)}
        />

        <Metric
          label="Stability limit"
          value={diagnostics.stability_limit.toExponential(
            3,
          )}
        />

        <Metric
          label="Simulated time"
          value={diagnostics.simulated_time.toFixed(
            4,
          )}
        />

        <Metric
          label="Initial peak"
          value={diagnostics.initial_max.toFixed(4)}
        />

        <Metric
          label="Final peak"
          value={diagnostics.final_max.toFixed(4)}
        />

        <Metric
          label="Steps"
          value={diagnostics.steps.toString()}
        />
      </div>
    </section>
  );
}

function Metric({
  label,
  value,
}: {
  label: string;
  value: string;
}) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}