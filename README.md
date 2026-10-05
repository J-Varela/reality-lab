# Reality Lab

Reality Lab is an experimental scientific-computing workspace for building
simulation, optimization, and autonomous discovery systems.

The project explores a simple idea:

> Can a computer propose designs, simulate their physical behavior, evaluate the
> results, and use what it learns to search for better designs?

The current system uses a two-dimensional diffusion problem as the first testbed for that workflow.

## Current Capabilities

Reality Lab currently includes:

- a Python scientific-computing core
- a 2D finite-difference diffusion simulator
- Gaussian initial conditions
- explicit stability checking
- time-resolved simulation trajectory capture
- a diffusion design oracle
- objective-based design evaluation
- reproducible random search
- adaptive coarse-to-fine optimization
- experiment history logging
- search trajectory and convergence visualizations
- a FastAPI service
- a Next.js web interface

## Autonomous Discovery Loop

The current discovery workflow is:

```text
design space
    ↓
generate candidate designs
    ↓
physics simulation
    ↓
measure physical response
    ↓
calculate objective
    ↓
select promising designs
    ↓
refine the search region
    ↓
repeat


