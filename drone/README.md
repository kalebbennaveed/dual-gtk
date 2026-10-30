# Drone navigation

Run the [root setup and example](../README.md#drone-navigation) first.
Commands below run from `drone/`.

## Compare methods

```bash
uv run --locked python -m scripts.compare_methods --scenario spatial_learning_course --true-drag 0.4 --seed 17 --disturbance-profile constant_positive --output output/comparison
uv run --locked python -m scripts.validate_results output/comparison --refine
uv run --locked python -m scripts.export_web_demo --scenario spatial_learning_course --source output/comparison --output output/web/index.html
uv run --locked python -m http.server 8878 --bind 127.0.0.1 --directory output/web
```

The comparison runs proposed, passive, weighted information, and no learning
(`robust`) with shared dynamics, sensing, constraints, and initial conditions.
Change `--scenario` to select another course:

| Scenario | Configuration | Included runs |
|---|---|---|
| Narrow passage | `informative_demo` | `results/focused_demo/` |
| Spatial course | `spatial_learning_course` | `results/spatial_course/` |
| Slalom | `slalom_course` | `results/slalom_course/` |

The included records can be exported together using `scripts.build_web_demo`,
as shown in the root README. To rerun all four methods, use `scripts.compare_methods`
with the corresponding scenario and an output directory. `--gamma` selects the
weighted baseline's information coefficient.

## Model and implementation

The simulator uses six-state translational dynamics:

```text
p_dot = v
v_dot = u + g - Cd * ||v|| * v + d
```

Robust tubes and ancillary feedback account for the bounded drag parameter and
disturbance. Informative candidates must satisfy the constraints, safely rejoin
the backup tube, and fit the exploration budget. SMID updates the drag interval
using executed measurements. Attitude and rotor dynamics are not simulated.

| Directory | Role |
|---|---|
| `dynamics/`, `environment/` | Drag model, obstacles, and collision checks |
| `control/` | Tube propagation and ancillary feedback |
| `estimation/` | SMID and predicted uncertainty reduction |
| `planning/` | Curve representation, optimization, and dual-gatekeeper selection |
| `simulation/`, `methods/` | Simulator, metrics, and method definitions |
| `configs/` | Shared controller settings and scenario geometry |

Paths in this table are relative to `drone_dual_gatekeeper/`.
