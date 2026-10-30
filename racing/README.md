# Autonomous racing

Run the [root setup and short example](../README.md#autonomous-racing) first.
Commands below run from `racing/`.

## Examples

One lap with the proposed method and the 5% failure tolerance:

```bash
uv run --locked drs-racing simulate --scenario traction_relaxed --method proposed --seconds 120 --output output/lap
uv run --locked drs-racing audit --input output/lap
uv run --locked drs-racing plot --input output/lap --output output/lap/comparison.png
```

`traction` uses a 1% failure tolerance; `traction_relaxed` uses 5%.
Both use 1,000 friction samples per candidate and the same circuit and dynamics.
These are per-certification tolerances under the stated distributional assumptions,
not whole-lap failure bounds. Details are in [docs/method.md](docs/method.md).

Run the paired method and weight comparisons at both tolerances:

```bash
uv run --locked python tools/run_mppi_demo.py --workers 3
uv run --locked drs-racing demo-suite --output output/web
uv run --locked python -m http.server 8879 --bind 127.0.0.1 --directory output/web
```

The variants are proposed, passive, no learning (`robust`), and weighted information
with γ = 0, 0.1, 0.4, and 2. Use `--risk 01` or `--risk 05` to select one tolerance,
`--only proposed passive` to select variants, or `--seconds 8` for a short run.
The proposed method uses a fixed informative generator; `--information-weight`
on `drs-racing simulate` applies to the weighted baseline.

For a smaller Monte Carlo study and generated figures/tables:

```bash
uv run --locked python tools/run_mppi_validation.py --trials 8 --workers 3
uv run --locked python tools/check_mppi_certificates.py
uv run --locked drs-racing mppi-report
```

The full comparison command above must run before building the combined report.
`tools/run_mppi_stress.py` runs the larger launch-trial study; use `--help` for its
trial count and output options.

## Main modules

| File | Role |
|---|---|
| `dynamics.py`, `model.py`, `track.py` | Physical inputs, bicycle model, and circuit geometry |
| `mppi.py` | Nominal and informative control-sequence optimization |
| `belief.py` | Executed-data SMID and predicted uncertainty reduction |
| `gatekeeper.py` | Candidate pairing, exploration charge, ranking, and commitment |
| `safety.py`, `control_safety.py` | Rollout constraints and binomial certification |
| `simulation.py`, `validation.py` | Closed-loop execution and independent replay audit |
| `scenarios.py` | Speed, sensing, safety, and budget settings |
| `web.py`, `web/` | Static browser replay generation and viewer |

Paths in this table are relative to `dual_gtk_drs_racing/`.
