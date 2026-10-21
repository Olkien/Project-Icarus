# Decision log

| Date | Decision | Why |
|------|----------|-----|
| 2026-10-15 | Python 3.11.9 for all code | Compatible with PyTorch, MuJoCo, SB3 |
| 2026-10-15 | Private repo until week 8 | Publish with the write-up |
| 2026-10-21 | Simulator is plain NumPy/Python with a Gymnasium wrapper | Fast, repeatable runs, and easy to replay in a browser |
| 2026-10-21 | Non-ML controllers live in `controllers/` and `fly/` | `env/` and `ml/` stay separate |
| 2026-10-21 | Simulator is flat (xy) first. Side view only after the fly passes the vertical-control test | The fly may not steer up and down. Test before building |
| 2026-10-21 | Python env is `.venv` | `env/` is the simulator folder |
| 2026-10-21 | Add `pyproject.toml` for pytest and ruff. `ml/` is excluded from ruff | Shared test and lint setup without touching Oliver's code |
| 2026-10-21 | Run logs go in `results/runs/` and are git-ignored. Figures and tables are tracked | Logs are large and can be rebuilt from seeds |
| 2026-10-21 | Tuning seeds 0-49. Scoring seeds 1000-1019 or more | No overlap between tuning and scoring |
| 2026-10-21 | Scoring uses metrics, not reward | Reward is only for training |
| 2026-10-21 | Same tuning budget for every controller, counted in episodes | Fair comparison |
| 2026-10-21 | Scenarios S5 and S6 are held out until all gains and mappings are frozen | No tuning on the test set |
