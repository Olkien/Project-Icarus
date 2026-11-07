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
| 2026-10-21 | S1 to S4 made harder after PID calibration | The PID succeeded everywhere, so the comparison had no room |
| 2026-10-22 | S1 to S4 recalibrated inside wind caps with sensor and actuator stress | Wind above top speed made success depend on target direction, not control |
| 2026-11-07 | Fly weight scale lowered from 0.6 to 0.55 | Lock-ins in 13 of 150 soak episodes at 0.6 and 0 of 150 at 0.55. The 0.5 row was stopped early at 12 of 150 (partial) |
| 2026-11-07 | Tuned fly config at 0.55 equals the calibrated gains | The gain search (40 evaluations x 8 episodes) found nothing better than the calibrated start point |
| 2026-11-07 | S4 failure traced to sensor delay (tuning seeds 0 to 4) | S1 plus delay alone slows the fly from 1.85 to 3.20 s per target (PID 1.37 to 1.48). Removing delay from S4 gives the fly 5 of 5 |
| 2026-11-07 | Decode smoothing (20, 40, 60 ms) rejected | It did not pass the fixed rule, and results got worse as smoothing grew |
| 2026-11-07 | Yaw brake rejected | A PID-sized brake wrecked every scenario because the fly's steering is about 7 to 10 times weaker. A matched brake (kd 0.05, 0.08, 0.11) raised S4 mean hits to 3.4 to 4.8 but broke S1 and delay-alone, so it failed the fixed rule |
| 2026-11-07 | Fly design frozen: weight scale 0.55, calibrated gains, decode_smooth_ms 0, yaw_damp 0 | The smoothing and brake options stay in the code, off by default, as diagnostics only. No further fly-side tuning before the scoring run |
| 2026-11-07 | S4 recorded as a limit of this fly design | The fly fails S4 mainly through sensor delay combined with the other S4 stresses, as the diagnostics above show |
| 2026-11-07 | Known flaw: range coupling rule passes at 24.8% against a 25% limit | Disclosed because the margin is small |
| 2026-11-07 | Known flaw: small-angle check rounding | 96 of 120 (exactly 80%) prints as a fail at c 35, a 25. The band choice does not change |
| 2026-11-07 | Known flaw: smoothing yaw-mean check | It failed the paired test by 0.003 to 0.005, so the smoothing flights were run as screening only |
| 2026-11-07 | Known flaw: first brake test | It used the PID's brake scale without first comparing steering strength |
