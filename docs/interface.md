# Controller interface

Both sides depend on this file. Change it only together, and log the change in decisions.md.

Draft v1 (Ken). Oliver to review. Mode: xy (flat plane). Other modes get their own section later.

## Timing

- Physics step: 1 ms.
- Control tick: 5 ms. A controller acts once per tick.
- Episode limit: 20 s (4000 ticks).

## Observations (what the controller sees)

Units are mm, s and rad. Left is positive (counter-clockwise).

| Key | Shape | Unit | Meaning |
|---|---|---|---|
| `bearing` | scalar | rad | Angle from heading to the current target. Range -pi to pi. Positive means the target is to the left. |
| `range` | scalar | mm | Distance to the current target. |
| `vel_body` | 2 | mm/s | Ground velocity in the body frame: forward, left. |
| `yaw_rate` | scalar | rad/s | Turn rate. Positive is counter-clockwise. |
| `air_rel` | 2 | mm/s | Velocity relative to the air in the body frame: forward, left. This is what the antennae would sense. |

Noise and delay are applied before the controller sees these. Values are in `configs/physics.json`.

Gym vector order: `bearing, range, vel_body[0], vel_body[1], yaw_rate, air_rel[0], air_rel[1]`. Raw units. Normalizing is up to each controller.

A controller never sees true position, wind, or the target position.

## Actions (what the controller outputs)

| Key | Range | Meaning |
|---|---|---|
| `thrust` | 0 to 1 | Forward thrust, as a fraction of maximum. |
| `yaw_torque` | -1 to 1 | Turn torque. Positive turns left. |

Actions go through an actuator lag (about 20 ms) and a slew limit before they reach the body.

## Reward (training only)

Per tick:

```
reward = 0.05 * (range_prev - range)
       + 10 * target_hit
       - 0.01
       - 0.002 * (thrust^2 + yaw_torque^2)
       - 20 * out_of_bounds
```

The progress term is skipped on the tick a target is hit. Oliver can change the reward for training. Scoring never uses reward. Scoring uses the metrics in the spec.

## Episode end

- Success: all 5 targets hit.
- Out of bounds: position beyond 800 mm from the origin on either axis. Counts as a failure.
- Time limit: 20 s. Counts as a failure.

A target is hit when the distance to it drops below 10 mm. The next target appears at once.

## Rules for every controller

- Same observations, same actions, same seeds, same scenarios.
- A controller gets `reset(seed, scenario)` once per episode and `act(obs, t)` once per tick.
- Tuning uses scenarios S1 to S4 and tuning seeds 0 to 49 only.
- Scoring uses seeds 1000 to 1019 or more. S5 and S6 are held out.
- Failures are counted, not dropped.
