# OSC Tuning Summary

**Outcome**: SATURATED  
**Evaluations**: 12 (10 scored, 2 early-aborted/drop-failed)  
**Target**: global_score ≤ 0.05, per-trial ≤ 0.1

## Best candidate

```json
{
  "kp_pos": 200.0,
  "kp_rot": 30.0,
  "kd_joint": 20.0,
  "lambda_damping": 0.01
}
```

**global_score**: 0.0164

Per-trial breakdown:

| Trial | Score |
|-------|-------|
| pos_+x_5cm | 0.0183 |
| pos_-x_5cm | 0.0247 |
| pos_+y_5cm | 0.0100 |
| pos_-y_5cm | 0.0071 |
| pos_+z_5cm | 0.0219 |

## Top 10 results by score

| Rank | global_score | kp_pos | kp_rot | kd_joint | lambda_damping | reason |
|------|--------------|--------|--------|----------|----------------|--------|
| 1 | 0.0164 | 200 | 30 | 20 | 0.01 | ok |
| 2 | 0.0175 | 200 | 30 | 20 | 0.001 | ok |
| 3 | 0.0182 | 200 | 30 | 20 | 0.0001 | ok |
| 4 | 0.0204 | 200 | 30 | 50 | 0.01 | ok |
| 5 | 0.0204 | 200 | 15 | 50 | 0.01 | ok |
| 6 | 0.0208 | 200 | 60 | 50 | 0.01 | ok |
| 7 | 0.0217 | 200 | 100 | 50 | 0.01 | ok |
| 8 | 0.0320 | 200 | 30 | 80 | 0.01 | ok |
| 9 | 0.0474 | 100 | 30 | 50 | 0.01 | ok |
| 10 | 0.0533 | 200 | 30 | 120 | 0.01 | ok |

## Diagnosis

Coordinate descent saturated: 10 consecutive evals without improvement of ≥ 0.005 on global_score. The best score is what the current controller formulation can reach with these knobs at hand. Next leads:
- Investigate the pos_-z baseline (likely IsaacLab workspace clip).
- Add a rotation-error term so the rotation trials aren't noise-dominated.
- Mirror Factory's joint_armature/friction values once the IsaacLab probe is rerun.
