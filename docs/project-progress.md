# Project 1 progress log

Each completed milestone gets a dedicated branch and Git worktree. Start the next milestone from the previous verified checkpoint, append its implementation and results to this log, then commit, push, and verify the remote branch. Keep earlier milestone branches available for comparison.

## Step 1 — Odometry-only baseline

**Status:** Complete. The baseline deliberately drives straight toward the goal so we can measure what localisation and obstacle avoidance improve.

- **Implementation.** The [mobile controller](../student_templates/mobile_controller.py) reads raw odometry as its estimate of position and heading. It points the base toward the goal at up to 0.3 m/s, slows inside 30 cm, and steers yaw toward zero. Only the base receives movement commands.
- **Measurements.** The runner logs both odometry and simulator ground truth, the 16 LiDAR ranges, true goal distance, cumulative contact steps, and arm drift from reset through the last control step. Ground truth is used for evaluation. The saved scan sequence lets us compare the EKF and raw odometry on the same observations.
- **Verified result.** In one `none` trial, the robot first touched a bin at 1.55 simulated seconds. After 400 steps (20 s) it had accumulated 370 contact steps and was 1.456 m from the goal. Final odometry error was 0.037 m in position and 3.99° in heading. Peak arm drift was 0.037 rad. This run shows the expected failure of driving straight across an obstructed route. It is one observation, so report aggregate performance after repeating trials.
- **Evidence.** [Recorded baseline CSV](results/01-odometry-baseline.csv), with 401 rows including the reset observation. Preserve this file and write new runs under `plots/`.

Run from the repository root with the project's Python environment active.

```bash
python student_templates/mobile_controller.py --human-mode none --steps 400 --log-csv plots/odometry_baseline.csv
```

For the macOS viewer, use the same environment's `mjpython` launcher and add `--render`.

## Next milestone — LiDAR-assisted localisation

Adapt the S08 EKF to predict pose from odometry increments and correct it with LiDAR ranges to mapped geometry. Compare both estimates on the saved baseline and on drives with more motion. Use position and heading error, especially near the goal, to decide whether the correction is ready for integration with planning.
