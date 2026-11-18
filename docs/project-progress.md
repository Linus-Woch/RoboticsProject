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

## Step 2 — LiDAR-assisted localisation

**Status:** Complete. Select the extended Kalman filter (EKF) with `--localisation ekf`. Raw odometry remains the default for reproducing the baseline.

- **Predict.** Start from the calibrated odometry pose. At each control step, add the change in odometry to the previous estimate. The simulator reports increments along the world axes, which makes the motion Jacobian the identity. Increase the covariance to represent the uncertainty added by motion and drift.
- **Predict the scan.** Cast each beam from its position on the LiDAR ring. Intersect it with mapped walls and table skirts to find the range the sensor should see. The range Jacobian tells the filter how each predicted range changes when the estimated position or heading moves.
- **Correct.** Compare measured and predicted ranges. Skip missing returns, hits near segment ends, grazing angles, and differences outside an uncertainty-scaled gate. The EKF uses the remaining differences to correct the pose and reduce its uncertainty. With fewer than two accepted beams it keeps the odometry prediction. A bin or person can still happen to produce a range that passes the gate.
- **Compare.** Replay the same raw odometry and LiDAR for both estimates. Use ground truth to calculate their errors after estimating. Verified the range Jacobian against finite differences and checked that a synthetic occlusion is rejected and missing returns leave the filter in prediction-only mode.

| Drive | Position RMSE, odometry → EKF | Heading RMSE, odometry → EKF | Position RMSE within 25 cm of goal, odometry → EKF |
| --- | ---: | ---: | ---: |
| [Step 1 baseline](results/01-odometry-baseline.csv) | 1.9 → 1.3 cm | 1.92 → 0.44° | — |
| [Fixed waypoints, seed 1](results/02-seeded-drive-1.csv) | 4.4 → 0.9 cm | 3.71 → 0.16° | 6.1 → 1.6 cm |
| [Fixed waypoints, seed 2](results/02-seeded-drive-2.csv) | 6.5 → 1.0 cm | 0.92 → 0.14° | 6.9 → 0.9 cm |
| [Fixed waypoints, seed 3](results/02-seeded-drive-3.csv) | 2.6 → 0.9 cm | 4.22 → 0.12° | 4.8 → 0.5 cm |

See the [baseline plot](results/02-baseline-localisation.png) and [waypoint-drive plot](results/02-seeded-drive-2.png) for predicted versus measured ranges and errors over time. The waypoint drives used a hand-chosen route and had zero contacts. They test localisation near the goal. A separate live EKF run confirmed that the controller can use the filtered pose, with 0.8 cm final localisation error before evaluation, while the straight-line policy still hit a bin. These are development checks in the no-person scene. Repeat trials and test pedestrian modes after adding planning.

With the project Python environment active, run the following from this worktree.

```bash
python student_templates/compare_localisation.py docs/results/01-odometry-baseline.csv
python student_templates/record_localisation_drive.py plots/waypoint-drive.csv --seed 2
python student_templates/compare_localisation.py plots/waypoint-drive.csv --output plots/waypoint-comparison.png
python student_templates/mobile_controller.py --localisation ekf --human-mode none --steps 400
```

## Next milestone — Path planning and avoidance

Use the EKF pose to place LiDAR hits on an occupancy grid and plan a collision-free route with A*. Check the base footprint against obstacles and replan when new detections block the route.
