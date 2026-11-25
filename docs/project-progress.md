# Project 1 progress log

Start each milestone from the latest `main` in a dedicated branch and Git worktree. Record the implementation and verification results in this log, then commit and push the milestone branch. Merge it into `main` with a separate merge commit, push `main`, and verify that GitHub has both updates. Keep the milestone branches available for browsing each step.

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

## Step 3 — A* path planning and bin avoidance

**Scope.** Navigate the randomized no-person course using the Step 2 EKF, the static map, and live LiDAR. Simulator ground truth is used to score runs and draw the driven path.

- **Map.** Divide the arena into 5 cm cells. Compare each measured LiDAR range with the predicted range to a wall or table skirt. Remember unexplained foreground hits for up to 500 steps (25 simulated seconds), so a bin stays on the map when it disappears between beams. Check the distance from a hit to mapped surfaces as well. This prevents small pose errors near a table corner from creating a false obstacle.
- **Clearance.** Expand obstacles into the positions where placing the base origin would cause the offset rectangular chassis to overlap them. Allow for the folded arm in the wall clearance and for the shoulder beside tabletops. Apply 9 cm of bin clearance plus a grid-cell allowance to the estimated 8 cm bin half-size.
- **Planning and tracking.** A* explores the grid with eight possible moves and a straight-line estimate of remaining distance. Diagonals cannot cross blocked corners. The robot follows visible waypoints and replans if the remaining route is blocked. If a late detection puts the base inside a bin’s safety padding, it can retreat slowly toward a recently traversed position. The retreat keeps the mapped wall and table clearance, checks the bin footprint with a smaller margin, and stops when no checked retreat is available. Travel speed is capped at 0.26 m/s and falls as the robot approaches its final target. The controller aims for 2.5 cm and 4° in its estimated pose to leave room for localisation error in the scored parking test.
- **Verification.** A small grid check confirmed that A* goes around a barrier, respects diagonal clearance, and reports when a goal is disconnected. The [example route plot](results/03-astar-route.png) shows the initial plan and the driven path after replanning.

**Results.** All 20 runs on the final code (seeds 1–20, no pedestrian, 1,000-step limit) parked without contact. Final distance was 1.3–2.6 cm, maximum absolute yaw was 0.2°, and travel took 11.85–25.45 simulated seconds (median 12.53 s). The pedestrian scenarios will need separate testing and shorter memory for moving detections.

Run from the project root.

```bash
.venv/bin/python student_templates/mobile_controller.py --navigation astar --human-mode none --seed 18 --steps 1000 --plot-plan plots/astar-route.png
```

Use `.venv/bin/mjpython` with `--render` to watch on macOS. Omit `--navigation astar` to rerun the straight-line baseline.

### Step 3 refinement — Startup LiDAR sweep

With A* enabled, the robot turns about 13° to either side of its starting heading, collecting scans the entire time, then returns to within 1° of its driving heading before translating. Splitting the sweep across both sides reduces arm joint motion during the turn. The measured angle spans more than the 22.5° between adjacent LiDAR rays, filling the gaps between the initial ray directions. Physical occlusions, objects outside sensor range, and the gap between successive samples still limit what the robot can detect.

The sweep updates the same occupancy grid used while driving. After the first route is planned, scans continue to add foreground detections. A* replans if the current route becomes blocked. Use the saved command in [`_testcommand.txt`](../_testcommand.txt) to watch the sweep. Add `--skip-startup-sweep` to compare against the original A* controller or to avoid the joint excursion associated with turning. In the simulator, rotating can shift the reported base position slightly and move the arm joints even with zero translation and arm commands.

**Verification.** All 20 no-person layouts (seeds 1–20) parked inside the distance and yaw limits with zero contacts. The true sweep spanned 24.3–25.4° and the largest gap between sampled azimuths was 1.03°. Median plans found dropped from 13 without the sweep to 4 with it, including the initial plan. The maximum arm joint excursion during the sweep was 7.3°, and the base reference point shifted by up to 4.7 cm as it pivoted. In seed 18, the sweep took 3.65 simulated seconds and the complete run took 27.45 s, versus 14.00 s with the sweep disabled. The earlier view of the bins can lead A* to take a longer route.

## Next milestone — Pedestrian avoidance

Extend A* navigation to the moving-person scenarios and validate it in the slow and fast modes.

## Run statistics

The mobile controller prints a summary after every run, including the scored result, why it stopped, final and closest goal distances, final yaw, elapsed simulation time, path length, average speed, and localisation error. Path length adds the distance between successive measured base positions, including any displacement during the startup turn. The summary also shows arm-joint movement and, when applicable, sweep and A* planning statistics.

One contact episode groups nearby reports for the same robot part and obstacle category. The console also reports the number of control steps with any contact and a breakdown by pair. Existing CSV traces retain their counts summed over pairs.
