"""Record a fixed waypoint drive for testing localisation in the no-person course."""

import argparse
from pathlib import Path

import numpy as np

from mobile_controller import ContactMonitor, MobileController, make_env, odometry_log_row, save_odometry_log


def record_drive(output, seed):
    # The bin sampler uses NumPy’s global generator.
    np.random.seed(seed)
    env = make_env(task="wipe", human_mode="none", seed=seed, odom_seed=seed, horizon=900)
    try:
        obs = env.reset()
        goal = env.manipulation_base_xy()
        monitor = ContactMonitor(env)
        arm_start = obs["robot0_joint_pos"].copy()
        arm_drift = 0.0
        rows = [odometry_log_row(0, env, obs, goal, monitor, arm_drift)]
        start, end = env.robots[0].composite_controller._action_split_indexes["base"]

        # Travel east of the pickup table, then approach the goal from the south.
        waypoints = iter([(1.65, 2.0), (1.65, -0.85), (goal[0], -0.85), goal])
        controller = MobileController(next(waypoints))
        action = np.zeros(env.action_dim)
        for step in range(1, 901):
            action[start:end] = controller.compute_action(obs)
            obs, _, done, _ = env.step(action)
            monitor.update()
            arm_drift = max(arm_drift, np.max(np.abs(obs["robot0_joint_pos"] - arm_start)))
            rows.append(odometry_log_row(step, env, obs, goal, monitor, arm_drift))
            if done or monitor.total_steps():
                break
            if controller.at_goal(obs, tolerance=0.08):
                target = next(waypoints, None)
                if target is None:
                    break
                controller = MobileController(target)
        save_odometry_log(rows, output)
        print(f"Seed {seed}: {step} steps, {monitor.total_steps()} contact steps, "
              f"final true goal distance {np.linalg.norm(env.true_base_pose()[:2] - goal):.3f} m")
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    record_drive(args.output, args.seed)
