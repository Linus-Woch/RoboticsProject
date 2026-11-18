# Copyright (c) 2026 Universitat Politècnica de Catalunya (UPC),
# Barcelona East School of Engineering (EEBE).
# Robotics Systems MSc course 295II235, 2026-2027.
# All rights reserved.

"""
mobile_manip.wipe_progress
==========================
"How much did my controller clean?" as a picture instead of one number.

    rec = WipeProgressRecorder(env)          # right after env.reset()
    for t in range(steps):
        ...
        obs, reward, done, info = env.step(action)
        rec.update(obs)                      # once per step
    print(rec.summary())                     # "Cleaned 23% (23 of 100 dirt spots) in 700 steps"
    rec.save("plots/my_wipe")                # -> plots/my_wipe/wipe_progress.png

The figure has two panels:
  * left  -- percent of the dirt cleaned over time (the task is done at 100%);
  * right -- a top-down view of the table: every dirt spot (brown = still dirty,
             coloured = cleaned, with the colour showing WHEN it was cleaned) and
             the path the end effector took. This is the panel that tells you
             WHY a run cleaned little: you can see whether the tool ever went
             near the dirt that is left.

Works for the "wipe" task only (the nut task has no dirt).
"""

import os

import numpy as np


class WipeProgressRecorder:
    """Records how much dirt is cleaned over time, and draws the figure described above.

    Create it right after `env.reset()` (wipe task only) and call `update(obs, done)` after every
    `env.step(...)`. Afterwards `summary()` gives one line of text, `save(folder)` writes the figure,
    and the lists `step`, `proportion` and `eef_xy` hold the raw data if you want to make your own plots.
    It reads the dirt positions from the simulator, so it is an evaluation tool: do not use it
    inside your controller.
    """

    def __init__(self, env):
        if not hasattr(env, "wiped_markers") or not hasattr(env.model.mujoco_arena, "markers"):
            raise ValueError("WipeProgressRecorder needs the 'wipe' task (this env has no dirt markers).")
        self.env = env
        self._markers = list(env.model.mujoco_arena.markers)
        sim = env.sim
        self.marker_xy = np.array([sim.data.body_xpos[sim.model.body_name2id(m.root_body)][:2] for m in self._markers])
        self.n_markers = len(self._markers)
        self.step = []                 # control step index of each update
        self.proportion = []           # fraction of markers wiped after that step
        self.eef_xy = []               # end-effector world (x, y) after that step
        self.wiped_at = np.full(self.n_markers, -1, dtype=int)   # step each marker was first wiped (-1 = not yet)
        self._t = 0
        self.ended_at = None           # step at which the environment ended the episode, if it did

    def update(self, obs, done=False):
        """Call once after every `env.step(...)`, passing its `done`. The wipe task ends an episode
        early (reward about -0.37) when the arm reaches a joint limit or touches the table with
        something other than the tool; passing `done` lets the plot show that, instead of a curve
        that just stops."""
        wiped = {id(m) for m in self.env.wiped_markers}
        for i, m in enumerate(self._markers):
            if self.wiped_at[i] < 0 and id(m) in wiped:
                self.wiped_at[i] = self._t
        self.step.append(self._t)
        self.proportion.append(len(wiped) / self.n_markers)
        self.eef_xy.append(np.asarray(obs["robot0_eef_pos"][:2], dtype=float).copy())
        if done and self.ended_at is None:
            self.ended_at = self._t
        self._t += 1

    @property
    def final_proportion(self):
        return self.proportion[-1] if self.proportion else 0.0

    def summary(self):
        """One line of text: percent cleaned, number of spots, steps, and whether the episode ended early."""
        done = int((self.wiped_at >= 0).sum())
        text = f"Cleaned {100 * self.final_proportion:.0f}% ({done} of {self.n_markers} dirt spots) in {self._t} steps"
        if self.ended_at is not None:
            text += " -- task complete" if self.final_proportion >= 0.999 else " -- ENDED EARLY by the environment (joint limit or collision)"
        return text

    def save(self, out_dir, name="wipe_progress", title=None):
        """Writes `<out_dir>/<name>.png` and returns its path."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import patches

        os.makedirs(out_dir, exist_ok=True)
        fig, (ax_t, ax_m) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1, 1.1]})

        # --- left: progress over time ---
        steps = np.asarray(self.step)
        pct = 100 * np.asarray(self.proportion)
        ax_t.plot(steps, pct, color="tab:green", linewidth=2)
        ax_t.fill_between(steps, pct, color="tab:green", alpha=0.15)
        ax_t.axhline(100, color="gray", linestyle=":", linewidth=1)
        if self.ended_at is not None:
            complete = self.final_proportion >= 0.999
            ax_t.axvline(self.ended_at, color="tab:green" if complete else "tab:red", linestyle="--", linewidth=1.5)
            ax_t.annotate("task complete" if complete else "episode ended by the environment\n(joint limit or collision)",
                          (self.ended_at, 50), rotation=90, va="center", ha="right", fontsize=8,
                          color="tab:green" if complete else "tab:red")
        ax_t.set_ylim(0, 105)
        ax_t.set_xlabel("step (0.05 s each)")
        ax_t.set_ylabel("dirt cleaned (%)")
        ax_t.set_title(self.summary().replace(" -- ", "\n"), fontsize=10)
        ax_t.grid(alpha=0.3)

        # --- right: where, and when ---
        env = self.env
        c, size = np.asarray(env.table_offset[:2]), np.asarray(env.table_full_size[:2])
        ax_m.add_patch(patches.Rectangle(c - size / 2, size[0], size[1], fill=False, edgecolor="peru", linewidth=2, label="table"))
        eef = np.array(self.eef_xy) if self.eef_xy else np.zeros((0, 2))
        if len(eef):
            ax_m.plot(eef[:, 0], eef[:, 1], color="tab:blue", linewidth=1, alpha=0.7, label="end-effector path")
            ax_m.scatter(*eef[0], c="green", s=60, zorder=5, label="start")
            ax_m.scatter(*eef[-1], c="red", marker="x", s=70, zorder=5, label="end")
        dirty = self.wiped_at < 0
        ax_m.scatter(self.marker_xy[dirty, 0], self.marker_xy[dirty, 1], s=34, color="saddlebrown", label=f"still dirty ({int(dirty.sum())})", zorder=3)
        if (~dirty).any():
            sc = ax_m.scatter(self.marker_xy[~dirty, 0], self.marker_xy[~dirty, 1], s=34, c=self.wiped_at[~dirty], cmap="viridis",
                              vmin=0, vmax=max(self._t, 1), edgecolors="black", linewidths=0.4, zorder=4, label=f"cleaned ({int((~dirty).sum())})")
            fig.colorbar(sc, ax=ax_m, fraction=0.046, pad=0.04, label="step when cleaned")
        # zoom to the dirt and the tool's path (the table is much bigger than either);
        # the table outline is still drawn, just clipped to this view
        pts = self.marker_xy if len(eef) == 0 else np.vstack([self.marker_xy, eef])
        lo, hi = pts.min(axis=0) - 0.07, pts.max(axis=0) + 0.07
        ax_m.set_xlim(lo[0], hi[0])
        ax_m.set_ylim(lo[1], hi[1])
        ax_m.set_aspect("equal")
        ax_m.set_xlabel("x (m)")
        ax_m.set_ylabel("y (m)")
        ax_m.set_title("top-down view of the dirt (table outline clipped)")
        ax_m.grid(alpha=0.3)
        ax_m.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=False)
        if title:
            fig.suptitle(title)
        fig.tight_layout()
        path = os.path.join(out_dir, f"{name}.png")
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        return path
