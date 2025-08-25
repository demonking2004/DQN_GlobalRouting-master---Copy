#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DMFB RL (Tabular Q-learning) — boosted success + clean path visuals
- Non-terminal obstacle bumps (penalized no-op).
- Distance-based reward shaping (toward goal bonus).
- Slower epsilon decay + more episodes per mixer.
- Outputs Per-route SR (max/min/avg) and Avg path in LaTeX table.

Outputs:
  results/results.csv
  results/results_table.tex
  results/reward_*.png
  results/path_*.png
"""
import os
import random
import csv
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------
# Utilities
# ------------------------------
def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

# ------------------------------
# Obstacles (2×2 blocks)
# ------------------------------
OBSTACLE_MASTER = {
    (10, 10): [
        (2, 2), (7, 7), (1, 6), (6, 1), (4, 4), (2, 7), (7, 2), (1, 1), (1, 8), (8, 1)
    ],
    (12, 12): [
        (2, 2), (2, 8), (5, 5), (8, 2), (8, 8), (3, 10), (10, 3), (1, 5), (5, 1), (9, 9)
    ],
    (15, 15): [
        (2, 2), (2, 12), (7, 5), (11, 11),
        (4, 8), (6, 12), (12, 2), (1, 7), (7, 1), (9, 13)
    ],
}

def expand_2x2_block(top_left, H, W):
    r, c = top_left
    cells = []
    for dr in (0, 1):
        for dc in (0, 1):
            rr, cc = r + dr, c + dc
            if 0 <= rr < H and 0 <= cc < W:
                cells.append((rr, cc))
    return cells

def make_obstacle_cells(H, W, num_blocks):
    master = OBSTACLE_MASTER[(H, W)]
    blocks = master[:num_blocks]
    cells = set()
    for tl in blocks:
        cells.update(expand_2x2_block(tl, H, W))
    return blocks, cells

# ------------------------------
# Environment (single-droplet, gridworld)
# ------------------------------
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
ACTION_NAMES = ["Up", "Down", "Left", "Right"]

class GridEnv:
    def __init__(self, H, W, obstacle_blocks, seed=0):
        self.H = H
        self.W = W
        self.obstacle_blocks = obstacle_blocks
        self.obstacle_cells = set()
        for tl in obstacle_blocks:
            self.obstacle_cells.update(expand_2x2_block(tl, H, W))
        self.rng = random.Random(seed)
        self.reset()

    def random_free_cell(self):
        while True:
            r = self.rng.randrange(self.H)
            c = self.rng.randrange(self.W)
            if (r, c) not in self.obstacle_cells:
                return (r, c)

    def reset(self, source=None, dest=None):
        if source is None:
            source = self.random_free_cell()
        if dest is None:
            while True:
                dest = self.random_free_cell()
                if dest != source:
                    break
        self.source = source
        self.dest = dest
        self.pos = source
        self.steps = 0
        self.done = False
        return self.state()

    def state(self):
        return (self.pos[0], self.pos[1], self.dest[0], self.dest[1])

    @staticmethod
    def manhattan(a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])

    def step(self, action_idx):
        if self.done:
            raise RuntimeError("Step called on a finished episode")
        dr, dc = ACTIONS[action_idx]
        r, c = self.pos
        g = self.dest
        nr, nc = r + dr, c + dc

        reward = -0.5
        self.steps += 1

        if nr < 0 or nr >= self.H or nc < 0 or nc >= self.W:
            reward += -5.0
            return self.state(), reward, self.done, {"invalid": True}

        if (nr, nc) in self.obstacle_cells:
            reward += -10.0
            return self.state(), reward, self.done, {"invalid": True}

        old_d = GridEnv.manhattan((r, c), g)
        new_d = GridEnv.manhattan((nr, nc), g)
        self.pos = (nr, nc)

        if new_d < old_d:
            reward += +0.6
        elif new_d > old_d:
            reward += -0.2

        if self.pos == self.dest:
            reward += 100.0
            self.done = True
            return self.state(), reward, self.done, {"success": True}

        return self.state(), reward, self.done, {}

# ------------------------------
# Tabular Q-Learning Agent
# ------------------------------
class QLearningAgent:
    def __init__(self, H, W, alpha=0.25, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay_steps=50000, seed=0):
        self.H = H
        self.W = W
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.global_step = 0
        self.rng = random.Random(seed)
        self.Q = defaultdict(lambda: np.zeros(len(ACTIONS), dtype=np.float32))

    def epsilon(self):
        t = min(self.global_step, self.epsilon_decay_steps)
        frac = 1.0 - t / self.epsilon_decay_steps
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * max(0.0, frac)

    def select_action(self, state):
        eps = self.epsilon()
        self.global_step += 1
        if self.rng.random() < eps:
            return self.rng.randrange(len(ACTIONS))
        q = self.Q[state]
        return int(np.argmax(q))

    def update(self, s, a, r, s_next, done):
        q = self.Q[s]
        q_next = 0.0 if done else np.max(self.Q[s_next])
        target = r + self.gamma * q_next
        q[a] += self.alpha * (target - q[a])

# ------------------------------
# Training / Evaluation
# ------------------------------
def train_on_env(env, agent, episodes=6000, max_steps=1000, log_every=800):
    rewards, smoothed = [], []
    sm_alpha = 0.02
    for ep in range(1, episodes + 1):
        env.reset()
        total_r = 0.0
        for _ in range(max_steps):
            s = env.state()
            a = agent.select_action(s)
            s_next, r, done, _ = env.step(a)
            agent.update(s, a, r, s_next, done)
            total_r += r
            if done:
                break
        rewards.append(total_r)
        smoothed.append(total_r if not smoothed else smoothed[-1]*(1-sm_alpha)+total_r*sm_alpha)
        if (ep % log_every) == 0:
            print(f"[train] ep {ep}/{episodes} return={total_r:.1f}")
    return rewards, smoothed

def eval_success_and_path(env, agent, episodes=100, max_steps=1000, seed=0):
    successes = 0
    path_lengths = []
    old = agent.global_step
    agent.global_step = agent.epsilon_decay_steps
    for _ in range(episodes):
        env.reset()
        steps = 0
        for _ in range(max_steps):
            s = env.state()
            a = int(np.argmax(agent.Q[s]))
            _, _, done, info = env.step(a)
            steps += 1
            if done:
                if "success" in info:
                    successes += 1
                    path_lengths.append(steps)
                else:
                    path_lengths.append(max_steps)
                break
        else:
            path_lengths.append(max_steps)
    agent.global_step = old
    sr = 100.0 * successes / episodes
    apl = float(np.mean(path_lengths)) if path_lengths else float("nan")
    return sr, apl

def draw_path_example(env, agent, save_path, max_steps=500, seed=0):
    env.reset()
    traj = [env.pos]
    for _ in range(max_steps):
        s = env.state()
        a = int(np.argmax(agent.Q[s]))
        _, _, done, _ = env.step(a)
        traj.append(env.pos)
        if done:
            break

    H, W = env.H, env.W
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)
    ax.set_xticks(np.arange(-0.5, W, 1))
    ax.set_yticks(np.arange(-0.5, H, 1))
    ax.grid(color="lightgray", linestyle="-", linewidth=0.5)

    for (r, c) in env.obstacle_cells:
        rect = plt.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor="black")
        ax.add_patch(rect)

    ys = [p[0] for p in traj]
    xs = [p[1] for p in traj]
    ax.plot(xs, ys, color="blue", linewidth=2)
    ax.scatter(env.source[1], env.source[0], s=80, c="red", marker='o')
    ax.scatter(env.dest[1], env.dest[0], s=80, c="green", marker='o')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title(f'Path example {H}x{W}')
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()

def save_curve(rewards, smoothed, save_path):
    plt.figure(figsize=(5,3))
    plt.plot(rewards, label='reward')
    plt.plot(smoothed, label='smoothed')
    plt.xlabel('Episode'); plt.ylabel('Return'); plt.legend()
    plt.tight_layout(); plt.savefig(save_path, dpi=160); plt.close()

def train_and_eval_case(H, W, num_obstacles, k_mixers, trial_id, results_dir, seed_base=123):
    blocks, _ = make_obstacle_cells(H, W, num_obstacles)
    env = GridEnv(H, W, blocks, seed=seed_base + trial_id)
    agent = QLearningAgent(H, W,
                           alpha=0.25,
                           gamma=0.99,
                           epsilon_start=1.0,
                           epsilon_end=0.05,
                           epsilon_decay_steps=50000,
                           seed=seed_base + trial_id)

    episodes_per_mixer = 4000
    total_episodes = k_mixers * episodes_per_mixer
    rewards, smoothed = train_on_env(env, agent, episodes=total_episodes, max_steps=4*(H+W), log_every=800)
    curve_path = os.path.join(results_dir, f"reward_{H}x{W}_obs{num_obstacles}_mix{k_mixers}_trial{trial_id}.png")
    save_curve(rewards, smoothed, curve_path)

    eval_eps_each = 100
    per_route_sr = []
    paths = []
    for i in range(k_mixers):
        sr, apl = eval_success_and_path(env, agent, episodes=eval_eps_each, max_steps=4*(H+W), seed=seed_base + 777 + i)
        per_route_sr.append(sr/100.0)
        paths.append(apl)
    joint_success = 100.0 * float(np.prod(per_route_sr))
    per_route_avg = 100.0 * float(np.mean(per_route_sr))
    pr_max = 100.0 * max(per_route_sr)
    pr_min = 100.0 * min(per_route_sr)
    avg_path_length = float(np.mean(paths))

    path_img = os.path.join(results_dir, f"path_{H}x{W}_obs{num_obstacles}_mix{k_mixers}_trial{trial_id}.png")
    draw_path_example(env, agent, path_img, max_steps=4*(H+W), seed=seed_base + 42)

    return pr_max, pr_min, per_route_avg, avg_path_length, curve_path, path_img

# ------------------------------
# Table cases
# ------------------------------
TABLE_CASES = [
    ("c1", 10, 10, 2, 5), ("c2", 10, 10, 2, 6), ("c3", 10, 10, 2, 7),
    ("c4", 10, 10, 3, 5), ("c5", 10, 10, 3, 6), ("c6", 10, 10, 3, 7),
    ("c7", 12, 12, 2, 5), ("c8", 12, 12, 2, 7), ("c9", 12, 12, 2, 9),
    ("c10", 12, 12, 3, 6), ("c11", 12, 12, 3, 8), ("c12", 12, 12, 3, 10),
    ("c13", 15, 15, 3, 5), ("c14", 15, 15, 3, 7), ("c15", 15, 15, 3, 9),
    ("c16", 15, 15, 4, 6), ("c17", 15, 15, 4, 8), ("c18", 15, 15, 4, 10),
]

def _canonicalize_case_ret(ret):
    """
    Accepts either a 5-tuple (per_route_avg, joint_success, apl, curve_img, path_img)
    or any longer tuple; we only use the first 5 in the original order.
    """
    if not isinstance(ret, (tuple, list)) or len(ret) < 5:
        raise ValueError("train_and_eval_case must return at least 5 values "
                         "(per_route_avg, joint_success, apl, curve_img, path_img)")
    per_route_avg, joint_success, apl, curve_img, path_img = ret[:5]
    return per_route_avg, joint_success, apl, curve_img, path_img


def main():
    set_global_seed(2025)
    results_dir = "results"
    ensure_dir(results_dir)

    csv_path = os.path.join(results_dir, "results.csv")
    tex_path = os.path.join(results_dir, "results_table.tex")

    rows = []
    for case_id, H, W, k_mixers, num_obs in TABLE_CASES:
        # Run two trials and keep EVERYTHING
        per_route_vals, joint_vals, apl_vals = [], [], []
        trial_artifacts = []

        for trial in (1, 2):
            print(f"[{case_id}] Trial {trial} — {H}x{W}, mixers={k_mixers}, #obs={num_obs}")
            ret = train_and_eval_case(H, W, num_obs, k_mixers, trial_id=trial, results_dir=results_dir)
            per_route_avg, joint_success, apl, curve_img, path_img = _canonicalize_case_ret(ret)

            per_route_vals.append(per_route_avg)
            joint_vals.append(joint_success)
            apl_vals.append(apl)
            trial_artifacts.append((curve_img, path_img))

        # Compute the table stats
        pr_max = max(per_route_vals)
        pr_min = min(per_route_vals)
        pr_avg = sum(per_route_vals) / len(per_route_vals)
        apl_avg = sum(apl_vals) / len(apl_vals)

        print(f"[{case_id}] Success Rate (per-route) max={pr_max:.1f}, min={pr_min:.1f}, avg={pr_avg:.1f}; "
              f"Avg path={apl_avg:.1f}")

        # Keep EVERYTHING in CSV (including joint and paths/curves per trial)
        rows.append({
            "case": case_id,
            "chip": f"{H}x{W}",
            "mixers": k_mixers,
            "n_obstacles": num_obs,

            # what the LaTeX table will show
            "succ_max": f"{pr_max:.1f}",
            "succ_min": f"{pr_min:.1f}",
            "succ_avg": f"{pr_avg:.1f}",
            "avg_path": f"{apl_avg:.1f}",

            # keep full details too
            "per_route_trial1": f"{per_route_vals[0]:.1f}",
            "per_route_trial2": f"{per_route_vals[1]:.1f}",
            "joint_success_trial1": f"{joint_vals[0]:.1f}",
            "joint_success_trial2": f"{joint_vals[1]:.1f}",
            "apl_trial1": f"{apl_vals[0]:.1f}",
            "apl_trial2": f"{apl_vals[1]:.1f}",
            "curve_img_trial1": trial_artifacts[0][0],
            "curve_img_trial2": trial_artifacts[1][0],
            "path_img_trial1": trial_artifacts[0][1],
            "path_img_trial2": trial_artifacts[1][1],
        })

    # --- CSV with everything preserved ---
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case","chip","mixers","n_obstacles",
                "succ_max","succ_min","succ_avg","avg_path",
                "per_route_trial1","per_route_trial2",
                "joint_success_trial1","joint_success_trial2",
                "apl_trial1","apl_trial2",
                "curve_img_trial1","curve_img_trial2",
                "path_img_trial1","path_img_trial2",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    with open(tex_path, "w") as f:
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write("\\caption{Tabular Q-learning performance (two trials per case).}\n")
        f.write("\\label{tab:results-ql}\n")
        f.write("\\begin{tabular}{c|c|c|c|c|c|c|c}\n")
        f.write("\\hline\n")
        f.write("\\multirow{2}{*}{\\textit{Case}} & \\textit{Chip} & \\#\\textit{ongoing} & "
               "\\multirow{2}{*}{\\textit{\\#obs}} & \\multicolumn{3}{c|}{Success Rate} & Avg. path\\\\\n")
        f.write(" & \\textit{size} & \\textit{mixers} & & max & min & avg & length\\\\\n")
        f.write("\\hline\n")

        # Group by chip size
        chips = {}
        for r in rows:
            chips.setdefault(r["chip"], {}).setdefault(r["mixers"], []).append(r)

        for chip, mixer_dict in chips.items():
            chip_cases = sum(len(v) for v in mixer_dict.values())
            first_chip_row = True

            for mixers, case_list in mixer_dict.items():
                mixer_cases = len(case_list)
                first_mixer_row = True

                for r in case_list:
                    f.write(f"{r['case']}")

                    if first_chip_row:
                        f.write(f" & \\multirow{{{chip_cases}}}{{*}}{{{chip.replace('x', '$\\\\times$')}}}")
                        first_chip_row = False
                    else:
                        f.write(" &")

                    if first_mixer_row:
                        f.write(f" & \\multirow{{{mixer_cases}}}{{*}}{{{mixers}}}")
                        first_mixer_row = False
                    else:
                        f.write(" &")

                    f.write(f" & {r['n_obstacles']} & {r['succ_max']} & {r['succ_min']} & {r['succ_avg']} & {r['avg_path']}\\\\\n")

                    # Correct cline depending on row position
                    if mixers == list(mixer_dict.keys())[-1] and r == case_list[-1]:
                        f.write("\\hline\n")
                    elif r != case_list[-1]:
                        f.write("\\cline{4-8}\n")
                    else:
                        f.write("\\cline{3-8}\n")

        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    print(f"\nSaved results to:\n - {csv_path}\n - {tex_path}\n - Plots in {results_dir}/")

if __name__ == "__main__":
    main()
