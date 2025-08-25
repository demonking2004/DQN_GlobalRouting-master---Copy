#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DMFB Double-DQN (PyTorch) — fresh training per test case
- Fixed 2×2 obstacle blocks (top-left coordinates) per grid size (first entries match ma'am's sketches).
- Random source/destination each episode (never inside obstacles).
- For each table case: 2 trials (fresh training), report success-rate (joint over k mixers) & avg path length.
- Saves curves, path snapshots, CSV, and a LaTeX table you can \input{}.

Requires: numpy, matplotlib, torch
"""

import os
import math
import random
import csv
from collections import deque, defaultdict

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------
# Repro / utils
# ------------------------------
def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------
# Obstacles (2×2 blocks) — master lists
# ------------------------------
OBSTACLE_MASTER = {
    # 10x10 (first two match the sketch: (2,2) and (7,7))
    (10, 10): [
        (2, 2), (7, 7), (1, 6), (6, 1), (4, 4), (2, 7), (7, 2), (1, 1), (1, 8), (8, 1)
    ],
    # 12x12
    (12, 12): [
        (2, 2), (2, 8), (5, 5), (8, 2), (8, 8), (3, 10), (10, 3), (1, 5), (5, 1), (9, 9)
    ],
    # 15x15 (first four match the sketch: (2,2), (2,12), (7,5), (11,11))
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
# Environment
# ------------------------------
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
ACTION_NAMES = ["Up", "Down", "Left", "Right"]

class GridEnv:
    """Single droplet; obstacles are terminal-fault cells; state is (grid tensor)."""
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
        return self.obs()

    def obs(self):
        # 3-channel grid: [agent, goal, obstacle] normalized to {0,1}
        grid = np.zeros((3, self.H, self.W), dtype=np.float32)
        ar, ac = self.pos
        gr, gc = self.dest
        grid[0, ar, ac] = 1.0
        grid[1, gr, gc] = 1.0
        for (r, c) in self.obstacle_cells:
            grid[2, r, c] = 1.0
        return grid

    def step(self, action_idx):
        if self.done:
            raise RuntimeError("Step called after done")
        dr, dc = ACTIONS[action_idx]
        nr, nc = self.pos[0] + dr, self.pos[1] + dc
        reward = -1.0  # step penalty
        self.steps += 1

        # outside grid
        if nr < 0 or nr >= self.H or nc < 0 or nc >= self.W:
            reward += -5.0
            next_obs = self.obs()  # pos unchanged
            return next_obs, reward, self.done, {"invalid": True}

        # into obstacle -> terminal fail
        if (nr, nc) in self.obstacle_cells:
            reward += -100.0
            self.pos = (nr, nc)
            self.done = True
            return self.obs(), reward, self.done, {"fault": True}

        # normal move
        self.pos = (nr, nc)
        # reach goal
        if self.pos == self.dest:
            reward += 100.0
            self.done = True
            return self.obs(), reward, self.done, {"success": True}

        return self.obs(), reward, self.done, {}

# ------------------------------
# DQN components
# ------------------------------
class QNet(nn.Module):
    """Small CNN encoder + linear head -> 4 actions"""
    def __init__(self, in_ch=3, H=10, W=10, n_actions=4):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        with torch.no_grad():
            dummy = torch.zeros(1, in_ch, H, W)
            x = self._forward_conv(dummy)
            flat_dim = x.view(1, -1).shape[1]
        self.fc1 = nn.Linear(flat_dim, 128)
        self.fc2 = nn.Linear(128, n_actions)

    def _forward_conv(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, s2, d):
        self.buf.append((s, a, r, s2, d))

    def sample(self, batch_size):
        idxs = np.random.choice(len(self.buf), size=batch_size, replace=False)
        s, a, r, s2, d = [], [], [], [], []
        for i in idxs:
            _s, _a, _r, _s2, _d = self.buf[i]
            s.append(_s); a.append(_a); r.append(_r); s2.append(_s2); d.append(_d)
        return (np.stack(s), np.array(a), np.array(r, dtype=np.float32),
                np.stack(s2), np.array(d, dtype=np.float32))

    def __len__(self): return len(self.buf)

class DQNAgent:
    def __init__(self, H, W, lr=1e-3, gamma=0.99,
                 eps_start=1.0, eps_end=0.05, eps_decay_steps=40_000,
                 target_sync=2000, batch_size=64, seed=0):
        self.H, self.W = H, W
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_steps = eps_decay_steps
        self.target_sync = target_sync
        self.batch_size = batch_size
        self.step_count = 0
        self.rng = random.Random(seed)

        self.q = QNet(3, H, W, n_actions=4).to(DEVICE)
        self.tgt = QNet(3, H, W, n_actions=4).to(DEVICE)
        self.tgt.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.replay = ReplayBuffer(100_000)
        self.loss_val = 0.0

    def epsilon(self):
        t = min(self.step_count, self.eps_decay_steps)
        frac = 1.0 - (t / self.eps_decay_steps)
        return self.eps_end + (self.eps_start - self.eps_end) * max(0.0, frac)

    def select_action(self, state_np):
        self.step_count += 1
        if self.rng.random() < self.epsilon():
            return self.rng.randrange(4)
        with torch.no_grad():
            s = torch.from_numpy(state_np).unsqueeze(0).to(DEVICE)  # (1,3,H,W)
            qvals = self.q(s)
            a = int(qvals.argmax(dim=1).item())
        return a

    def optimize(self):
        if len(self.replay) < self.batch_size:
            return
        s, a, r, s2, d = self.replay.sample(self.batch_size)
        s  = torch.from_numpy(s).to(DEVICE)
        a  = torch.from_numpy(a).long().to(DEVICE)
        r  = torch.from_numpy(r).to(DEVICE)
        s2 = torch.from_numpy(s2).to(DEVICE)
        d  = torch.from_numpy(d).to(DEVICE)

        # Double DQN target
        with torch.no_grad():
            a_max = self.q(s2).argmax(dim=1)
            q_tgt_next = self.tgt(s2).gather(1, a_max.unsqueeze(1)).squeeze(1)
            y = r + (1.0 - d) * self.gamma * q_tgt_next

        q_vals = self.q(s).gather(1, a.unsqueeze(1)).squeeze(1)
        loss = F.mse_loss(q_vals, y)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 5.0)
        self.opt.step()
        self.loss_val = float(loss.item())

        if (self.step_count % self.target_sync) == 0:
            self.tgt.load_state_dict(self.q.state_dict())

# ------------------------------
# Train / eval
# ------------------------------
def train_agent(env, agent, episodes=6000, max_steps=1000, warmup=1000, log_every=200):
    rewards, smoothed = [], []
    sm_alpha = 0.02
    # Fill replay with random actions (stabilize)
    s = env.reset()
    for _ in range(warmup):
        a = random.randrange(4)
        s2, r, done, _ = env.step(a)
        agent.replay.push(s, a, r, s2, float(done))
        s = s2 if not done else env.reset()

    for ep in range(1, episodes + 1):
        s = env.reset()
        ep_r = 0.0
        for t in range(max_steps):
            a = agent.select_action(s)
            s2, r, done, _ = env.step(a)
            agent.replay.push(s, a, r, s2, float(done))
            agent.optimize()
            s = s2
            ep_r += r
            if done: break
        rewards.append(ep_r)
        if len(smoothed)==0: smoothed.append(ep_r)
        else: smoothed.append(smoothed[-1]*(1-sm_alpha) + ep_r*sm_alpha)
    return rewards, smoothed

def save_curve(rewards, smoothed, save_path):
    plt.figure(figsize=(5,3))
    plt.plot(rewards, label='reward')
    plt.plot(smoothed, label='smoothed')
    plt.xlabel('Episode'); plt.ylabel('Return'); plt.legend()
    plt.tight_layout(); plt.savefig(save_path, dpi=160); plt.close()

def eval_greedy(env, agent, episodes=100, max_steps=1000, seed=0):
    rng = random.Random(seed)
    successes = 0
    path_lengths = []
    for ep in range(episodes):
        s = env.reset()
        steps = 0
        for t in range(max_steps):
            with torch.no_grad():
                ss = torch.from_numpy(s).unsqueeze(0).to(DEVICE)
                a = int(agent.q(ss).argmax(dim=1).item())
            s, r, done, info = env.step(a)
            steps += 1
            if done:
                if "success" in info: successes += 1; path_lengths.append(steps)
                else: path_lengths.append(max_steps)
                break
        else:
            path_lengths.append(max_steps)
    return 100.0 * successes / episodes, float(np.mean(path_lengths))

def draw_path_example(env, agent, save_path, max_steps=500, seed=0):
    # Reset environment and collect trajectory
    env.reset()
    traj = [env.pos]
    for _ in range(max_steps):
        with torch.no_grad():
            s = env.obs()
            a = int(agent.q(torch.from_numpy(s).unsqueeze(0).to(DEVICE)).argmax(dim=1).item())
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
    ax.grid(color="black", linestyle="-", linewidth=0.6)  # darker grid lines

    # Draw obstacles as black squares
    for (r, c) in env.obstacle_cells:
        rect = plt.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor="black")
        ax.add_patch(rect)

    # Draw path
    ys = [p[0] for p in traj]
    xs = [p[1] for p in traj]
    ax.plot(xs, ys, color="blue", linewidth=2)

    # Draw start and goal
    ax.scatter(env.source[1], env.source[0], s=80, c="red", marker='o')
    ax.scatter(env.dest[1], env.dest[0], s=80, c="green", marker='o')

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_facecolor("white")  # white background
    ax.set_title(f'Path example {H}x{W}')
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()

def train_and_eval_case(H, W, num_obstacles, k_mixers, trial_id, results_dir, seed_base=123):
    # Build env + agent
    blocks, _cells = make_obstacle_cells(H, W, num_obstacles)
    env = GridEnv(H, W, blocks, seed=seed_base + trial_id)
    agent = DQNAgent(H, W,
                     lr=1e-3, gamma=0.99,
                     eps_start=1.0, eps_end=0.05, eps_decay_steps=60_000,
                     target_sync=2000, batch_size=64, seed=seed_base + trial_id)

    # Training budget scales with mixers
    episodes_per_mixer = 2500
    total_episodes = k_mixers * episodes_per_mixer
    rewards, smoothed = train_agent(env, agent, episodes=total_episodes, max_steps=4*(H+W), warmup=2000)

    curve_path = os.path.join(results_dir, f"reward_{H}x{W}_obs{num_obstacles}_mix{k_mixers}_trial{trial_id}.png")
    save_curve(rewards, smoothed, curve_path)

    # Evaluate k sequences (approx joint success = product of per-route success)
    eval_eps_each = 100
    succ_fracs, paths = [], []
    for i in range(k_mixers):
        sr, apl = eval_greedy(env, agent, episodes=eval_eps_each, max_steps=4*(H+W), seed=seed_base+777+i)
        succ_fracs.append(sr/100.0); paths.append(apl)
    joint_success = 100.0 * float(np.prod(succ_fracs))
    avg_path_length = float(np.mean(paths))

    path_img = os.path.join(results_dir, f"path_{H}x{W}_obs{num_obstacles}_mix{k_mixers}_trial{trial_id}.png")
    draw_path_example(env, agent, path_img, max_steps=4*(H+W), seed=seed_base + 42)

    return joint_success, avg_path_length, curve_path, path_img

# ------------------------------
# Table cases (as in your LaTeX)
# ------------------------------
TABLE_CASES = [
    # 10x10, mixers 2 (obs 5,6,7), mixers 3 (obs 5,6,7)
    ("c1", 10, 10, 2, 5), ("c2", 10, 10, 2, 6), ("c3", 10, 10, 2, 7),
    ("c4", 10, 10, 3, 5), ("c5", 10, 10, 3, 6), ("c6", 10, 10, 3, 7),

    # 12x12, mixers 2 (obs 5,7,9), mixers 3 (obs 6,8,10)
    ("c7", 12, 12, 2, 5), ("c8", 12, 12, 2, 7), ("c9", 12, 12, 2, 9),
    ("c10", 12, 12, 3, 6), ("c11", 12, 12, 3, 8), ("c12", 12, 12, 3, 10),

    # 15x15, mixers 3 (obs 5,7,9), mixers 4 (obs 6,8,10)
    ("c13", 15, 15, 3, 5), ("c14", 15, 15, 3, 7), ("c15", 15, 15, 3, 9),
    ("c16", 15, 15, 4, 6), ("c17", 15, 15, 4, 8), ("c18", 15, 15, 4, 10),
]

def main():
    set_global_seed(2025)
    results_dir = "results"
    ensure_dir(results_dir)

    csv_path = os.path.join(results_dir, "results.csv")
    tex_path = os.path.join(results_dir, "results_table.tex")

    rows = []
    for case_id, H, W, k_mixers, num_obs in TABLE_CASES:
        stats = []
        for trial in (1, 2):
            print(f"[{case_id}] DQN trial {trial} — {H}x{W}, mixers={k_mixers}, #obs={num_obs}")
            sr, apl, curve_img, path_img = train_and_eval_case(
                H, W, num_obs, k_mixers, trial_id=trial, results_dir=results_dir
            )
            stats.append((sr, apl, curve_img, path_img))

        srs = [x[0] for x in stats]
        apls = [x[1] for x in stats]
        sr_max, sr_min, sr_avg = max(srs), min(srs), sum(srs)/len(srs)
        apl_avg = sum(apls)/len(apls)

        print(f"[{case_id}] success% max={sr_max:.1f}, min={sr_min:.1f}, avg={sr_avg:.1f}; avg path={apl_avg:.1f}")

        rows.append({
            "case": case_id,
            "chip": f"{H}x{W}",
            "mixers": k_mixers,
            "n_obstacles": num_obs,
            "succ_max": f"{sr_max:.1f}",
            "succ_min": f"{sr_min:.1f}",
            "succ_avg": f"{sr_avg:.1f}",
            "avg_path": f"{apl_avg:.1f}"
        })

    # --- CSV output ---
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "case","chip","mixers","n_obstacles","succ_max","succ_min","succ_avg","avg_path"
        ])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # --- Grouped LaTeX table (matches your pasted format) ---
    with open(tex_path, "w") as f:
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write("\\caption{Result of our DQN method before wire-routing (two trials per case).}\n")
        f.write("\\label{tab:results-dqn}\n")
        f.write("\\begin{tabular}{c|c|c|c|c|c|c|c}\n")
        f.write("\\hline\n")
        f.write("\\multirow{2}{*}{\\textit{Case}} & \\textit{Chip} & \\#\\textit{ongoing} & "
                "\\multirow{2}{*}{\\textit{\\#obs}} & \\multicolumn{3}{c|}{Success Rate} & Avg. path\\\\\\cline{5-7}\n")
        f.write(" & \\textit{size} & \\textit{mixers} & & max & min & avg & length\\\\\n")
        f.write("\\hline\n")

        # Group rows by chip and mixers
        chips = {}
        for r in rows:
            chips.setdefault(r["chip"], {}).setdefault(r["mixers"], []).append(r)

        for chip, mixer_dict in chips.items():
            chip_cases = sum(len(v) for v in mixer_dict.values())
            first_chip_row = True
            mixer_items = list(mixer_dict.items())

            for m_idx, (mixers, case_list) in enumerate(mixer_items):
                mixer_cases = len(case_list)
                first_mixer_row = True

                for c_idx, r in enumerate(case_list):
                    # case id
                    f.write(f"{r['case']}")

                    # chip size multirow
                    if first_chip_row:
                        f.write(f" & \\multirow{{{chip_cases}}}{{*}}{{{chip.replace('x', '$\\\\times$')}}}")
                        first_chip_row = False
                    else:
                        f.write(" &")

                    # mixers multirow
                    if first_mixer_row:
                        f.write(f" & \\multirow{{{mixer_cases}}}{{*}}{{{mixers}}}")
                        first_mixer_row = False
                    else:
                        f.write(" &")

                    # rest of columns
                    f.write(f" & {r['n_obstacles']} & {r['succ_max']} & {r['succ_min']} & {r['succ_avg']} & {r['avg_path']}\\\\\n")

                    # clines
                    if c_idx < mixer_cases - 1:
                        f.write("\\cline{4-8}\n")
                    elif m_idx < len(mixer_items) - 1:
                        f.write("\\cline{3-8}\n")
                    else:
                        f.write("\\hline\n")

        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    print(f"\nSaved results to:\n - {csv_path}\n - {tex_path}\n - Plots in {results_dir}/")


if __name__ == "__main__":
    main()
