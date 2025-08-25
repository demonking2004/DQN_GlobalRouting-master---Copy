#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dmfb_dqn_fast.py

Double-DQN runner for DMFB routing experiments (single-droplet).
- Fixed 2x2 obstacles per-grid (master lists).
- Random source/destination each episode (not inside obstacles).
- Fresh training per-case, 2 trials each (configurable).
- Saves: results CSV, LaTeX table fragment, reward curves, example paths.
- Uses GPU automatically if available.
- Same network architecture as before; optimized training loop and IO.
"""
from __future__ import annotations
import os
import csv
import time
import math
import random
from collections import deque
from typing import List, Tuple, Set, Dict

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# Config (tweak here)
# ---------------------------
RESULTS_DIR = "results"
SEED = 2025
TRIALS_PER_CASE = 2
EPISODES_PER_MIXER = 2500   # original script used 2500 per mixer; keep similar
WARMUP_STEPS = 2000         # original warmup
EPS_DECAY = 60000
TARGET_SYNC = 2000
BATCH_SIZE = 64
REPLAY_CAP = 100_000
LR = 1e-3
GAMMA = 0.99
MAX_EPISODE_STEPS_FACTOR = 4  # max_steps = factor * (H+W)
EVAL_EPISODES = 100
PRINT_EVERY = 1  # print per-case progress

# Keep network size same as previous DQN example (3 conv layers -> fc)
# ---------------------------

# Use GPU if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# Obstacles master lists (2x2 blocks)
# First entries match the sketches as discussed earlier.
# ---------------------------
OBSTACLE_MASTER = {
    (10, 10): [
        (2, 2), (7, 7), (1, 6), (6, 1), (4, 4), (2, 7), (7, 2), (1, 1), (1, 8), (8, 1)
    ],
    (12, 12): [
        (2, 2), (2, 8), (5, 5), (8, 2), (8, 8), (3, 10), (10, 3), (1, 5), (5, 1), (9, 9)
    ],
    (15, 15): [
        (2, 2), (2, 12), (7, 5), (11, 11), (4, 8), (6, 12), (12, 2), (1, 7), (7, 1), (9, 13)
    ],
}

# ---------------------------
# Helpers for obstacles
# ---------------------------
def expand_2x2_block(top_left: Tuple[int,int], H: int, W: int) -> List[Tuple[int,int]]:
    r, c = top_left
    out = []
    for dr in (0,1):
        for dc in (0,1):
            rr, cc = r+dr, c+dc
            if 0 <= rr < H and 0 <= cc < W:
                out.append((rr, cc))
    return out

def make_obstacle_cells(H: int, W: int, num_blocks: int):
    master = OBSTACLE_MASTER.get((H,W))
    if master is None:
        raise ValueError(f"No obstacle master for {H}x{W}")
    blocks = master[:num_blocks]
    cells = set()
    for tl in blocks:
        cells.update(expand_2x2_block(tl, H, W))
    return blocks, cells

# ---------------------------
# Environment
# ---------------------------
ACTIONS = [(-1,0), (1,0), (0,-1), (0,1)]
ACTION_NAMES = ["Up","Down","Left","Right"]

class GridEnv:
    def __init__(self, H:int, W:int, obstacle_blocks:List[Tuple[int,int]], seed:int=0):
        self.H = H
        self.W = W
        self.obstacle_blocks = list(obstacle_blocks)
        self.obstacle_cells = set()
        for b in self.obstacle_blocks:
            self.obstacle_cells.update(expand_2x2_block(b, H, W))
        self.rng = random.Random(seed)
        self.reset()

    def random_free_cell(self):
        # sample uniformly among free cells
        while True:
            r = self.rng.randrange(self.H)
            c = self.rng.randrange(self.W)
            if (r,c) not in self.obstacle_cells:
                return (r,c)

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
        # 3 x H x W
        grid = np.zeros((3, self.H, self.W), dtype=np.float32)
        ar, ac = self.pos
        gr, gc = self.dest
        grid[0, ar, ac] = 1.0
        grid[1, gr, gc] = 1.0
        for (r,c) in self.obstacle_cells:
            grid[2, r, c] = 1.0
        return grid

    def step(self, a_idx:int):
        if self.done:
            raise RuntimeError("step after done")
        dr, dc = ACTIONS[a_idx]
        nr, nc = self.pos[0]+dr, self.pos[1]+dc
        reward = -1.0
        self.steps += 1
        # outside
        if nr < 0 or nr >= self.H or nc < 0 or nc >= self.W:
            reward += -5.0
            return self.obs(), reward, self.done, {"invalid": True}
        # obstacle
        if (nr, nc) in self.obstacle_cells:
            reward += -100.0
            self.pos = (nr, nc)
            self.done = True
            return self.obs(), reward, self.done, {"fault": True}
        # normal
        self.pos = (nr, nc)
        if self.pos == self.dest:
            reward += 100.0
            self.done = True
            return self.obs(), reward, self.done, {"success": True}
        return self.obs(), reward, self.done, {}

# ---------------------------
# Network, replay, agent (same architecture)
# ---------------------------
class QNet(nn.Module):
    def __init__(self, in_ch=3, H=10, W=10, n_actions=4):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        # compute flattened dim dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, in_ch, H, W)
            x = F.relu(self.conv1(dummy))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            flat_dim = x.view(1, -1).shape[1]
        self.fc1 = nn.Linear(flat_dim, 128)
        self.fc2 = nn.Linear(128, n_actions)

    def forward(self, x):
        # x: (B, C, H, W)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class ReplayBuffer:
    def __init__(self, capacity=REPLAY_CAP):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, s2, d):
        self.buf.append((s, a, r, s2, d))

    def sample(self, batch_size):
        idxs = np.random.choice(len(self.buf), size=batch_size, replace=False)
        s_batch = np.stack([self.buf[i][0] for i in idxs]).astype(np.float32)
        a_batch = np.array([self.buf[i][1] for i in idxs], dtype=np.int64)
        r_batch = np.array([self.buf[i][2] for i in idxs], dtype=np.float32)
        s2_batch = np.stack([self.buf[i][3] for i in idxs]).astype(np.float32)
        d_batch = np.array([self.buf[i][4] for i in idxs], dtype=np.float32)
        return s_batch, a_batch, r_batch, s2_batch, d_batch

    def __len__(self):
        return len(self.buf)

class DQNAgent:
    def __init__(self, H, W, lr=LR, gamma=GAMMA, eps_start=1.0, eps_end=0.05, eps_decay=EPS_DECAY,
                 target_sync=TARGET_SYNC, batch_size=BATCH_SIZE, seed=0):
        self.H = H; self.W = W
        self.gamma = gamma
        self.eps_start = eps_start; self.eps_end = eps_end; self.eps_decay = eps_decay
        self.target_sync = target_sync
        self.batch_size = batch_size
        self.step_count = 0
        self.rng = random.Random(seed)

        self.q = QNet(3, H, W, 4).to(DEVICE)
        self.tgt = QNet(3, H, W, 4).to(DEVICE)
        self.tgt.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.replay = ReplayBuffer()
        self.loss_val = 0.0

    def epsilon(self):
        t = min(self.step_count, self.eps_decay)
        frac = 1.0 - (t / self.eps_decay)
        return self.eps_end + (self.eps_start - self.eps_end) * max(0.0, frac)

    def select_action(self, state_np):
        self.step_count += 1
        if self.rng.random() < self.epsilon():
            return self.rng.randrange(4)
        with torch.no_grad():
            s = torch.from_numpy(state_np).unsqueeze(0).to(DEVICE)  # (1,3,H,W)
            qvals = self.q(s)
            return int(qvals.argmax(dim=1).item())

    def optimize(self):
        if len(self.replay) < self.batch_size:
            return
        s, a, r, s2, d = self.replay.sample(self.batch_size)
        s = torch.from_numpy(s).to(DEVICE)
        a = torch.from_numpy(a).long().to(DEVICE)
        r = torch.from_numpy(r).to(DEVICE)
        s2 = torch.from_numpy(s2).to(DEVICE)
        d = torch.from_numpy(d).to(DEVICE)

        # double dqn target
        with torch.no_grad():
            a_max = self.q(s2).argmax(dim=1, keepdim=True)
            q_tgt_next = self.tgt(s2).gather(1, a_max).squeeze(1)
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

# ---------------------------
# Training / evaluation utility functions
# ---------------------------
def save_curve(rewards, smoothed, save_path):
    plt.figure(figsize=(5,3))
    plt.plot(rewards, label='reward')
    plt.plot(smoothed, label='smoothed')
    plt.xlabel('Episode'); plt.ylabel('Return'); plt.legend()
    plt.tight_layout(); plt.savefig(save_path, dpi=160); plt.close()

def draw_path_example(env:GridEnv, agent:DQNAgent, save_path:str, max_steps:int=500):
    env.reset()
    traj = [env.pos]
    for t in range(max_steps):
        with torch.no_grad():
            s = env.obs()
            a = int(agent.q(torch.from_numpy(s).unsqueeze(0).to(DEVICE)).argmax(dim=1).item())
        _, _, done, _ = env.step(a)
        traj.append(env.pos)
        if done:
            break
    H, W = env.H, env.W
    grid = np.zeros((H, W), dtype=np.int32)
    for (r,c) in env.obstacle_cells:
        grid[r,c] = -1
    plt.figure(figsize=(4,4))
    plt.xticks([])  # Remove x-axis tick labels
    plt.yticks([])  # Remove y-axis tick labels
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    plt.imshow(grid, origin='upper')
    ys = [p[0] for p in traj]; xs = [p[1] for p in traj]
    plt.plot(xs, ys, linewidth=2)
    plt.scatter(env.source[1], env.source[0], s=60, marker='o')
    plt.scatter(env.dest[1], env.dest[0], s=60, marker='X')
    plt.title(f'Path example {H}x{W}')
    plt.tight_layout(); plt.savefig(save_path, dpi=160); plt.close()

def eval_greedy(env:GridEnv, agent:DQNAgent, episodes:int=EVAL_EPISODES, max_steps:int=1000, seed:int=0):
    rng = random.Random(seed)
    successes = 0
    path_lengths = []
    for ep in range(episodes):
        env.reset()
        steps = 0
        for t in range(max_steps):
            with torch.no_grad():
                s = env.obs()
                a = int(agent.q(torch.from_numpy(s).unsqueeze(0).to(DEVICE)).argmax(dim=1).item())
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
    success_rate = 100.0 * successes / episodes
    avg_path = float(np.mean(path_lengths)) if path_lengths else float('nan')
    return success_rate, avg_path

def train_agent(env:GridEnv, agent:DQNAgent, episodes:int, max_steps:int, warmup_steps:int):
    rewards = []
    smoothed = []
    sm_alpha = 0.02
    # Warmup random filling of replay buffer
    s = env.reset()
    for _ in range(warmup_steps):
        a = random.randrange(4)
        s2, r, done, _ = env.step(a)
        agent.replay.push(s, a, r, s2, float(done))
        s = s2 if not done else env.reset()
    # Main training
    for ep in range(1, episodes+1):
        s = env.reset()
        total_r = 0.0
        for t in range(max_steps):
            a = agent.select_action(s)
            s2, r, done, _ = env.step(a)
            agent.replay.push(s, a, r, s2, float(done))
            agent.optimize()
            s = s2
            total_r += r
            if done:
                break
        rewards.append(total_r)
        if not smoothed:
            smoothed.append(total_r)
        else:
            smoothed.append(smoothed[-1]*(1-sm_alpha) + total_r*sm_alpha)
    return rewards, smoothed

# ---------------------------
# Case list (same as earlier table)
# ---------------------------
TABLE_CASES = [
    ("c1", 10,10,2,5), ("c2",10,10,2,6), ("c3",10,10,2,7),
    ("c4", 10,10,3,5), ("c5",10,10,3,6), ("c6",10,10,3,7),

    ("c7", 12,12,2,5), ("c8",12,12,2,7), ("c9",12,12,2,9),
    ("c10",12,12,3,6), ("c11",12,12,3,8), ("c12",12,12,3,10),

    ("c13",15,15,3,5), ("c14",15,15,3,7), ("c15",15,15,3,9),
    ("c16",15,15,4,6), ("c17",15,15,4,8), ("c18",15,15,4,10),
]

# ---------------------------
# Main high-level runner for one case (fresh training)
# ---------------------------
def train_and_eval_case(H:int, W:int, num_obstacles:int, k_mixers:int, trial_id:int, results_dir:str, seed_base:int=123):
    blocks, _cells = make_obstacle_cells(H, W, num_obstacles)
    env = GridEnv(H, W, blocks, seed=seed_base+trial_id)
    agent = DQNAgent(H, W, lr=LR, gamma=GAMMA, eps_decay=EPS_DECAY, target_sync=TARGET_SYNC, batch_size=BATCH_SIZE, seed=seed_base+trial_id)
    episodes = k_mixers * EPISODES_PER_MIXER
    max_steps = MAX_EPISODE_STEPS_FACTOR * (H + W)
    rewards, smoothed = train_agent(env, agent, episodes=episodes, max_steps=max_steps, warmup_steps=WARMUP_STEPS)
    # Save curve
    curve_path = os.path.join(results_dir, f"reward_{H}x{W}_obs{num_obstacles}_mix{k_mixers}_trial{trial_id}.png")
    save_curve(rewards, smoothed, curve_path)
    # Evaluate per-mixer success -> joint success approx (product)
    succ_fracs = []
    path_lengths = []
    for i in range(k_mixers):
        sr, avgp = eval_greedy(env, agent, episodes=EVAL_EPISODES, max_steps=max_steps, seed=seed_base+777+i)
        succ_fracs.append(sr/100.0); path_lengths.append(avgp)
    joint_success = 100.0 * float(np.prod(succ_fracs))
    avg_path_length = float(np.mean(path_lengths)) if path_lengths else float('nan')
    path_img = os.path.join(results_dir, f"path_{H}x{W}_obs{num_obstacles}_mix{k_mixers}_trial{trial_id}.png")
    draw_path_example(env, agent, path_img, max_steps=max_steps)
    return joint_success, avg_path_length, curve_path, path_img

# ---------------------------
# Runner
# ---------------------------
def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows = []

    t0 = time.time()
    for idx, (case_id, H, W, k_mixers, num_obs) in enumerate(TABLE_CASES, start=1):
        stats = []
        if PRINT_EVERY:
            print(f"[{case_id}] Starting case {idx}/{len(TABLE_CASES)}: {H}x{W}, mixers={k_mixers}, obs={num_obs}")
        for trial in range(1, TRIALS_PER_CASE+1):
            start = time.time()
            sr, apl, curve, path = train_and_eval_case(H, W, num_obs, k_mixers, trial, RESULTS_DIR, seed_base=1000 + idx*10)
            stats.append((sr, apl, curve, path))
            if PRINT_EVERY:
                print(f"  trial {trial}: success%={sr:.1f}, avg_path={apl:.1f} (time {time.time()-start:.1f}s)")
        srs = [s for s,_,_,_ in stats]
        apls = [a for _,a,_,_ in stats]
        sr_max = max(srs); sr_min = min(srs); sr_avg = sum(srs)/len(srs)
        apl_avg = sum(apls)/len(apls)
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
        if PRINT_EVERY:
            print(f"[{case_id}] Completed: max={sr_max:.1f}, min={sr_min:.1f}, avg={sr_avg:.1f}, path={apl_avg:.1f}")

    # write CSV
    csv_path = os.path.join(RESULTS_DIR, "results_dqn.csv")
    with open(csv_path, "w", newline="") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=["case","chip","mixers","n_obstacles","succ_max","succ_min","succ_avg","avg_path"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # write LaTeX table fragment
    tex_path = os.path.join(RESULTS_DIR, "results_table_dqn.tex")
    with open(tex_path, "w") as f:
        f.write("\\begin{table}[t]\n")
        f.write("    \\caption{Result of our DQN method (two trials per case).}\n")
        f.write("    \\label{tab:results-dqn}\n")
        f.write("    \\centering\n")
        f.write("    \\begin{tabular}{c|c|c|c|c|c|c|c}\n")
        f.write("    \\hline\n")
        f.write("    \\multirow{2}{*}{\\textit{Case}} & \\textit{Chip} & \\#\\textit{ongoing} & \\multirow{2}{*}{\\textit{\\#obs}} & \\multicolumn{3}{c|}{Success Rate} & Avg. path\\\\\\cline{5-7}\n")
        f.write("     & \\textit{size} & \\textit{mixers} & & max & min & avg & length\\\\\n")
        f.write("    \\hline\n")
        for r in rows:
            f.write(f"    {r['case']} & {r['chip']} & {r['mixers']} & {r['n_obstacles']} & {r['succ_max']} & {r['succ_min']} & {r['succ_avg']} & {r['avg_path']}\\\\\n")
        f.write("    \\hline\n")
        f.write("    \\end{tabular}\n")
        f.write("\\end{table}\n")

    print(f"\nAll cases finished in {time.time()-t0:.1f}s")
    print(f"CSV saved to: {csv_path}")
    print(f"LaTeX table fragment saved to: {tex_path}")
    print(f"Plots & images saved in folder: {RESULTS_DIR}")

if __name__ == "__main__":
    main()
