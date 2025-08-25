import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import csv

NUM_POINTS = 300.0

def plot(prefix, rewards):
    x_gap = len(rewards) / NUM_POINTS if len(rewards) > NUM_POINTS else 1
    x_vals = np.arange(0, len(rewards), x_gap).astype(int)
    rewards = np.array(rewards)

    for name, axis_label, func in [
        ('sum', 'Reward Sum (to date)', points_sum),
        ('avg', 'Reward Average (next 100)', points_avg)
    ]:
        y_vals = func(rewards, x_vals)
        for logscale in [True, False]:
            plt.yscale('log' if logscale else 'linear')
            plt.plot(x_vals + 1, y_vals)
            plt.xlabel('Episodes')
            plt.ylabel(axis_label)
            plt.grid(which='Both')
            plt.tight_layout()
            plt.savefig(prefix + '_' + name + '_' + ('log' if logscale else 'lin') + '.png')
            plt.close()

def points_sum(rewards, x_vals):
    return np.array([np.sum(rewards[0:val]) for val in x_vals])

def points_avg(rewards, x_vals):
    return np.array([
        np.mean(rewards[val:min(len(rewards), val + 100)])
        for val in x_vals
    ])

def load_rewards_from_csv(csv_file="C:\Personal\Machine Learning\DQN_GlobalRouting-master - Copy\GlobalRoutingRL\report.csv"):
    rewards = []
    if os.path.exists(csv_file):
        with open(csv_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    rewards.append(float(row["Reward"]))
                except (KeyError, ValueError):
                    continue
    return rewards

if __name__ == '__main__':
    rewards = load_rewards_from_csv()

    if not rewards:
        print("⚠️ No report.csv found. Using synthetic rewards instead.")
        episodes = 200
        # Simulated DQN-like learning curve
        rewards = np.clip(
            np.linspace(-50, 200, episodes) + np.random.randn(episodes) * 20,
            -100, 250
        ).tolist()

    plot("Reward_plot", rewards)
    print("✅ Plots saved (from CSV if available, otherwise dummy data).")
