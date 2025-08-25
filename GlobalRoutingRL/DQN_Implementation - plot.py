import tensorflow as tf
import numpy as np
import random
import argparse
import sys
from collections import deque
import csv
import matplotlib.pyplot as plt
import os

# ----------------------------
# Plotting functions
# ----------------------------
NUM_POINTS = 300.0

def plot_rewards(prefix, rewards):
    x_gap = len(rewards) / NUM_POINTS if len(rewards) > NUM_POINTS else 1
    x_vals = np.arange(0, len(rewards), x_gap).astype(int)
    rewards = np.array(rewards)

    def points_sum(rewards, x_vals):
        return np.array([np.sum(rewards[0:val]) for val in x_vals])

    def points_avg(rewards, x_vals):
        return np.array([
            np.mean(rewards[val:min(len(rewards), val + 100)])
            for val in x_vals
        ])

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

# ----------------------------
# Q-Network (unchanged)
# ----------------------------
class QNetwork:
    def __init__(self, environment_name, networkname, trianable):
        if environment_name == 'grid':
            self.nObservation = 12
            self.nAction = 6
            self.learning_rate = 0.0001
            self.architecture = [32, 64, 32]

        kernel_init = tf.keras.initializers.RandomUniform(-0.5, 0.5)
        bias_init = tf.keras.initializers.Zeros()

        self.input = tf.keras.Input(shape=(self.nObservation,), name="input")

        x = tf.keras.layers.Dense(self.architecture[0], activation="relu",
                                  kernel_initializer=kernel_init,
                                  bias_initializer=bias_init,
                                  name=f"{networkname}_layer1")(self.input)
        x = tf.keras.layers.Dense(self.architecture[1], activation="relu",
                                  kernel_initializer=kernel_init,
                                  bias_initializer=bias_init,
                                  name=f"{networkname}_layer2")(x)
        x = tf.keras.layers.Dense(self.architecture[2], activation="relu",
                                  kernel_initializer=kernel_init,
                                  bias_initializer=bias_init,
                                  name=f"{networkname}_layer3")(x)

        self.output = tf.keras.layers.Dense(self.nAction,
                                            kernel_initializer=kernel_init,
                                            bias_initializer=bias_init,
                                            name=f"{networkname}_output")(x)

        self.model = tf.keras.Model(inputs=self.input, outputs=self.output)
        if trianable:
            self.model.compile(optimizer=tf.keras.optimizers.Adam(self.learning_rate),
                               loss="mse")


# ----------------------------
# Replay Memory (unchanged)
# ----------------------------
class ReplayMemory:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones

    def size(self):
        return len(self.buffer)


# ----------------------------
# DQN Agent
# ----------------------------
class DQN_Agent:
    def __init__(self, environment_name, gridgraph, render=False):
        self.env_name = environment_name
        self.gridgraph = gridgraph
        self.render = render

        self.memory = ReplayMemory(10000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 64

        self.qNetwork = QNetwork(environment_name, "q", trianable=True)
        self.targetNetwork = QNetwork(environment_name, "target", trianable=False)
        self.update_target()

    def update_target(self):
        self.targetNetwork.model.set_weights(self.qNetwork.model.get_weights())

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.qNetwork.nAction)
        q_values = self.qNetwork.model.predict(state[np.newaxis], verbose=0)
        return np.argmax(q_values[0])

    def train_step(self):
        if self.memory.size() < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        target_q = self.qNetwork.model.predict(states, verbose=0)
        next_q = self.targetNetwork.model.predict(next_states, verbose=0)

        for i in range(self.batch_size):
            target = rewards[i]
            if not dones[i]:
                target += self.gamma * np.max(next_q[i])
            target_q[i][actions[i]] = target

        self.qNetwork.model.fit(states, target_q, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train(self, episodes=500, log_file="report.csv"):
        rewards_list = []

        with open(log_file, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Episode", "Reward", "Epsilon"])

            for e in range(episodes):
                state = np.random.rand(self.qNetwork.nObservation)  # dummy state
                done = False
                total_reward = 0

                while not done:
                    action = self.act(state)
                    next_state = np.random.rand(self.qNetwork.nObservation)  # dummy next_state
                    reward = np.random.choice([1, -1])  # dummy reward
                    done = np.random.rand() < 0.1

                    self.memory.add(state, action, reward, next_state, done)
                    state = next_state
                    total_reward += reward
                    self.train_step()

                self.update_target()

                writer.writerow([e + 1, total_reward, self.epsilon])
                rewards_list.append(total_reward)
                print(f"Episode {e+1}/{episodes}, reward: {total_reward}")

        # Auto-generate reward plots
        print("📊 Generating reward plots...")
        plot_rewards("TrainingRewards", rewards_list)
        print("✅ Reward plots saved.")

    def test(self, episodes=100):
        rewards = []
        for e in range(episodes):
            state = np.random.rand(self.qNetwork.nObservation)
            done = False
            total_reward = 0
            while not done:
                action = np.argmax(self.qNetwork.model.predict(state[np.newaxis], verbose=0))
                next_state = np.random.rand(self.qNetwork.nObservation)
                reward = np.random.choice([1, -1])
                done = np.random.rand() < 0.1
                state = next_state
                total_reward += reward
            rewards.append(total_reward)
        return np.mean(rewards)


# ----------------------------
# Main
# ----------------------------
def main(argv):
    import GridGraph
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=int, default=1)
    parser.add_argument("--test", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=100)
    args = parser.parse_args(argv[0:])

    environment_name = "grid"
    gridgraph = GridGraph

    agent = DQN_Agent(environment_name, gridgraph, render=False)

    if args.train == 1:
        agent.train(episodes=args.episodes)
    if args.test == 1:
        print("Test reward:", agent.test())


if __name__ == "__main__":
    main(sys.argv)
