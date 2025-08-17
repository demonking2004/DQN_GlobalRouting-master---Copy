#!/usr/bin/env python3
"""
dqn_tf2.py
TensorFlow 2.x DQN single-file implementation with replay memory and target network.
Usage examples:
  python dqn_tf2.py --env CartPole-v1 --train 1 --episodes 500 --savepath ./models
  python dqn_tf2.py --env CartPole-v1 --test 1 --model ./models/q_network_weights.h5 --episodes 20
"""
import os
import argparse
import random
from collections import deque
import numpy as np
import tensorflow as tf
from keras import layers, Model
import gym

# Reproducibility
SEED = 10701
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)


class QNetwork(Model):
    def __init__(self, n_observation, n_action, architecture=(64, 64), learning_rate=1e-4, trainable=True):
        super(QNetwork, self).__init__()
        self.n_observation = n_observation
        self.n_action = n_action
        self.architecture = list(architecture)
        self.trainable_flag = trainable

        # Build layers
        self.hidden_layers = []
        for units in self.architecture:
            self.hidden_layers.append(layers.Dense(units, activation='relu'))
        self.out_layer = layers.Dense(n_action, activation=None)

        # optimizer and loss only really used for trainable network
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate) if trainable else None
        self.loss_fn = tf.keras.losses.MeanSquaredError()

    def call(self, inputs, training=False):
        x = tf.convert_to_tensor(inputs, dtype=tf.float32)
        # ensure batch dim
        if len(x.shape) == 1:
            x = tf.expand_dims(x, 0)
        for layer in self.hidden_layers:
            x = layer(x)
        return self.out_layer(x)

    def q_values(self, states):
        return self.call(states)


class ReplayMemory:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def append(self, transition):
        # transition: (state, action, reward, next_state, done)
        self.buffer.append(transition)

    def sample(self, batch_size):
        idx = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        batch = [self.buffer[i] for i in idx]
        s, a, r, s2, d = map(np.array, zip(*batch))
        return s, a, r, s2, d

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(
        self,
        env,
        architecture=(64, 64),
        gamma=0.99,
        lr=1e-4,
        batch_size=64,
        memory_size=50000,
        min_replay_size=1000,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=50000,
        target_update_freq=1000,
        seed=SEED,
    ):
        self.env = env
        # observation and action sizes
        obs_space = env.observation_space
        act_space = env.action_space
        if hasattr(obs_space, "shape") and len(obs_space.shape) > 0:
            self.n_obs = int(np.prod(obs_space.shape))
        else:
            self.n_obs = int(obs_space.n)
        self.n_act = act_space.n

        # networks
        self.q_net = QNetwork(self.n_obs, self.n_act, architecture=architecture, learning_rate=lr, trainable=True)
        self.target_net = QNetwork(self.n_obs, self.n_act, architecture=architecture, learning_rate=lr, trainable=False)

        # build networks by calling once
        dummy = np.zeros((1, self.n_obs), dtype=np.float32)
        _ = self.q_net(dummy)
        _ = self.target_net(dummy)
        # copy weights
        self.target_net.set_weights(self.q_net.get_weights())

        # replay and hyperparams
        self.replay = ReplayMemory(capacity=memory_size)
        self.batch_size = batch_size
        self.min_replay_size = min_replay_size
        self.gamma = gamma

        # epsilon schedule
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.epsilon_decay = (epsilon_start - epsilon_end) / max(1, epsilon_decay_steps)
        self.total_steps = 0

        self.target_update_freq = target_update_freq

    def select_action(self, state, greedy=False):
        # state: shape (n_obs,) or (1,n_obs)
        if (not greedy) and (np.random.rand() < self.epsilon):
            return self.env.action_space.sample()
        q_vals = self.q_net.q_values(np.array(state, dtype=np.float32))
        q_vals = np.squeeze(q_vals)
        return int(np.argmax(q_vals))

    def process_transition(self, s, a, r, s2, done):
        self.replay.append((s, a, r, s2, float(done)))

    def update_epsilon(self):
        if self.total_steps < self.epsilon_decay_steps:
            self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay)
        else:
            self.epsilon = self.epsilon_end

    def train_step(self):
        if len(self.replay) < self.min_replay_size or len(self.replay) < self.batch_size:
            return None

        s, a, r, s2, d = self.replay.sample(self.batch_size)
        s = s.astype(np.float32)
        s2 = s2.astype(np.float32)
        a = a.astype(np.int32)
        r = r.astype(np.float32)
        d = d.astype(np.float32)

        # compute target: y = r + gamma * (1-d) * max_a' Q_target(s', a')
        q_next = self.target_net.q_values(s2).numpy()
        max_q_next = np.max(q_next, axis=1)
        y = r + self.gamma * (1.0 - d) * max_q_next

        with tf.GradientTape() as tape:
            q_pred = self.q_net.q_values(s)
            # gather q values for taken actions
            indices = tf.stack([tf.range(self.batch_size, dtype=tf.int32), a], axis=1)
            q_taken = tf.gather_nd(q_pred, indices)
            loss = tf.reduce_mean(tf.square(y - q_taken))

        grads = tape.gradient(loss, self.q_net.trainable_variables)
        self.q_net.optimizer.apply_gradients(zip(grads, self.q_net.trainable_variables))

        return float(loss.numpy())

    def sync_target(self):
        self.target_net.set_weights(self.q_net.get_weights())

    def burn_in(self, burn_in_steps=1000):
        # Fill replay buffer with random actions
        s = self.env.reset()
        for _ in range(burn_in_steps):
            a = self.env.action_space.sample()
            s2, r, done, _ = self.env.step(a)
            self.process_transition(s, a, r, s2, done)
            if done:
                s = self.env.reset()
            else:
                s = s2

    def train(
        self,
        num_episodes=500,
        max_steps_per_episode=1000,
        savepath="./models",
        save_every_episodes=50,
        burn_in_steps=None,
    ):
        os.makedirs(savepath, exist_ok=True)
        if burn_in_steps is None:
            burn_in_steps = max(self.min_replay_size, 1000)
        print(f"Burn-in with {burn_in_steps} random steps...")
        self.burn_in(burn_in_steps)

        episode_rewards = []
        global_step = 0
        for ep in range(1, num_episodes + 1):
            s = self.env.reset()
            ep_reward = 0.0
            for t in range(max_steps_per_episode):
                a = self.select_action(s, greedy=False)
                s2, r, done, _ = self.env.step(a)
                self.process_transition(s, a, r, s2, done)

                loss = self.train_step()
                self.total_steps += 1
                global_step += 1
                self.update_epsilon()

                # target network update (by steps)
                if self.total_steps % self.target_update_freq == 0:
                    self.sync_target()

                s = s2
                ep_reward += r
                if done:
                    break

            episode_rewards.append(ep_reward)
            if ep % 10 == 0 or ep == 1:
                avg_10 = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 1 else ep_reward
                print(f"Episode {ep} | Reward: {ep_reward:.2f} | Avg10: {avg_10:.2f} | Epsilon: {self.epsilon:.3f} | Steps: {self.total_steps}")

            if ep % save_every_episodes == 0:
                weights_file = os.path.join(savepath, f"q_network_ep{ep}.h5")
                self.q_net.save_weights(weights_file)
                print(f"Saved weights to {weights_file}")

        # final save
        final_weights = os.path.join(savepath, "q_network_final.h5")
        self.q_net.save_weights(final_weights)
        print(f"Training finished. Final weights saved to {final_weights}")
        return episode_rewards

    def test(self, model_path=None, num_episodes=20, max_steps_per_episode=1000, render=False):
        if model_path:
            self.q_net.load_weights(model_path)
            self.sync_target()
            print(f"Loaded weights from {model_path}")

        rewards = []
        for ep in range(num_episodes):
            s = self.env.reset()
            ep_reward = 0.0
            done = False
            for t in range(max_steps_per_episode):
                if render:
                    self.env.render()
                a = self.select_action(s, greedy=True)
                s, r, done, _ = self.env.step(a)
                ep_reward += r
                if done:
                    break
            rewards.append(ep_reward)
            print(f"Test Episode {ep+1} | Reward: {ep_reward}")
        return rewards


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CartPole-v1", help="Gym environment name")
    parser.add_argument("--train", type=int, default=1, help="Train if 1")
    parser.add_argument("--test", type=int, default=0, help="Test if 1")
    parser.add_argument("--episodes", type=int, default=500, help="Number of training episodes")
    parser.add_argument("--savepath", type=str, default="./models", help="Folder to save model weights")
    parser.add_argument("--model", type=str, default=None, help="Path to weights file for testing")
    parser.add_argument("--render", action="store_true", help="Render env during test")
    return parser.parse_args()


def main():
    args = parse_args()
    env = gym.make(args.env)
    env.seed(SEED)
    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    agent = DQNAgent(
        env,
        architecture=(64, 64),
        gamma=0.99,
        lr=1e-4,
        batch_size=64,
        memory_size=50000,
        min_replay_size=1000,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=50000,
        target_update_freq=1000,
    )

    if args.train:
        agent.train(num_episodes=args.episodes, savepath=args.savepath)

    if args.test:
        model_path = args.model or os.path.join(args.savepath, "q_network_final.h5")
        print("Testing with model:", model_path)
        agent.test(model_path=model_path, num_episodes=20, render=args.render)

    env.close()


if __name__ == "__main__":
    main()
