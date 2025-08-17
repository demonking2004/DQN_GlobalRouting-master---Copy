import numpy as np
import tensorflow as tf
import DQN_Implementation as dqn

np.random.seed(42); tf.random.set_seed(42)

def run_test():
    qnet = dqn.QNetwork(environment_name="grid", networkname="test_q", trianable=True)
    x = np.random.rand(1, qnet.nObservation)
    q_vals = qnet.model.predict(x, verbose=0)
    print("Q-values:", q_vals)
    y = np.zeros((1, qnet.nAction)); y[0,0] = 1.0
    loss = qnet.model.train_on_batch(x, y)
    print("Loss after one step:", float(loss))

if __name__ == "__main__":
    run_test()
