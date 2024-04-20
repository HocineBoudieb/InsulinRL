import gym
from RL.InsulinEnv import InsulinEnv

# Hyperparameters of the PPO algorithm
steps_per_epoch = 4000
epochs = 30
gamma = 0.99
clip_ratio = 0.2
policy_learning_rate = 3e-4
value_function_learning_rate = 1e-3
train_policy_iterations = 80
train_value_iterations = 80
lam = 0.97
target_kl = 0.01
hidden_sizes = (64, 64)

# True if you want to render the environment
render = False

# Initialize the environment
gym.register(id='InsulinEnv-v0', entry_point=InsulinEnv)

env = gym.make('InsulinEnv-v0')
num_states = env.observation_space.shape[0]