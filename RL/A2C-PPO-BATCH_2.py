# BATCH ACTOR-CRITIC (A2C) (NOT ONLINE)

# conda install swig # needed to build Box2D in the pip install
# pip install box2d-py # a repackaged version of pybox2d

# lessons learned
#    - rewards must be discounted
#    - rewards must be normalized
#    - entorpy loss should be added
#    - there should be a maximum number of steps, otherwise you'll get strange results

import torch
import torch.nn as nn
from torch.distributions import Categorical
import gym
from termcolor import cprint
from RL import PPO, ActorCritic, Memory
from tqdm import trange
from constants import bar_format

############## Hyperparameters ##############
render = True
_ppo = True
from_scratch = False

# env_name = "CartPole-v1"
env_name = "LunarLander-v2"
# creating environment
env = gym.make(env_name)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
solved_reward = 299  # stop training if avg_reward > solved_reward
log_interval = 20  # print avg reward in the interval
max_episodes = 50000  # max training episodes
max_timesteps = 300  # max timesteps in one episode
n_latent_var = 64  # number of variables in hidden layer
update_timestep = 2000  # update policy every n timesteps
lr = 0.002
betas = (0.9, 0.999)
gamma = 0.99  # discount factor
K_epochs = 4  # update policy for K epochs
eps_clip = 0.2  # clip parameter for PPO
random_seed = None

#############################################

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if random_seed:
    torch.manual_seed(random_seed)
    env.seed(random_seed)

memory = Memory()
ppo = PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip, device)
if not from_scratch:
    cprint('loading policy weights', 'grey', 'on_yellow')
    ppo.policy.load_state_dict(torch.load("{}.pth".format(env_name), map_location=torch.device('cpu')))
    ppo.policy_old.load_state_dict(torch.load("{}.pth".format(env_name), map_location=torch.device('cpu')))

print(lr, betas)

# logging variables
running_reward = 0
avg_length = 0
timestep = 0

# training loop
for i_episode in trange(1, max_episodes + 1, bar_format=bar_format):
    state = env.reset()
    for t in range(max_timesteps):
        timestep += 1

        # Running policy_old:
        action = ppo.policy_old.act(state, memory)
        state, reward, done, _ = env.step(action)

        # Saving reward and is_terminal:
        memory.rewards.append(reward)
        memory.is_terminals.append(done)

        # update if its time
        if timestep % update_timestep == 0:
            ppo.update(memory, _ppo)
            memory.clear_memory()
            timestep = 0

        running_reward += reward
        if render:
            env.render()
        if done:
            break

    avg_length += t

    # stop training if avg_reward > solved_reward
    if running_reward > (log_interval * solved_reward):
        print("########## Solved! ##########")
        torch.save(ppo.policy.state_dict(), './PPO_{}.pth'.format(env_name))
        break

    # logging
    if i_episode % log_interval == 0:
        avg_length = int(avg_length / log_interval)
        running_reward = int((running_reward / log_interval))

        print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, avg_length, running_reward))
        running_reward = 0
        avg_length = 0

        torch.save(ppo.policy.state_dict(), "{}.pth".format(env_name))
