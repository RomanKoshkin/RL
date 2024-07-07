from math import sqrt, log
from copy import deepcopy
from tqdm import trange
from replay_buffer import ReplayBuffer
from networks import PolicyV, PolicyP
import gym
import random
import time
import wandb
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

c = 1.0


class Node:
    """
    The Node class represents a node of the MCTS tree.
    It contains the information needed for the algorithm to run its search.
    """

    _node_id = -1

    def __init__(self, game, done, parent, observation, action_index):
        # child nodes
        self.child = None
        Node._node_id += 1
        self.id = Node._node_id

        # total rewards from MCTS exploration
        self.T = 0

        # visit count
        self.N = 0

        # the environment
        self.game = game

        # observation of the environment
        self.observation = observation

        # if game is won/loss/draw
        self.done = done

        # link to parent node
        self.parent = parent

        # action index that leads to this node
        self.action_index = action_index

        # the value of the node according to nn
        self.nn_v = 0

        # the next probabilities
        self.nn_p = None

    def getUCBscore(self):
        """
        This is the formula that gives a value to the node.
        The MCTS will pick the nodes with the highest value.
        """

        # Unexplored nodes have maximum values so we favour exploration
        if self.N == 0:
            return float("inf")

        # We need the parent node of the current node
        top_node = self
        if top_node.parent:
            top_node = top_node.parent

        value_score = self.T / self.N

        prior_score = c * self.parent.nn_p[self.action_index] * sqrt(log(top_node.N) / self.N)

        # We use both the Value(s) and Policy from the neural network estimations for calculating the node value
        return value_score + prior_score

    def detach_parent(self):
        # free memory detaching nodes
        del self.parent
        self.parent = None

    def create_child(self):
        """
        We create one child for each possible action of the game,
        then we apply such action to a copy of the current node enviroment
        and create such child node with proper information returned from the
        action executed
        """

        if self.done:
            return

        actions = []
        games = []
        for i in range(ACTION_SPACE):
            actions.append(i)

        for i in range(len(actions)):
            new_game = deepcopy(self.game)
            games.append(new_game)

        child = {}
        action_index = 0
        for action, game in zip(actions, games):
            observation, reward, done, _, _ = game.step(action)
            child[action] = Node(game, done, self, observation, action_index)
            action_index += 1

        self.child = child

    def explore(self):
        """
        The search along the tree is as follows:
        - from the current node, recursively pick the children which
        maximizes the value according to the MCTS formula
        - when a leaf is reached:
            - if it has never been explored before, do a rollout
            and update its current value
            - otherwise, expand the node creating its children,
            pick one child at random, do a rollout and update its value
        - backpropagate the updated statistics up the tree until the root:
        update both value and visit counts
        """

        # find a leaf node by choosing nodes with max U.
        current = self
        while current.child:
            child = current.child
            max_U = max(c.getUCBscore() for c in child.values())
            actions = [a for a, c in child.items() if c.getUCBscore() == max_U]
            if len(actions) == 0:
                print("error zero length ", max_U)
            action = random.choice(actions)
            current = child[action]

        # play a random game, or expand if needed
        if current.N < 1:
            current.nn_v, current.nn_p = current.rollout()
            current.T = current.T + current.nn_v
        else:
            current.create_child()  # expand the node (by creating 2 children)
            if current.child:
                current = random.choice(current.child)
            current.nn_v, current.nn_p = current.rollout()
            current.T = current.T + current.nn_v  # update the values of the node

        current.N += 1  # increment the number of visits to the current node

        # update statistics and backpropagate
        parent = current
        while parent.parent:
            parent = parent.parent
            parent.N += 1
            parent.T = parent.T + current.T

    def rollout(self):
        """
        NOTE the difference between "rollout" in alphazero and "rollout" in MCTS:
        - in MCTS, the rollout is a random simulation of the game from the current node
        - in AlphaZero, the rollout is where we use the neural network estimations
        to approximate the Value and Policy of a given node.
        With the trained neural network, it will give us a
        good approximation even in large state spaces searches.
        """

        if self.done:
            return 0, None
        else:
            obs = np.array([self.observation]).astype(np.float32)
            obs = torch.from_numpy(obs).to(torch.float32)

            v = policy_v(obs).detach().numpy().flatten()
            p = policy_p(obs).detach().numpy().flatten()

            return v, p

    def next(self):
        """
        Once we have done enough search in the tree, the values contained in
        it should be statistically accurate. We will at some point then ask
        for the next action to play from the current node, and this is what
        this function does. There may be different ways on how to choose such
        action, in this implementation the strategy is as follows:
        - pick at random one of the node which has the maximum visit
        count, as this means that it will have a good value anyway.
        """

        if self.done:
            raise ValueError("game has ended")

        if not self.child:
            raise ValueError("no children found and game hasn't ended")

        child = self.child

        max_N = max(node.N for node in child.values())

        probs = [node.N / max_N for node in child.values()]
        probs /= np.sum(probs)

        next_children = random.choices(list(child.values()), weights=probs)[0]

        return next_children, next_children.action_index, next_children.observation, probs, self.observation


def Policy_Player_MCTS(mytree):
    """
    Our strategy for using AlphaZero is quite simple:
    - in order to pick the best move from the current node:
        - explore the tree starting from that node for a certain number of iterations to collect reliable statistics
        - pick the node that, according to AlphaZero, is the best possible next action
    """

    for i in range(MCTS_POLICY_EXPLORE):
        mytree.explore()

    next_tree, next_action, obs, p, p_obs = mytree.next()

    # note that here we are detaching the current node and returning the sub-tree
    # that starts from the node rooted at the choosen action.
    # The next search, hence, will not start from scratch but will already have collected information and statistics
    # about the nodes, so we can reuse such statistics to make the search even more reliable!
    next_tree.detach_parent()

    return next_tree, next_action, obs, p, p_obs


def show_image(image, game_finished=False):
    """
    This function is used to display the image of the game.
    """
    plt.imshow(image)  # Display the image
    plt.draw()
    plt.pause(0.02)  # Pause to render the image, adjust time as needed
    plt.clf()  # Cl
    if game_finished:
        plt.close()


def linear_warmup(optimizer, current_epoch, warmup_epochs, base_lr):
    lr = base_lr * (current_epoch / warmup_epochs)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def train_step_v(inputs, targets):
    optimizer_v.zero_grad()
    outputs = policy_v(inputs.squeeze())
    loss = loss_fn_v(outputs, targets)
    loss.backward()
    optimizer_v.step()
    return loss.item()


def train_step_p(inputs, targets, epoch):
    global optimizer_p
    optimizer_p.zero_grad()
    outputs = policy_p(inputs.squeeze())
    loss = loss_fn_p(outputs, targets.squeeze())
    loss.backward()
    if epoch < POLICY_WARMUP_EPOCHS:
        linear_warmup(optimizer_p, epoch, warmup_epochs=POLICY_WARMUP_EPOCHS, base_lr=POLICY_BASE_LR)
    optimizer_p.step()
    return loss.item()


RENDER = False
SAVE_MODELS = True  # NOTE: this will overwrite the models
CONTINUE_TRAINING = False

GAME_NAME = "CartPole-v1"
MCTS_POLICY_EXPLORE = 200

ACTION_SPACE = 2  # FIXME
OBSERVATION_DIM = 4  # FIXME
HIDDEN_DIM = 64  # FIXME

BUFFER_SIZE = int(1000)  # replay buffer size
BATCH_SIZE = 128  # minibatch size
UPDATE_EVERY = 1
POLICY_BASE_LR = 0.001
POLICY_WARMUP_EPOCHS = 3

EPISODES = 500

rewards = []
moving_average = []
v_losses = []
p_losses = []

# the maximum reward of the current game to scale the values
MAX_REWARD = 500

# Create the replay buffer
replay_buffer = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE)

# Create the Value neural network.
# The loss is the Mean Squared Error between predicted and actual values.
policy_v = PolicyV(OBSERVATION_DIM, HIDDEN_DIM)

# Create the Policy neural network.
# The loss is the Categorical Crossentropy between the predicted and actual policy according to visit counts.
policy_p = PolicyP(ACTION_SPACE, OBSERVATION_DIM, HIDDEN_DIM)

if CONTINUE_TRAINING:
    policy_v = torch.load("value_net_full.pth")
    policy_p = torch.load("policy_net_full.pth")

# Optimizers
optimizer_v = optim.Adam(policy_v.parameters(), lr=0.001)  # You can adjust learning rate as needed
optimizer_p = optim.Adam(policy_p.parameters(), lr=0.001)

# Loss functions
loss_fn_v = nn.MSELoss()  # Mean Squared Error Loss for the Value Network
loss_fn_p = nn.CrossEntropyLoss()  # Cross-Entropy Loss for the Policy Network


"""
Here we are experimenting with our implementation:
- we play a certain number of episodes of the game
- for deciding each move to play at each step, we will apply our AlphaZero algorithm
- we will collect and plot the rewards to check if the AlphaZero is actually working.
- For CartPole-v1, in particular, 500 is the maximum possible reward. 
"""

wandb.init(
    project="AlphaZero",
    entity="nightdude",
    config=dict(
        GAME_NAME=GAME_NAME,
        MCTS_POLICY_EXPLORE=MCTS_POLICY_EXPLORE,
        ACTION_SPACE=ACTION_SPACE,
        OBSERVATION_DIM=OBSERVATION_DIM,
        HIDDEN_DIM=HIDDEN_DIM,
        BUFFER_SIZE=BUFFER_SIZE,
        BATCH_SIZE=BATCH_SIZE,
        UPDATE_EVERY=UPDATE_EVERY,
        EPISODES=EPISODES,
        c=c,
    ),
    save_code=False,
)

for e in trange(EPISODES):

    start_time = time.time()
    reward_e = 0
    game = gym.make(GAME_NAME, render_mode="rgb_array")
    observation, _ = game.reset()
    done = False

    new_game = deepcopy(game)
    mytree = Node(new_game, False, 0, observation, 0)

    obs = []
    ps = []
    p_obs = []

    step = 0

    while not done:

        step = step + 1

        mytree, action, ob, p, p_ob = Policy_Player_MCTS(mytree)

        obs.append(ob)
        ps.append(p)
        p_obs.append(p_ob)

        _, reward, done, _, _ = game.step(action)

        reward_e = reward_e + reward

        if RENDER:
            show_image(game.render(), game_finished=done)

        if done:
            for i in range(len(obs)):
                replay_buffer.add(obs[i], reward_e, p_obs[i], ps[i])
            game.close()
            print("episode #" + str(e + 1))
            break

    print("reward " + str(reward_e))
    rewards.append(reward_e)
    moving_average.append(np.mean(rewards[-100:]))

    if (e + 1) % UPDATE_EVERY == 0 and len(replay_buffer) > BATCH_SIZE:

        # update and train the neural networks
        experiences = replay_buffer.sample()

        # Each current state has as target value the total rewards of the episode
        # obs are observeration p_{1, ...,  N}
        # they have rewards {1, ..., N} associated with them
        # because rewards are given for an action and at step 0 no action has been taken yet
        inputs = torch.from_numpy(np.array([[experience.obs] for experience in experiences])).float()
        targets = torch.from_numpy(np.array([[experience.v / MAX_REWARD] for experience in experiences])).float()

        loss_v = train_step_v(inputs, targets)
        v_losses.append(loss_v)

        # Each state has as target policy the policy according to visit counts
        # p_obs are observeration p_{0, ...,  N-1}
        # they have actions at steps a_{0, ..., N-1} associated with them (because no action at the last step)
        inputs = torch.from_numpy(np.array([[experience.p_obs] for experience in experiences])).float()
        targets = torch.from_numpy(np.array([[experience.p] for experience in experiences])).float()

        loss_p = train_step_p(inputs, targets, e)
        p_losses.append(loss_p)

        wandb.log(
            dict(
                episode=e,
                v_loss=loss_v,
                p_loss=loss_p,
                reward=reward_e,
                ep_wall_t=np.round(time.time() - start_time, 2),
                lr_p=optimizer_p.param_groups[0]["lr"],
            )
        )

if SAVE_MODELS:
    torch.save(policy_v, "value_net_full.pth")
    torch.save(policy_p, "policy_net_full.pth")


# plot rewards, value losses and policy losses
plt.plot(rewards)
plt.plot(moving_average)
plt.show()

plt.plot(v_losses)
plt.show()

plt.plot(p_losses)
plt.show()

print("moving average: " + str(np.mean(rewards[-20:])))
