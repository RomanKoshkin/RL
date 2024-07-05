import gym
import random
from math import sqrt, log
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt


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

    def getUCBscore(self, c=1.0):
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

        # We use one of the possible MCTS formula for calculating the node
        # value
        return (self.T / self.N) + c * sqrt(log(top_node.N) / self.N)

    def detach_parent(self):
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
        for i in range(GAME_ACTIONS):
            actions.append(i)
            new_game = deepcopy(self.game)
            games.append(new_game)

        child = {}
        for action, game in zip(actions, games):
            observation, reward, done, _, _ = game.step(action)
            child[action] = Node(game, done, self, observation, action)

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
            # update the node value
            current.T = current.T + current.rollout()
        else:
            current.create_child()  # expand the node (by creating 2 children)
            if current.child:
                current = random.choice(list(current.child.values()))  # randomly pick a child
            current.T = current.T + current.rollout()

        current.N += 1  # increment the number of visits to the current node

        # update statistics and backpropagate
        parent = current
        while parent.parent:
            parent = parent.parent
            parent.N += 1
            parent.T = parent.T + current.T

    def rollout(self):
        """
        The rollout is a random play from a copy of the environment
        of the current node using random moves.This will give us a
        value for the current node. Taken alone, this value is quite random,
        but, the more rollouts we will do for such node,the more accurate the
        average of the value for such node will be. This is at the core of the
        MCTS algorithm.
        """

        if self.done:
            return 0

        v = 0
        done = False
        new_game = deepcopy(self.game)
        counter = 0
        while not done:
            action = new_game.action_space.sample()
            observation, reward, done, _, _ = new_game.step(action)
            print(
                f"rollout | from node: {self.id} | step: {counter} | action: {action} | reward: {reward} | done: {done}"
            )
            v = v + reward
            if done:
                new_game.reset()
                new_game.close()
                break
            counter += 1
        return v

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

        max_children = [c for a, c in child.items() if c.N == max_N]

        if len(max_children) == 0:
            print("error zero length ", max_N)

        max_child = random.choice(max_children)

        return max_child, max_child.action_index


def Policy_Player_MCTS(mytree):
    """
    Our strategy for using the MCTS is quite simple:
    - in order to pick the best move from the current node:
        - explore the tree starting from that node for a certain number
        of iterations to collect reliable statistics
        - pick the node that, according to MCTS, is the
        best possible next action
    """

    for i in range(MCTS_POLICY_EXPLORE):
        mytree.explore()

    # choose the next action based on the _value_ of the child nodes
    next_tree, next_action = mytree.next()

    # note that here we are detaching the current node and returning
    # the sub-tree that starts from the node rooted at the choosen action.
    # The next search, hence, will not start from scratch but will already
    # have collected information and statistics about the nodes, so we
    # can reuse such statistics to make the search even more reliable!
    next_tree.detach_parent()

    return next_tree, next_action


def show_image(image):
    """
    This function is used to display the image of the game.
    """
    plt.imshow(image)  # Display the image
    plt.draw()
    plt.pause(0.02)  # Pause to render the image, adjust time as needed
    plt.clf()  # Cl


RENDER = True
rewards = []
moving_average = []
GAME_NAME = "CartPole-v0"
# GAME_NAME = "Acrobot-v1"
# GAME_NAME = "LunarLander-v2"  # won't work, because it's observation space is (210, 160, 3)

# MCTS exploring constant: the higher, the more reliable, but slower in execution time
MCTS_POLICY_EXPLORE = 8

"""
Here we are experimenting with our implementation:
- we play a certain number of episodes of the game
- for deciding each move to play at each step, we will apply our MCTS algorithm
- we will collect and plot the rewards to check if the MCTS is actually
working.
- For CartPole-v0, in particular, 200 is the maximum possible reward.
"""

reward_e = 0
game = gym.make(GAME_NAME, render_mode="rgb_array")

GAME_ACTIONS = game.action_space.n
GAME_OBS = game.observation_space.shape[0]

observation = game.reset()
done = False

new_game = deepcopy(game)

# create the root node of the MCTS tree. It doesn't have children yet.
mytree = Node(new_game, False, 0, observation, 0)

while not done:
    # update the expected value of future moves using MCTS
    # Policy_Player_MCTS will play random games until the end and update the values of the tree nodes
    # each time, Policy_Player_MCTS returns a tree with the root detached (removed),
    # so that the next optimal step of the actual game (not to be confused with the random MCTS games)
    # becomes the root of the tree
    mytree, action = Policy_Player_MCTS(mytree)
    observation, reward, done, _, _ = game.step(action)
    reward_e = reward_e + reward

    if RENDER:
        show_image(game.render())

    if done:
        print("reward_e " + str(reward_e))
        game.close()
        break

rewards.append(reward_e)
moving_average.append(np.mean(rewards[-100:]))

plt.plot(rewards)
plt.plot(moving_average)
plt.show()
print("moving average: " + str(np.mean(rewards[-20:])))
