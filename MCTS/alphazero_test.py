import gym
import matplotlib.pyplot as plt
import torch
import argparse


argparser = argparse.ArgumentParser()
argparser.add_argument("--runid", type=int, default=None)
args = argparser.parse_args()
runid = args.runid

name_suffix = '_' + str(runid) if runid is not None else ''

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


RENDER = True
GAME_NAME = "CartPole-v1"

policy_v = torch.load(f"WEIGHTS/value_net_full{name_suffix}.pth")
policy_p = torch.load(f"WEIGHTS/policy_net_full{name_suffix}.pth")

game = gym.make(GAME_NAME, render_mode="rgb_array")
observation, _ = game.reset()
observation = torch.from_numpy(observation).float().unsqueeze(0)
done = False

while not done:
    action = policy_p(observation).argmax().item()
    observation, reward, done, _, _ = game.step(action)
    observation = torch.from_numpy(observation).float().unsqueeze(0)
    if RENDER:
        show_image(game.render(), game_finished=done)
