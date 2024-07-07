import gym
import matplotlib.pyplot as plt
import torch


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

# policy_v = torch.load("value_net_full.pth")
policy_p = torch.load("policy_net_full.pth")
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
