import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyV(nn.Module):
    """
    The Value Neural Network will approximate the Value of the node,
    given a State of the game.
    """

    def __init__(self, OBSERVATION_DIM, HIDDEN_DIM):
        super(PolicyV, self).__init__()

        # Layers
        self.dense1 = nn.Linear(in_features=OBSERVATION_DIM, out_features=HIDDEN_DIM)
        self.dense2 = nn.Linear(in_features=HIDDEN_DIM, out_features=HIDDEN_DIM)
        self.v_out = nn.Linear(in_features=HIDDEN_DIM, out_features=1)

        for layer in [self.dense1, self.dense2, self.v_out]:
            nn.init.kaiming_uniform_(layer.weight, mode="fan_in", nonlinearity="relu")

    def forward(self, x):
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = self.v_out(x)
        return x


class PolicyP(nn.Module):
    """
    The Policy Neural Network will approximate the MCTS policy for the
    choice of nodes, given a State of the game.
    """

    def __init__(self, ACTION_SPACE, OBSERVATION_DIM, HIDDEN_DIM):
        super(PolicyP, self).__init__()

        # Layers
        self.dense1 = nn.Linear(in_features=OBSERVATION_DIM, out_features=HIDDEN_DIM)
        self.dense2 = nn.Linear(in_features=HIDDEN_DIM, out_features=HIDDEN_DIM)
        self.p_out = nn.Linear(in_features=HIDDEN_DIM, out_features=ACTION_SPACE)
        for layer in [self.dense1, self.dense2, self.p_out]:
            nn.init.kaiming_uniform_(layer.weight, mode="fan_in", nonlinearity="relu")

    def forward(self, x):
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = self.p_out(x)
        x = F.softmax(x, dim=-1)  # NOTE: although in PyTorch this is not necessary, you need this for correct UCB1
        return x


# Example use
if __name__ == "__main__":

    HIDDEN_DIM = 64
    OBSERVATION_DIM = 4  # Define this based on your game's specific observation space
    ACTION_SPACE = 2  # Define this based on your game's specific action space

    # Example input tensor
    input_tensor = torch.randn(1, HIDDEN_DIM)  # Batch size of 1, and input features

    # Value Network
    value_net = PolicyV(OBSERVATION_DIM, HIDDEN_DIM)
    value_output = value_net(input_tensor)
    print("Value Output:", value_output)

    # Policy Network
    policy_net = PolicyP(ACTION_SPACE, OBSERVATION_DIM, HIDDEN_DIM)
    policy_output = policy_net(input_tensor)
    print("Policy Output:", policy_output)
