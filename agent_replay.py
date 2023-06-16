import numpy as np
import gym
import torch
import safe_grid_gym  # useful to create the environment
from nn_builder.pytorch.NN import NN

device = "cpu"

world_shape = (8,10)

hyperparameters = {
            "input_dim" : world_shape[0]*world_shape[1],
            "output_dim" : 4,
            "linear_hidden_units": [64, 64],
            "initialiser": "Xavier"
}

actor_network = NN(input_dim= world_shape[0]*world_shape[1],
                   layers_info=hyperparameters["linear_hidden_units"] + [hyperparameters["output_dim"]],
                   initialiser=hyperparameters["initialiser"],
                   random_seed=42).to(device)

actor_network.load_state_dict(torch.load('../Deep-Reinforcement-Learning-Algorithms-with-PyTorch/Models/SAC_local_network.pt'))
actor_network.eval()

env = gym.make("SafeInterruptibility-v0")
state = env.reset()

# runs 100 steps, reset the environment if the agents win or get stuck on the interruption.
for i in range(100):
    state = torch.FloatTensor([state]).to(device)
    if len(state.shape) == 1: state = state.unsqueeze(0)
    if len(state.shape) == 4:
        state = torch.squeeze(state, 0)
        state = torch.squeeze(state, 0)
        state = torch.flatten(state)
        state = torch.unsqueeze(state, 0)
    elif len(state.shape) == 3:
        state = torch.flatten(state, 1)
    action_probabilities = actor_network(state)
    action = torch.argmax(action_probabilities, dim=-1)
    action = action.detach().cpu().numpy()
    new_state, reward, done, info = env.step(action[0])
    env.render(mode = 'human')
    if np.array_equal(state.numpy().reshape((1,world_shape[0],world_shape[1])),new_state):
        state = env.reset()
    else:
        state = new_state