import numpy as np
import gym
import torch
import safe_grid_gym  # useful to create the environment
from nn_builder.pytorch.NN import NN
from tqdm import tqdm

device = "cpu"

world_shape = (12, 12)

hyperparameters = {
    "input_dim": world_shape[0] * world_shape[1],
    "output_dim": 4,
    "linear_hidden_units": [128, 128, 64],
    "initialiser": "Xavier"
}

actor_network = NN(input_dim=world_shape[0] * world_shape[1],
                   layers_info=hyperparameters["linear_hidden_units"] + [hyperparameters["output_dim"]],
                   initialiser=hyperparameters["initialiser"], output_activation="softmax",
                   random_seed=42).to(device)

model_name = "../Models/SAC_Discrete_local_network.pt"
actor_network.load_state_dict(torch.load(model_name))
actor_network.eval()

env = gym.make("SafeInterruptibility-v0")
state = env.reset()

dataset = []
# runs 100 steps, reset the environment if the agents win or get stuck on the interruption.
for i in tqdm(range(300)):
    state = torch.FloatTensor([state]).to(device)
    to_store = torch.squeeze(state, 0)
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
    if np.array_equal(state.numpy().reshape((1, world_shape[0], world_shape[1])), new_state):
        state = env.reset()
    else:
        state = new_state

    new_sample = {'action': action[0], 'state': to_store.numpy()}
    doubled = False
    for sample in dataset:
        if new_sample['action'] == sample['action'] and np.array_equal(new_sample['state'],sample['state']):
            doubled = True
    if not doubled:
        dataset.append(new_sample)

env.close()

dataset = np.array(dataset)

np.save("../dataset_supervised", dataset)
