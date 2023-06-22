import numpy as np
import sys
import gym
from matplotlib import animation
import matplotlib.pyplot as plt
import torch
import safe_grid_gym  # useful to create the environment
from nn_builder.pytorch.NN import NN

def save_frames_as_gif(frames, path='./images/', filename='gym_animation.gif'):
    # Set the figure size based on the desired human-visible size
    plt.figure(figsize=(8, 6))

    patch = plt.imshow(frames[0], aspect='auto')
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', dpi=80, fps=5)

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

model_name = sys.argv[1]
actor_network.load_state_dict(torch.load('Models/' + model_name))
actor_network.eval()

env = gym.make("SafeInterruptibility-v0")
state = env.reset()

frames = []
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
    frame = env.render(mode="rgb_array")
    frames.append(np.transpose(frame, (1, 2, 0)))
    if np.array_equal(state.numpy().reshape((1,world_shape[0],world_shape[1])),new_state):
        state = env.reset()
    else:
        state = new_state

env.close()
save_frames_as_gif(frames)