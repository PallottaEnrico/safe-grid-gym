import os

from omlt.io import write_onnx_model_with_bounds
import torch

from nn_builder.pytorch.NN import NN

device = "cpu"

world_shape = (8,10)

hyperparameters = {
            "input_dim" : world_shape[0]*world_shape[1],
            "output_dim" : 4,
            "linear_hidden_units": [64, 64],
            "initialiser": "Xavier"
}

actor_network = NN(input_dim= hyperparameters['input_dim'],
                   layers_info=hyperparameters["linear_hidden_units"] + [hyperparameters["output_dim"]],
                   initialiser=hyperparameters["initialiser"],
                   random_seed=42).to(device)

actor_network.load_state_dict(torch.load('../Deep-Reinforcement-Learning-Algorithms-with-PyTorch/Models/SAC_local_network.pt'))
actor_network.eval()

# a forward pass is required to make the export work
dummy_input = torch.randn(1, hyperparameters["input_dim"], device=device)
actor_network(dummy_input)

input_bounds = {(0, i): (0, 5) for i in range(hyperparameters["input_dim"])}
os.makedirs("../onnx_models", exist_ok=True)
file_path = "../onnx_models/SAC_Discrete_actor_network.onnx"

# model input used for exporting
torch.onnx.export(
    actor_network,
    dummy_input,
    file_path,
    verbose=True,
    input_names=['state'] + ["learned_%d" % i for i in range(6)],
    output_names=['action'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

write_onnx_model_with_bounds(file_path, None, input_bounds)
print(f"Wrote PyTorch model to {file_path}")