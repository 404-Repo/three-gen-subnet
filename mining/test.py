import torch
from mining import forward, load_models
from neurons.protocol import TextTo3D

device = torch.device("cuda:0")
cache = "cache"

models = load_models(device, cache)
synapse = TextTo3D(prompt_in="A Golden Poison Dart Frog")
forward(synapse, models)
with open("output.ply", "wb") as f:
    f.write(synapse.mesh_out)
