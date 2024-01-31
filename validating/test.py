import torch
import time
from neurons.protocol import TextTo3D

from validating import score_responses, load_models

with open("output.ply", "rb") as f:
    mesh = f.read()

# _render_images(mesh)

device = torch.device("cuda:0")
models = load_models(device, "cache")

prompt = "A Golden Poison Dart Frog"
synapse = TextTo3D(
    prompt_in=prompt,
    mesh_out=mesh,
)

start = time.time()
scores = score_responses(prompt, [synapse, synapse], device, models)
end = time.time()

print(scores)
print(end - start)
