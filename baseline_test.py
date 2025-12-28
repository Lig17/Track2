import torch
import bdh

device = "cpu"

# load model
config = bdh.BDHConfig(
    n_layer=3,
    n_embd=128,
)
model = bdh.BDH(config)
model.load_state_dict(torch.load("bdh_baseline.pt", map_location=device))
model.eval()

def encode(text):
    return torch.tensor(
        list(text.encode("utf-8")),
        dtype=torch.long
    ).unsqueeze(0)

def decode(tensor):
    return bytes(tensor.squeeze(0).tolist()).decode(errors="ignore")

# CHANGE PROMPT HERE
prompt = encode("What is a glip?\n")


out = model.generate(prompt, max_new_tokens=60, top_k=5)
print(decode(out))
