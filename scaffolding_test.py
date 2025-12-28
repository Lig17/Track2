import torch
import bdh

device = "cpu"

config = bdh.BDHConfig(
    n_layer=3,
    n_embd=128,
)

model = bdh.BDH(config)
state = torch.load("bdh_baseline.pt", map_location=device)
model.load_state_dict(state, strict=False)
model.eval()   # IMPORTANT: learning happens only in eval()

def encode(text):
    return torch.tensor(
        list(text.encode("utf-8")),
        dtype=torch.long
    ).unsqueeze(0)

def decode(tensor):
    return bytes(tensor.squeeze(0).tolist()).decode(errors="ignore")

# --- Exposure phase ---
fact = encode("A glip is a small blue bird.\n")

for _ in range(5):
    model(fact)

# Save synaptic memory
model.attn.save_synapses("glip_memory.pt")

print("Exposure done. Synapses saved.")
print(model.attn.get_diagnostics())
