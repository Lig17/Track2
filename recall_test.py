import torch
import bdh

device = "cpu"
config = bdh.BDHConfig()

model = bdh.BDH(config).to(device)
model.eval()

model.attn.load_synapses("glip_memory.pt")

prompt = torch.tensor(
    [bytearray("What is a glip?", "utf-8")],
    dtype=torch.long
)

out = model.generate(prompt, max_new_tokens=20)
print(bytes(out[0].tolist()).decode(errors="ignore"))
