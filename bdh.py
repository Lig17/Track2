# Copyright 2025 Pathway Technology, Inc.

import dataclasses
import math

import torch
import torch.nn.functional as F
from torch import nn


# -------------------------------------------------
# Config
# -------------------------------------------------

@dataclasses.dataclass
class BDHConfig:
    n_layer: int = 3
    n_embd: int = 128
    dropout: float = 0.1
    n_head: int = 4
    mlp_internal_dim_multiplier: int = 128
    vocab_size: int = 256


# -------------------------------------------------
# Rotary frequencies (unchanged)
# -------------------------------------------------

def get_freqs(n, theta, dtype):
    def quantize(t, q=2):
        return (t / q).floor() * q

    return (
        1.0
        / (theta ** (quantize(torch.arange(0, n, 1, dtype=dtype)) / n))
        / (2 * math.pi)
    )


# -------------------------------------------------
# Attention with Synaptic Scaffolding
# -------------------------------------------------

class Attention(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        nh = config.n_head
        D = config.n_embd
        N = config.mlp_internal_dim_multiplier * D // nh

        self.freqs = torch.nn.Buffer(
            get_freqs(N, theta=2**16, dtype=torch.float32).view(1, 1, 1, N)
        )

        # --- Synaptic Scaffolding state ---
        self.register_buffer("sigma", torch.zeros(nh, N, N))
        self.register_buffer("H", torch.zeros(nh, N, N))

        # hyperparameters (conservative defaults)
        self.eta = 0.05
        self.lambda_base = 0.01
        self.alpha = 0.1
        self.out_proj = nn.Linear(N, config.n_embd, bias=False)

    # -------------------------------------------------
    # Rotary helpers (unchanged)
    # -------------------------------------------------

    @staticmethod
    def phases_cos_sin(phases):
        phases = (phases % 1) * (2 * math.pi)
        return torch.cos(phases), torch.sin(phases)

    @staticmethod
    def rope(phases, v):
        v_rot = torch.stack((-v[..., 1::2], v[..., ::2]), dim=-1).view(*v.size())
        phases_cos, phases_sin = Attention.phases_cos_sin(phases)
        return (v * phases_cos).to(v.dtype) + (v_rot * phases_sin).to(v.dtype)

    # -------------------------------------------------
    # Hebbian synaptic update (core contribution)
    # -------------------------------------------------

      
    @torch.no_grad()
    def update_synapses(self, x):
        """
        x: (B, nh, N) sparse neuron firing at one timestep
        """

        # Compute activity FIRST
        active = (x > 0).float()
        activity = active.mean().item()

        print("Synapse update triggered, activity =", activity)

        # Sparsity gate
        if activity > 0.30:
            return
        # Gate based on top-k density, not raw firing


        # Top-k sparsification
        k = 32
        topk_vals, topk_idx = torch.topk(x, k=k, dim=-1)

        sparse = torch.zeros_like(x)
        sparse.scatter_(-1, topk_idx, topk_vals)

        # Hebbian update
        hebb = torch.einsum("bhn,bhm->hnm", sparse, sparse)

        Lambda = self.lambda_base * torch.exp(-self.alpha * self.H)

        self.sigma += self.eta * hebb - Lambda * self.sigma
        self.sigma.clamp_(min=0.0)

        self.H += (hebb > 0).float()

    def save_synapses(self, path):
        torch.save(
            {
                "sigma": self.sigma,
                "H": self.H,
            },
            path,
        )

    def load_synapses(self, path):
        state = torch.load(path, map_location=self.sigma.device)
        self.sigma.copy_(state["sigma"])
        self.H.copy_(state["H"])

    # -------------------------------------------------
    # Diagnostics (optional but useful)
    # -------------------------------------------------

    def get_diagnostics(self):
        with torch.no_grad():
            return {
                "sigma_norm": torch.norm(self.sigma).item(),
                "stiff_synapses": (self.H > 5).float().mean().item(),
                "avg_decay": torch.mean(
                    self.lambda_base * torch.exp(-self.alpha * self.H)
                ).item(),
            }

    # -------------------------------------------------
    # Forward (attention replaced with synaptic reasoning)
    # -------------------------------------------------

    def forward(self, Q, K, V):
        assert self.freqs.dtype == torch.float32
        assert K is Q

        B, nh, T, N = Q.size()

        # rotary position encoding (unchanged)
        r_phases = (
            torch.arange(
                0,
                T,
                device=self.freqs.device,
                dtype=self.freqs.dtype,
            ).view(1, 1, -1, 1)
        ) * self.freqs
        Q = self.rope(r_phases, Q)

        outputs = []

        for t in range(T):
            x_t = Q[:, :, t, :]  # (B, nh, N)

            # synaptic read
            y_t = torch.einsum("bhn,hnm->bhm", x_t, self.sigma)

            # learning only during inference
            if not self.training and not getattr(self, "freeze_learning", False):
                self.update_synapses(x_t)

            outputs.append(y_t.unsqueeze(2))

        Y = torch.cat(outputs, dim=2)  # (B, nh, T, N)

        # project back to value space
        # Aggregate across heads
        Y_agg = Y.sum(dim=1)        # (B, T, N)

        # Expand to match value shape
        Y_agg = Y_agg.unsqueeze(1)  # (B, 1, T, N)

        # Project by modulation with V (same interface as original BDH)
        Y_proj = self.out_proj(Y_agg)  # (B, 1, T, D)
        return Y_proj


# -------------------------------------------------
# BDH Model (UNCHANGED except using new Attention)
# -------------------------------------------------

class BDH(nn.Module):
    def __init__(self, config: BDHConfig):
        super().__init__()
        assert config.vocab_size is not None
        self.config = config
        nh = config.n_head
        D = config.n_embd
        N = config.mlp_internal_dim_multiplier * D // nh

        self.decoder = nn.Parameter(torch.zeros((nh * N, D)).normal_(std=0.02))
        self.encoder = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))
        self.encoder_v = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))

        self.attn = Attention(config)

        self.ln = nn.LayerNorm(D, elementwise_affine=False, bias=False)
        self.embed = nn.Embedding(config.vocab_size, D)
        self.drop = nn.Dropout(config.dropout)

        self.lm_head = nn.Parameter(
            torch.zeros((D, config.vocab_size)).normal_(std=0.02)
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        C = self.config
        B, T = idx.size()
        D = C.n_embd
        nh = C.n_head
        N = D * C.mlp_internal_dim_multiplier // nh

        x = self.embed(idx).unsqueeze(1)
        x = self.ln(x)  # B, 1, T, D

        for _ in range(C.n_layer):
            x_latent = x @ self.encoder
            x_sparse = F.relu(x_latent)  # B, nh, T, N

            yKV = self.attn(Q=x_sparse, K=x_sparse, V=x)
            yKV = self.ln(yKV)

            y_latent = yKV @ self.encoder_v
            y_sparse = F.relu(y_latent)

            xy_sparse = self.drop(x_sparse * y_sparse)

            yMLP = (
                xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ self.decoder
            )
            y = self.ln(yMLP)
            x = self.ln(x + y)

        logits = x.view(B, T, D) @ self.lm_head

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
