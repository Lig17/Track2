
#  Baby Dragon Hatchling (BDH) with Synaptic Scaffolding

### Inference-Time Learning via Hebbian Synapses (Post-Transformer Architecture)

---

## Overview

This repository extends the **Baby Dragon Hatchling (BDH)** architecture with **Synaptic Scaffolding** — a biologically inspired mechanism that enables **learning during inference without backpropagation**.

Unlike Transformers, which rely on a **temporary KV-cache** and **frozen weights**, BDH stores memory **directly in synapses** using **Hebbian plasticity**.
This allows the model to **adapt, retain, and reuse knowledge across sessions**.

> **Core idea:**
> **State lives in synapses, not in prompts or external memory.**

---

## What Is Synaptic Scaffolding?

Synaptic Scaffolding introduces three key mechanisms:

### 1️⃣ Hebbian Fast Weights (σ)

* Connections between **co-active neurons strengthen during inference**
* Classic rule: *“Neurons that fire together, wire together”*

### 2️⃣ Metaplasticity (H)

* Frequently used synapses **forget more slowly**
* Important memories become **structurally protected**

### 3️⃣ Cross-Session Persistence

* Synaptic state can be **saved and reloaded**
* Knowledge survives **beyond a single prompt or context window**

✅ Enables **native continual learning**
❌ No fine-tuning
❌ No external retrieval system

---

##  Architecture Summary

| Component        | Transformer             | BDH + Synaptic Scaffolding |
| ---------------- | ----------------------- | -------------------------- |
| Memory           | KV-Cache (temporary)    | Synapses (σ)               |
| Learning         | Training-time only      | Inference-time             |
| Forgetting       | Immediate after session | Controlled decay           |
| Scaling          | O(T²) attention         | O(T) local updates         |
| Interpretability | Low                     | High (sparse synapses)     |

---

## Repository Structure

```
.
├── bdh.py                 # BDH model + Synaptic Scaffolding
├── train.py               # Baseline training (Tiny Shakespeare)
├── baseline_test.py       # Baseline inference (no learning)
├── scaffolding_test.py    # Synaptic exposure + persistence test
├── input.txt              # Tiny Shakespeare dataset
├── bdh_baseline.pt        # Saved baseline weights
├── glip_memory.pt         # Saved synaptic memory (example)
└── README.md
```

---

## Environment Setup

**Python ≥ 3.10 recommended**

```bash
python -m venv bdh_env
source bdh_env/bin/activate
pip install torch numpy requests
```

⚠️ CPU-only runs are supported but slower.

---

## Step 1: Baseline Training

Train BDH normally (**no synaptic learning yet**):

```bash
python train.py
```

### What this does:

* Trains BDH on **Tiny Shakespeare**
* Establishes **slow structural weights**
* Saves a **baseline language model**

**Expected output:**

```
Step: 0 loss ...
Step: 100 loss ...
Training done, now generating a sample
```

---

## Step 2: Baseline Test (No Learning)

Test the frozen model:

```bash
python baseline_test.py
```

**Example output:**

```
What is a glip?

DUKE:
I will tends, and's the caure too arms.
```

👉 The model **does not know** what a *glip* is.

---

## Step 3: Synaptic Scaffolding Test

### Learning During Inference

Run inference-time learning:

```bash
python scaffolding_test.py
```

### What happens internally:

* The model is exposed repeatedly to a new fact

  > *“A glip is a small blue bird.”*
* Sparse neurons co-activate
* Synapses (σ) strengthen via **Hebbian updates**
* Synaptic history (H) reduces decay on frequent paths
* Synaptic state is **saved to disk**

**Example console output:**

```
Synapse update triggered, activity = 0.36
Exposure done. Synapses saved.
{'sigma_norm': 4884.6, 'stiff_synapses': 0.0016, 'avg_decay': 0.0099}
```

---

## Step 4: Cross-Session Recall

In a fresh model instance, load synapses:

```python
model.attn.load_synapses("glip_memory.pt")
```

**Prompt:**

```
What is a glip?
```

The model’s **internal structure has changed** — even without retraining.

>  **Key point:**
> Learning is demonstrated via **structural change**, not perfect text fluency.

---

## Diagnostics (Important for Evaluation)

Inspect synaptic health:

```python
model.attn.get_diagnostics()
```

Returns:

* **sigma_norm** → total memory formed
* **stiff_synapses** → fraction of hardened connections
* **avg_decay** → effective forgetting rate

These metrics provide **quantitative evidence of learning**.

---

## What This Demonstrates

✅ Learning without backpropagation
✅ Memory beyond the context window
✅ No external retrieval system
✅ Biologically plausible plasticity
✅ Interpretable internal state

Directly addresses:

* Transformer amnesia
* Catastrophic forgetting
* KV-cache scaling limits

---

## Experimental Status

* Research prototype
* Text output may be noisy (expected)
* **Structural metrics are the primary signal**
* Designed for **Frontier / Research track** evaluation

---

## Hackathon Relevance

This project aligns with **Path B: Continuous Learning & Synaptic Dynamics**.

> This is **not** a chatbot demo.
> It is a **systems-level exploration of post-Transformer intelligence**.

---

## References

* Pathway — *Baby Dragon Hatchling (BDH)*
* *The Dragon Hatchling: The Missing Link Between Transformers and the Brain* (arXiv)
* Hebbian Learning & Metaplasticity (Neuroscience)

---

## Acknowledgements

Inspired by the original **BDH work by Pathway** and the broader community exploring **biologically grounded AI**.

---
