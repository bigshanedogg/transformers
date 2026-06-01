Thank you for the thoughtful review and for the suggestion to reuse the Granite model class. We deeply appreciate the effort to keep the codebase lean, and we fully share that goal. However, after thorough investigation, we'd like to respectfully present some concrete findings that we hope will be useful context, and that suggest a dedicated implementation may be necessary for correctness.

---

### 1. Structural incompatibility: Peri-Layer Normalization

HyperCLOVAX introduces **Peri-Layer Normalization (Peri-LN)** — an additional RMSNorm applied after each sub-layer output (both attention and MLP). This architectural feature is described in the HyperCLOVAX technical report and is controlled by `use_post_norm=True` in the config.

The actual released checkpoint carries `post_norm1.weight` and `post_norm2.weight` in every decoder layer. Attempting to load it with `GraniteForCausalLM` produces the following:

```
Key                                     | Status     |
----------------------------------------+------------+
model.layers.{0...37}.post_norm2.weight | UNEXPECTED |
model.layers.{0...37}.post_norm1.weight | UNEXPECTED |
```

These are not extraneous weights — they are load-bearing parameters that change the forward pass. Since Granite has no equivalent, this mismatch cannot be bridged via a config alias alone.

---

### 2. Logits scaling: opposite semantics

Both models share a `logits_scaling` parameter, but they apply it in **opposite directions**:

```python
# Granite
logits = logits / self.config.logits_scaling   # division

# HyperCLOVAX
logits = self.lm_head(...) * self.logits_scaling   # multiplication
```

The actual HyperCLOVAX checkpoint uses `logits_scaling = 0.125`. Under Granite's semantics this becomes a division by 0.125, scaling the logits **×8** — the inverse of what was intended. Under HyperCLOVAX's semantics it correctly scales logits **×0.125**, consistent with the MuP training setup.

This produces dramatically different generation behavior, as confirmed empirically:

**With `GraniteForCausalLM`** (inverted logits scale + no Peri-LN):
```
역대 역대 역대 역대 역대 역대 역대 역대 역대 역대 역대 역대 역대 역대 ...
```

**With `HyperCLOVAXForCausalLM`** (correct implementation):
```
Okay, so the user wants a detailed explanation of my capabilities. Let me start by
breaking down what I can do. First, I need to outline the main areas where I can
assist users. These areas typically include natural language processing (NLP),
information retrieval, text generation, and more...
```

---

### 3. Benchmark reproduction validates the implementation

The draft PR's implementation — with `use_post_norm=True` and multiplicative `logits_scaling` — successfully reproduces the benchmark scores reported in the official HyperCLOVAX technical report. This gives us confidence that both features are essential for correctness, not implementation detail.

---

### Summary

| Feature | Granite | HyperCLOVAX |
|---|---|---|
| Peri-Layer Normalization | ✗ | ✓ (`use_post_norm`) |
| Logits scaling direction | `logits / scale` | `logits * scale` |
| `logits_scaling` in actual checkpoint | N/A | `0.125` (×8 error under Granite) |
| Can load `think_14b_hf` checkpoint? | ✗ (UNEXPECTED keys) | ✓ |
| Reproduces tech report benchmarks? | ✗ (degenerate output) | ✓ |

---

We agree with the broader principle of minimizing redundant model classes, and we're genuinely open to exploring whether Granite could be extended to optionally support Peri-LN, or whether HyperCLOVAX could be restructured to inherit from Granite. We'd welcome any guidance on the preferred direction. That said, given the two concrete issues above — a hard weight-loading failure and a numerically inverted logits scaling — we believe some form of dedicated handling is needed to correctly serve existing HyperCLOVAX checkpoints.

We hope these findings are helpful, and we're happy to discuss further or adjust the approach based on your feedback. Thank you again for your time and thorough review.
