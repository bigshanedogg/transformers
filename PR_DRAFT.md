> **Draft PR — waiting for issue approval.** This PR is opened alongside the issue request.
> It will be marked ready for review after a maintainer gives the go-ahead on the issue.

# What does this PR do?

Adds native Transformers support for **[HyperCLOVA X SEED Think 14B](https://huggingface.co/naver-hyperclovax/HyperCLOVAX-SEED-Think-14B)**,
a 14.74B-parameter Korean reasoning LLM developed by NAVER Cloud.
- related issue: #44957 

## Architecture

LLaMA-style decoder-only transformer with two modifications:

- **Peri-Layer Normalization** (`use_post_norm`): an extra `RMSNorm` is applied *after* each
  sub-layer output (both attention and MLP), in addition to the standard pre-norm.
- **Maximal Update Parametrization (μP)**: four per-config scaling factors replace fixed constants:
  - `attention_multiplier` — replaces `1/sqrt(head_dim)` in attention
  - `residual_multiplier` — scales each sub-layer output before adding to the residual stream
  - `embedding_multiplier` — scales the token embedding output
  - `logits_scaling` — scales final logits before softmax / sampling

## Implementation approach

Following the maintainer's guidance in #44957, this PR uses the **modular system** (`modular_hyperclovax.py`) to minimise LOC and make the diff easy to review-iterate. (Roughly 59% of lines are generated rather than manually maintained.)

The maintainer suggested inheriting the decoder layer with post-norms from **GLM4**. After evaluation, **Granite** was chosen as the decoder layer base instead, for the following reasons:

- `use_post_norm` is **optional** (`False` by default). GLM4's decoder layer has post-norms **always on** — inheriting from it would require logic to conditionally disable `post_self_attn_layernorm` / `post_mlp_layernorm`, adding complexity rather than reducing it.
- Granite's decoder layer already provides `residual_multiplier` (always-active MuP). When `use_post_norm=False`, `HyperCLOVAXDecoderLayer` is identical to `GraniteDecoderLayer` — zero extra code.
- Using GLM4 would require adding both `residual_multiplier` **and** conditionally disabling its built-in norms — two changes in opposite directions for no net gain in code reuse.

All other modules (RMSNorm, MLP, Attention, etc.) are inherited from Granite unchanged. The modular file is a few hundred LOC as suggested.

## (WIP) Benchmark validation

| Tasks | Metric | vLLM | Huggingface (this PR) }
|---|---|---|---|
| hellaswag (non-think) | acc_norm | 0.6521 | - |
| gsm8k (non-think) | flexible-extract | 0.9151 | - 

## External support

- **Huggingface hub**: [naver-hyperclovax/HyperCLOVAX-SEED-Think-14B](https://huggingface.co/naver-hyperclovax/HyperCLOVAX-SEED-Think-14B)
- **Technical report**: [arXiv 2506.22403](https://arxiv.org/abs/2506.22403)
- **vLLM upstream**: [vllm-project/vllm#37107](https://github.com/vllm-project/vllm/pull/37107) (merged 2026-03-16)

## Code Agent Policy

- [x] I confirm that this is not a pure code agent PR.

A code agent was used for mechanical tasks such as aligning docstrings and comments. The core implementation was written by the submitter directly, who has reviewed every changed line and personally run the tests including benchmark validation.

## Before submitting
- [ ] This PR fixes a typo or improves the docs (you can dismiss the other checks if that's the case).
- [x] Did you read the [contributor guideline](https://github.com/huggingface/transformers/blob/main/CONTRIBUTING.md#create-a-pull-request),
      Pull Request section?
- [x] Was this discussed/approved via a Github issue or the [forum](https://discuss.huggingface.co/)? Please add a link
      to it if that's the case.
  - #44957 
- [x] Did you make sure to update the documentation with your changes? Here are the
      [documentation guidelines](https://github.com/huggingface/transformers/tree/main/docs), and
      [here are tips on formatting docstrings](https://github.com/huggingface/transformers/tree/main/docs#writing-source-documentation).
- [ ] Did you write any new necessary tests?
