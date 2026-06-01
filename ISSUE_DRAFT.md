# GitHub Issue 초안 — HyperCLOVAX 모델 추가 요청

이슈를 열기 전 중복 확인:

```bash
gh issue list --repo huggingface/transformers --state open --search "HyperCLOVA"
gh pr list   --repo huggingface/transformers --state open --search "HyperCLOVA"
```

**Title**: `Add HyperCLOVA X SEED Think 14B`

Label: `New model`

---

## 이슈 본문 (아래 내용을 GitHub에 그대로 붙여넣기)

---

### Model description

It would be great to add native support for **HyperCLOVA X SEED Think 14B** to the Transformers library, so users can load it without `trust_remote_code=True`.

**HyperCLOVA X SEED Think 14B** is a 14.74B-parameter reasoning LLM developed by NAVER Cloud. It is a LLaMA-style decoder-only transformer with two architectural modifications not present in standard LLaMA:

- **Peri-Layer Normalization**: an extra RMSNorm is applied *after* each sub-layer output (in addition to the standard pre-norm), controlled by a `use_post_norm` config flag.
- **Maximal Update Parametrization (μP)**: per-config scaling factors (`attention_multiplier`, `residual_multiplier`, `embedding_multiplier`, `logits_scaling`) replace the standard fixed scaling, enabling stable training across model sizes.

The model supports dual-mode reasoning: **Think** (chain-of-thought before answering) and **Non-Think** (direct answer), switchable via `apply_chat_template(force_reasoning=True/False)`. It also supports function calling via a custom ChatML dialect. The model is [supported in vLLM](https://github.com/vllm-project/vllm/pull/37107) as of March 2026.

We (the NAVER Cloud HyperCLOVA X team) have a working implementation ready. Draft PR: #YYYY

**License**: Released under the [HyperCLOVA X SEED Model License Agreement](https://huggingface.co/naver-hyperclovax/HyperCLOVAX-SEED-Think-14B/blob/main/LICENSE) (custom Naver license, not Apache 2.0).

---

### Open source status

- [x] The model implementation is available
- [x] The model weights are available

---

### Provide useful links for the implementation

- **Model on Hub**: https://huggingface.co/naver-hyperclovax/HyperCLOVAX-SEED-Think-14B
- **Technical report**: https://arxiv.org/abs/2506.22403

---

## 작성 배경 및 주의사항

### 이슈 구조 근거

GitHub의 `.github/ISSUE_TEMPLATE/new-model-addition.yml` 양식을 따랐습니다 (3개 필드: Model description / Open source status / Links).

### 승인 가능성을 높이는 요소 (과거 사례 분석)

| 요소 | 이 이슈의 상황 |
|------|----------------|
| 아키텍처 차별점 명시 | ✅ Peri-LN + μP, LLaMA와 구분 |
| 구현·가중치 모두 공개 | ✅ Hub에서 이용 가능 |
| 논문 링크 | ✅ arXiv 2506.22403 |
| 구현 준비 완료 | ✅ Draft PR 링크 첨부 |
| 타 프레임워크 지원 확인 | ✅ vLLM upstream merged #37107 (inline 링크) |
| 라이선스 선제 공개 | ✅ 커스텀 라이선스 명시 (비교 사례 제거) |

### 유사 모델 비교

- **Granite (IBM)** — LLaMA + μP attention multiplier, [#31502](https://github.com/huggingface/transformers/pull/31502)로 merged. HyperCLOVAX는 여기에 Peri-LN이 추가된 구조.
- **Qwen2.5 / Qwen3** — 한국어 14B 규모 비교 모델. HyperCLOVAX는 한국어 벤치마크에서 우위.

### 거절 패턴과 이 이슈의 대응

| 과거 거절 사유 | 대응 |
|---------------|------|
| "remote code로 충분하다" | Peri-LN + μP는 기존 LLaMA 코드로 처리 불가, 별도 클래스 필요 |
| 아키텍처 차이가 미미 | μP (Granite과 동일 방식) + Peri-LN (신규) 명시 |
| 가중치 미공개 | Hub에서 공개, monthly downloads 25k+ |
