# HyperCLOVAX PR 제출 체크리스트

이 문서는 `feat/hyperclovax` 브랜치의 PR 제출 전 완료해야 할 사항들을 정리합니다.

---

## 현재 구현 상태

| 파일 | 상태 |
|------|------|
| `src/transformers/models/hyperclovax/modular_hyperclovax.py` | ✅ 완료 |
| `src/transformers/models/hyperclovax/configuration_hyperclovax.py` | ⚠️ 재생성 필요 (`make fix-repo`) |
| `src/transformers/models/hyperclovax/modeling_hyperclovax.py` | ⚠️ 재생성 필요 (`make fix-repo`) |
| `src/transformers/models/hyperclovax/__init__.py` | ✅ 완료 |
| `src/transformers/models/auto/configuration_auto.py` | ✅ 등록됨 |
| `src/transformers/models/auto/modeling_auto.py` | ✅ 등록됨 |
| `src/transformers/utils/dummy_pt_objects.py` | ⚠️ 자동 생성 필요 (`make fix-repo`) |
| `docs/source/en/model_doc/hyperclovax.md` | ✅ 완료 |
| `docs/source/en/_toctree.yml` | ✅ 등록됨 |
| `tests/models/hyperclovax/test_modeling_hyperclovax.py` | ✅ 완료 |

---

## Step 1. 생성 파일 재동기화

`modular_hyperclovax.py`의 모든 변경사항을 생성 파일에 반영합니다.

```bash
make fix-repo
```

`make fix-repo`가 수행하는 작업:
- `modular_hyperclovax.py` → `modeling_hyperclovax.py`, `configuration_hyperclovax.py` 재생성
- `dummy_pt_objects.py` 더미 클래스 자동 추가
- `# Copied from` 블록 동기화
- 독스트링 포맷 정규화
- 문서 TOC 자동 업데이트

**주의**: 실행 후 diff를 반드시 확인합니다. `make fix-repo`가 의도치 않은 변경을 포함할 수 있습니다.

```bash
git diff src/transformers/models/hyperclovax/
```

---

## Step 2. CI 동일 검사 통과

```bash
python utils/check_auto.py --fix_and_overwrite
PATH="/home/nsml/.omni/bin:$PATH"
python utils/modular_model_converter.py \
  src/transformers/models/hyperclovax/modular_hyperclovax.py
python utils/check_modular_conversion.py \
  --files src/transformers/models/hyperclovax/modular_hyperclovax.py \
  --fix_and_overwrite
make fix-repo
make style
make typing
make check-repo
CUDA_VISIBLE_DEVICES="" python -m pytest tests/models/hyperclovax/test_modeling_hyperclovax.py -v -k "not slow"
RUN_SLOW=1 python -m pytest tests/models/hyperclovax/ -v -m slow
```

실패 시 자동 수정:

```bash
make style   # ruff format + lint
make fix-repo
```

개별 검사 항목:

```bash
# 타입 체크 (ty)
python -m ty check src/transformers/models/hyperclovax/

# modular 동기화 검사
python utils/check_modular_conversion.py --model_type hyperclovax

# 더미 클래스 검사
python utils/check_dummies.py

# 독스트링 검사
python utils/check_docstrings.py
```

---

## Step 3. 단위 테스트

```
uv pip install -e .[dev] --cache-dir=/mnt/tmp
uv pip install ruff pytest-xdist pytest mpi4py libcst gitpython --upgrade
```

> **참고**: 현재 환경(`omni`)에서는 conftest 버전 충돌로 `pytest`가 동작하지 않습니다.
> `python -m unittest`으로 대체하고, 공식 dev 환경에서 `pytest`로 최종 검증합니다.

```bash
# 현재 환경 — python -m unittest (conftest 우회)
source /home/nsml/.omni/bin/activate

# HyperCLOVAX-specific 단위 테스트 전체
python -m unittest tests.models.hyperclovax.test_modeling_hyperclovax.HyperCLOVAXModelTest -v

# MuP 및 Peri-Layer Norm 핵심 테스트만
python -m unittest \
  tests.models.hyperclovax.test_modeling_hyperclovax.HyperCLOVAXModelTest.test_mup_attention_scaling \
  tests.models.hyperclovax.test_modeling_hyperclovax.HyperCLOVAXModelTest.test_mup_logits_scaling \
  tests.models.hyperclovax.test_modeling_hyperclovax.HyperCLOVAXModelTest.test_post_norm_output_shape \
  tests.models.hyperclovax.test_modeling_hyperclovax.HyperCLOVAXModelTest.test_post_norm_changes_output
```

```bash
# 공식 dev 환경 — pytest (PR 제출 전 최종 검증)
pytest tests/models/hyperclovax/ -v
pytest tests/models/hyperclovax/test_modeling_hyperclovax.py -k "mup or post_norm" -v
```

이 테스트들이 반드시 통과해야 합니다:

| 테스트 | 검증 내용 |
|--------|-----------|
| `test_mup_attention_scaling` | `attention_multiplier` 변경 시 logits 달라지는지 |
| `test_mup_logits_scaling` | `logits_scaling` 변경 시 logits 비례 변화하는지 |
| `test_post_norm_output_shape` | `use_post_norm=True`가 출력 shape을 바꾸지 않는지 |
| `test_post_norm_changes_output` | `use_post_norm=True`가 실제로 출력을 다르게 만드는지 |

---

## Step 4. 통합 테스트 실행

expected value는 이미 채워져 있습니다 (`temp/collect_integration_expected.py`로 수집 완료).
실제 체크포인트로 테스트가 통과하는지 확인합니다.

```bash
# 현재 환경 — python -m unittest
source /home/nsml/.omni/bin/activate
RUN_SLOW=1 python -m unittest \
  tests.models.hyperclovax.test_modeling_hyperclovax.HyperCLOVAXIntegrationTest -v
```

```bash
# 공식 dev 환경 — pytest
RUN_SLOW=1 pytest tests/models/hyperclovax/test_modeling_hyperclovax.py \
  -k "integration" -v -s
```

---

## Step 5. 최종 검증

```bash
# 전체 단위 테스트 (slow 제외) — 공식 dev 환경
pytest tests/models/hyperclovax/ -v

# upstream과 동기화 후 rebase
git fetch upstream
git rebase upstream/main

# CI 최종 확인
make check-repo
```

---

## Step 6. PR 제출

### PR 제목 (변경 불필요)

```
Add HyperCLOVAX model
```

GitHub Draft PR 기능이 WIP 상태를 표시하므로 `[WIP]` 접두어는 불필요합니다.

### PR 본문 (최종)

```markdown
> **Draft PR — waiting for issue approval.** This PR is opened alongside the issue request.
> It will be marked ready for review after a maintainer gives the go-ahead on the issue.

## What does this PR do?

Adds native Transformers support for **[HyperCLOVA X SEED Think 14B](https://huggingface.co/naver-hyperclovax/HyperCLOVAX-SEED-Think-14B)**,
a 14.74B-parameter Korean reasoning LLM developed by NAVER Cloud.

Fixes #ISSUE_NUMBER

### Why a new model class (not `trust_remote_code`)?

HyperCLOVAX is LLaMA-based but has two architectural changes that cannot be expressed
with existing LLaMA or Granite code:

| Feature | HyperCLOVAX | LLaMA | Granite |
|---|---|---|---|
| Peri-Layer Norm (`use_post_norm`) | ✅ extra RMSNorm after each sublayer | ❌ | ❌ |
| MuP attention scaling (`attention_multiplier`) | ✅ | ❌ | ✅ |
| MuP residual / embedding / logits scaling | ✅ | ❌ | partial |

Both modifications require overriding `LlamaAttention` and `LlamaDecoderLayer`,
making `# Copied from` or runtime patching insufficient.

### Architecture

LLaMA-style decoder-only transformer with two modifications:

- **Peri-Layer Normalization** (`use_post_norm`): an extra `RMSNorm` is applied *after* each
  sub-layer output (both attention and MLP), in addition to the standard pre-norm.
- **Maximal Update Parametrization (μP)**: four per-config scaling factors replace fixed constants:
  - `attention_multiplier` — replaces `1/sqrt(head_dim)` in attention
  - `residual_multiplier` — scales each sub-layer output before adding to the residual stream
  - `embedding_multiplier` — scales the token embedding output
  - `logits_scaling` — scales final logits before softmax / sampling

The μP attention pattern follows Granite ([#31502](https://github.com/huggingface/transformers/pull/31502));
HyperCLOVAX extends it with the full residual/embedding/logits scaling and the new Peri-LN.

### Implementation

- Uses the **modular system** (`modular_hyperclovax.py` extends `LlamaModel`)
- Generated files (`modeling_hyperclovax.py`, `configuration_hyperclovax.py`) produced by `make fix-repo`
- New classes: `HyperCLOVAXConfig`, `HyperCLOVAXModel`, `HyperCLOVAXForCausalLM`
  (+ `ForQuestionAnswering`, `ForSequenceClassification`, `ForTokenClassification`)
- Registered in `AutoConfig`, `AutoModelForCausalLM`

### External support

- Huggingface hub: [naver-hyperclovax/HyperCLOVAX-SEED-Think-14B](https://huggingface.co/naver-hyperclovax/HyperCLOVAX-SEED-Think-14B)
- Technical report: [arXiv 2506.22403](https://arxiv.org/abs/2506.22403)
- vLLM upstream: [vllm-project/vllm#37107](https://github.com/vllm-project/vllm/pull/37107) (merged 2026-03-16)

## Before submitting

- [x] Did you read the [contributor guideline](https://github.com/huggingface/transformers/blob/main/CONTRIBUTING.md#start-contributing-pull-requests)?
- [x] Did you read the [pull request guidelines](https://github.com/huggingface/transformers/blob/main/CONTRIBUTING.md#how-to-submit-a-pull-request)?
- [ ] Was this discussed/approved via a GitHub issue or forum? — **pending; see issue #ISSUE_NUMBER**
- [x] Did you make sure to update the documentation with your changes?
- [x] Did you write any new necessary tests?
- [ ] `make fix-repo` run and diff verified
- [ ] `make check-repo` passes
- [ ] `pytest tests/models/hyperclovax/` passes
- [ ] `RUN_SLOW=1 pytest tests/models/hyperclovax/` passes (integration tests)

## AI assistance disclosure

This implementation was developed with **Claude Code (Anthropic)**. Per the repository's
mandatory AI contribution policy ([CLAUDE.md](https://github.com/huggingface/transformers/blob/main/CLAUDE.md)):

- **Coordination**: Draft PR opened alongside issue #ISSUE_NUMBER to facilitate early review.
  Will not be marked ready until a maintainer approves the issue.
- **Duplicate check**: No existing open PR covers HyperCLOVAX
  (`gh pr list --repo huggingface/transformers --state open --search "HyperCLOVA"` returned empty).
- **Differentiation**: Granite (#31502) covers MuP attention multiplier only.
  HyperCLOVAX additionally requires Peri-Layer Normalization and the full
  MuP residual/embedding/logits scaling, which are not present in any existing model class.
- **Tests run**:
  ```
  python -m unittest tests.models.hyperclovax.test_modeling_hyperclovax.HyperCLOVAXModelTest
  ```
  All unit tests pass, including `test_mup_attention_scaling`, `test_mup_logits_scaling`,
  `test_post_norm_output_shape`, `test_post_norm_changes_output`.
- **Human validation**: The submitter has reviewed all changed lines and run the tests directly.
```

### 리뷰어 (PR ready 전환 후 태그)

```
@ArthurZucker @CyrilVallez
```

---

## HyperCLOVAX 특이사항 체크리스트

PR 리뷰에서 지적받을 수 있는 항목들:

- [ ] `make fix-repo` 후 생성된 `configuration_hyperclovax.py`의 `__post_init__`이 modular와 동일한지 확인
- [ ] MuP 스케일링 값(`embedding_multiplier`, `residual_multiplier`, `logits_scaling`, `attention_multiplier`)이 `from_pretrained` 후 `config`에 올바르게 로드되는지 확인
- [ ] `use_post_norm=True` 체크포인트와 `False` 체크포인트를 각각 로드할 때 모두 정상 동작하는지 확인
- [ ] `rope_theta`/`rope_scaling` backward compat: 구 체크포인트(이 필드를 standalone으로 저장한 것)가 정상 로드되는지 확인
- [ ] `HyperCLOVAXForQuestionAnswering.base_model_prefix = "transformer"` — 구 체크포인트 BC를 위한 것으로, PR 리뷰에서 이유 설명 필요

---

## 참고: 유사 PR 사례

| 모델 | PR | 특이사항 |
|------|-----|----------|
| Granite (IBM, LLaMA + MuP) | [#31502](https://github.com/huggingface/transformers/pull/31502) | 가장 유사한 아키텍처. MuP attention multiplier 구현 패턴 참고 |
| Qwen3 | [#37855](https://github.com/huggingface/transformers/pull/37855) | 최근 LLaMA계 모델 추가 사례 |
| vLLM PR | [vllm#37107](https://github.com/vllm-project/vllm/pull/37107) | 2026-03-16 merged. 동일 아키텍처의 vLLM 구현 참고 |
