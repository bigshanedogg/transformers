---
name: HyperCLOVAX CI Analysis
description: CI failure analysis for feat/hyperclovax branch - distinguishes maskformer regression from hyperclovax-specific issues
type: project
---

CI 분석 결과 (2026-05-06):

**Why:** HyperCLOVAX PR 준비 과정에서 CI 실패 원인 파악 필요

**결론**: HyperCLOVAX 테스트는 모두 통과 (151 passed, 120 skipped, 553 subtests). CI 실패는 HyperCLOVAX 코드가 아닌 working tree의 maskformer regression이 원인.

## CI 실패 원인

### 1. [차단됨] MaskFormerMaskFormerDetrConfig ImportError
- **파일**: `src/transformers/models/maskformer/modeling_maskformer.py`
- **원인**: working tree(unstaged)에서 `MaskFormerMaskFormerDetrConfig`로 revert, HEAD(committed)는 `MaskFormerDetrConfig`가 정확
- **출처**: `make fix-repo` 또는 modular 생성 스크립트가 잘못된 class name으로 파일을 재생성함
- **HyperCLOVAX 책임?**: NO - hyperclovax 변경과 무관
- **main 브랜치 책임?**: NO - local main(`a609966c06`)은 maskformer modular conversion이 없음
- **수정**: `git checkout HEAD -- src/transformers/models/maskformer/modeling_maskformer.py`로 복원함

### 2. Working Tree 광범위 unstaged 변경 (65개 파일)
- `make fix-repo` 또는 `make style` 실행 부산물로 추정
- 주요 regression: `auto_mappings.py`에서 `("detr", "DetrConfig")` → `("detr", "MaskFormerDetrConfig")`로 잘못 변경
- 이 변경들은 commit되지 않았음

### 3. Upstream 커밋의 Known Issue
- 커밋 `09832b2ae5 Dynamic weight conversion is recursive (#44300)` 커밋 메시지에 "i'll need to fix maskformer later" 주석 있음
- 이 브랜치에 upstream HF commits(#41250, #44300, #44803, #44953)이 포함됨

## HyperCLOVAX 테스트 결과
- test_model: PASS
- test_config: PASS  
- test_mup_attention_scaling: PASS
- test_mup_logits_scaling: PASS
- test_post_norm: PASS
- test_tp_forward/backward/generation: PASS
- 전체: 151 passed, 120 skipped (GPU/slow 테스트), 0 failed

**How to apply:** PR 제출 전에 working tree의 unstaged 변경 중 의도된 것만 stage하고, maskformer 관련 regression은 commit하지 말 것.
