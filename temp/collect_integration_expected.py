"""
Run this script on a machine with the 14B model loaded to collect
the hardcoded expected values for HyperCLOVAXIntegrationTest.

Usage:
    python temp/collect_integration_expected.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "naver-hyperclovax/HyperCLOVAX-SEED-Think-14B"
CACHE_DIR = "/mnt/ocr-nfsx1/public_datasets/.cache/huggingface/hub"
INPUT_TEXT = ["서울에서 부산까지 기차로 걸리는 시간은 ", "The travel time by train from Seoul to Busan"]
# Short Korean sentence — tokenized at runtime to avoid using wrong (LLaMA-style) placeholder IDs
LOGIT_TEXT = "대한민국의 수도는 서울입니다."

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
LOGIT_INPUT_IDS = tokenizer.encode(LOGIT_TEXT, add_special_tokens=True)

# ── 1. Logits (수치 정밀도 검증용) ────────────────────────────────────────────
print("=" * 60)
print("# 1. LOGITS — paste into test_model_seed_think_14b_logits_bf16")
print("=" * 60)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    dtype=torch.bfloat16, 
    attn_implementation="eager", 
    device_map="auto",
    cache_dir=CACHE_DIR,
)
model.eval()

with torch.no_grad():
    out = model(torch.tensor([LOGIT_INPUT_IDS]).to(model.device))

mean = out.logits.float().mean(-1)
slc = out.logits[0, 0, :15].float()

print(f"# In the test file — test_model_seed_think_14b_logits_bf16:")
print(f'LOGIT_INPUT_IDS = {LOGIT_INPUT_IDS}')
print(f'expected_mean  = torch.tensor({mean.tolist()})')
print(f'expected_slice = torch.tensor({slc.tolist()})')

del model
torch.cuda.empty_cache()

# ── 2. Generated string (end-to-end 검증용) ──────────────────────────────────
print()
print("=" * 60)
print("# 2. GENERATED STRINGS — paste into test_model_seed_think_14b_bf16 / sdpa")
print("=" * 60)

for attn_impl in ("eager", "sdpa"):
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        attn_implementation=attn_impl, 
        device_map="auto",
        cache_dir=CACHE_DIR,
    )
    model.eval()

    inputs = tokenizer(INPUT_TEXT, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    texts = tokenizer.batch_decode(output, skip_special_tokens=False)

    print(f"\n# attn_implementation={attn_impl!r}  →  test_model_seed_think_14b_bf16{'_sdpa' if attn_impl == 'sdpa' else ''}")
    print("# input_text[0] = Korean, input_text[1] = English")
    print("EXPECTED_TEXTS = [")
    for t in texts:
        print(f"    {t!r},")
    print("]")

    del model
    torch.cuda.empty_cache()
