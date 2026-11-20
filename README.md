# FinTech SMS-to-JSON Parser via Knowledge Distillation (Qwen 2.5 3B)

A local, privacy-preserving Indian banking SMS parser distilled from **Gemini-3.5-flash-lite** into a 4-bit **Qwen 2.5 3B Instruct** student model using **QLoRA** and **Unsloth** on a single free Tesla T4 GPU.

Through iterative error analysis, deterministic data cleaning, and response-only loss masking, **100% Exact Match accuracy** on a held-out 30-example Golden Test Set improved from **16.7% (Base)** to **90.0% (Distilled v3)** while training only **0.96% of model parameters** (29.9M / 3.11B).

---

## Problem Statement

Sending raw banking and UPI SMS alerts to external cloud LLM APIs creates user privacy risks, recurring API costs, and network latency. Conversely, compact open-weight models (3B parameters) struggle out-of-the-box with Indian banking formats (SBI, HDFC, ICICI, Axis, Kotak), masked account numbers (`X4821`, `XX3310`), date standardisation, and rejecting non-transaction traps (OTPs, failed transactions, balance alerts, reminders).

This project solves that using **Sequence-Level Knowledge Distillation**: generating structured training data from a frontier teacher model, cleaning teacher-induced noise, and fine-tuning a small local student model.

---

## Target JSON Schema (7 Fields)

Every SMS is parsed into strict JSON with exactly 7 keys:
`is_transaction`, `type`, `amount`, `merchant`, `payment_mode`, `account_last4`, `date`.

For non-transactions (OTPs, promos, failed alerts, balance checks), the model outputs `"is_transaction": false` and `null` for the remaining 6 fields.

**Example Input:**
> `Dear UPI user A/C X4821 debited by 245.50 on date 18Nov26 trf to ZOMATO Refno 632189451234. If not u? call 1800111109 -SBI`

**Example Output:**
```json
{
  "is_transaction": true,
  "type": "debit",
  "amount": 245.5,
  "merchant": "ZOMATO",
  "payment_mode": "UPI",
  "account_last4": "4821",
  "date": "2026-11-18"
}
```

---

## Benchmark Progression (30-Example Golden Test Set)

All evaluations were run with greedy decoding (`do_sample=False`) and verified zero train/test leakage.

| Metric | Base Qwen 2.5 3B | v1 (Raw Data) | v2 (Cleaned Data) | v3 (Loss-Masked) | Total Gain |
|---|---|---|---|---|---|
| **100% Exact Match** | 16.7% (5/30) | 60.0% (18/30) | 73.3% (22/30) | **90.0% (27/30)** | **+73.3%** |
| **Strict JSON Validity** | 100.0% | 100.0% | 100.0% | **100.0%** | — |
| **`is_transaction`** | 93.3% | 100.0% | 96.7% | **100.0%** | +6.7% |
| **`type`** | 60.0% | 100.0% | 96.7% | **100.0%** | +40.0% |
| **`amount`** | 93.3% | 100.0% | 96.7% | **100.0%** | +6.7% |
| **`account_last4`** | 43.3% | 80.0% | 96.7% | **100.0%** | +56.7% |
| **`date`** | 26.7% | 100.0% | 93.3% | **100.0%** | +73.3% |
| **`payment_mode`** | 63.3% | 76.7% | 86.7% | **96.7%** | +33.4% |
| **`merchant`** | 70.0% | 93.3% | 86.7% | **93.3%** | +23.3% |

---

## Key Engineering Iterations

1. **Synthetic Data Generation (444 Examples):** Prompted `gemini-3.5-flash-lite` across 5 Indian banking scenarios (UPI, Cards, NetBanking/IMPS, ATM withdrawals, and non-transaction traps) yielding a balanced dataset (236 transactions / 208 traps) with 0 test-set leaks.
2. **Error-Driven Data Cleaning (`v1 -> v2`, +13.3%):** Error analysis on v1 revealed 10 of 12 failures came from Teacher-induced noise (`X` prefixes left in `account_last4` and inconsistent labels like `UPI/P2M`, `Credit Card`, `IMPS`). Deterministically normalising `account_last4` to 4 digits and enforcing a strict 4-value `payment_mode` enum (`UPI`, `Card`, `NetBanking`, `ATM`) via `clean_data.py` lifted `account_last4` from 80.0% to 96.7%.
3. **Response-Only Loss Masking (`v2 -> v3`, +16.7%):** Applied Unsloth's `train_on_responses_only` to mask prompt and SMS tokens (`-100`) so 100% of gradient updates focused on the 7-field JSON output, paired with `packing=False`, cosine LR scheduling, and 3 epochs.

---

## Known Limitations & Error Analysis

Across the 3 remaining mismatches in v3 (out of 30) and live unseen SMS tests:
* **Implicit Payment Mode:** When an SMS omits explicit keywords (`UPI`, `IMPS`, `Card`)—for example, *"Sent Rs.120.00 From HDFC Bank A/C *9012 To SWIGGY INSTAMART"*—the model can default to `"Card"` instead of inferring `"UPI"`.
* **ATM Merchant Naming:** On ATM cash withdrawals, the model occasionally extracts the physical location (`VISHNUPURI NANDED`) instead of the bank ATM label (`SBI ATM`).

---

## Training Setup

* **Base Model:** `unsloth/Qwen2.5-3B-Instruct-bnb-4bit` (4-bit NF4)
* **LoRA Config:** `r=16`, `lora_alpha=16`, `lora_dropout=0`, all linear layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`)
* **Trainable Parameters:** 29,933,568 / 3,115,872,256 (0.96%)
* **Optimiser & Schedule:** `adamw_8bit`, `lr=2e-4`, cosine LR scheduler, effective batch size 16, 3 epochs, response-only loss masking
* **Hardware:** Single NVIDIA Tesla T4 (15 GB VRAM, Kaggle)

---

## Repository Structure

```text
fintech-sms-distillation-qwen3b/
├── data/
│   └── golden_test.jsonl          # 30 hand-verified Indian banking SMS test cases
├── notebooks/
│   ├── 01_teacher_data_gen.ipynb  # Gemini-3.5-flash-lite data generation
│   └── 02_qlora_distill_eval.ipynb # QLoRA training (Base -> v1 -> v2 -> v3)
├── clean_data.py                  # Deterministic Teacher-noise cleaning & chat formatting
├── evaluate.py                    # Reproducible Golden Test Set evaluator
├── inference.py                   # Standalone CLI & Python parser for single SMS
├── upload_to_hf.py                # Helper script to host LoRA weights on Hugging Face Hub
├── requirements.txt
└── README.md
```

---

## Model Weights (v3 LoRA Adapter)

Download `sms_lora_adapter_v3` from:
* **GitHub Releases / Hugging Face Hub:** `https://github.com/PradhyumnaAI/fintech-sms-distillation-qwen3b/releases/tag/v3.0`

---

## Quickstart

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Single-SMS Inference
```bash
python inference.py \
  --adapter path/to/sms_lora_adapter_v3 \
  --sms "A/C X4821 debited by Rs 349.00 on 20Nov26 transfer to JIO PREPAID Ref No 632411223344."
```

### 3. Reproduce the Golden Test Set Benchmark
```bash
python evaluate.py \
  --adapter path/to/sms_lora_adapter_v3 \
  --golden data/golden_test.jsonl
```
