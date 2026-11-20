import argparse
import json
import time
import unsloth  # noqa: F401
import torch
from inference import SYSTEM_PROMPT, load_model

REQUIRED_KEYS = [
    "is_transaction",
    "type",
    "amount",
    "merchant",
    "payment_mode",
    "account_last4",
    "date",
]


def normalize_value(key: str, val):
    if val is None:
        return None
    if key == "amount":
        try:
            return round(float(val), 2)
        except (ValueError, TypeError):
            return val
    if isinstance(val, str):
        return val.strip().lower()
    return val


def evaluate(model, tokenizer, golden_path: str, show_mistakes: bool = True):
    with open(golden_path, "r", encoding="utf-8") as f:
        golden_data = [json.loads(line) for line in f if line.strip()]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    preds = []
    t0 = time.time()

    for item in golden_data:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": item["sms"]},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=160,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        preds.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())

    avg_ms = (time.time() - t0) / len(golden_data) * 1000
    total = len(golden_data)
    valid_json, schema_match, exact_match = 0, 0, 0
    field_correct = {k: 0 for k in REQUIRED_KEYS}

    for i, (item, raw_pred) in enumerate(zip(golden_data, preds)):
        exp = item["expected"]
        try:
            pred = json.loads(raw_pred)
            if not isinstance(pred, dict):
                continue
            valid_json += 1
        except json.JSONDecodeError:
            if show_mistakes:
                print(f"Mistake #{i + 1}: Invalid JSON -> {raw_pred}")
            continue

        if set(pred.keys()) == set(REQUIRED_KEYS):
            schema_match += 1

        diffs = {}
        for k in REQUIRED_KEYS:
            if normalize_value(k, exp.get(k)) == normalize_value(k, pred.get(k)):
                field_correct[k] += 1
            else:
                diffs[k] = (exp.get(k), pred.get(k))

        if not diffs:
            exact_match += 1
        elif show_mistakes:
            print(f"Mistake #{i + 1} | SMS: {item['sms'][:80]}... | Diffs: {diffs}")

    print("\n" + "=" * 48)
    print(f"EVALUATION SCORECARD ({total} examples)")
    print("=" * 48)
    print(f"Strict JSON Validity : {valid_json}/{total} ({valid_json / total * 100:.1f}%)")
    print(f"Exact Schema Match   : {schema_match}/{total} ({schema_match / total * 100:.1f}%)")
    print(f"100% Exact Match     : {exact_match}/{total} ({exact_match / total * 100:.1f}%)")
    print("-" * 48)
    for k in REQUIRED_KEYS:
        print(f"  {k:<15}: {field_correct[k]:>2}/{total} ({field_correct[k] / total * 100:.1f}%)")
    print("=" * 48)
    print(f"Avg inference time: {avg_ms:.0f} ms / SMS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a Qwen 2.5 3B LoRA adapter on golden_test.jsonl."
    )
    parser.add_argument("--adapter", type=str, required=True, help="Adapter path or HF repo ID")
    parser.add_argument("--golden", type=str, default="data/golden_test.jsonl", help="Path to golden_test.jsonl")
    args = parser.parse_args()

    model, tokenizer = load_model(args.adapter)
    evaluate(model, tokenizer, args.golden)
