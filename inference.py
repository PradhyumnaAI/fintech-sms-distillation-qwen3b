import argparse
import json
import time
import unsloth  # noqa: F401 - must be imported before transformers/trl/peft
import torch
from unsloth import FastLanguageModel

SYSTEM_PROMPT = (
    "You are a financial SMS parser. Extract transaction details into strict JSON with exactly 7 keys: "
    "is_transaction, type, amount, merchant, payment_mode, account_last4, date. "
    "Rules: payment_mode must be ONLY one of 'UPI', 'Card', 'NetBanking', 'ATM', or null. "
    "account_last4 must be 4 digits only (no X or *). "
    "If not a completed money movement, set is_transaction to false and all other 6 fields to null. "
    "Output ONLY valid JSON, no extra text."
)


def load_model(adapter_path: str, max_seq_length: int = 512):
    """Loads the 4-bit Qwen 2.5 3B student model with the trained v3 LoRA adapter."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_path,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def parse_sms(model, tokenizer, sms_text: str, verbose: bool = False) -> dict:
    """Parses a raw banking SMS string into a 7-field dictionary."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": sms_text.strip()},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=160,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    latency_ms = (time.time() - t0) * 1000

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    raw_output = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    if verbose:
        print(f"Latency: {latency_ms:.0f} ms")

    try:
        return json.loads(raw_output)
    except json.JSONDecodeError:
        return {"error": "Invalid JSON", "raw_output": raw_output}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse Indian banking SMS alerts into strict 7-field JSON."
    )
    parser.add_argument(
        "--adapter",
        type=str,
        required=True,
        help="Local path or Hugging Face repo ID for sms_lora_adapter_v3",
    )
    parser.add_argument(
        "--sms",
        type=str,
        required=True,
        help="Raw banking SMS text to parse",
    )
    args = parser.parse_args()

    model, tokenizer = load_model(args.adapter)
    result = parse_sms(model, tokenizer, args.sms, verbose=True)
    print(json.dumps(result, indent=2))
