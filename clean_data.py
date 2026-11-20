import argparse
import json
import random
import re
from collections import Counter

SYSTEM_PROMPT_V2 = (
    "You are a financial SMS parser. Extract transaction details into strict JSON with exactly 7 keys: "
    "is_transaction, type, amount, merchant, payment_mode, account_last4, date. "
    "Rules: payment_mode must be ONLY one of 'UPI', 'Card', 'NetBanking', 'ATM', or null. "
    "account_last4 must be 4 digits only (no X or *). "
    "If not a completed money movement, set is_transaction to false and all other 6 fields to null. "
    "Output ONLY valid JSON, no extra text."
)


def clean_payment_mode(mode, sms: str):
    if mode is None:
        return None
    m = str(mode).strip().lower()
    sms_l = sms.lower()
    if "atm" in m or "cash withdrawal" in sms_l or "withdrawn from" in sms_l:
        return "ATM"
    if "upi" in m:
        return "UPI"
    if "card" in m:
        return "Card"
    if any(k in m for k in ["net", "imps", "neft", "rtgs", "sip", "emi", "auto"]):
        return "NetBanking"
    return mode


def clean_account_last4(val):
    if val is None:
        return None
    digits = re.sub(r"\D", "", str(val))
    return digits[-4:] if len(digits) >= 4 else (digits or None)


def build_clean_chat_dataset(raw_train_path: str, golden_path: str, output_path: str, seed: int = 42):
    with open(golden_path, "r", encoding="utf-8") as f:
        golden_sms = {json.loads(line)["sms"].strip().lower() for line in f if line.strip()}

    with open(raw_train_path, "r", encoding="utf-8") as f:
        raw_train = [json.loads(line) for line in f if line.strip()]

    cleaned = []
    leaks, fixed_acct, fixed_mode = 0, 0, 0

    for item in raw_train:
        if item["sms"].strip().lower() in golden_sms:
            leaks += 1
            continue
        exp = dict(item["expected"])
        if exp.get("is_transaction"):
            new_acct = clean_account_last4(exp.get("account_last4"))
            if new_acct != exp.get("account_last4"):
                fixed_acct += 1
            exp["account_last4"] = new_acct

            new_mode = clean_payment_mode(exp.get("payment_mode"), item["sms"])
            if new_mode != exp.get("payment_mode"):
                fixed_mode += 1
            exp["payment_mode"] = new_mode

        cleaned.append({"sms": item["sms"], "expected": exp})

    random.seed(seed)
    random.shuffle(cleaned)

    with open(output_path, "w", encoding="utf-8") as f:
        for item in cleaned:
            record = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT_V2},
                    {"role": "user", "content": item["sms"]},
                    {"role": "assistant", "content": json.dumps(item["expected"])},
                ]
            }
            f.write(json.dumps(record) + "\n")

    print(f"Raw examples: {len(raw_train)} | Leaks removed: {leaks} | Clean saved: {len(cleaned)}")
    print(f"Fixed account_last4: {fixed_acct} | Fixed payment_mode: {fixed_mode}")
    print("payment_mode distribution:", dict(Counter(x["expected"]["payment_mode"] for x in cleaned)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean Teacher noise and format 3-role chat JSONL.")
    parser.add_argument("--raw", type=str, required=True, help="Path to train_raw.jsonl")
    parser.add_argument("--golden", type=str, default="data/golden_test.jsonl", help="Path to golden_test.jsonl")
    parser.add_argument("--out", type=str, default="data/train_chat_v2.jsonl", help="Output chat JSONL path")
    args = parser.parse_args()

    build_clean_chat_dataset(args.raw, args.golden, args.out)
