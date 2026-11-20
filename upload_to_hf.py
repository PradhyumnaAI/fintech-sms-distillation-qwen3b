"""
Helper script to upload your unzipped sms_lora_adapter_v3 folder to Hugging Face Hub.
GitHub blocks regular git files over 100 MB (and browser uploads over 25 MB), so hosting
LoRA weights on Hugging Face Hub (or GitHub Releases) is the standard workflow.

Usage:
    pip install huggingface_hub
    python upload_to_hf.py --folder path/to/sms_lora_adapter_v3 --repo your-hf-username/qwen2.5-3b-indian-sms-lora-v3
"""
import argparse
from huggingface_hub import HfApi


def main():
    parser = argparse.ArgumentParser(description="Upload LoRA adapter folder to Hugging Face Hub.")
    parser.add_argument("--folder", type=str, required=True, help="Path to unzipped sms_lora_adapter_v3 folder")
    parser.add_argument("--repo", type=str, required=True, help="Target HF repo ID (username/repo-name)")
    parser.add_argument("--token", type=str, default=None, help="Optional HF write token (or run huggingface-cli login)")
    args = parser.parse_args()

    api = HfApi(token=args.token)
    api.create_repo(repo_id=args.repo, repo_type="model", exist_ok=True)
    api.upload_folder(
        folder_path=args.folder,
        repo_id=args.repo,
        repo_type="model",
    )
    print(f"Uploaded successfully! Model URL: https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
