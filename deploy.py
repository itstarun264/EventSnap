import os
from huggingface_hub import HfApi

def deploy():
    token = os.environ.get('HF_TOKEN')
    if not token:
        token = input("Enter your Hugging Face Write Token: ").strip()
    if not token:
        print("Token is required.")
        return
        
    repo_id = "tarunrathod2643/EventSnap"
    print(f"Uploading files to Hugging Face Space: {repo_id}...")
    
    try:
        api = HfApi(token=token)
        api.upload_folder(
            folder_path=".",
            repo_id=repo_id,
            repo_type="space",
            ignore_patterns=[
                ".git*",
                "venv*",
                "__pycache__*",
                "*.db",
                "instance*",
                "deploy.py"
            ]
        )
        print("\n✅ Success! Your application files have been uploaded to Hugging Face.")
        print(f"🔗 View deployment progress here: https://huggingface.co/spaces/{repo_id}")
    except Exception as e:
        print(f"\n❌ Deployment failed: {e}")

if __name__ == "__main__":
    deploy()
