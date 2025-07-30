from huggingface_hub import create_repo, upload_folder

# 모델 이름 (Hugging Face에 생성될 경로: https://huggingface.co/your-username/LLaDA-sft-s1k-merged)
repo_id = "Gredora/LLaDA-sft-s1k-merged"

# (1) 모델 저장소 생성
create_repo(repo_id, private=False)  # 공개하려면 private=False

# (2) 폴더 업로드
upload_folder(
    folder_path="LLaDA-sft-s1k-merged",
    repo_id=repo_id,
    repo_type="model"
)