import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse
import os

def merge_lora_adapter(base_model_path: str, adapter_path: str, output_path: str):
    """
    SFT로 학습된 LoRA 어댑터를 베이스 모델과 병합하고 새로운 디렉토리에 저장합니다.

    Args:
        base_model_path (str): 원본 베이스 모델의 경로 또는 Hugging Face Hub 이름.
                               (예: 'meta-llama/Llama-2-7b-hf')
        adapter_path (str): SFT로 학습된 LoRA 어댑터 파일들이 있는 디렉토리 경로.
        output_path (str): 병합된 모델을 저장할 새로운 디렉토리 경로.
                           (예: 'LLaDA-sft-s1k-merged')
    """
    print(f"Loading base model from: {base_model_path}")
    
    # Hugging Face 토큰이 필요한 경우, 미리 로그인해두세요.
    # 터미널에서 `huggingface-cli login` 실행
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,  # 원본 모델 로드 시 사용한 dtype과 맞추는 것이 좋습니다.
        device_map="auto", # GPU 메모리가 충분하다면 'auto' 또는 'cuda'로 설정
        trust_remote_code=True,
    )
    
    # 베이스 모델의 토크나이저도 함께 로드해야 합니다.
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True,)

    print(f"Loading LoRA adapter from: {adapter_path}")
    # PeftModel을 사용해 베이스 모델 위에 어댑터를 로드합니다.
    model_with_adapter = PeftModel.from_pretrained(base_model, adapter_path)

    print("Merging adapter weights with the base model...")
    # .merge_and_unload()를 호출하여 가중치를 병합합니다.
    # 이 함수는 어댑터를 병합한 새로운 모델 객체를 반환합니다.
    merged_model = model_with_adapter.merge_and_unload()
    print("Merge complete.")

    # 출력 디렉토리가 없으면 생성
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"Created output directory: {output_path}")

    print(f"Saving merged model to: {output_path}")
    # 병합된 모델과 토크나이저를 지정된 경로에 저장합니다.
    # 이 디렉토리는 이제 하나의 완전한 모델로 기능합니다.
    merged_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    print("\nProcess finished successfully!")
    print(f"The merged model is ready to be used from the directory: '{output_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter with a base model.")
    parser.add_argument(
        "--base_model_path", 
        type=str, 
        required=True, 
        help="Path or Hugging Face Hub name of the base model (e.g., 'meta-llama/Llama-2-7b-hf')."
    )
    parser.add_argument(
        "--adapter_path", 
        type=str, 
        required=True, 
        help="Path to the directory containing the SFT-trained LoRA adapter files."
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        required=True, 
        help="Path to the new directory where the merged model will be saved."
    )

    args = parser.parse_args()

    merge_lora_adapter(
        base_model_path=args.base_model_path,
        adapter_path=args.adapter_path,
        output_path=args.output_path
    )
# python merge_adapter.py --base_model_path "GSAI-ML/LLaDA-8B-Instruct" --adapter_path "/home/jyjang/d1/SFT/sft_outputs/llada-s1/checkpoint-2460/" --output_path "/home/jyjang/d1/LLaDA-sft-s1k-merged"