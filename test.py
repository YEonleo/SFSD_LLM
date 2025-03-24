import torch
import sys
import os
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

# (필요하다면) 커스텀 레이어(Injection) 클래스를 import
# from layers import ModuleInjection, DecomposeLinearSVD, DecomposeLinearEigen

def main():
    ################################################################
    # 1) 원본 모델 로드
    #    - device_map="auto" + FP16
    #    - 여러 GPU에서 자동 분산(shard)
    ################################################################
    model_name = "meta-llama/Llama-2-7b-hf"
    base_model = torch.load("decompose_llama2_cnndm.pt")
    base_model.to("cuda")
    
    base_model.eval()
    total_params = sum(p.numel() for p in base_model.parameters())
    print(f"[INFO] Total number of parameters: {total_params:,}")
    
    print("[INFO] Model loaded")

    ################################################################
    # 5) 토크나이저 로드
    ################################################################
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ################################################################
    # 6) 예시 입력(prompt) & 생성
    ################################################################
    input_text = (
        "Article: LONDON, England (Reuters) -- Harry Potter star Daniel Radcliffe gains access to a reported £20 million ($41.1 million) fortune as he turns 18 on Monday, but he insists the money won't cast a spell on him. "
        "Daniel Radcliffe as Harry Potter in 'Harry Potter and the Order of the Phoenix' To the disappointment of gossip columnists around the world, the young actor says he has no plans to fritter his cash away on fast cars, drink and celebrity parties. "
        "\"I don't plan to be one of those people who, as soon as they turn 18, suddenly buy themselves a massive sports car collection or something similar,\" he told an Australian interviewer earlier this month. "
        "\"I don't think I'll be particularly extravagant. The things I like buying are things that cost about 10 pounds -- books and CDs and DVDs.\" "
        "At 18, Radcliffe will be able to gamble in a casino, buy a drink in a pub or see the horror film 'Hostel: Part II,' currently six places below his number one movie on the UK box office chart. "
        "Details of how he'll mark his landmark birthday are under wraps. His agent and publicist had no comment on his plans. "
        "\"I'll definitely have some sort of party,\" he said in an interview. \"Hopefully none of you will be reading about it.\" "
        "Radcliffe's earnings from the first five Potter films have been held in a trust fund which he has not been able to touch. "
        "Despite his growing fame and riches, the actor says he is keeping his feet firmly on the ground. "
        "\"People are always looking to say 'kid star goes off the rails,'\" he told reporters last month. \"But I try very hard not to go that way because it would be too easy for them.\" "
        "His latest outing as the boy wizard in 'Harry Potter and the Order of the Phoenix' is breaking records on both sides of the Atlantic and he will reprise the role in the last two films. "
        "There is life beyond Potter, however. The Londoner has filmed a TV movie called 'My Boy Jack,' about author Rudyard Kipling and his son, due for release later this year. "
        "He will also appear in 'December Boys,' an Australian film about four boys who escape an orphanage. "
        "Earlier this year, he made his stage debut playing a tortured teenager in Peter Shaffer's 'Equus.' "
        "Meanwhile, he is braced for even closer media scrutiny now that he's legally an adult: \"I just think I'm going to be more sort of fair game,\" he told Reuters. "
        "E-mail to a friend. Copyright 2007 Reuters. All rights reserved. This material may not be published, broadcast, rewritten, or redistributed.\n"
        "Q: Summarize the above article briefly in 2-3 sentences.\nA:"
    )


    print("[INFO] Encoding input text")
    # 아래에서 to("cuda") -> GPU로 텐서 이동
    # device_map="auto" 상태이므로, 사실상 첫 번째 GPU에서 forward가 시작됨
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to("cuda:0")

    print("[INFO] Generating text...")
    with torch.no_grad():
        output_ids = base_model.generate(
            input_ids,
            max_new_tokens=128,
            do_sample=True,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.1,
        )

    ################################################################
    # 7) 결과 디코딩 & 출력
    ################################################################
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print("=== Generated text ===")
    print(generated_text)
    print("======================")

if __name__ == "__main__":
    main()
