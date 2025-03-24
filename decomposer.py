import torch
import sys
import pickle
sys.path.append('../')
import os
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader
import transformers
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
)
from datasets import load_dataset
from preprocess import get_combination
from preprocess import get_bookcorpus
import argparse
from tqdm import tqdm
from layers import ModuleInjection, DecomposeLinearSVD, DecomposeLinearEigen
from lm_eval import evaluator
from preprocess import *
import json
from dataset_ppl import get_wikitext2


########################################
# 1) 인자 설정
########################################
parser = argparse.ArgumentParser("Combined Decomposition and Evaluation")
# [분해 관련 인자]
parser.add_argument("--layers", type=str, default="o_proj,q_proj,v_proj,k_proj,gate_proj,up_proj,down_proj",
                    help="분해할 레이어 이름(콤마로 구분)")
parser.add_argument("--dataset", type=str, default="piqa", help="데이터셋 이름")
parser.add_argument("--batch_size", type=int, default=512, help="배치 사이즈")
parser.add_argument("--seq_len", type=int, default=128, help="시퀀스 길이")
parser.add_argument("--log_path", type=str, default="surgical_logs.txt", help="로그 파일 경로")
parser.add_argument("--algo", type=str, default="eigen", help="분해 알고리즘")
parser.add_argument("--weights_name", type=str, default="decomposed_model_mistral_combination.pt",
                    help="분해된 모델 저장 파일명")
parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf", help="모델 이름")
parser.add_argument("--layer_cache_dir", type=str, default="./cache_dir", help="레이어 분해 결과 캐시 디렉토리")
parser.add_argument("--cache_dir", type=str, default="../cache_dir", help="모델 캐시 디렉토리")

# [평가/압축 관련 인자]
parser.add_argument("--delta", type=float, default=0.0, help="허용 정확도 감소량")
parser.add_argument("--start_layer", type=int, default=28, help="압축 시작할 레이어 인덱스")
# 평가 시 사용할 베이스 모델 파일 (분해된 모델로 저장한 파일)
parser.add_argument("--base_model", type=str, default="decomposed_model_mistral_combination.pt",
                    help="평가용 베이스 모델 파일")

args = parser.parse_args()

with open(args.log_path, "a") as file:
    file.write(json.dumps(str(args)))
    file.write("\n")

print(f"[INFO] Using model: {args.model}")


########################################
# 1. 모델 분해 (Decomposition) 단계
########################################
# 원본 모델 로드 (분해 전)
base_model = AutoModelForCausalLM.from_pretrained(
    args.model,
    torch_dtype=torch.float32,
    device_map="auto",
    trust_remote_code=True,
    cache_dir=args.cache_dir,
    # load_in_8bit=True,  # 메모리 절감 필요 시 활성화
)
print("[INFO] Base model loaded.")

# 분해 가능한 레이어 수집
decomposable_layers_base = []
max_rank = []
layer_names = [x.strip() for x in args.layers.split(',')]
for name, module in base_model.named_modules():
    if isinstance(module, nn.Linear):
        # 최대 랭크 계산 (일반적인 근사치)
        max_rank_val = int(module.weight.data.shape[0] * module.weight.data.shape[1] / (module.weight.data.shape[0] + module.weight.data.shape[1]))
        max_rank.append(max_rank_val)
        for eligible_layer in layer_names:
            if eligible_layer in name:
                tokens = name.strip().split(".")
                layer = base_model
                for t in tokens[:-1]:
                    if not t.isnumeric():
                        layer = getattr(layer, t)
                    else:
                        layer = layer[int(t)]
                decomposable_layers_base.append([layer, tokens[-1]])
                break

print(f"[INFO] Found {len(decomposable_layers_base)} decomposable linear layers.")


########################################
# 4) 토크나이저 & data_collator
########################################
tokenizer = AutoTokenizer.from_pretrained(
    args.model,
    trust_remote_code=True,
    torch_dtype="auto",
)
tokenizer.pad_token = tokenizer.eos_token

data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    pad_to_multiple_of=8,
    return_tensors="pt",
    padding=True
)


########################################
# 2. 토크나이저 및 데이터 준비
########################################
tokenizer = AutoTokenizer.from_pretrained(
    args.model,
    trust_remote_code=True,
    torch_dtype="auto",
)
tokenizer.pad_token = tokenizer.eos_token

data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    pad_to_multiple_of=8,
    return_tensors="pt",
    padding=True
)

def tokenize(prompt, add_eos_token=True):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=args.seq_len,
        padding='max_length',
        return_tensors=None,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < 2048
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)
    result["labels"] = result["input_ids"].copy()
    return result

def generate_and_tokenize_prompt(data_point):
    full_prompt = data_point.get("text", "")
    return tokenize(full_prompt)

# 데이터셋 로드 (데이터셋 이름에 따라 분기)
if args.dataset == 'wikitext2':
    dataset = get_wikitext2(tokenizer, seq_len=args.seq_len)
    dataloader = DataLoader(dataset, batch_size=args.batch_size)
elif args.dataset == 'combination':
    dataset, _, _ = get_combination(args.batch_size)
    dataset = dataset.map(generate_and_tokenize_prompt)
    dataset = dataset.select_columns(["input_ids", "attention_mask", "labels"])
    dataloader = DataLoader(dataset, collate_fn=data_collator, batch_size=args.batch_size)
elif args.dataset == 'bookcorp':
    data = get_bookcorpus(tokenizer, args.batch_size, args.seq_len)
else:
    # get_dataset 함수는 preprocess 모듈에 있다고 가정
    dataset, _, _ = get_dataset(args.dataset)
    dataset = dataset.map(generate_and_tokenize_prompt)
    dataset = dataset.select_columns(["input_ids", "attention_mask", "labels"])
    print(f"Dataset columns after select: {dataset.column_names}")
    dataloader = DataLoader(dataset, collate_fn=data_collator, batch_size=args.batch_size)

print("[INFO] Done Loading Data.")

########################################
# 3. 분해 레이어 주입 (Injection)
########################################
for index in tqdm(range(len(decomposable_layers_base)), desc="Injecting decomposable layers"):
    parent_layer, last_token = decomposable_layers_base[index]
    setattr(
        parent_layer,
        last_token,
        ModuleInjection.make_decomposable(
            getattr(parent_layer, last_token), max_rank[index], args.algo
        ),
    )
    # 모든 파라미터 학습 비활성화
    for _, param in base_model.named_parameters():
        param.requires_grad = False

print("[INFO] Finished injecting decomposable layers.")

########################################
# 4. 모델 정상 동작 확인 (한 번 forward 수행)
########################################
if args.dataset == 'wikitext2':
    for inputs in tqdm(dataloader, desc="Forward on wikitext2"):
        for k in list(inputs.keys()):
            if inputs[k] is None:
                del inputs[k]
        inputs = {k: inputs[k].to(base_model.device) for k in inputs}
        _ = base_model(**inputs)
        break
elif args.dataset == 'combination':
    for i, inputs in enumerate(tqdm(dataloader, desc="Forwarding combination dataset", total=len(dataloader))):
        for k in list(inputs.keys()):
            if inputs[k] is None:
                del inputs[k]
        inputs = {k: v.to(base_model.device) for k, v in inputs.items()}
        _ = base_model(**inputs)
elif args.dataset == 'bookcorp':
    data = data.to(base_model.device)
    _ = base_model(data)
else:
    for i, inputs in enumerate(tqdm(dataloader, desc=f"Forward on {args.dataset}", total=len(dataloader))):
        for k in list(inputs.keys()):
            if inputs[k] is None:
                del inputs[k]
        inputs = {k: inputs[k].to(base_model.device) for k in inputs}
        _ = base_model(**inputs)
        break

########################################
# 6. 평가 및 추가 압축 단계 (Evaluation & Pruning)
########################################
# 평가를 위해 분해된 베이스 모델 로드
eval_base_model = torch.load(args.base_model)
tokenizer = AutoTokenizer.from_pretrained(args.model)
print("[INFO] Loaded base model for evaluation.")
print(eval_base_model)


########################################
# 8) 한 번 forward 수행 (or 전체 forward)
########################################
if args.dataset == 'wikitext2':
    for inputs in tqdm(dataloader, desc="Forward on wikitext2"):
        for k in list(inputs.keys()):
            if inputs[k] is None:
                del inputs[k]
        inputs = {k: inputs[k].to(base_model.device) for k in inputs}
        _ = base_model(**inputs)
        break

elif args.dataset == 'combination':
    for i, inputs in enumerate(tqdm(dataloader, desc="Forwarding combination dataset", total=len(dataloader))):
        for k in list(inputs.keys()):
            if inputs[k] is None:
                del inputs[k]
        inputs = {k: v.to(base_model.device) for k, v in inputs.items()}
        _ = base_model(**inputs)
        # break

elif args.dataset == 'bookcorp':
    data = data.to(base_model.device)
    _ = base_model(data)

else:
    for i, inputs in enumerate(tqdm(dataloader, desc=f"Forward on {args.dataset}", total=len(dataloader))):
        for k in list(inputs.keys()):
            if inputs[k] is None:
                del inputs[k]
        inputs = {k: inputs[k].to(base_model.device) for k in inputs}
        _ = base_model(**inputs)
        break

########################################
# 9) 모델 저장 (전체 모델)
########################################
save_path = args.weights_name
# for module in base_model.modules():
#     if hasattr(module, "_forward_hooks"):
#         module._forward_hooks.clear()
base_model.half()
base_model.save_pretrained(save_path)
torch.save(base_model, save_path)
# print(f"[INFO] Saved decomposed model => {save_path}")
# torch.save(base_model.state_dict(), save_path)
