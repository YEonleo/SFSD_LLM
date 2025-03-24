import os
import sys
import gc
import time
import copy
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
)
from datasets import load_dataset

# 내부 프로젝트 모듈 (환경에 맞게 수정)
from preprocess import *
from dataset_ppl import get_wikitext2, get_loaders_chunk, get_loaders_end
from layers import ModuleInjection, DecomposeLinearSVD, DecomposeLinearEigen
from lm_eval import evaluator
from evaluator_modified import simple_evaluate_chunk, full_evaluate
from lm_eval.models.huggingface import HFLM

# wandb 추가
import wandb

########################################
# 평가 함수들
########################################
def evaluate(model, chunk, limit, chunk_ratio, tokenizer, reduce, args):
    lm = HFLM(pretrained=model, tokenizer=tokenizer, device="auto")
    results = simple_evaluate_chunk(
        model=lm,
        chunk_num=chunk,
        tasks=[args.dataset],
        num_fewshot=0,
        batch_size=4,
        device="auto",
        chunk_ratio=chunk_ratio,
        limit=limit,
        reduce=reduce
    )
    if reduce is not None:
        metric_key = 'llt'
    else:
        if args.dataset in ['xsum', 'cnn_dailymail']:
            metric_key = 'rouge,none'
        elif args.dataset in ['gsm8k']:
            metric_key = 'exact_match,flexible-extract'
        else:
            metric_key = 'acc,none'
    acc = results['results'][args.dataset][metric_key]
    params = sum(param.numel() for _, param in model.named_parameters())
    print(f"Chunk {chunk} acc: {acc}, params: {params}")
    return acc, params

def evaluate_full(model, limit, tokenizer, reduce, args):
    lm = HFLM(pretrained=model, tokenizer=tokenizer, device="auto")
    results = full_evaluate(
        model=lm,
        tasks=[args.dataset],
        num_fewshot=0,
        batch_size=4,
        device="auto",
        limit=limit,
        reduce=reduce
    )
    if reduce is not None:
        metric_key = 'llt'
    else:
        if args.dataset in ['xsum', 'cnn_dailymail']:
            metric_key = 'rouge,none'
        elif args.dataset in ['gsm8k']:
            metric_key = 'exact_match,flexible-extract'
        else:
            metric_key = 'acc,none'
    acc = results['results'][args.dataset][metric_key]
    params = sum(param.numel() for _, param in model.named_parameters())
    print(f"Full eval acc: {acc}, params: {params}")
    return acc, params

def evaluate_vanilla(model, tokenizer, args):
    lm = HFLM(pretrained=model, tokenizer=tokenizer, device="auto")
    results = evaluator.simple_evaluate(
        model=lm,
        tasks=[args.dataset],
        num_fewshot=0,
        batch_size=4,
        device="auto",
        limit=0.3
    )
    if args.dataset in ['xsum', 'cnn_dailymail']:
        metric_key = 'rouge,none'
    elif args.dataset in ['gsm8k']:
        metric_key = 'exact_match,flexible-extract'
    else:
        metric_key = 'acc,none'
    acc = results['results'][args.dataset][metric_key]
    params = sum(param.numel() for _, param in model.named_parameters())
    print(f"Vanilla eval acc: {acc}, params: {params}")
    return acc, params

########################################
# 유틸리티 함수 (로깅)
########################################
def log_message(msg, log_path):
    with open(log_path, "a") as f:
        f.write(json.dumps(msg) + "\n")

########################################
# 인자 파싱
########################################
def parse_arguments():
    parser = argparse.ArgumentParser("Combined Decomposition and Evaluation")
    # 분해 관련 인자
    parser.add_argument("--layers", type=str, default="o_proj,q_proj,v_proj,k_proj,gate_proj,up_proj,down_proj",
                        help="분해할 레이어 이름(콤마로 구분)")
    parser.add_argument("--dataset", type=str, default="piqa", help="데이터셋 이름")
    parser.add_argument("--batch_size", type=int, default=512, help="배치 사이즈")
    parser.add_argument("--seq_len", type=int, default=128, help="시퀀스 길이")
    parser.add_argument("--log_path", type=str, default="surgical_logs.txt", help="로그 파일 경로")
    parser.add_argument("--algo", type=str, default="eigen", help="분해 알고리즘")
    # 저장 파일명에 max_ratio가 포함되도록 기본값 수정 가능
    parser.add_argument("--weights_name", type=str, default="decomposed_model_mistral_combination", help="분해된 모델 저장 파일명 (확장자 제외)")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf", help="모델 이름")
    parser.add_argument("--layer_cache_dir", type=str, default="./cache_dir", help="레이어 분해 결과 캐시 디렉토리")
    parser.add_argument("--cache_dir", type=str, default="../cache_dir", help="모델 캐시 디렉토리")
    # 평가/압축 관련 인자
    parser.add_argument("--delta", type=float, default=0.0, help="허용 정확도 감소량")
    parser.add_argument("--start_layer", type=int, default=28, help="압축 시작할 레이어 인덱스")
    parser.add_argument("--base_model", type=str, default="decomposed_model_mistral_combination.pt",
                        help="평가용 베이스 모델 파일")
    # 새로 추가된 인자: 검색 공간의 최대 비율 (예: 0.6이면 최대 60%까지 고려)
    parser.add_argument("--max_ratio", type=float, default=1.0, help="검색 공간의 최대 비율 (예: 0.6이면 60%까지)")
    return parser.parse_args()

########################################
# 모델 로드 및 분해 가능한 레이어 수집
########################################
def load_base_model(args):
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
        cache_dir=args.cache_dir,
    )
    print("[INFO] Base model loaded.")
    return model

def get_decomposable_layers(model, layer_names):
    decomposable_layers = []
    max_ranks = []
    for module_name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            max_val = int(module.weight.data.shape[0] * module.weight.data.shape[1] /
                          (module.weight.data.shape[0] + module.weight.data.shape[1]))
            max_ranks.append(max_val)
            for eligible in layer_names:
                if eligible in module_name:
                    tokens = module_name.split(".")
                    layer = model
                    for t in tokens[:-1]:
                        # 숫자인 경우 indexing으로 접근
                        if t.isdigit():
                            layer = layer[int(t)]
                        else:
                            layer = getattr(layer, t)
                    decomposable_layers.append([layer, tokens[-1]])
                    break
    print(f"[INFO] Found {len(decomposable_layers)} decomposable linear layers.")
    return decomposable_layers, max_ranks

########################################
# 토크나이저 및 데이터 준비
########################################
def prepare_tokenizer_and_data(args):
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
        if result["input_ids"][-1] != tokenizer.eos_token_id and len(result["input_ids"]) < 2048 and add_eos_token:
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)
        result["labels"] = result["input_ids"].copy()
        return result

    def generate_and_tokenize_prompt(data_point):
        return tokenize(data_point.get("text", ""))
    
    def generate_and_tokenize_prompt_summary(example):
        enc = tokenizer(example["text"], truncation=True, max_length=512)
        return enc
    
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
        dataloader = DataLoader([data], batch_size=args.batch_size)
    elif args.dataset == 'cnn_dailymail':
        dataset, _, _ = get_dataset(args.dataset)
        dataset = dataset.map(preprocess_function_cnndm, num_proc=64)
        dataset = dataset.map(generate_and_tokenize_prompt_summary, num_proc=64)
        dataset = dataset.select_columns(["input_ids", "attention_mask"])
        print(f"Dataset columns after select: {dataset.column_names}")
        dataloader = DataLoader(dataset, collate_fn=data_collator, batch_size=args.batch_size)
    elif args.dataset == 'xsum':
        dataset, _, _ = get_dataset(args.dataset)
        dataset = dataset.map(preprocess_function_xsum, num_proc=64)
        dataset = dataset.map(generate_and_tokenize_prompt_summary, num_proc=64)
        dataset = dataset.select_columns(["input_ids", "attention_mask"])
        print(f"Dataset columns after select: {dataset.column_names}")
        dataloader = DataLoader(dataset, collate_fn=data_collator, batch_size=args.batch_size)
    else:
        dataset, _, _ = get_dataset(args.dataset)
        dataset = dataset.map(generate_and_tokenize_prompt)
        dataset = dataset.select_columns(["input_ids", "attention_mask", "labels"])
        print(f"Dataset columns after select: {dataset.column_names}")
        dataloader = DataLoader(dataset, collate_fn=data_collator, batch_size=args.batch_size)

    print("[INFO] Data loading complete.")
    return tokenizer, dataloader

########################################
# 분해 레이어 주입
########################################
def inject_decomposable_layers(model, decomposable_layers, max_ranks, args):
    for idx in tqdm(range(len(decomposable_layers)), desc="Injecting decomposable layers"):
        parent_layer, last_token = decomposable_layers[idx]
        new_layer = ModuleInjection.make_decomposable(
            getattr(parent_layer, last_token), max_ranks[idx], args.algo
        )
        setattr(parent_layer, last_token, new_layer)
    for _, param in model.named_parameters():
        param.requires_grad = False
    print("[INFO] Decomposable layers injected.")

########################################
# Forward 테스트
########################################
def forward_test(model, dataloader, args):
    device = model.device
    model.eval()
    if args.dataset == 'wikitext2':
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Forward on wikitext2"):
                batch = {k: v.to(device) for k, v in batch.items() if v is not None}
                _ = model(**batch)
                break
    elif args.dataset == 'combination':
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Forward on combination"):
                batch = {k: v.to(device) for k, v in batch.items() if v is not None}
                _ = model(**batch)
    elif args.dataset == 'bookcorp':
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Forward on bookcorp"):
                batch = {k: v.to(device) for k, v in batch.items() if v is not None}
                _ = model(**batch)
    else:
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Forward on {args.dataset}"):
                batch = {k: v.to(device) for k, v in batch.items() if v is not None}
                print(batch["input_ids"].shape)
                _ = model(**batch)
                break

    print("[INFO] Forward test complete.")

########################################
# 모델 저장 (선택 사항)
########################################
def save_decomposed_model(model, args):
    model.half()
    # 저장 경로에 max_ratio 정보를 포함하여 파일명 생성
    weights_name = f"{args.weights_name}_maxratio_{args.max_ratio:.1f}"
    out_dir = os.path.dirname(weights_name)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(weights_name)
    torch.save(model, weights_name + ".pt")
    print(f"[INFO] Saved decomposed model to {weights_name}")

########################################
# 평가용 모델 준비
########################################
def prepare_evaluation_models(base_model, args, layer_names):
    eval_base_model = base_model
    eval_base_model.to("cuda")
    decomposable_layers_eval, max_ranks_eval = get_decomposable_layers(eval_base_model, layer_names)
    
    new_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        device_map="cuda",
        cache_dir="../SpeculativeDecoding/cache_dir",
        torch_dtype=torch.float16
    )
    decomposable_layers_new, _ = get_decomposable_layers(new_model, layer_names)
    
    return eval_base_model, new_model, decomposable_layers_eval, max_ranks_eval, decomposable_layers_new

########################################
# Baseline 평가
########################################
def baseline_evaluation(new_model, tokenizer, args):
    if args.dataset in ['xsum', 'cnn_dailymail', 'gsm8k']:
        _, ppl_loader = get_loaders_chunk(args.dataset, 0, 0.2, tokenizer, seq_len=128, batch_size=8)
        baseline_perplexity = llama_eval(new_model, ppl_loader, 'cuda')
        log_message(f"Baseline perplexity: {baseline_perplexity}", args.log_path)
        wandb.log({"baseline_perplexity": baseline_perplexity})
        return None, baseline_perplexity
    else:
        baseline_accs = []
        for i in range(3):
            acc, _ = evaluate(new_model, chunk=i, limit=0.0666, chunk_ratio=1/3,
                                tokenizer=tokenizer, reduce=None, args=args)
            baseline_accs.append(acc)
        old_acc, _ = evaluate(new_model, chunk=0, limit=0.2, chunk_ratio=1/3,
                              tokenizer=tokenizer, reduce=None, args=args)
        entire_acc, _ = evaluate_full(new_model, limit=0.2, tokenizer=tokenizer,
                                      reduce=None, args=args)
        log_message(f"Baseline disjoint acc {entire_acc}, 20% set acc {old_acc}", args.log_path)
        log_message(f"Chunk accs: {baseline_accs}", args.log_path)
        wandb.log({"baseline_entire_acc": entire_acc,
                   "baseline_20pct_acc": old_acc,
                   "chunk_accs": baseline_accs})
        return baseline_accs, old_acc

@torch.no_grad()
def llama_eval(model, test_lodaer, device):
    nlls = []
    for batch in tqdm(test_lodaer):
        batch = batch.to(device)
        output = model(batch)
        lm_logits = output.logits
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = batch[:, 1:].contiguous()
        
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.view(-1))
        nlls.append(loss)
    ppl = np.exp(torch.cat(nlls, dim=-1).mean().item())
    return ppl.item()

########################################
# 레이어 압축 (Pruning) 진행
########################################
def prune_layers(eval_base_model, new_model, baseline_accs, max_ranks_eval, 
                 decomposable_layers_eval, decomposable_layers_new, tokenizer, args):
    if args.dataset in ['xsum', 'cnn_dailymail', 'gsm8k']:
        _, ppl_loader = get_loaders_chunk(args.dataset, 0, 0.2, tokenizer, seq_len=128, batch_size=8)
        baseline_metric = llama_eval(new_model, ppl_loader, 'cuda')
        print(f"[INFO] Baseline perplexity: {baseline_metric}")
    else:
        baseline_metric = None

    acc_30_cal = []
    acc_20_cal = []
    layer_ind = []
    params_ = []
    pre_surgical_params = sum(param.numel() for _, param in new_model.named_parameters())
    wandb.log({"pre_surgical_param_count": pre_surgical_params})
    
    # 새로 추가: 레이어별로 최종 선택된 rank를 기록하기 위한 딕셔너리
    chosen_ranks = {}

    for index in tqdm(reversed(range(len(decomposable_layers_eval))), desc="Evaluating layers"):
        if index < args.start_layer:
            continue

        parent_layer_base, last_token_base = decomposable_layers_eval[index]
        layer_base = copy.deepcopy(getattr(parent_layer_base, last_token_base)).cuda().half()

        parent_layer_new, last_token_new = decomposable_layers_new[index]
        layer_old = copy.deepcopy(getattr(parent_layer_new, last_token_new)).cuda().half()

        setattr(parent_layer_new, last_token_new, layer_base)
        layer_new = getattr(parent_layer_new, last_token_new)

        split_rank = []
        # parser로 전달된 max_ratio를 사용하여 검색 공간 조절
        # 예: 0.1 ~ 0.5라면 [1, (0.1*max_rank), (0.2*max_rank) ...]
        search_space = [1] + list((np.arange(0.5, args.max_ratio + 0.1, 0.1) * max_ranks_eval[index]).astype(np.int32))
        print(f"[Debug] Layer {index} search_space: {search_space}")

        for i in range(3):
            ind = len(search_space) - 1
            if split_rank and max(split_rank) == search_space[-1]:
                break
            for j in range(len(search_space)):
                rank = search_space[j]
                V = copy.deepcopy(layer_base.V[:, -rank:]).cuda().half()
                layer_new.weight2.data = V
                layer_new.weight1.data = (torch.transpose(V, 1, 0)
                                            .to(layer_base.weight.device)
                                            .half() @ layer_base.weight).cuda().half()

                V_prune = copy.deepcopy(layer_base.V[:, :-rank]).to(torch.float32).cuda()
                layer_base.Y_sub = layer_base.Y_sub.to(torch.float32).cuda()
                layer_new.bias.data = layer_base.b1.cuda().half()
                temp = (V_prune @ V_prune.transpose(1, 0) @ layer_base.Y_sub.transpose(1, 0)).transpose(1, 0).half()
                layer_new.bias.data += temp

                if args.dataset in ['xsum', 'cnn_dailymail', 'gsm8k']:
                    new_metric = llama_eval(new_model, ppl_loader, 'cuda')
                    condition = (new_metric <= baseline_metric * (1.005)**(224-index))
                else:
                    new_metric, _ = evaluate(new_model, chunk=i, limit=0.0666, 
                                               chunk_ratio=1/3, tokenizer=tokenizer, reduce=None, args=args)
                    condition = (new_metric >= baseline_accs[i] - args.delta)
                
                if condition:
                    ind = j
                    log_message(f"Layer {index}: new metric {new_metric}, baseline {baseline_metric if baseline_metric is not None else baseline_accs[i]}, chunk {i}, rank {search_space[j]}", args.log_path)
                    break
            split_rank.append(search_space[ind])
        final_rank = max(split_rank)
        rank = final_rank

        # 최종 rank로 다시 설정
        V = copy.deepcopy(layer_base.V[:, -rank:]).cuda().half()
        layer_new.weight2.data = V
        layer_new.weight1.data = (torch.transpose(V, 1, 0)
                                    .to(layer_base.weight.device)
                                    .half() @ layer_base.weight).cuda().half()
        V_prune = copy.deepcopy(layer_base.V[:, :-rank]).to(torch.float32).cuda()
        layer_base.Y_sub = layer_base.Y_sub.to(torch.float32).cuda()
        layer_new.bias.data = layer_base.b1.cuda().half() + \
            (V_prune @ V_prune.transpose(1, 0) @ layer_base.Y_sub.transpose(1, 0)).transpose(1, 0).cuda().half()

        if args.dataset in ['xsum', 'cnn_dailymail', 'gsm8k']:
            new_metric = llama_eval(new_model, ppl_loader, 'cuda')
            final_condition = (final_rank != search_space[-1]) and (new_metric <= baseline_metric * (1.005)**(224-index))
        else:
            new_metric, _ = evaluate(new_model, chunk=0, limit=0.2, tokenizer=tokenizer, reduce=None, args=args)
            final_condition = (final_rank != search_space[-1]) and (new_metric >= baseline_accs[0] - args.delta)

        if not final_condition:
            # 압축 실패 -> 원상 복구
            setattr(parent_layer_new, last_token_new, layer_old)
            del layer_new
            log_message(f"Layer {index} unchanged", args.log_path)
            chosen_ranks[index] = None  # 해당 레이어는 압축 안 됨
        else:
            # 압축 성공
            layer_new.V = None
            layer_new.Y_sub = None
            layer_new.weight = None
            log_message(f"Layer {index} max compression {final_rank}", args.log_path)
            chosen_ranks[index] = final_rank  # 최종 rank 기록

        if (index + 1) % 7 == 0:
            if args.dataset in ['xsum', 'cnn_dailymail', 'gsm8k']:
                curr_metric = llama_eval(new_model, ppl_loader, 'cuda')
                pm = sum(param.numel() for _, param in new_model.named_parameters())
            else:
                curr_metric, pm = evaluate_full(new_model, limit=0.2, tokenizer=tokenizer, reduce=None, args=args)
            acc_30_cal.append(curr_metric)
            acc_20_cal.append(curr_metric)
            layer_ind.append(index)
            params_.append(pm)
            p = np.hstack((
                np.array(layer_ind).reshape((-1, 1)),
                np.array(acc_30_cal).reshape((-1, 1)),
                np.array(acc_20_cal).reshape((-1, 1)),
                np.array(params_).reshape((-1, 1))
            ))
            print(p)
            p_df = pd.DataFrame(p, columns=["layer_ind", "metric_30_cal", "metric_20_cal", "params"])
            log_csv = f"logs_{args.dataset}_llama_3.csv"
            p_df.to_csv(log_csv, index=False)
            log_message(f"Decomposed till {index}: final metric {curr_metric}, params {pm}", args.log_path)
            wandb.log({
                "surgical_layer_index": index,
                "current_metric": curr_metric,
                "param_count": pm
            })
            torch.save(new_model.half(), f"final_max_comp_{args.dataset}_{args.max_ratio}_llama_3.pt")
        torch.cuda.empty_cache()
        gc.collect()

    print("[INFO] Evaluation and compression finished.")

    # -------------------------------------------------
    # 추가: 레이어별 최종 rank 선택 결과 요약
    # -------------------------------------------------
    log_message(f"Chosen Ranks by Layer: {chosen_ranks}", args.log_path)

    # (예) None 제외하고 실제 선택된 rank만 모아서 빈도 계산
    actual_ranks = [r for r in chosen_ranks.values() if r is not None]
    rank_freq = {}
    for r in actual_ranks:
        rank_freq[r] = rank_freq.get(r, 0) + 1

    # 로컬 로그 기록
    log_message(f"Rank Frequency: {rank_freq}", args.log_path)

    # wandb 로그
    # 1) 전체 딕셔너리 형태로 한번에 로깅 (wandb 상에서는 key-value를 펼쳐 보거나 table 형태가 될 수 있음)
    wandb.log({"chosen_rank_map": chosen_ranks})
    # 2) 빈도도 각각 기록
    for rank_val, freq_count in rank_freq.items():
        wandb.log({f"rank_{rank_val}_freq": freq_count})

########################################
# Main
########################################
def main():
    args = parse_arguments()
    
    # run name 예시로 구성
    run_name = f"{args.model.split('/')[-1]}_{args.dataset}_maxratio_{args.max_ratio}_delta_{args.delta}_start_{args.start_layer}"
    wandb.init(project="SFSD", config=vars(args), name=run_name)
    
    log_message(str(args), args.log_path)
    print(f"[INFO] Using model: {args.model}")

    if not os.path.exists(args.weights_name + ".pt"):
        print(f"[INFO] Decomposed model file {args.weights_name} not found. Running forward test and saving model.")
        base_model = load_base_model(args)
        layer_names = [name.strip() for name in args.layers.split(',')]
        decomposable_layers_base, max_ranks = get_decomposable_layers(base_model, layer_names)

        tokenizer, dataloader = prepare_tokenizer_and_data(args)

        inject_decomposable_layers(base_model, decomposable_layers_base, max_ranks, args)

        forward_test(base_model, dataloader, args)

        save_decomposed_model(base_model, args)
    else:
        print(f"[INFO] Found decomposed model file {args.weights_name}. Loading model directly.")
        base_model = torch.load(args.weights_name + ".pt")
        layer_names = [name.strip() for name in args.layers.split(',')]

    eval_base_model, new_model, decomposable_layers_eval, max_ranks_eval, decomposable_layers_new = \
        prepare_evaluation_models(base_model, args, layer_names)

    tokenizer, _ = prepare_tokenizer_and_data(args)
    baseline_accs, old_acc = baseline_evaluation(new_model, tokenizer, args)
    
    pre_surgical_params = sum(param.numel() for _, param in new_model.named_parameters())
    wandb.log({"pre_surgical_param_count_final": pre_surgical_params})

    prune_layers(eval_base_model, new_model, baseline_accs, max_ranks_eval,
                 decomposable_layers_eval, decomposable_layers_new,
                 tokenizer, args)
    
    post_surgical_params = sum(param.numel() for _, param in new_model.named_parameters())
    wandb.log({"post_surgical_param_count": post_surgical_params})
    
    wandb.finish()

if __name__ == "__main__":
    main()
