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
# 유틸리티 함수 (로깅)
########################################
def log_message(msg, log_path):
    with open(log_path, "a") as f:
        f.write(json.dumps(msg) + "\n")


########################################
# 인자 파싱
########################################
def parse_arguments():
    parser = argparse.ArgumentParser("Combined Decomposition and Surgical Pruning")
    parser.add_argument("--mode", type=str, default="decompose", 
                        help="실행 모드 (decompose / prune)")
    parser.add_argument("--layers", type=str, default="o_proj,q_proj,v_proj,k_proj,gate_proj,up_proj,down_proj",
                        help="분해할 레이어 이름 (콤마로 구분)")
    parser.add_argument("--dataset", type=str, default="piqa", help="데이터셋 이름")
    parser.add_argument("--batch_size", type=int, default=512, help="배치 사이즈")
    parser.add_argument("--seq_len", type=int, default=128, help="시퀀스 길이")
    parser.add_argument("--log_path", type=str, default="surgical_logs.txt", help="로그 파일 경로")
    parser.add_argument("--algo", type=str, default="eigen", help="분해 알고리즘")
    parser.add_argument("--weights_name", type=str, default="decomposed_model_mistral_combination",
                        help="분해된 모델 저장 파일명 (확장자 제외)")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf", help="모델 이름")
    parser.add_argument("--layer_cache_dir", type=str, default="./cache_dir", help="레이어 분해 결과 캐시 디렉토리")
    parser.add_argument("--cache_dir", type=str, default="../cache_dir", help="모델 캐시 디렉토리")
    parser.add_argument("--delta", type=float, default=0.0, help="허용 정확도 감소량")
    parser.add_argument("--start_layer", type=int, default=28, help="압축 시작할 레이어 인덱스")
    parser.add_argument("--base_model", type=str, default="decomposed_model_mistral_combination.pt",
                        help="평가용 베이스 모델 파일 (분해 후 저장된 모델)")
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
            for eligible in layer_names.split(","):
                if eligible in module_name:
                    tokens = module_name.split(".")
                    layer = model
                    for t in tokens[:-1]:
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

    def generate_and_tokenize_prompt_cnn(example):
        enc = tokenizer(example["article"], truncation=True, max_length=512)
        return enc

    def generate_and_tokenize_prompt_xsum(example):
        enc = tokenizer(example["document"], truncation=True, max_length=512)
        return enc

    if args.dataset == 'wikitext2':
        dataset = get_wikitext2(tokenizer, seq_len=args.seq_len)
        dataloader = DataLoader(dataset, batch_size=args.batch_size)
    elif args.dataset in ['cnn_dailymail', 'xsum']:
        dataset, _, _ = get_dataset(args.dataset)
        if args.dataset == 'cnn_dailymail':
            dataset = dataset.map(preprocess_function_cnndm, num_proc=64)
            dataset = dataset.map(generate_and_tokenize_prompt_cnn, num_proc=64)
        else:
            dataset = dataset.map(preprocess_function_xsum, num_proc=64)
            dataset = dataset.map(generate_and_tokenize_prompt_xsum, num_proc=64)
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
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Forward on {args.dataset}"):
            batch = {k: v.to(device) for k, v in batch.items() if v is not None}
            _ = model(**batch)
            break
    print("[INFO] Forward test complete.")


########################################
# 모델 저장 (분해된 모델 전체 저장)
########################################
def save_decomposed_model(model, args):
    weights_name = f"{args.weights_name}_maxratio_{args.max_ratio:.1f}"
    torch.save(model, weights_name + ".pt")
    torch.save(base_model.half(),weights_name + "_half.pt")
    print(f"[INFO] Saved decomposed model to {weights_name+'.pt'}")


########################################
# 평가용 모델 준비 (새 모델 불러오기)
########################################
def prepare_evaluation_models(base_model, args, layer_names):
    eval_base_model = base_model
    decomposable_layers_eval, max_ranks_eval = get_decomposable_layers(eval_base_model, layer_names)
    
    new_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        device_map="cuda",
        cache_dir=args.cache_dir,
        torch_dtype=torch.float16
    )
    decomposable_layers_new, _ = get_decomposable_layers(new_model, layer_names)
    
    return eval_base_model, new_model, decomposable_layers_eval, max_ranks_eval, decomposable_layers_new


########################################
# Baseline 평가 (퍼플렉서티 기준)
########################################
def baseline_evaluation(new_model, tokenizer, args):
    if args.dataset in ['xsum', 'cnn_dailymail', 'gsm8k']:
        _, ppl_loader = get_loaders_chunk(args.dataset, 0, 0.0001, tokenizer, seq_len=128, batch_size=8)
        baseline_perplexity = llama_eval(new_model, ppl_loader, 'cuda')
        log_message(f"Baseline perplexity: {baseline_perplexity}", args.log_path)
        wandb.log({"baseline_perplexity": baseline_perplexity})
        return None, baseline_perplexity
    else:
        # 퍼플렉서티 기준으로만 평가하는 경우
        print("[INFO] Evaluating baseline model (perplexity).")
        return None, llama_eval(new_model, get_loaders_chunk(args.dataset, 0, 0.01, tokenizer, seq_len=128, batch_size=8)[0], 'cuda')

def param_counter(model):
        params = 0
        for _, param in model.named_parameters():
            params+=param.numel()
        return params

@torch.no_grad()
def llama_eval(model, test_loader, device):
    nlls = []
    for batch in tqdm(test_loader):
        batch = batch.to(device)
        output = model(batch)
        lm_logits = output.logits
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = batch[:, 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.view(-1))
        nlls.append(loss)
    ppl = np.exp(torch.cat(nlls, dim=-1).mean().item())
    return ppl


########################################
# 레이어 압축 (Surgical Pruning)
########################################
def prune_layers(eval_base_model, new_model, baseline_accs, max_ranks_eval,
                 decomposable_layers_eval, decomposable_layers_new,
                 tokenizer, args):
    """
    레이어별로 랭크(차원)를 점진적으로 줄여가며(per-layer compression),
    퍼플렉서티(PPL) 기준을 만족하는지 검사. 
    모델 전체 파라미터 감소량에 따라 중간/최종 체크포인트도 저장.
    모든 로깅은 wandb를 사용함.
    """

    # ---------------------------------------
    # 1) 20% 데이터셋 로더 & baseline 계산
    # ---------------------------------------
    if args.dataset in ['xsum', 'cnn_dailymail', 'gsm8k']:
        # 20% chunk 로더
        _, acc_20_loader = get_loaders_chunk(args.dataset, 0, 0.2, tokenizer, seq_len=128, batch_size=8)
        acc_20 = llama_eval(new_model, acc_20_loader, 'cuda')  
        print(f"[INFO] Baseline (20% subset) perplexity: {acc_20}")
        wandb.log({"initial_20pct_ppl": acc_20})
    else:
        # wikitext2 예시
        acc_20_loader, _ = get_loaders_chunk('wikitext2', 0, 0.2, tokenizer, seq_len=128, batch_size=8)
        acc_20 = llama_eval(new_model, acc_20_loader, 'cuda')
        print(f"[INFO] Baseline (20% subset) perplexity (wikitext2): {acc_20}")
        wandb.log({"initial_20pct_ppl": acc_20})

    # 파라미터 개수 등 초기 정보 로깅
    pre_surgical_params = param_counter(new_model)
    wandb.log({"pre_surgical_param_count": pre_surgical_params})
    print(f"[INFO] Pre-surgical total params: {pre_surgical_params}")

    # 파라미터 예산체크 플래그
    flag_80, flag_90 = True, True
    total_ = pre_surgical_params

    # 중간 기록용 리스트
    acc_30_cal = []
    layer_ind = []
    params_ = []
    chosen_ranks = {}

    # ---------------------------------------
    # 2) 레이어를 역순으로 순회하여 Pruning
    # ---------------------------------------
    for index in tqdm(reversed(range(len(decomposable_layers_eval))), desc="Surgical Pruning"):
        # start_layer보다 작은 인덱스는 건너뜀
        if index < args.start_layer:
            continue

        # (A) 분해된 모델에서 해당 레이어 복사
        parent_layer_base, last_token_base = decomposable_layers_eval[index]
        layer_base = copy.deepcopy(getattr(parent_layer_base, last_token_base)).cuda().half()

        # (B) 새 모델에서 해당 레이어 백업
        parent_layer_new, last_token_new = decomposable_layers_new[index]
        layer_old = copy.deepcopy(getattr(parent_layer_new, last_token_new)).cuda().half()

        # (C) 새 모델 레이어 교체(분해 레이어로)
        setattr(parent_layer_new, last_token_new, layer_base)
        layer_new = getattr(parent_layer_new, last_token_new)

        # (D) rank 후보(search space) 정의
        search_space = [1] + list((np.arange(0.1, args.max_ratio + 0.1, 0.1) * max_ranks_eval[index]).astype(np.int32))
        search_space = sorted(set(search_space))  # 혹시 중복 제거
        print(f"[Debug] Layer {index} search_space: {search_space}")

        final_rank = search_space[-1]
        chosen_index = len(search_space) - 1
        current_ppl = None

        # (E) rank 후보 순회
        for j, rank_candidate in enumerate(search_space):
            # (i) weight2, weight1, bias를 rank_candidate 에 맞게 수정
            V = copy.deepcopy(layer_base.V[:, -rank_candidate:]).cuda().half()
            layer_new.weight2.data = V
            layer_new.weight1.data = (
                torch.transpose(V, 1, 0).to(layer_base.weight.device).half() @ layer_base.weight
            ).cuda().half()

            V_prune = copy.deepcopy(layer_base.V[:, :-rank_candidate]).float()
            layer_base.Y_sub = layer_base.Y_sub.float()

            layer_new.bias.data = layer_base.b1.cuda().half()
            temp = (V_prune @ V_prune.transpose(1, 0) @ layer_base.Y_sub.transpose(1, 0)).transpose(1, 0).cuda().half()
            layer_new.bias.data += temp

            # (ii) 새로 압축한 레이어로 현재 PPL 계산
            current_ppl = llama_eval(new_model, acc_20_loader, 'cuda')

            # (iii) 기준: baseline(=acc_20)에 비해 (1.005)^(224-index) 이하로만 증가해야 OK
            if current_ppl <= acc_20 * (1.005)**(224 - index):
                chosen_index = j
                wandb.log({
                    "layer_index": index,
                    "candidate_rank": rank_candidate,
                    "candidate_ppl": current_ppl,
                    "msg": "Accepted rank"
                })
                break
            else:
                wandb.log({
                    "layer_index": index,
                    "candidate_rank": rank_candidate,
                    "candidate_ppl": current_ppl,
                    "msg": "Rejected rank"
                })

        final_rank = search_space[chosen_index]
        print(f"[INFO] Layer {index} final chosen rank={final_rank}, ppl={current_ppl}")

        # (F) 최종 점검: 만약 rank가 끝까지 갔거나 PPL이 기준 초과면 원복
        if (final_rank == search_space[-1]) or (current_ppl is not None and current_ppl > acc_20*(1.005)**(224 - index)):
            setattr(parent_layer_new, last_token_new, layer_old)
            del layer_new
            msg = (f"Layer {index}, Unchanged. Permitted ppl <= "
                   f"{acc_20*(1.005)**(224-index):.4f}, got {current_ppl:.4f}")
            print(msg)
            wandb.log({
                "layer_index": index,
                "final_decision": "unchanged",
                "final_ppl": current_ppl,
                "ppl_threshold": acc_20*(1.005)**(224 - index)
            })
            chosen_ranks[index] = None
        else:
            # 압축한 레이어를 유지하고, 원본 weight/버퍼 정리
            layer_new.V = None
            layer_new.Y_sub = None
            layer_new.weight = None
            msg = (f"Layer {index} compressed to rank={final_rank}, ppl={current_ppl:.4f}, "
                   f"Permitted threshold ~ {acc_20*(1.005)**(224-index):.4f}")
            print(msg)
            wandb.log({
                "layer_index": index,
                "final_decision": "compressed",
                "final_rank": final_rank,
                "final_ppl": current_ppl
            })
            chosen_ranks[index] = final_rank

        # (G) 파라미터 예산 체크 (예: 80%, 90%)
        pm = param_counter(new_model)
        if pm < total_ * 0.815 and flag_80:
            torch.save(new_model.half(), 'budget_80_mistral_ppl.pt')
            print(f"[INFO] Saved 80% budget checkpoint at layer index {index}")
            wandb.log({"checkpoint_80pct_layer": index, "current_param_count": pm})
            flag_80 = False

        if pm < total_ * 0.91 and flag_90:
            torch.save(new_model.half(), 'budget_90_mistral_ppl.pt')
            print(f"[INFO] Saved 90% budget checkpoint at layer index {index}")
            wandb.log({"checkpoint_90pct_layer": index, "current_param_count": pm})
            flag_90 = False

        # (H) 7씩 구간으로 평가/로깅
        if (index + 1) % 7 == 0:
            # 여기서는 20% ppl을 그대로 사용(원하면 풀세트 ppl도 가능)
            curr_metric = current_ppl if current_ppl is not None else -1
            pm_now = param_counter(new_model)

            acc_30_cal.append(curr_metric)
            layer_ind.append(index)
            params_.append(pm_now)

            # CSV 저장이나 다른 방식을 쓰고 싶다면...
            p = np.column_stack([layer_ind, acc_30_cal, params_])
            p_df = pd.DataFrame(p, columns=["layer_ind", "metric", "params"])
            p_df.to_csv(f"logs_{args.dataset}_llama_3.csv", index=False)

            wandb.log({
                "surgical_layer_index": index,
                "current_metric": curr_metric,
                "param_count": pm_now
            })
            print(f"[INFO] Decomposed up to {index}, metric={curr_metric}, params={pm_now}")

            # 중간 체크포인트
            torch.save(new_model.half(), f"final_max_comp_{args.dataset}_{args.max_ratio}_llama_3.pt")

        torch.cuda.empty_cache()
        gc.collect()

    print("[INFO] All layers processed. Surgical pruning finished.")

    # ---------------------------------------
    # 3) 결과 요약 및 최종 모델 저장
    # ---------------------------------------
    # chosen_ranks 정보를 분석해 wandb에 로깅
    rank_freq = {}
    for r in [val for val in chosen_ranks.values() if val is not None]:
        rank_freq[r] = rank_freq.get(r, 0) + 1

    # wandb에 최종 정보 기록
    wandb.log({"chosen_ranks": chosen_ranks, "rank_frequency": rank_freq})

    # 최종 모델 저장
    final_save_name = f"final_max_comp_{args.dataset}_{args.max_ratio}_mistral.pt"
    torch.save(new_model.half(), final_save_name)
    print(f"[INFO] Final pruned model saved: {final_save_name}")
    wandb.log({"final_model_saved": final_save_name})



########################################
# 메인
########################################
def main():
    args = parse_arguments()
    
    run_name = f"{args.model.split('/')[-1]}_{args.dataset}_maxratio_{args.max_ratio}_delta_{args.delta}_start_{args.start_layer}"
    wandb.init(project="SFSD", config=vars(args), name=run_name)
    log_message(str(args), args.log_path)
    print(f"[INFO] Using model: {args.model}")

    layer_names = [name.strip() for name in args.layers.split(',')]
    decomposed_weights_name = f"{args.weights_name}_maxratio_{args.max_ratio:.1f}"
    param_file = decomposed_weights_name + ".pt"

    if args.mode == "decompose":
        # Decomposition Phase
        print(f"[INFO] Decomposed model not found or mode==decompose. Running decomposition.")
        base_model = load_base_model(args)
        decomposable_layers_base, max_ranks = get_decomposable_layers(base_model, args.layers)
        tokenizer, dataloader = prepare_tokenizer_and_data(args)
        inject_decomposable_layers(base_model, decomposable_layers_base, max_ranks, args)
        forward_test(base_model, dataloader, args)
        save_decomposed_model(base_model, args)
    elif args.mode == "prune":
        # Surgical Pruning Phase: load decomposed model first
        print(f"[INFO] Loading decomposed model from {param_file}")
        base_model = torch.load(param_file)
        print("[INFO] Decomposed model loaded.")
    else:
        print("Invalid mode. Use 'decompose' or 'prune'.")
        return

    # 평가용 모델 준비 (동일한 기준으로 새 모델을 로드)
    eval_base_model, new_model, decomposable_layers_eval, max_ranks_eval, decomposable_layers_new = prepare_evaluation_models(base_model, args, args.layers)
    tokenizer, _ = prepare_tokenizer_and_data(args)
    baseline_accs, old_acc = baseline_evaluation(new_model, tokenizer, args)
    pre_surgical_params = sum(param.numel() for _, param in new_model.named_parameters())
    wandb.log({"pre_surgical_param_count_final": pre_surgical_params})
    prune_layers(eval_base_model, new_model, baseline_accs, max_ranks_eval, decomposable_layers_eval, decomposable_layers_new, tokenizer, args)
    post_surgical_params = sum(param.numel() for _, param in new_model.named_parameters())
    wandb.log({"post_surgical_param_count": post_surgical_params})
    wandb.finish()

if __name__ == "__main__":
    main()
