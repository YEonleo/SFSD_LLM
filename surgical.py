import torch
import os
import sys
import numpy as np
import gc
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader
import transformers
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    AutoTokenizer,
    TrainingArguments,
    T5ForConditionalGeneration,
    DataCollatorForSeq2Seq,
)
import copy
from datasets import load_dataset
from preprocess import get_combination, get_bookcorpus
import argparse
from tqdm import tqdm
from layers import ModuleInjection
from lm_eval import evaluator
from evaluator_modified import simple_evaluate_chunk, full_evaluate
from preprocess import *
from lm_eval.models.huggingface import HFLM
import json
import time

# 인자 파싱
parser = argparse.ArgumentParser("main")
parser.add_argument("--dataset", type=str, default="piqa")
parser.add_argument("--layers", type=str, default="o_proj,q_proj,v_proj,k_proj,gate_proj,up_proj,down_proj")
parser.add_argument("--log_path", type=str, default="surgical_logs.txt")
parser.add_argument("--algo", type=str, default="eigen")
parser.add_argument("--delta", type=float, default=0.0)
parser.add_argument("--start_layer", type=int, default=28)
parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf")
parser.add_argument("--base_model", type=str, default="decompose_llama2_cnndm.pt")
args = parser.parse_args()

log_name = f"logs_{args.dataset}_llama_3.csv"
with open(args.log_path, "a") as file:
    file.write(json.dumps(f"Max Compression for Delta : {args.delta}\n"))
    file.write(json.dumps(str(args)))
    file.write("\n")

# 베이스 모델 불러오기 (devicemap="auto" 사용)
# base_model = AutoModelForCausalLM.from_pretrained(
#     args.base_model,
#     device_map="auto",
# ).half()
base_model = torch.load(args.base_model)
tokenizer = AutoTokenizer.from_pretrained(args.model)
print("[INFO] load_state_dict done.")


decomposable_layers_base = []
max_rank = []
for name, l in base_model.named_modules():
    if isinstance(l, nn.Linear):
        max_rank.append(int(l.weight.data.shape[0]*l.weight.data.shape[1]/(l.weight.data.shape[0]+l.weight.data.shape[1])))
        for eligible_layer in args.layers:
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
    # 평가 metric key 결정 (xsum이나 cnn_dailymail의 경우 'rouge,none' 사용)
    if reduce is not None:
        metric_key = 'llt'
    else:
        if args.dataset in ['xsum', 'cnn_dailymail']:
            metric_key = 'rouge,none'
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
    else:
        metric_key = 'acc,none'
    acc = results['results'][args.dataset][metric_key]
    params = sum(param.numel() for _, param in model.named_parameters())
    print(f"Vanilla eval acc: {acc}, params: {params}")
    return acc, params



new_model = AutoModelForCausalLM.from_pretrained(
    args.model,
    trust_remote_code=True,
    device_map="cuda",
    cache_dir="../SpeculativeDecoding/cache_dir",
    torch_dtype=torch.float16
)
decomposable_layers_new = []

for name, l in new_model.named_modules():
    if isinstance(l, nn.Linear):
        for eligible_layer in args.layers:
            if eligible_layer in name:
                tokens = name.strip().split(".")
                layer = new_model
                for t in tokens[:-1]:
                    if not t.isnumeric():
                        layer = getattr(layer, t)
                    else:
                        layer = layer[int(t)]

                decomposable_layers_new.append([layer, tokens[-1]])
                break


baseline_accs = []
for i in range(3):
    base_acc,_  = evaluate(new_model, chunk = i, limit = 0.666, chunk_ratio=1/3, tokenizer=tokenizer,reduce = None)
    baseline_accs.append(base_acc)


old_acc,_ = evaluate(new_model, chunk=0, limit=0.2, tokenizer=tokenizer,reduce = None)
entire_acc,_ = evaluate_full(new_model)
acc_30_cal = []
acc_20_cal = []
layer_ind = []
params_ = []

with open(args.log_path, "a") as file:
    file.write(json.dumps(f"Baseline test set acc disjoint {entire_acc} acc on 20% {old_acc} "))
    file.write("\n")
    file.write(json.dumps(f"Chunk 0 {baseline_accs[0]} Chunk 1 {baseline_accs[1]} Chunk 2 {baseline_accs[2]}"))
    file.write("\n")

for index in tqdm(reversed((range(len(decomposable_layers_base)-1)))):
    if(index<args.start_layer):
        continue

    parent_layer_base, last_token_base = decomposable_layers_base[index]
    layer_base = copy.deepcopy(getattr(parent_layer_base, last_token_base)).cuda().half()
    parent_layer_new, last_token_new = decomposable_layers_new[index]
    layer_old = copy.deepcopy(getattr(parent_layer_new, last_token_new)).cuda().half()
    setattr(parent_layer_new, last_token_new, layer_base)
    layer_new = getattr(parent_layer_new, last_token_new)
    split_rank = []
    search_space = [1] + list((np.arange(0.1, 1.1, 0.1)*max_rank[index]).astype(np.int32))
    print(f"[Debug] Search_space {search_space}")
    for i in range(3):
        ind = len(search_space) -1
        if(len(split_rank)>0 and max(split_rank) == search_space[-1]):
            break
        for j in range(len(search_space)):

            rank = search_space[j]

            V = copy.deepcopy(layer_base.V[:, -rank:]).cuda().half()

            layer_new.weight2.data =  V
            layer_new.weight1.data = (
                torch.transpose (V, 1, 0).to(layer_base.weight.device).half() @ layer_base.weight
            ).cuda().half()

            V_prune = copy.deepcopy(layer_base.V[:, :-rank])
            V_prune = V_prune.to(torch.float32).cuda()
            layer_base.Y_sub = layer_base.Y_sub.to(torch.float32).cuda()
            layer_new.bias.data = layer_base.b1.cuda().half()
            
            temp =  (V_prune @ V_prune.transpose(1,0) @ layer_base.Y_sub.transpose(1,0)).transpose(1,0).half()
            layer_new.bias.data += temp
            acc,_ = evaluate(new_model, chunk=i, limit=0.0666,chunk_ratio=1/3, tokenizer=tokenizer, reduce = None)
            if(acc>=baseline_accs[i] - args.delta):
                ind = j
                with open(args.log_path, "a") as file:
                    file.write(json.dumps(f"Layer index {index} new  {(acc)}  old  {baseline_accs[i]}  chunk {i} and rank {search_space[j]}"))
                    file.write("\n")
                break
        split_rank.append(search_space[ind])   
    final_rank = max(split_rank)
    rank = final_rank

    V = copy.deepcopy(layer_base.V[:, -rank:]).cuda().half()

    layer_new.weight2.data =  V
    layer_new.weight1.data = (
        torch.transpose (V, 1, 0).to(layer_base.weight.device).half() @ layer_base.weight
    ).cuda().half()

    V_prune = copy.deepcopy(layer_base.V[:, :-rank])
    V_prune = V_prune.to(torch.float32)
    layer_base.Y_sub = layer_base.Y_sub.to(torch.float32)
    layer_new.bias.data = layer_base.b1.cuda().half() + (V_prune @ V_prune.transpose(1,0) @ layer_base.Y_sub.transpose(1,0)).transpose(1,0).cuda().half()
    
    acc,_ = evaluate(new_model, chunk=0, size=0.2, reduce = None)
    if(final_rank == search_space[-1] or acc < old_acc - args.delta):
        setattr(parent_layer_new, last_token_new, layer_old)
        del layer_new
        with open(args.log_path, "a") as file:
            file.write(json.dumps(f"Layer index {index}, Unchanged"))
            file.write("\n")
    else:
        layer_new.V = None
        layer_new.Y_sub = None
        layer_new.weight = None
        with open(args.log_path, "a") as file:
            file.write(json.dumps(f"Layer index {index} max compression {final_rank}"))
            file.write("\n")
    

    if((index+1)%7 == 0):
        with open(args.log_path, "a") as file:
            curr_acc,pm = evaluate_full(new_model)
            # if(curr_acc>=entire_acc - entire_acc*0.05):
                # torch.save(new_model.half(), f"delta_perf_max_comp_{args.dataset}_mistral_3.pt")
                # file.write(json.dumps(f"New delta perf checkpoint with {curr_acc} params {pm}"))
            acc,pm = evaluate(new_model, chunk = 0, size = 0.2, reduce = None)
            acc_30_cal.append(curr_acc)
            acc_20_cal.append(acc)
            layer_ind.append(index)
            params_.append(pm)
            p = np.hstack((np.array(layer_ind).reshape((len(layer_ind),1)), np.array(acc_30_cal).reshape((len(layer_ind),1)), np.array(acc_20_cal).reshape((len(layer_ind),1)),np.array(params_).reshape((len(layer_ind),1))))
            print(p)
            p = pd.DataFrame(p, columns=["layer_ind", "acc_30_cal", "acc_20_cal","params"])
            p.to_csv(log_name, index=False)
            file.write(json.dumps(f"Decomposed till {index} 80% disjoint acc {curr_acc} 20% set acc {acc} params {pm}"))
            file.write("\n")
        torch.save(new_model.half(), f"final_max_comp_{args.dataset}_mistral_3.pt")
    torch.cuda.empty_cache()
    gc.collect()