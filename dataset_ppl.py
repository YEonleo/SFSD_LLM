import random
import numpy as np
import torch

from datasets import load_dataset
from torch.utils.data.dataset import Dataset

def get_wikitext2(seq_len, tokenizer):
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    return traindata, testdata

def get_ptb(seq_len, tokenizer):
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    valdata = load_dataset('ptb_text_only', 'penn_treebank', split='validation')
    return traindata, valdata

class IndexDataset(Dataset):
    def __init__(self, tensors):
        self.tensors = tensors

    def __getitem__(self, index):
        return self.tensors[index]

    def __len__(self):
        return len(self.tensors)

def process_data(samples, tokenizer, seq_len, field_name):
    # samples[field_name]가 리스트이면 join, 문자열이면 그대로 사용
    field_content = samples[field_name]
    if isinstance(field_content, list):
        text = "\n\n".join(field_content)
    else:
        text = field_content
    test_ids = tokenizer(text, return_tensors='pt').input_ids[0]
    test_ids_batch = []
    nsamples = test_ids.numel() // seq_len
    for i in range(nsamples):
        batch = test_ids[(i * seq_len):((i + 1) * seq_len)]
        test_ids_batch.append(batch)
    test_ids_batch = torch.stack(test_ids_batch)
    return IndexDataset(tensors=test_ids_batch)

def get_loaders_chunk(name, chunk, size, tokenizer, seq_len=2048, batch_size=8):
    if 'wikitext2' in name:
        _, test_data = get_wikitext2(seq_len, tokenizer)
        num_samples = len(test_data)
        assert(size < 1.0)
        num_eval = int(size * num_samples)
        start = chunk * num_eval
        end = min(start + num_eval, num_samples)
        print(f"Start {start} to {end}")
        test_dataset = process_data(test_data[start:end], tokenizer, seq_len, 'text')
    elif 'ptb' in name:
        _, test_data = get_ptb(seq_len, tokenizer)
        test_dataset = process_data(test_data, tokenizer, seq_len, 'sentence')
    elif 'gsm8k' in name:
        test_data = load_dataset('gsm8k', 'main', split='test')
        num_samples = len(test_data)
        assert(size < 1.0)
        num_eval = int(size * num_samples)
        start = chunk * num_eval
        end = min(start + num_eval, num_samples)
        print(f"Start {start} to {end}")
        # GSM8K에서는 'question' 필드를 사용
        test_dataset = process_data(test_data[start:end], tokenizer, seq_len, 'question')
    elif 'xsum' in name:
        test_data = load_dataset('xsum', split='test')
        num_samples = len(test_data)
        assert(size < 1.0)
        num_eval = int(size * num_samples)
        start = chunk * num_eval
        end = min(start + num_eval, num_samples)
        print(f"Start {start} to {end}")
        test_dataset = process_data(test_data[start:end], tokenizer, seq_len, 'document')
    elif 'cnn_dailymail' in name:
        test_data = load_dataset('cnn_dailymail', '3.0.0', split='test')
        num_samples = len(test_data)
        assert(size < 1.0)
        num_eval = int(size * num_samples)
        start = chunk * num_eval
        end = min(start + num_eval, num_samples)
        print(f"Start {start} to {end}")
        test_dataset = process_data(test_data[start:end], tokenizer, seq_len, 'article')
    else:
        dataset = load_dataset(name, split='test')
        num_samples = len(dataset)
        assert(size < 1.0)
        num_eval = int(size * num_samples)
        start = chunk * num_eval
        end = min(start + num_eval, num_samples)
        print(f"Start {start} to {end}")
        test_dataset = process_data(dataset[start:end], tokenizer, seq_len, 'text')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return None, test_loader

def get_loaders_end(name, tokenizer, chunk=1, size=0.2, seq_len=2048, batch_size=8):
    if 'wikitext2' in name:
        _, test_data = get_wikitext2(seq_len, tokenizer)
        num_samples = len(test_data)
        assert(size < 1.0)
        num_eval = int(size * num_samples)
        start = chunk * num_eval
        print(f"Start {start} to {len(test_data)}")
        test_dataset = process_data(test_data[start:], tokenizer, seq_len, 'text')
    elif 'ptb' in name:
        _, test_data = get_ptb(seq_len, tokenizer)
        test_dataset = process_data(test_data, tokenizer, seq_len, 'sentence')
    elif 'gsm8k' in name:
        test_data = load_dataset('gsm8k', 'main', split='test')
        num_samples = len(test_data)
        assert(size < 1.0)
        num_eval = int(size * num_samples)
        start = chunk * num_eval
        print(f"Start {start} to {len(test_data)}")
        # GSM8K에서는 'question' 필드를 사용
        test_dataset = process_data(test_data[start:], tokenizer, seq_len, 'question')
    elif 'xsum' in name:
        test_data = load_dataset('xsum', split='test')
        num_samples = len(test_data)
        assert(size < 1.0)
        num_eval = int(size * num_samples)
        start = chunk * num_eval
        print(f"Start {start} to {len(test_data)}")
        test_dataset = process_data(test_data[start:], tokenizer, seq_len, 'document')
    elif 'cnn_dailymail' in name:
        test_data = load_dataset('cnn_dailymail', '3.0.0', split='test')
        num_samples = len(test_data)
        assert(size < 1.0)
        num_eval = int(size * num_samples)
        start = chunk * num_eval
        print(f"Start {start} to {len(test_data)}")
        test_dataset = process_data(test_data[start:], tokenizer, seq_len, 'article')
    else:
        dataset = load_dataset(name, split='test')
        num_samples = len(dataset)
        assert(size < 1.0)
        num_eval = int(size * num_samples)
        start = chunk * num_eval
        print(f"Start {start} to {len(dataset)}")
        test_dataset = process_data(dataset[start:], tokenizer, seq_len, 'text')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return None, test_loader
