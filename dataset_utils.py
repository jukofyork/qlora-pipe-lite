import os
import os.path
import sys

import torch
import datasets
from tqdm import tqdm
import yaml

from utils import *


def yield_sequences_from_token_batch(tokenizer, token_batch, sequence_len):
    # Initialize sequence_tokens with BOS token if it exists
    sequence_tokens = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
    for tokens in tqdm(token_batch):
        tokens = tokens.tolist()
        assert len(tokens) > 0, 'Empty tokens list'
        if tokens[-1] != tokenizer.eos_token_id:
            tokens.append(tokenizer.eos_token_id)
        idx = 0
        # Skip the auto-generated BOS token if present
        if tokenizer.bos_token_id is not None and tokens[0] == tokenizer.bos_token_id:
            idx += 1
        while idx < len(tokens):
            # Calculate how many tokens are needed to fill the sequence
            need = sequence_len - len(sequence_tokens)
            taken = tokens[idx : idx + need]
            idx += len(taken)
            sequence_tokens.extend(taken)
            if len(sequence_tokens) >= sequence_len:
                assert len(sequence_tokens) == sequence_len
                yield sequence_tokens
                # Reset sequence_tokens with BOS token if it exists
                sequence_tokens = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
    # Discard anything remaining to ensure all are exactly sequence_len in length...


def load_single_dataset(dataset_path, tokenizer, sequence_len):
    base_dir   = os.path.dirname(dataset_path.split("*", 1)[0])
    cache_dir  = os.path.join(base_dir, "hf_cache")

    dataset = datasets.load_dataset(
        "text",
        data_files=dataset_path,
        sample_by="document",
        cache_dir=cache_dir,
    )["train"]

    dataset.set_format(type='torch')
    
    NUM_PROC = max(os.cpu_count() // 2, 1)

    dataset = dataset.map(
        lambda x: tokenizer(x['text']),
        batched=True,
        batch_size=10,
        remove_columns=dataset.column_names,
        desc='tokenizing',
        num_proc=NUM_PROC,
    )
    dataset = dataset.map(
        lambda x: {'input_ids': list(yield_sequences_from_token_batch(tokenizer, x['input_ids'], sequence_len))},
        batched=True,
        batch_size=100,
        remove_columns=dataset.column_names,
        desc='splitting',
        num_proc=NUM_PROC,
    )
    dataset = dataset.map(
        lambda x: {'attention_mask': torch.ones_like(x['input_ids']), 'labels': x['input_ids']},
        desc='adding attention_mask and labels',
    )
    dataset = dataset.map(
        lambda x: {'length': len(x['input_ids'])},
        desc='adding length field'
    )

    return dataset


def load_datasets(config, tokenizer, sequence_len, eval_fraction):
    if 'datasets' not in config:
        raise ValueError('Need to specify at least one dataset')
    assert sequence_len > 0
    assert 0 < eval_fraction < 1

    with zero_first(is_main_process()):
        datasets_list = []
        for dataset_config in config['datasets']:
            dataset = load_single_dataset(
                dataset_config['dataset_path'],
                tokenizer,
                sequence_len
            )
            datasets_list.append(dataset)
        combined_dataset = datasets.concatenate_datasets(datasets_list)
        split_datasets = combined_dataset.train_test_split(test_size=eval_fraction, shuffle=True, seed=42)
        train_dataset = split_datasets['train']
        eval_dataset = split_datasets['test']

    if is_main_process():
        print(f'train_data size: {len(train_dataset)}')
        print(f'eval_data size: {len(eval_dataset)}')

    return train_dataset, eval_dataset