from tqdm import tqdm
import datasets
import os
import os.path
import torch

from utils import *

# Dataset preprocessing batch sizes
TOKENIZE_BATCH_SIZE = 10
SPLIT_BATCH_SIZE = 100

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
            taken = tokens[idx: idx + need]
            idx += len(taken)
            sequence_tokens.extend(taken)
            if len(sequence_tokens) >= sequence_len:
                assert len(sequence_tokens) == sequence_len
                yield sequence_tokens
                # Reset sequence_tokens with BOS token if it exists
                sequence_tokens = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
    # Discard anything remaining to ensure all are exactly sequence_len in length...

def load_single_dataset(dataset_path, tokenizer, sequence_len):
    base_dir = os.path.dirname(dataset_path.split("*", 1)[0])
    cache_dir = os.path.join(base_dir, "hf_cache")

    dataset = datasets.load_dataset(
        "text",
        data_files=dataset_path,
        sample_by="document",
        cache_dir=cache_dir,
    )["train"]

    dataset.set_format(type='torch')

    num_proc = min(os.cpu_count(), len(dataset))

    dataset = dataset.map(
        lambda x: tokenizer(x['text']),
        batched=True,
        batch_size=TOKENIZE_BATCH_SIZE,
        remove_columns=dataset.column_names,
        desc='tokenizing',
        num_proc=num_proc,
    )
    dataset = dataset.map(
        lambda x: {'input_ids': [torch.as_tensor(seq) for seq in yield_sequences_from_token_batch(tokenizer, x['input_ids'], sequence_len)]},
        batched=True,
        batch_size=SPLIT_BATCH_SIZE,
        remove_columns=dataset.column_names,
        desc='splitting and converting to tensors',
        num_proc=num_proc,
    )

    return dataset

def load_datasets(config, tokenizer):
    if 'sequence_len' not in config:
        raise ValueError('Need to specify a sequence_len')
    sequence_len = config['sequence_len']
    assert sequence_len > 0
    # A100 wants sequence lengths to be multiples of 64, other cards are efficient with smaller, so just do 64
    assert sequence_len % 64 == 0, f"sequence_len ({sequence_len}) must be a multiple of 64 for optimal GPU performance"

    if 'datasets' not in config:
        raise ValueError('Need to specify at least one dataset')

    eval_fraction = config.get('eval_fraction', 0.01)
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