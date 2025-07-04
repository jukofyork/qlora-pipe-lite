from tqdm import tqdm
import datasets
import hashlib
import json
import os
import os.path
import sys
import torch

from constants import DATASET_MAP_BATCH_SIZE, DEFAULT_EVAL_FRACTION
from utils.utils import is_main_process, zero_first, log

def tokenize_and_add_eos(batch, tokenizer):
    result = tokenizer(batch['text'])
    # Add EOS token to the end of each text field if missing
    for i, tokens in enumerate(result['input_ids']):
        if tokens[-1] != tokenizer.eos_token_id:
            result['input_ids'][i] = tokens + [tokenizer.eos_token_id]
    return result

def slice_into_sequences(dataset, tokenizer, sequence_len, cache_dir, sample_weight=1.0, max_sequences=sys.maxsize):
    # Use the dataset's built-in fingerprint - it's already computed and deterministic
    cache_key = (
        f"{dataset._fingerprint}_{sequence_len}_"
        f"{tokenizer.bos_token_id}_{tokenizer.eos_token_id}_{tokenizer.pad_token_id}_"
        f"{sample_weight}_{max_sequences}"
    )
    cache_path = os.path.join(cache_dir, f"sliced_sequences_{cache_key}")

    # Try to load from cache first
    if os.path.exists(cache_path):
        log(f"Loading sliced sequences from cache: {cache_path}")
        cached_dataset = datasets.load_from_disk(cache_path)
        cached_dataset.set_format(type='torch')
        return cached_dataset

    all_sequences = []
    sequence_count = 0

    # Initialize sequence_tokens with BOS token if it exists
    sequence_tokens = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []

    # Process dataset item by item (streaming)
    for item in tqdm(dataset, desc="Creating sequences"):
        if sequence_count >= max_sequences:
            break

        tokens = item['input_ids'].tolist()
        assert len(tokens) > 0, 'Empty tokens list'
        idx = 0
        # Skip the auto-generated BOS token if present
        if tokenizer.bos_token_id is not None and tokens[0] == tokenizer.bos_token_id:
            idx += 1
        while idx < len(tokens):
            if sequence_count >= max_sequences:
                break

            # Calculate how many tokens are needed to fill the sequence
            need = sequence_len - len(sequence_tokens)
            taken = tokens[idx: idx + need]
            idx += len(taken)
            sequence_tokens.extend(taken)
            if len(sequence_tokens) >= sequence_len:
                assert len(sequence_tokens) == sequence_len

                # Create the triplet
                input_ids = torch.as_tensor(sequence_tokens, dtype=torch.long)
                attention_mask = torch.ones(sequence_len, dtype=torch.long)
                labels = input_ids.clone()
                sample_weights = torch.full((sequence_len,), sample_weight, dtype=torch.float32)

                # Mask out BOS tokens in labels (if they exist)
                if tokenizer.bos_token_id is not None:
                    labels[labels == tokenizer.bos_token_id] = -100

                # Mask out EOS tokens in labels
                if tokenizer.eos_token_id is not None:
                    labels[labels == tokenizer.eos_token_id] = -100

                # Mask out pad tokens in both labels AND attention_mask
                if tokenizer.pad_token_id is not None:
                    pad_mask = input_ids == tokenizer.pad_token_id
                    labels[pad_mask] = -100
                    attention_mask[pad_mask] = 0

                all_sequences.append({
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'labels': labels,
                    'sample_weights': sample_weights
                })

                sequence_count += 1  # Increment counter

                # Reset sequence_tokens with BOS token if it exists
                sequence_tokens = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []

    # Create the dataset
    result_dataset = datasets.Dataset.from_dict({
        'input_ids': [seq['input_ids'] for seq in all_sequences],
        'attention_mask': [seq['attention_mask'] for seq in all_sequences],
        'labels': [seq['labels'] for seq in all_sequences],
        'sample_weights': [seq['sample_weights'] for seq in all_sequences]
    })
    result_dataset.set_format(type='torch')

    # Save to cache
    os.makedirs(cache_dir, exist_ok=True)
    result_dataset.save_to_disk(cache_path)

    return result_dataset

def load_single_dataset(dataset_path, tokenizer, sequence_len, sample_weight=1.0, max_sequences=sys.maxsize):
    base_dir = os.path.dirname(dataset_path.split("*", 1)[0])
    cache_dir = os.path.join(base_dir, "hf_cache")

    if dataset_path.endswith('.txt'):
        dataset = datasets.load_dataset(
            "text",
            data_files=dataset_path,
            sample_by="document",
            split="train",
            cache_dir=cache_dir,
        )
    elif dataset_path.endswith('.json') or dataset_path.endswith('.jsonl'):
        dataset = datasets.load_dataset(
            "json",
            data_files=dataset_path,
            split="train",
            cache_dir=cache_dir,
        )
    elif dataset_path.endswith('.parquet'):
        dataset = datasets.load_dataset(
            "parquet",
            data_files=dataset_path,
            split="train",
            cache_dir=cache_dir,
        )
    else:
        raise NotImplementedError()

    dataset = dataset.map(
        lambda x: tokenize_and_add_eos(x, tokenizer),
        batched=True,
        batch_size=DATASET_MAP_BATCH_SIZE,
        remove_columns=dataset.column_names,
        desc='tokenizing',
        num_proc=min(os.cpu_count(), len(dataset)),
    )

    dataset = dataset.shuffle(seed=42)

    # Set torch format after tokenization when only token data remains
    dataset.set_format(type='torch')

    return slice_into_sequences(dataset, tokenizer, sequence_len, cache_dir, sample_weight, max_sequences)

def load_datasets(config, tokenizer):
    if 'sequence_len' not in config:
        raise ValueError('Need to specify a sequence_len')
    sequence_len = config['sequence_len']
    assert sequence_len > 0, "sequence_len must be positive"
    # A100 wants sequence lengths to be multiples of 64, other cards are efficient with smaller, so just do 64
    assert sequence_len % 64 == 0, f"sequence_len ({sequence_len}) must be multiple of 64"

    if 'datasets' not in config:
        raise ValueError('Need to specify at least one dataset')

    eval_fraction = config.get('eval_fraction', DEFAULT_EVAL_FRACTION)
    assert 0 < eval_fraction < 1, "eval_fraction must be between 0 and 1"

    with zero_first(is_main_process()):
        datasets_list = []
        for dataset_config in config['datasets']:
            max_sequences = dataset_config.get('max_sequences', sys.maxsize)
            assert max_sequences > 0, "max_sequences must be positive"
            sample_weight = dataset_config.get('sample_weight', 1.0)
            assert sample_weight != 0, "sample_weight cannot be zero"
            dataset = load_single_dataset(
                dataset_config['dataset_path'],
                tokenizer,
                sequence_len,
                sample_weight,
                max_sequences
            )
            datasets_list.append(dataset)
        combined_dataset = datasets.concatenate_datasets(datasets_list)
        split_datasets = combined_dataset.train_test_split(test_size=eval_fraction, shuffle=True, seed=42)
        train_dataset = split_datasets['train']
        eval_dataset = split_datasets['test']

    log(f'train data size: {len(train_dataset)} sequences ({len(train_dataset) * sequence_len} tokens)')
    log(f'eval data size: {len(eval_dataset)} sequences ({len(eval_dataset) * sequence_len} tokens)')

    return train_dataset, eval_dataset