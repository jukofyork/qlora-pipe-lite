from tqdm import tqdm
import datasets
import glob
import hashlib
import json
import os
import os.path
import sys
import torch

from constants import DATASET_MAP_BATCH_SIZE, DEFAULT_EVAL_FRACTION
from utils.utils import main_process_first, log, log_all

def tokenize_and_add_separator(batch, tokenizer, separator=None):
    if separator is None:
        # Default behavior: tokenize first, then add EOS token
        result = tokenizer(batch['text'])
        # Add EOS token to the end of each text field if missing
        for i, tokens in enumerate(result['input_ids']):
            if len(tokens) == 0:
                # Handle empty tokenization - just add EOS token (should not really happen)
                result['input_ids'][i] = [tokenizer.eos_token_id]
            elif tokens[-1] != tokenizer.eos_token_id:
                result['input_ids'][i] = tokens + [tokenizer.eos_token_id]
        return result
    elif separator == "":
        # Empty separator: just tokenize without adding anything
        return tokenizer(batch['text'])
    else:
        # Custom separator: add to text before tokenizing
        modified_texts = [text + separator for text in batch['text']]
        return tokenizer(modified_texts)

def create_dataset_cache_key(dataset_path, tokenizer, sequence_len, control_class, max_sequences, separator, drop_tails):
    """Create a deterministic cache key from input parameters."""

    # Get all matching files (handles both wildcards and literal paths)
    matching_files = sorted(glob.glob(dataset_path))  # Sort for deterministic order
    if not matching_files:
        raise FileNotFoundError(f"No files found matching pattern: {dataset_path}")

    # Get combined file info from all matching files
    file_infos = []
    for file_path in matching_files:
        file_stat = os.stat(file_path)
        file_infos.append(f"{file_path}_{file_stat.st_size}_{file_stat.st_mtime}")
    file_info = "_".join(file_infos)

    # Tokenizer info
    tokenizer_info = f"{tokenizer.vocab_size}_{tokenizer.bos_token_id}_{tokenizer.eos_token_id}_{tokenizer.pad_token_id}"

    # Processing parameters
    params = f"{sequence_len}_{control_class}_{max_sequences}_{separator}_{drop_tails}"

    # Create hash of all components
    cache_string = f"{file_info}_{tokenizer_info}_{params}"
    cache_hash = hashlib.md5(cache_string.encode()).hexdigest()

    return cache_hash

def slice_into_sequences(
        dataset,
        tokenizer,
        sequence_len,
        cache_dir,
        control_class=1,
        max_sequences=sys.maxsize,
        separator=None,
        drop_tails=False
):
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

                # Create the data tensors for this sequence
                input_ids = torch.as_tensor(sequence_tokens, dtype=torch.long)
                attention_mask = torch.ones(sequence_len, dtype=torch.int8)
                labels = input_ids.clone()

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
                    'control_class': control_class,
                    'labels': labels
                })

                sequence_count += 1

                # Reset sequence_tokens with BOS token if it exists
                sequence_tokens = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []

                # If asked to, drop the remaining tokens of this document to ensure each sequence starts with a fresh document
                if drop_tails:
                    idx = len(tokens)

    # Create the dataset
    result_dataset = datasets.Dataset.from_dict({
        'input_ids': [seq['input_ids'] for seq in all_sequences],
        'attention_mask': [seq['attention_mask'] for seq in all_sequences],
        'control_class': [seq['control_class'] for seq in all_sequences],
        'labels': [seq['labels'] for seq in all_sequences]
    })
    result_dataset.set_format(type='torch')

    return result_dataset

def load_single_dataset(
        dataset_path,
        tokenizer,
        sequence_len,
        control_class=1,
        max_sequences=sys.maxsize,
        separator=None,
        drop_tails=False
):
    base_dir = os.path.dirname(dataset_path.split("*", 1)[0])
    cache_dir = os.path.join(base_dir, "hf_cache")

    # Create cache key from input parameters and check cache first
    cache_key = create_dataset_cache_key(
        dataset_path, tokenizer, sequence_len, control_class, max_sequences, separator, drop_tails
    )
    cache_path = os.path.join(cache_dir, f"processed_dataset_{cache_key}")

    # Load from cache if available
    if os.path.exists(cache_path):
        # log_all(f"Loading processed dataset from cache: {cache_path}")
        cached_dataset = datasets.load_from_disk(cache_path)
        cached_dataset.set_format(type='torch')
        return cached_dataset

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
        lambda x: tokenize_and_add_separator(x, tokenizer, separator),
        batched=True,
        batch_size=DATASET_MAP_BATCH_SIZE,
        remove_columns=dataset.column_names,
        desc='tokenizing',
        num_proc=min(os.cpu_count(), len(dataset)),
    )

    dataset = dataset.shuffle(seed=42)

    # Set torch format after tokenization when only token data remains
    dataset.set_format(type='torch')

    result_dataset = slice_into_sequences(
        dataset,
        tokenizer,
        sequence_len,
        cache_dir,
        control_class,
        max_sequences,
        separator,
        drop_tails
    )

    # Save to cache
    os.makedirs(cache_dir, exist_ok=True)
    result_dataset.save_to_disk(cache_path)

    return result_dataset

def load_datasets(config, tokenizer, run_dir):
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

    data_dir = os.path.join(run_dir, "data")
    train_subdir = os.path.join(data_dir, "train")
    eval_subdir = os.path.join(data_dir, "eval")

    # Only main process loads/processes data and saves final datasets, others wait then load
    with main_process_first():
        # Check if final datasets already exist
        if os.path.exists(data_dir):
            train_dataset = datasets.load_from_disk(train_subdir)
            eval_dataset = datasets.load_from_disk(eval_subdir)
            train_dataset.set_format(type='torch')
            eval_dataset.set_format(type='torch')
        else:
            datasets_list = []
            for dataset_config in config['datasets']:
                max_sequences = dataset_config.get('max_sequences', sys.maxsize)
                assert max_sequences > 0, f"max_sequences must be positive, got {max_sequences}"
                control_class = dataset_config.get('control_class', 1)
                assert control_class in [-1, 1], f"control_class must be -1 or 1, got {control_class}"
                separator = dataset_config.get('separator', None)  # None --> tokenize first, then add EOS token
                drop_tails = dataset_config.get('drop_tails', False)
                dataset = load_single_dataset(
                    dataset_config['dataset_path'],
                    tokenizer,
                    sequence_len,
                    control_class,
                    max_sequences,
                    separator,
                    drop_tails
                )
                datasets_list.append(dataset)
            combined_dataset = datasets.concatenate_datasets(datasets_list)
            split_datasets = combined_dataset.train_test_split(test_size=eval_fraction, shuffle=True, seed=42)
            train_dataset = split_datasets['train']
            eval_dataset = split_datasets['test']

            # Save final datasets to run directory
            os.makedirs(data_dir, exist_ok=True)
            train_dataset.save_to_disk(train_subdir)
            eval_dataset.save_to_disk(eval_subdir)

    log(f'train data size: {len(train_dataset)} sequences ({len(train_dataset) * sequence_len} tokens)')
    log(f'eval data size: {len(eval_dataset)} sequences ({len(eval_dataset) * sequence_len} tokens)')

    return train_dataset, eval_dataset