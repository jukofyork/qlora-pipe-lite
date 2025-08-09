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

def slice_into_sequences(
        dataset,
        tokenizer,
        sequence_len,
        max_sequences=sys.maxsize,
        drop_tails=False
):
    all_sequences = []
    sequence_count = 0

    # Initialize sequence with BOS token if it exists
    sequence_tokens = [(tokenizer.bos_token_id, 0)] if tokenizer.bos_token_id is not None else []

    # Process dataset item by item (streaming)
    for item in tqdm(dataset, desc="Creating sequences"):
        if sequence_count >= max_sequences:
            break

        tokens = item['input_ids'].tolist()
        document_control_class = item['control_class'].item()  # Extract scalar from tensor
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

            # Assign document's control class to each token taken
            sequence_tokens.extend([(token, document_control_class) for token in taken])

            if len(sequence_tokens) >= sequence_len:
                assert len(sequence_tokens) == sequence_len

                # Unzip tuples into separate lists
                input_ids, control_classes = zip(*sequence_tokens)

                # Convert to torch tensors
                input_ids = torch.as_tensor(input_ids, dtype=torch.long)
                control_classes = torch.as_tensor(control_classes, dtype=torch.int8)

                # Create the other tensors for this sequence
                attention_mask = torch.ones(sequence_len, dtype=torch.int8)
                labels = input_ids.clone()

                # Mask out the class and label for all special tokens
                if hasattr(tokenizer, 'all_special_ids') and tokenizer.all_special_ids:
                    special_token_mask = torch.isin(input_ids, torch.tensor(tokenizer.all_special_ids, device=input_ids.device))
                    control_classes[special_token_mask] = 0
                    labels[special_token_mask] = -100

                all_sequences.append({
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'control_classes': control_classes,
                    'labels': labels
                })

                sequence_count += 1

                # Reset sequence with BOS token if it exists
                sequence_tokens = [(tokenizer.bos_token_id, 0)] if tokenizer.bos_token_id is not None else []

                # If asked to, drop the remaining tokens of this document to ensure each sequence starts with a fresh document
                if drop_tails:
                    idx = len(tokens)

    # Create the dataset
    result_dataset = datasets.Dataset.from_dict({
        'input_ids': [seq['input_ids'] for seq in all_sequences],
        'attention_mask': [seq['attention_mask'] for seq in all_sequences],
        'control_classes': [seq['control_classes'] for seq in all_sequences],
        'labels': [seq['labels'] for seq in all_sequences]
    })
    result_dataset.set_format(type='torch')

    return result_dataset

def tokenize(batch, tokenizer, separator=None, control_class=1):
    """
    Tokenizes a batch of text and assigns control class to each document.

    Args:
        batch: Dict with 'text' field containing list of text strings
        tokenizer: HuggingFace tokenizer
        separator: Optional separator to add to text before tokenizing
        control_class: Control class value to assign to each document

    Returns:
        Dict with 'input_ids' (lists of token IDs) and 'control_class' (scalar per document)
    """
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
    elif separator == "":
        # Empty separator: just tokenize without adding anything
        result = tokenizer(batch['text'])
    else:
        # Custom separator: add to text before tokenizing
        modified_texts = [text + separator for text in batch['text']]
        result = tokenizer(modified_texts)

    # Add control_class field (one scalar per document)
    result['control_class'] = [control_class] * len(batch['text'])

    return result

def load_single_dataset(
        dataset_path,
        tokenizer,
        separator=None,
        control_class=1
):
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
        lambda x: tokenize(x, tokenizer, separator, control_class),
        batched=True,
        batch_size=DATASET_MAP_BATCH_SIZE,
        remove_columns=dataset.column_names,
        desc='tokenizing',
        num_proc=min(os.cpu_count(), len(dataset)),
    )

    return dataset

def load_datasets(config, tokenizer, run_dir):
    if 'sequence_len' not in config:
        raise ValueError('Need to specify a sequence_len')
    sequence_len = config['sequence_len']
    assert sequence_len > 0, "sequence_len must be positive"
    # A100 wants sequence lengths to be multiples of 64, other cards are efficient with smaller, so just do 64
    assert sequence_len % 64 == 0, f"sequence_len ({sequence_len}) must be multiple of 64"

    max_sequences = config.get('max_sequences', sys.maxsize)
    drop_tails = config.get('drop_tails', False)

    eval_fraction = config.get('eval_fraction', DEFAULT_EVAL_FRACTION)
    assert 0 < eval_fraction < 1, "eval_fraction must be between 0 and 1"

    if 'datasets' not in config:
        raise ValueError('Need to specify at least one dataset')

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
                control_class = dataset_config.get('control_class', 1)
                assert control_class in [-1, 1], f"control_class must be -1 or 1, got {control_class}"
                separator = dataset_config.get('separator', None)  # None --> tokenize first, then add EOS token

                dataset = load_single_dataset(
                    dataset_config['dataset_path'],
                    tokenizer,
                    separator,
                    control_class
                )
                datasets_list.append(dataset)

            combined_dataset = datasets.concatenate_datasets(datasets_list)

            combined_dataset = combined_dataset.shuffle(seed=42)

            combined_dataset.set_format(type='torch')

            combined_dataset = slice_into_sequences(
                combined_dataset,
                tokenizer,
                sequence_len,
                max_sequences,
                drop_tails
            )

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