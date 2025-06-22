from tqdm import tqdm
import datasets
import os
import os.path
import psutil
import sys
import torch

from utils import is_main_process, zero_first, log

TOKENIZE_BATCH_SIZE = 10

def tokenize_with_eos(batch, tokenizer):
    result = tokenizer(batch['text'])
    # Add EOS token to each text field if missing
    for i, tokens in enumerate(result['input_ids']):
        if tokens[-1] != tokenizer.eos_token_id:
            result['input_ids'][i] = tokens + [tokenizer.eos_token_id]
    return result

def slice_into_sequences(dataset, tokenizer, sequence_len):
    all_sequences = []
    # Initialize sequence_tokens with BOS token if it exists
    sequence_tokens = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []

    # Process dataset item by item (streaming)
    for item in tqdm(dataset, desc="Creating sequences"):
        tokens = item['input_ids'].tolist()
        assert len(tokens) > 0, 'Empty tokens list'
        # EOS already added in tokenize_with_eos, so skip that check
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
                all_sequences.append(torch.as_tensor(sequence_tokens))
                # Reset sequence_tokens with BOS token if it exists
                sequence_tokens = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []

                # Print memory usage every 100 sequences
                if len(all_sequences) % 100 == 0:
                    seq_tokens_size = sys.getsizeof(sequence_tokens) / (1024 ** 2)  # MB
                    all_seqs_size = sys.getsizeof(all_sequences) / (1024 ** 3)  # GB
                    process_memory = process.memory_info().rss / (1024 ** 3)  # GB
                    log(f"Sequences {len(all_sequences)}: sequence_tokens={seq_tokens_size:.1f}MB, all_sequences={all_seqs_size:.1f}GB, process_memory={process_memory:.1f}GB")

    # Discard the final partial sequence to ensure all are exactly sequence_len in length...
    return datasets.Dataset.from_dict({'input_ids': all_sequences})

def load_single_dataset(dataset_path, tokenizer, sequence_len):
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

    dataset.set_format(type='torch')

    num_proc = min(os.cpu_count(), len(dataset))

    dataset = dataset.map(
        lambda x: tokenize_with_eos(x, tokenizer),
        batched=True,
        batch_size=TOKENIZE_BATCH_SIZE,
        remove_columns=dataset.column_names,
        desc='tokenizing',
        num_proc=num_proc,
    )

    return slice_into_sequences(dataset, tokenizer, sequence_len)

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

    log(f'train data size: {len(train_dataset)} sequences ({len(train_dataset) * sequence_len} tokens)')
    log(f'eval data size: {len(eval_dataset)} sequences ({len(eval_dataset) * sequence_len} tokens)')

    return train_dataset, eval_dataset