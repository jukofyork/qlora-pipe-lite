from tqdm import tqdm
import datasets
import os
import sys
import torch

from constants import DATASET_MAP_BATCH_SIZE, DEFAULT_EVAL_FRACTION
from utils.utils import main_process_first, log

class DatasetBuilder:
    """
    Builds fixed-length sequence datasets (train/eval) from one or more source files.

    Behavior:
    - Loads text/json/jsonl/parquet sources via HF datasets
    - Tokenizes with optional document suffix per source
    - Optionally randomizes control classes when control_class=0
    - Shuffles per-source, optionally again after concatenation, and shuffles the
      train/eval split (deterministic with seed=42)
    - Slices into fixed-length sequences with optional prefix and token masking
    - Splits into train/eval, caches to run_dir/data, and returns torch-formatted datasets

    Note:
    - The returned train and eval datasets are pre-shuffled. The training dataloader
      does not reshuffle; it expects datasets to already be in random order.

    Parameters:
        config    : Dict containing sequence_len, datasets list, and options
        tokenizer : HuggingFace tokenizer
        run_dir   : Run directory used to cache processed datasets

    Public API:
        build() -> (train_dataset, eval_dataset)
    """

    def __init__(self, config, tokenizer, run_dir):
        self.config = config
        self.tokenizer = tokenizer
        self.run_dir = run_dir

        if 'sequence_len' not in config:
            raise ValueError('Need to specify a sequence_len')

        self.sequence_len = config['sequence_len']
        assert self.sequence_len > 0, "sequence_len must be positive"
        assert self.sequence_len % 64 == 0, f"sequence_len ({self.sequence_len}) must be multiple of 64"

        if 'datasets' not in config:
            raise ValueError('Need to specify at least one dataset')

        self.eval_fraction = config.get('eval_fraction', DEFAULT_EVAL_FRACTION)
        assert 0 < self.eval_fraction < 1, "eval_fraction must be between 0 and 1"

        self.max_sequences = config.get('max_sequences', sys.maxsize)
        self.drop_tails = config.get('drop_tails', False)
        self.mix_datasets = config.get('mix_datasets', False)
        self.sequence_prefix = config.get('sequence_prefix', None)  # None --> add BOS if exists
        self.mask_tokens = config.get('mask_tokens', None)  # None --> no masking
        self.document_suffix = config.get('document_suffix', None)  # None --> add EOS if exists
        self.datasets_cfg = config['datasets']

        self.data_dir = os.path.join(run_dir, "data")
        self.train_subdir = os.path.join(self.data_dir, "train")
        self.eval_subdir = os.path.join(self.data_dir, "eval")

    def build(self):
        """
        Build or load cached train/eval datasets.

        Returns:
            (train_dataset, eval_dataset)
        """
        with main_process_first():
            if self._cache_exists():
                train_dataset, eval_dataset = self._load_cached()
            else:
                datasets_list = []
                for dataset_config in self.datasets_cfg:
                    control_class = dataset_config.get('control_class', 1)
                    assert control_class in [-1, 0, 1], f"control_class must be -1, 0, or 1, got {control_class}"
                    max_tokens = dataset_config.get('max_tokens', sys.maxsize)
                    ds = self._load_single_dataset(
                        dataset_path=dataset_config['dataset_path'],
                        control_class=control_class,
                        document_suffix=self.document_suffix,
                        max_tokens=max_tokens
                    )
                    datasets_list.append(ds)

                combined_dataset = datasets.concatenate_datasets(datasets_list)
                if self.mix_datasets:
                    combined_dataset = combined_dataset.shuffle(seed=42)
                combined_dataset.set_format(type='torch')

                sequences = self._slice_into_sequences(
                    combined_dataset,
                    sequence_len=self.sequence_len,
                    max_sequences=self.max_sequences,
                    drop_tails=self.drop_tails,
                    sequence_prefix=self.sequence_prefix,
                    mask_tokens=self.mask_tokens
                )

                split = sequences.train_test_split(test_size=self.eval_fraction, shuffle=True, seed=42)
                train_dataset = split['train']
                eval_dataset = split['test']
                assert len(train_dataset) > 0 and len(eval_dataset) > 0, "Empty train/eval split"

                self._save_cached(train_dataset, eval_dataset)

        log(f'train data size: {len(train_dataset)} sequences ({len(train_dataset) * self.sequence_len} tokens)')
        log(f'eval data size: {len(eval_dataset)} sequences ({len(eval_dataset) * self.sequence_len} tokens)')

        # Always ensure torch-format on all ranks
        train_dataset.set_format(type='torch')
        eval_dataset.set_format(type='torch')
        return train_dataset, eval_dataset

    def _cache_exists(self):
        return os.path.exists(self.train_subdir) and os.path.exists(self.eval_subdir)

    def _load_cached(self):
        train_dataset = datasets.load_from_disk(self.train_subdir)
        eval_dataset = datasets.load_from_disk(self.eval_subdir)
        return train_dataset, eval_dataset

    def _save_cached(self, train_dataset, eval_dataset):
        os.makedirs(self.data_dir, exist_ok=True)
        train_dataset.save_to_disk(self.train_subdir)
        eval_dataset.save_to_disk(self.eval_subdir)

    def _load_single_dataset(self, dataset_path, control_class=1, document_suffix=None, max_tokens=sys.maxsize):
        """
        Load one dataset file/pattern, tokenize, assign control class, shuffle, prune to max_tokens.
        """
        base_dir = os.path.dirname(dataset_path.split("*", 1)[0])
        cache_dir = os.path.join(base_dir, "hf_cache")

        if dataset_path.endswith('.txt'):
            ds = datasets.load_dataset(
                "text",
                data_files=dataset_path,
                sample_by="document",
                split="train",
                cache_dir=cache_dir
            )
        elif dataset_path.endswith('.json') or dataset_path.endswith('.jsonl'):
            ds = datasets.load_dataset(
                "json",
                data_files=dataset_path,
                split="train",
                cache_dir=cache_dir
            )
        elif dataset_path.endswith('.parquet'):
            ds = datasets.load_dataset(
                "parquet",
                data_files=dataset_path,
                split="train",
                cache_dir=cache_dir
            )
        else:
            raise NotImplementedError()

        ds = ds.map(
            lambda x: self._tokenize(x, self.tokenizer, control_class, document_suffix),
            batched=True,
            batch_size=DATASET_MAP_BATCH_SIZE,
            remove_columns=ds.column_names,
            desc='tokenizing',
            num_proc=min((os.cpu_count() or 1), len(ds))
        )

        if control_class == 0:
            ds = self._randomize_control_classes(ds)

        ds = ds.shuffle(seed=42)

        if max_tokens < sys.maxsize:
            total_tokens = 0
            indices_to_keep = []
            for i in range(len(ds)):
                doc_tokens = len(ds[i]['input_ids'])
                if total_tokens + doc_tokens <= max_tokens:
                    total_tokens += doc_tokens
                    indices_to_keep.append(i)
                else:
                    break
            ds = ds.select(indices_to_keep)

        return ds

    @staticmethod
    def _tokenize(batch, tokenizer, control_class=1, document_suffix=None):
        """
        Tokenizes a batch of text and assigns control class to each document.
        Mirrors previous tokenize() behavior.
        """
        result = {'input_ids': []}

        if isinstance(document_suffix, str) and document_suffix != "":
            for text in batch['text']:
                tokens = tokenizer.encode(text + document_suffix, add_special_tokens=False)
                if len(tokens) > 0:
                    result['input_ids'].append(tokens)
        else:
            if document_suffix is None:
                suffix_tokens = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
            elif isinstance(document_suffix, str) and document_suffix == "":
                suffix_tokens = []
            elif isinstance(document_suffix, int):
                suffix_tokens = [document_suffix]
            elif isinstance(document_suffix, list):
                suffix_tokens = document_suffix
            else:
                raise ValueError(f"Invalid document_suffix type: {type(document_suffix)}. Must be None, str, int, or list of ints.")

            for text in batch['text']:
                tokens = tokenizer.encode(text, add_special_tokens=False) + suffix_tokens
                if len(tokens) > 0:
                    result['input_ids'].append(tokens)

        result['control_class'] = [control_class] * len(result['input_ids'])
        return result

    @staticmethod
    def _randomize_control_classes(dataset):
        """
        Randomly assign control classes (-1 or +1) deterministically per index.
        """

        def assign_random_classes(batch, indices):
            batch['control_class'] = [1 if ((idx * 1664525 + 1013904223) & 0xFFFFFFFF) >> 31 else -1 for idx in indices]
            return batch

        return dataset.map(assign_random_classes, batched=True, with_indices=True, desc='class0 randomization')

    def _slice_into_sequences(
            self,
            dataset,
            sequence_len,
            max_sequences=sys.maxsize,
            drop_tails=False,
            sequence_prefix=None,
            mask_tokens=None
    ):
        """
        Slice tokenized documents into fixed-length sequences.
        Mirrors previous slice_into_sequences() behavior.
        """

        def initialize_sequence():
            if sequence_prefix is None:
                prefix_tokens = [self.tokenizer.bos_token_id] if self.tokenizer.bos_token_id is not None else []
            elif isinstance(sequence_prefix, str) and sequence_prefix == "":
                prefix_tokens = []
            elif isinstance(sequence_prefix, str) and sequence_prefix != "":
                prefix_tokens = self.tokenizer.encode(sequence_prefix, add_special_tokens=False)
            elif isinstance(sequence_prefix, int):
                prefix_tokens = [sequence_prefix]
            elif isinstance(sequence_prefix, list):
                prefix_tokens = sequence_prefix
            else:
                raise ValueError(f"Invalid sequence_prefix type: {type(sequence_prefix)}. Must be None, str, int, or list of ints.")
            return [(token_id, 0) for token_id in prefix_tokens]

        all_sequences = []
        sequence_count = 0
        sequence_tokens = initialize_sequence()

        for item in tqdm(dataset, desc="Creating sequences"):
            if sequence_count >= max_sequences:
                break

            tokens = item['input_ids'].tolist()
            document_control_class = item['control_class'].item()
            assert len(tokens) > 0, 'Empty tokens list'

            idx = 0
            if self.tokenizer.bos_token_id is not None and tokens[0] == self.tokenizer.bos_token_id:
                idx += 1

            while idx < len(tokens):
                if sequence_count >= max_sequences:
                    break
                need = sequence_len - len(sequence_tokens)
                taken = tokens[idx: idx + need]
                idx += len(taken)
                sequence_tokens.extend([(token, document_control_class) for token in taken])

                if len(sequence_tokens) >= sequence_len:
                    assert len(sequence_tokens) == sequence_len
                    input_ids, control_classes = zip(*sequence_tokens)
                    input_ids = torch.as_tensor(input_ids, dtype=torch.long)
                    control_classes = torch.as_tensor(control_classes, dtype=torch.int8)
                    attention_mask = torch.ones(sequence_len, dtype=torch.int8)
                    labels = input_ids.clone()

                    if mask_tokens:
                        token_mask = None
                        if mask_tokens is True:
                            if hasattr(self.tokenizer, 'all_special_ids') and self.tokenizer.all_special_ids:
                                token_mask = torch.isin(
                                    input_ids,
                                    torch.tensor(self.tokenizer.all_special_ids, device=input_ids.device)
                                )
                        elif isinstance(mask_tokens, int):
                            token_mask = input_ids == mask_tokens
                        elif isinstance(mask_tokens, list):
                            token_mask = torch.isin(input_ids, torch.tensor(mask_tokens, device=input_ids.device))
                        else:
                            raise ValueError(f"Invalid mask_tokens type: {type(mask_tokens)}. Must be bool, int, or list of ints.")
                        if token_mask is not None:
                            control_classes[token_mask] = 0
                            labels[token_mask] = -100

                    all_sequences.append({
                        'input_ids': input_ids,
                        'attention_mask': attention_mask,
                        'control_classes': control_classes,
                        'labels': labels
                    })
                    sequence_count += 1
                    sequence_tokens = initialize_sequence()
                    if drop_tails:
                        idx = len(tokens)

        result_dataset = datasets.Dataset.from_dict({
            'input_ids': [seq['input_ids'] for seq in all_sequences],
            'attention_mask': [seq['attention_mask'] for seq in all_sequences],
            'control_classes': [seq['control_classes'] for seq in all_sequences],
            'labels': [seq['labels'] for seq in all_sequences]
        })
        result_dataset.set_format(type='torch')
        return result_dataset