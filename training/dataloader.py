"""
Data loading utilities for distributed, data-parallel training with pipeline/micro-batching.

This module exposes PipelineDataLoader, a thin wrapper around torch.utils.data.DataLoader
that:
- Uses a custom per-rank batch sampler to evenly shard batches across data-parallel ranks
- Materializes full local batches of size (batch_size * gradient_accumulation_steps)
- Splits each local batch into micro-batches of size batch_size for gradient accumulation
- Tracks/resumes iteration state across epochs and supports epoch synchronization
"""
from deepspeed import comm as dist
from torch.utils.data import DataLoader, Sampler
import accelerate
import torch

from utils.utils import log

class PipelineDataLoader:
    """
    Pipeline-aware data loader that yields micro-batches suitable for gradient accumulation.

    The underlying DataLoader fetches local batches of size:
        local_batch_size = batch_size * gradient_accumulation_steps

    Each local batch is split into `gradient_accumulation_steps` micro-batches of size `batch_size`,
    which are yielded one-by-one to the caller. The class keeps an internal notion of `epoch`
    and supports saving/restoring iteration state.

    Parameters:
        dataset: A torch.utils.data.Dataset returning dicts with the keys:
                 'input_ids', 'attention_mask', 'control_classes', 'labels'
        batch_size: Number of samples per micro-batch (per rank)
        gradient_accumulation_steps: Number of micro-batches per optimizer step
        data_parallel_world_size: Total number of data-parallel ranks
        data_parallel_rank: This process's data-parallel rank (0-based)
        shuffle: Whether to shuffle samples each epoch (default: True)

    Attributes:
        data_sampler: The per-rank batch sampler used by the internal DataLoader
        dataloader:   The internal torch.utils.data.DataLoader
        epoch:        Current epoch number (1-based)
        num_batches_pulled: Count of full local batches pulled from the internal DataLoader
        next_micro_batch:   Prefetched next micro-batch to return from __next__()
        recreate_dataloader: Flag set by load_state_dict to recreate the dataloader after 1st pass
        data:         Generator yielding micro-batches
    """

    def __init__(self, dataset, batch_size, gradient_accumulation_steps, data_parallel_world_size, data_parallel_rank, shuffle=True):
        """Initialize the loader and prime the first epoch iterator."""
        assert data_parallel_rank < data_parallel_world_size
        self.dataset = dataset
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.data_parallel_world_size = data_parallel_world_size

        self.data_sampler = self.DistributedBatchSampler(
            dataset=dataset,
            batch_size=self.batch_size,
            batch_size_multiplier=self.gradient_accumulation_steps,
            num_replicas=data_parallel_world_size,
            rank=data_parallel_rank,
            shuffle=shuffle,
        )

        self.epoch = 1
        self.num_batches_pulled = 0
        self.next_micro_batch = None
        self.recreate_dataloader = False
        self._create_dataloader()
        self.data = self._pull_batches_from_dataloader()

    def reset(self):
        """Reset internal iteration state and start from the beginning of the dataset."""
        self.epoch = 1
        self.num_batches_pulled = 0
        self.next_micro_batch = None
        self.data = self._pull_batches_from_dataloader()

    def __iter__(self):
        """Return self as the iterator."""
        return self

    def __len__(self):
        """Total number of micro-batches per epoch."""
        return len(self.data_sampler) * self.gradient_accumulation_steps

    def __next__(self):
        """
        Return the next micro-batch; rotate to the next epoch when the current one completes.

        This method prefetches the next micro-batch to minimize control overhead. When an epoch
        is exhausted, it optionally recreates the dataloader once (after load_state_dict) and
        advances the epoch counter.
        """
        if self.next_micro_batch == None:
            self.next_micro_batch = next(self.data)
        ret = self.next_micro_batch
        try:
            self.next_micro_batch = next(self.data)
        except StopIteration:
            if self.recreate_dataloader:
                self._create_dataloader()
                self.recreate_dataloader = False
            self.data = self._pull_batches_from_dataloader()
            self.num_batches_pulled = 0
            self.next_micro_batch = next(self.data)
            self.epoch += 1
        return ret

    def _pull_batches_from_dataloader(self):
        """
        Internal generator:
        - Iterate full local batches from the internal DataLoader
        - Split each into gradient_accumulation_steps micro-batches
        - Yield micro-batches one-by-one
        """
        for batch in self.dataloader:
            self.num_batches_pulled += 1
            for micro_batch in self.split_batch(batch, self.gradient_accumulation_steps, self.data_parallel_world_size):
                yield micro_batch

    def _create_dataloader(self):
        """(Re)create the underlying DataLoader bound to the current sampler and collate function."""

        def collate_fn(examples):
            """Stack per-sample dicts into batched tensors expected by the model."""
            input_ids = torch.stack([ex['input_ids'] for ex in examples])
            attention_mask = torch.stack([ex['attention_mask'] for ex in examples])
            control_classes = torch.stack([ex['control_classes'] for ex in examples])
            labels = torch.stack([ex['labels'] for ex in examples])
            return ((input_ids, attention_mask, control_classes, labels), None)

        self.dataloader = DataLoader(
            self.dataset,
            pin_memory=True,
            batch_sampler=self.data_sampler,
            collate_fn=collate_fn,
        )

    def state_dict(self):
        """Return a minimal state dict to resume iteration deterministically."""
        return {
            'epoch': self.epoch,
            'num_batches_pulled': self.num_batches_pulled,
        }

    def load_state_dict(self, state_dict):
        """
        Restore iteration state and set up the internal dataloader to skip already-consumed batches.

        Notes:
        - One batch is preloaded by __next__, so we subtract 1 from the persisted counter.
        - accelerate.skip_first_batches is used only for the first pass after restore; afterward,
          the dataloader is recreated so future epochs proceed normally.
        """
        self.epoch = state_dict['epoch']
        # -1 because by preloading the next micro_batch, it's always going to have one more batch
        # pulled than the actual number of batches iterated by the caller.
        self.num_batches_pulled = state_dict['num_batches_pulled'] - 1
        self._create_dataloader()
        self.dataloader = accelerate.skip_first_batches(self.dataloader, self.num_batches_pulled)
        self.data = self._pull_batches_from_dataloader()
        # Recreate the dataloader after the first pass so that it won't skip
        # batches again (we only want it to skip batches the first time).
        self.recreate_dataloader = True

    # Only the first and last stages in the pipeline pull from the dataloader. Parts of the code need
    # to know the epoch, so we synchronize the epoch so the processes that don't use the dataloader
    # know the current epoch.
    def sync_epoch(self):
        """
        Synchronize epoch number across all data-parallel ranks.

        This is used by pipeline stages that do not own a dataloader to learn the current epoch.
        The maximum epoch observed across ranks is adopted to ensure monotonic progress.
        """
        process_group = dist.get_world_group()
        result = [None] * dist.get_world_size(process_group)
        torch.distributed.all_gather_object(result, self.epoch, group=process_group)
        max_epoch = -1
        for epoch in result:
            max_epoch = max(epoch, max_epoch)
        self.epoch = max_epoch

    @staticmethod
    def split_batch(batch, gradient_accumulation_steps, data_parallel_world_size):
        """
        Split a local batch into gradient_accumulation_steps micro-batches.

        Parameters:
            batch: Tuple of (example_tuple, None) where example_tuple is a tuple of tensors
                   with batch dimension at dim=0
            gradient_accumulation_steps: Number of micro-batches to produce
            data_parallel_world_size: Number of data-parallel ranks (for logging only)

        Returns:
            List of tuples [(micro_example_tuple, None), ...] where each micro_example_tuple
            contains the sliced tensors corresponding to one micro-batch.
        """
        example_tuple, _ = batch

        local_batch_size = example_tuple[0].size(0)

        batch_size = local_batch_size * data_parallel_world_size
        log(f'before GAS splitting, batch size: {batch_size} sequences')

        split_size = local_batch_size // gradient_accumulation_steps

        # Split tensors along batch dimension (dim=0)
        splits_per_field = [torch.split(tensor, split_size, dim=0) for tensor in example_tuple]

        # Zip splits together to form micro-batches
        split_examples = zip(*splits_per_field)

        # Return list of (micro_batch, None) tuples
        return [(ex, None) for ex in split_examples]

    class DistributedBatchSampler(Sampler):
        """
        Per-rank sampler that:
        - Shuffles and pads the dataset to be divisible by the global batch size
        - Groups indices into global batches
        - Optionally shuffles the order of global batches
        - Emits only the slice of each global batch that belongs to this rank

        The per-rank batch size is:
            samples_per_rank = batch_size * batch_size_multiplier
        where batch_size_multiplier is typically gradient_accumulation_steps.
        """

        def __init__(self, dataset, batch_size, num_replicas, rank, batch_size_multiplier=1, shuffle=True, seed=0):
            """
            Parameters:
                dataset: Dataset to sample from
                batch_size: Per-micro-batch size
                num_replicas: Number of data-parallel ranks
                rank: This process's data-parallel rank (0-based)
                batch_size_multiplier: Multiplier to form a full local batch
                shuffle: Shuffle samples and global batch order each epoch
                seed: Base seed for deterministic shuffles
            """
            self.dataset = dataset
            self.batch_size = batch_size
            self.batch_size_multiplier = batch_size_multiplier
            self.num_replicas = num_replicas
            self.rank = rank
            self.shuffle = shuffle
            self.seed = seed

            dataset_size = len(dataset)

            # Create indices and shuffle if needed
            indices = list(range(dataset_size))
            if self.shuffle:
                indices = self.shuffle_list(indices, self.seed)

            # Calculate global batch size
            global_batch_size = self.batch_size * self.batch_size_multiplier * self.num_replicas

            # Pad dataset to make it evenly divisible by global_batch_size
            if dataset_size % global_batch_size != 0:
                padding_needed = global_batch_size - (dataset_size % global_batch_size)
                # Repeat indices to pad
                padding_indices = (indices * ((padding_needed // dataset_size) + 1))[:padding_needed]
                indices.extend(padding_indices)

            # Split into global batches
            global_batches = []
            for i in range(0, len(indices), global_batch_size):
                global_batches.append(indices[i:i + global_batch_size])

            # Shuffle global batches if needed
            if self.shuffle:
                global_batches = self.shuffle_list(global_batches, self.seed + 1)

            # Extract batches for this rank
            samples_per_rank = self.batch_size * self.batch_size_multiplier
            self.indices = []
            for global_batch in global_batches:
                start_idx = self.rank * samples_per_rank
                end_idx = start_idx + samples_per_rank
                rank_batch = global_batch[start_idx:end_idx]
                self.indices.append(rank_batch)

        def __iter__(self):
            """Yield per-rank batches (lists of indices) for each global batch."""
            return iter(self.indices)

        def __len__(self):
            """Number of per-rank batches (i.e., number of global batches) per epoch."""
            return len(self.indices)

        @staticmethod
        def shuffle_list(l, seed):
            """Deterministically shuffle a sequence using a dedicated torch.Generator seeded with `seed`."""
            g = torch.Generator()
            g.manual_seed(seed)
            shuffle_idx = torch.randperm(len(l), generator=g).tolist()
            new_l = [l[i] for i in shuffle_idx]
            return new_l