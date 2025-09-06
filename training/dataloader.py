from torch.utils.data import DataLoader, Sampler
import torch

class PipelineDataLoader:
    """
    Pipeline-aware data loader that yields micro-batches suitable for gradient accumulation.

    Behavior:
    - Underlying DataLoader fetches local batches of size:
        local_batch_size = batch_size * gradient_accumulation_steps
    - Each local batch is split into `gradient_accumulation_steps` micro-batches (size `batch_size`)
    - Supports saving/restoring iteration position via a one-shot local-batch skip counter
    - Uses a per-rank batch sampler that truncates remainders to only emit full global batches

    Parameters:
        dataset                     : torch.utils.data.Dataset returning dicts with keys:
                                      'input_ids', 'attention_mask', 'control_classes' and 'labels'
        batch_size                  : Number of samples per micro-batch (per rank)
        gradient_accumulation_steps : Number of micro-batches per optimizer step
        data_parallel_world_size    : Total number of data-parallel ranks
        data_parallel_rank          : This process's data-parallel rank (0-based)

    Attributes:
        data_sampler       : Per-rank batch sampler
        dataloader         : torch.utils.data.DataLoader bound to the sampler
        num_batches_pulled : Number of full local batches consumed in the current pass
        next_micro_batch   : Prefetched next micro-batch for __next__()
        data               : Generator yielding micro-batches
    """

    def __init__(self, dataset, batch_size, gradient_accumulation_steps, data_parallel_world_size, data_parallel_rank):
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
        )

        def collate_fn(examples):
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

        self.num_batches_pulled = 0
        self.next_micro_batch = None
        self._skip_local_batches_once = 0
        self.data = self._pull_batches_from_dataloader()

    def reset(self):
        """Reset internal iteration state and start from the beginning of the sampler."""
        self.num_batches_pulled = 0
        self.next_micro_batch = None
        self._skip_local_batches_once = 0
        self.data = self._pull_batches_from_dataloader()

    def __iter__(self):
        """Return self as the iterator."""
        return self

    def __len__(self):
        """Total number of micro-batches produced by one full traversal of the sampler."""
        return len(self.data_sampler) * self.gradient_accumulation_steps

    def __next__(self):
        """
        Return the next micro-batch; rotate to a new pass when the current one completes.

        Notes:
        - This method prefetches the next micro-batch to minimize control overhead.
        - When an internal pass is exhausted, it starts a fresh pass over the dataloader.
        """
        if self.next_micro_batch is None:
            self.next_micro_batch = next(self.data)
        ret = self.next_micro_batch
        try:
            self.next_micro_batch = next(self.data)
        except StopIteration:
            self.data = self._pull_batches_from_dataloader()
            self.num_batches_pulled = 0
            self.next_micro_batch = next(self.data)
        return ret

    def state_dict(self):
        """Return a minimal state dict to resume iteration deterministically."""
        return {
            'num_batches_pulled': self.num_batches_pulled,
        }

    def load_state_dict(self, state_dict):
        """
        Restore iteration state and set up the internal generator to skip already-consumed local batches.

        Notes:
        - One batch is preloaded by __next__, so we subtract 1 from the persisted counter.
        - A one-shot local-batch skip counter is applied on the next pass only.
        """
        self.num_batches_pulled = state_dict['num_batches_pulled'] - 1
        self._skip_local_batches_once = max(0, self.num_batches_pulled)
        self.data = self._pull_batches_from_dataloader()

    def _pull_batches_from_dataloader(self):
        """
        Internal generator:
        - Iterate full local batches from the internal DataLoader
        - Split each into gradient_accumulation_steps micro-batches
        - Yield micro-batches one-by-one
        """
        to_skip = self._skip_local_batches_once
        self._skip_local_batches_once = 0
        for batch in self.dataloader:
            if to_skip > 0:
                to_skip -= 1
                continue
            self.num_batches_pulled += 1
            for micro_batch in self.split_batch(batch, self.gradient_accumulation_steps):
                yield micro_batch

    @staticmethod
    def split_batch(batch, gradient_accumulation_steps):
        """
        Split a local batch into gradient_accumulation_steps micro-batches and yield them.
        """
        example_tuple, _ = batch
        local_batch_size = example_tuple[0].size(0)
        split_size = local_batch_size // gradient_accumulation_steps

        splits_per_field = [torch.split(tensor, split_size, dim=0) for tensor in example_tuple]
        for ex in zip(*splits_per_field):
            yield (ex, None)

    class DistributedBatchSampler(Sampler):
        """
        Per-rank sampler that:
        - Truncates the dataset remainder to only emit full global batches
        - Groups indices into global batches
        - Emits only the slice of each global batch that belongs to this rank

        The per-rank batch size is:
            samples_per_rank = batch_size * batch_size_multiplier
        where batch_size_multiplier is typically gradient_accumulation_steps.
        """

        def __init__(self, dataset, batch_size, num_replicas, rank, batch_size_multiplier=1):
            """
            Parameters:
                dataset: Dataset to sample from
                batch_size: Per-micro-batch size
                num_replicas: Number of data-parallel ranks
                rank: This process's data-parallel rank (0-based)
                batch_size_multiplier: Multiplier to form a full local batch
            """
            self.dataset = dataset
            self.batch_size = batch_size
            self.batch_size_multiplier = batch_size_multiplier
            self.num_replicas = num_replicas
            self.rank = rank

            dataset_size = len(dataset)

            # Create indices
            indices = list(range(dataset_size))

            # Calculate global batch size
            global_batch_size = self.batch_size * self.batch_size_multiplier * self.num_replicas

            # Fail if dataset is smaller than one global batch
            if dataset_size < global_batch_size:
                raise ValueError(f'Dataset size ({dataset_size}) is smaller than one global batch ({global_batch_size}).')

            # Split into full global batches (truncate any remainder)
            full_size = (len(indices) // global_batch_size) * global_batch_size
            global_batches = []
            for i in range(0, full_size, global_batch_size):
                global_batches.append(indices[i:i + global_batch_size])

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
            """Number of per-rank batches (i.e., number of global batches) per pass."""
            return len(self.indices)