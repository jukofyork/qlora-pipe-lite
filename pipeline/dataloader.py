from deepspeed import comm as dist
from torch.utils.data import DataLoader
import accelerate
import math
import torch

from utils.utils import log

def split_batch(batch, gradient_accumulation_steps, data_parallel_world_size):
    example_tuple, labels = batch

    local_batch_size = example_tuple[0].size(0)
    local_tokens = example_tuple[0].numel()

    batch_size = local_batch_size * data_parallel_world_size
    batch_tokens = local_tokens * data_parallel_world_size
    log(f'before GAS splitting, batch size: {batch_size} sequences ({batch_tokens} tokens)')

    split_size = local_batch_size // gradient_accumulation_steps

    def split_or_broadcast(t):
        # Split tensors with a batch dimension; broadcast scalars/0-D tensors
        if torch.is_tensor(t) and t.dim() >= 1:
            return torch.split(t, split_size, dim=0)
        else:
            return (t,) * gradient_accumulation_steps

    splits_per_field = [split_or_broadcast(t) for t in example_tuple]
    split_examples = zip(*splits_per_field)
    return [(ex, None) for ex in split_examples]

def shuffle_list(l, seed):
    g = torch.Generator()
    g.manual_seed(seed)
    shuffle_idx = torch.randperm(len(l), generator=g).tolist()
    new_l = [l[i] for i in shuffle_idx]
    return new_l

class DistributedBatchSampler(torch.utils.data.Sampler):

    def __init__(self, dataset, batch_size, num_replicas, rank, batch_size_multiplier=1, shuffle=True, seed=0):
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
            indices = shuffle_list(indices, self.seed)

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
            global_batches = shuffle_list(global_batches, self.seed + 1)

        # Extract batches for this rank
        samples_per_rank = self.batch_size * self.batch_size_multiplier
        self.indices = []
        for global_batch in global_batches:
            start_idx = self.rank * samples_per_rank
            end_idx = start_idx + samples_per_rank
            rank_batch = global_batch[start_idx:end_idx]
            self.indices.append(rank_batch)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

class PipelineDataLoader:

    def __init__(self, dataset, batch_size, gradient_accumulation_steps, data_parallel_world_size, data_parallel_rank, shuffle=True):
        assert data_parallel_rank < data_parallel_world_size
        self.dataset = dataset
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.data_parallel_world_size = data_parallel_world_size

        self.data_sampler = DistributedBatchSampler(
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
        self.epoch = 1
        self.num_batches_pulled = 0
        self.next_micro_batch = None
        self.data = self._pull_batches_from_dataloader()

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.data_sampler) * self.gradient_accumulation_steps

    def __next__(self):
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
        for batch in self.dataloader:
            self.num_batches_pulled += 1
            for micro_batch in split_batch(batch, self.gradient_accumulation_steps, self.data_parallel_world_size):
                yield micro_batch

    def _create_dataloader(self):

        def collate_fn(examples):
            input_ids = torch.stack([ex['input_ids'] for ex in examples])
            attention_mask = torch.stack([ex['attention_mask'] for ex in examples])
            control_classes = torch.stack([ex['control_classes'] for ex in examples])
            labels = torch.stack([ex['labels'] for ex in examples])

            # input_ids shape here is [batch_size * GAS, seq_len] per rank
            batch_size, seq_len = input_ids.shape
            local_tokens = batch_size * seq_len  # already includes GAS
            n_tokens = local_tokens * self.data_sampler.num_replicas  # multiply by DP only
            n_tokens = torch.tensor(n_tokens, dtype=torch.long, device=input_ids.device)

            return ((input_ids, attention_mask, control_classes, labels, n_tokens), None)

        self.dataloader = DataLoader(
            self.dataset,
            pin_memory=True,
            batch_sampler=self.data_sampler,
            collate_fn=collate_fn,
        )

    def state_dict(self):
        return {
            'epoch': self.epoch,
            'num_batches_pulled': self.num_batches_pulled,
        }

    def load_state_dict(self, state_dict):
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
        process_group = dist.get_world_group()
        result = [None] * dist.get_world_size(process_group)
        torch.distributed.all_gather_object(result, self.epoch, group=process_group)
        max_epoch = -1
        for epoch in result:
            max_epoch = max(epoch, max_epoch)
        self.epoch = max_epoch