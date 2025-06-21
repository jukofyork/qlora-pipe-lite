import math
import torch
from torch.utils.data import DataLoader
import transformers
import accelerate
from deepspeed import comm as dist
from tqdm import tqdm

from utils import *


def split_batch(batch, pieces):
    example_tuple, labels = batch
    if is_main_process():
        print(f'before GAS splitting, batch size: {example_tuple[0].size(0)}, total tokens: {example_tuple[0].numel()}')
    split_size = example_tuple[0].size(0) // pieces
    split_examples = zip(*(torch.split(tensor, split_size) for tensor in example_tuple))
    return [(ex, None) for ex in split_examples]


def shuffle_list(l, seed):
    g = torch.Generator()
    g.manual_seed(seed)
    shuffle_idx = torch.randperm(len(l), generator=g).tolist()
    new_l = [l[i] for i in shuffle_idx]
    return new_l


class DistributedBatchSamper(torch.utils.data.Sampler):
    def __init__(self, dataset, batch_size, num_replicas, rank, batch_size_multiplier=1, shuffle=True, seed=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.batch_size_multiplier = batch_size_multiplier
        self.num_replicas = num_replicas
        self.rank = rank
        # every global batch must be evenly divisible by this amount
        self.chunk_size = self.num_replicas * self.batch_size_multiplier
        self.shuffle = shuffle
        self.seed = seed

        # Make list of (index, size). Shuffle if needed.
        indices = list(enumerate(self.dataset['length']))
        if self.shuffle:
            indices = shuffle_list(indices, self.seed)

        # Group indices together into global batches.
        global_batches = []
        current_batch = []
        for i in range(0, len(indices), self.chunk_size):
            slice = indices[i:i+self.chunk_size]
            if len(slice) < self.chunk_size:
                # pad with random examples if slice is too small
                padding_size = self.chunk_size - len(slice)
                shuffled_indices = shuffle_list(indices, self.seed+1)
                if padding_size < len(shuffled_indices):
                    slice += shuffled_indices[:padding_size]
                else:
                    slice += (shuffled_indices * math.ceil(padding_size / len(shuffled_indices)))[:padding_size]

            if self.should_emit_current_batch(current_batch, slice):
                global_batches.append(current_batch)
                current_batch = []
            current_batch.extend(slice)

        # Emit anything remaining
        if len(current_batch) > 0:
            global_batches.append(current_batch)

        if self.shuffle:
            global_batches = shuffle_list(global_batches, self.seed+2)

        batches_for_this_rank = [global_batch[self.rank:len(global_batch):self.num_replicas] for global_batch in global_batches]
        self.indices = [[i for i, _ in batch] for batch in batches_for_this_rank]

    def should_emit_current_batch(self, current_batch, slice):
        batch_size_after_appending = len(current_batch) // self.chunk_size + 1
        if batch_size_after_appending > self.batch_size:
            return True
        else:
            return False

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class PipelineDataLoader:
    def __init__(self, dataset, tokenizer, batch_size, gradient_accumulation_steps, data_parallel_world_size, data_parallel_rank, shuffle=True):
        assert data_parallel_rank < data_parallel_world_size
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.data_sampler = DistributedBatchSamper(
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
            for micro_batch in split_batch(batch, self.gradient_accumulation_steps):
                yield micro_batch

    def _create_dataloader(self):
        def collate_fn(examples):
            # Simple collation since all sequences are same length
            input_ids = torch.stack([torch.tensor(ex['input_ids']) for ex in examples])
            attention_mask = torch.stack([torch.tensor(ex['attention_mask']) for ex in examples])
            labels = torch.stack([torch.tensor(ex['labels']) for ex in examples])
            
            return ((input_ids, attention_mask, labels), None)
        
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