from deepspeed import comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime import utils as ds_utils
from deepspeed.runtime.activation_checkpointing import checkpointing as ds_checkpointing
from deepspeed.runtime.config import DeepSpeedConfig
from deepspeed.runtime.pipe import schedule
from deepspeed.runtime.pipe.engine import (
    PipelineEngine,
    TRAIN_BATCH_TIMER,
    PIPE_SEND_OUTPUT_TIMER,
    PIPE_SEND_GRAD_TIMER,
    PIPE_RECV_INPUT_TIMER,
    PIPE_RECV_GRAD_TIMER
)
from deepspeed.runtime.pipe.module import LayerSpec
from deepspeed.runtime.pipe.module import PipelineModule
from deepspeed.runtime.pipe.topology import ProcessTopology
from deepspeed.runtime.utils import PartitionedTensor
from torch import nn
import time
import torch

from utils.utils import log, seconds_to_time_str

def initialize(config=None,
               args=None,
               model=None,
               model_parameters=None,
               optimizer=None):
    assert model is not None, "deepspeed.initialize requires a model"

    dist_backend = get_accelerator().communication_backend_name()
    dist.init_distributed(dist_backend=dist_backend)

    ds_config = {
        'train_micro_batch_size_per_gpu': 1,
        'gradient_accumulation_steps': config.get('gradient_accumulation_steps', 1),
        'gradient_clipping': 1.0,
        'steps_per_print': 1,
    }

    mpu = model.mpu()
    config_class = DeepSpeedConfig(ds_config, mpu)
    engine = CustomPipelineEngine(
        args=args,
        model=model,
        optimizer=optimizer,
        model_parameters=model_parameters,
        mpu=mpu,
        config=ds_config,
        config_class=config_class
    )

    return engine, engine.optimizer

class CustomPipelineEngine(PipelineEngine):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.total_steps = None
        self.start_time = None
        self.start_step = None

    def train_batch(self):
        if not torch._C.is_grad_enabled():
            raise RuntimeError(f'train_batch() requires gradients enabled. Use eval_batch() instead.')

        # Start timing from first training step to avoid startup overhead bias
        if self.global_rank == 0 and self.start_time is None:
            self.start_time = time.time()

        # sequence length may change between macro batches (but not between gradient accumulation steps)
        self.reset_activation_shape()

        self.module.train()
        self._compute_loss = True

        # Do the work
        self.timers(TRAIN_BATCH_TIMER).start()
        sched = schedule.TrainSchedule(micro_batches=self.micro_batches,
                                       stages=self.num_stages,
                                       stage_id=self.stage_id)
        self._exec_schedule(sched)
        agg_losses = self._aggregate_total_losses()
        # Actual training loss is always the first item.
        self.agg_train_loss = agg_losses[0].mean()

        self.timers(TRAIN_BATCH_TIMER).stop()

        # Set start_step to track the step number before timing began
        if self.global_rank == 0 and self.start_step is None:
            self.start_step = self.global_steps - 1

        if self.global_steps % self.steps_per_print() == 0:
            if self.global_rank == 0:
                iter_elapsed = self.timers(TRAIN_BATCH_TIMER).elapsed(reset=True) / 1000.0
                iter_time = iter_elapsed / self.steps_per_print()
                iter_throughput = self.train_batch_size() / iter_time

                # Calculate ETA based on actual steps completed since timing started
                total_elapsed = time.time() - self.start_time
                steps_completed = self.global_steps - self.start_step
                eta = (total_elapsed / steps_completed) * (self.total_steps - self.global_steps)

                log(f'step: {self.global_steps} / {self.total_steps}, '
                    f'loss: {self.agg_train_loss:0.4f}, '
                    f'throughput: {iter_throughput:0.3f} sequences/s, '
                    f'elapsed: {seconds_to_time_str(total_elapsed)}, '
                    f'eta: {seconds_to_time_str(eta)}')
            else:
                self.timers(TRAIN_BATCH_TIMER).elapsed(reset=True)

        if self.wall_clock_breakdown() and self.global_steps % self.steps_per_print() == 0:
            self.timers.log([
                PIPE_SEND_OUTPUT_TIMER,
                PIPE_SEND_GRAD_TIMER,
                PIPE_RECV_INPUT_TIMER,
                PIPE_RECV_GRAD_TIMER,
            ])

        return agg_losses

    def eval_batch(self, data_iter):
        # sequence length may change between macro batches (but not between gradient accumulation steps)
        self.reset_activation_shape()

        self.module.eval()
        self._compute_loss = True

        # Use the provided data iterator
        train_iterator = self.data_iterator
        self.set_dataiterator(data_iter)

        # Do the work
        sched = schedule.InferenceSchedule(micro_batches=self.micro_batches,
                                           stages=self.num_stages,
                                           stage_id=self.stage_id)

        # prevent dead-lock with multiple evals sequence
        dist.barrier()

        with torch.no_grad():
            self._exec_schedule(sched)

        # list of losses
        agg_eval_losses = self._aggregate_total_losses()

        # Restore the training iterator
        self.set_dataiterator(train_iterator)

        return agg_eval_losses

    def _aggregate_total_losses(self):
        all_agg_outputs = []
        # gather each output for all the gradient accumulation steps
        grouped_outputs = [list(x) for x in zip(*self.fwd_outputs)]
        # if any are scalar, make them dim 1 so we can concat across DP ranks
        for outputs in grouped_outputs:
            for i, output in enumerate(outputs):
                if output.dim() == 0:
                    outputs[i] = torch.unsqueeze(output, 0)

        if self.is_last_stage():
            agg_sizes = []
            # loop to gather all the outputs across DP ranks
            for outputs in grouped_outputs:
                # concat all the grad_accum_steps
                concat_outputs = torch.cat(outputs)
                if self.is_data_parallel:
                    # might be different sizes across DP ranks, so, gather all the sizes
                    sizes = [None] * self.grid.get_data_parallel_world_size()
                    torch.distributed.all_gather_object(sizes, concat_outputs.size(), group=self.grid.get_data_parallel_group())
                    # once we know all the sizes we can gather the results across DP ranks
                    gather_result = [torch.zeros(size).to(self.device) for size in sizes]
                    dist.all_gather(gather_result, concat_outputs, group=self.grid.get_data_parallel_group())
                    # and finally, concat
                    agg_output = torch.cat(gather_result)
                else:
                    agg_output = concat_outputs
                agg_sizes.append(agg_output.size())
                all_agg_outputs.append(agg_output)

            # send the sizes, then broadcast to the PP ranks
            if self.is_pipe_parallel:
                torch.distributed.broadcast_object_list([agg_sizes], src=self.global_rank, group=self.grid.get_pipe_parallel_group())
                for agg_output in all_agg_outputs:
                    dist.broadcast(tensor=agg_output, src=self.global_rank, group=self.grid.get_pipe_parallel_group())
        else:
            # get the outputs from the last stage
            src_rank = self.grid.stage_to_global(self.num_stages - 1)
            assert src_rank in self.grid.pp_group
            result = [None]
            torch.distributed.broadcast_object_list(result, src=src_rank, group=self.grid.get_pipe_parallel_group())
            agg_sizes = result[0]
            for agg_size in agg_sizes:
                agg_output = torch.zeros(agg_size).to(self.device)
                dist.broadcast(tensor=agg_output, src=src_rank, group=self.grid.get_pipe_parallel_group())
                all_agg_outputs.append(agg_output)

        return all_agg_outputs

    # We override this to handle the model returning a list of "losses", but only doing backprop on the first.
    def _exec_forward_pass(self, buffer_id):
        self.tput_timer.start()
        self.mem_status('BEFORE FWD', reset_max=True)

        if isinstance(self.pipe_buffers['inputs'][buffer_id], tuple):
            inputs = tuple(t.clone() for t in self.pipe_buffers['inputs'][buffer_id])
        else:
            inputs = self.pipe_buffers['inputs'][buffer_id].clone()

        # collect the partitioned input from the previous stage
        if self.is_pipe_partitioned and not self.is_first_stage():
            part_input = PartitionedTensor.from_meta(meta=inputs[0],
                                                     local_part=inputs[1],
                                                     group=self.grid.get_slice_parallel_group())

            inputs = (part_input.full(), *inputs[2:])
            inputs[0].requires_grad = True
            # skip mask
            # inputs[1].requires_grad = True
            part_input = None
            inputs = inputs[0] if len(inputs) == 1 else inputs
            self.pipe_buffers['inputs'][buffer_id] = inputs

        # inputs has no gradient because it is from a cloned tensor
        outputs = super(PipelineEngine, self).forward(inputs)

        # Reset activation checkpointing buffers.
        # Need to call this between evaluation iterations
        if not self.module.training:
            ds_checkpointing.reset()

        # Partition the outputs if we are not the last stage
        if self.is_pipe_partitioned and not self.is_last_stage():
            if isinstance(outputs, tuple):
                first_output = outputs[0]
                # TODO: Improve pipe partitioning to pass multiple tensors that require grads
                assert all([torch.is_tensor(elt) and elt.requires_grad is False for elt in outputs[1:]])
                outputs_tail = outputs[1:]
            elif torch.is_tensor(outputs):
                first_output = outputs
                outputs_tail = []
            else:
                raise ValueError("expecting a tensor or a tuple of tensors")
            part = PartitionedTensor(tensor=first_output, group=self.grid.get_slice_parallel_group())
            # Clear the large output data, but save the computation graph
            first_output.data = torch.zeros(1)
            self.pipe_buffers['output_tensors'][buffer_id] = first_output
            # Inject the partitioned tensor into the output before sending
            outputs = (part.to_meta(), part.data(), *outputs_tail)
            part = None

        self.pipe_buffers['outputs'][buffer_id] = outputs

        # Optionally compute loss on the last device
        if self.is_last_stage():
            if self._compute_loss and self.module.loss_fn is not None:
                labels = self.pipe_buffers['labels'][buffer_id]
                losses = self.module.loss_fn(outputs, labels)
            else:
                # Some models just return loss from forward()
                losses = outputs
            if self.eval_return_logits:
                self.outputs = outputs
            if isinstance(losses, torch.Tensor):
                self.loss = losses
                self.fwd_outputs.append([self.loss.detach()])
            else:
                self.loss = losses[0]
                self.fwd_outputs.append([l.detach() for l in losses])

    # make our forward pass method apply
    PipelineEngine._INSTRUCTION_MAP[schedule.ForwardPass] = _exec_forward_pass

class CustomPipelineModule(PipelineModule):

    def __init__(self, layers, use_column_major_topology, **kwargs):
        if use_column_major_topology:
            self._set_column_major_topology(kwargs)
        super().__init__(layers, **kwargs)

    def _set_column_major_topology(self, kwargs):
        """
        Set a topology specialisation for hybrid data+pipeline parallelism optimized for LoRA training:
        - Sends high-volume "per token" hidden states over PCIe/NVLink.
        - Sends lower-volume "per step" LoRA gradient reductions over Ethernet/InfiniBand.
        """

        class ColumnMajorParallelTopology(ProcessTopology):

            def __init__(self, num_pp, num_dp):
                # Swap the axes and dims to change the rank mapping
                super().__init__(axes=['data', 'pipe'], dims=[num_dp, num_pp])

        world_size = dist.get_world_size()
        num_stages = kwargs.get('num_stages')
        if num_stages > 1 and world_size > 1:
            assert world_size % num_stages == 0, f"world_size {world_size} not divisible by num_stages {num_stages}"
            num_dp = world_size // num_stages
            kwargs['topology'] = ColumnMajorParallelTopology(num_pp=num_stages, num_dp=num_dp)

    def _partition_layers(self, method='estimated_size'):
        assert method == 'estimated_size', f"Only 'estimated_size' partitioning method is supported, got '{method}'"

        num_stages = self._topo.get_dim('pipe')
        stage_id = self._topo.get_coord(self.global_rank).pipe

        log(f'Partitioning pipeline stages with estimated_size method')

        # Use estimated_size for partitioning
        estimated_sizes = [getattr(l, 'estimated_size', 0) for l in self._layer_specs]
        self.parts = ds_utils.partition_balanced(weights=estimated_sizes, num_parts=num_stages)

        # Print some information on the partitioning.
        if self.global_rank == 0:
            for stage in range(num_stages):
                start = self.parts[stage]
                stop = self.parts[stage + 1]
                log(f'stage={stage} layers={stop - start}')
                for idx, layer in enumerate(self._layer_specs[start:stop]):
                    name = str(layer)
                    if isinstance(layer, LayerSpec):
                        name = layer.typename.__name__
                    if isinstance(layer, nn.Module):
                        name = layer.__class__.__name__
                    else:
                        try:
                            name = layer.__name__
                        except AttributeError:
                            pass
                    es = estimated_sizes[idx + start]
                    log(f'    {idx+start:2d}: {name}, estimated size: {es}')
            if self.loss_fn:
                try:
                    log(f'  loss: {self.loss_fn.__name__}')
                except AttributeError:
                    log(f'  loss: {self.loss_fn.__class__.__name__}')

        self._set_bounds(start=self.parts[stage_id], stop=self.parts[stage_id + 1])