import argparse
import os
from datetime import datetime, timedelta, timezone
import shutil
import glob
import time
import itertools
from contextlib import contextmanager
import json
import gc

import torch
from torch.utils.tensorboard import SummaryWriter
import transformers
from peft import LoraConfig, get_peft_model
from peft.optimizers import create_loraplus_optimizer
import deepspeed
from deepspeed.runtime.pipe.module import LayerSpec
import toml
import bitsandbytes
import optimi

from dataset_utils import load_datasets
import dataloader
from saver import Saver
from utils import is_main_process
import engine
import llama_pipe
import unsloth_utils

parser = argparse.ArgumentParser()
parser.add_argument('--config',
                    help='Path to TOML configuration file.')
parser.add_argument('--local_rank', type=int, default=-1,
                    help='local rank passed from distributed launcher')
parser.add_argument('--resume_from_checkpoint', action='store_true', default=None,
                    help='resume training from the most recent checkpoint')
parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()


def get_most_recent_run_dir(output_dir):
    return list(sorted(glob.glob(os.path.join(output_dir, '*'))))[-1]


def write_metrics(tb_writer, prefix, metrics, step):
    loss = metrics[0].mean().item()
    tb_writer.add_scalar(f'{prefix}/loss', loss, step)

    if len(metrics) > 1:
        losses = metrics[1].view(-1)
        positive_losses = (losses > 0)
        tb_writer.add_histogram(f'{prefix}/log_loss_hist',  torch.log(losses[positive_losses]), step)

    if len(metrics) > 2:
        entropy = metrics[2].view(-1)
        tb_writer.add_scalar(f'{prefix}/entropy', entropy.mean().item(), step)

    if len(metrics) > 3:
        normalised_entropy = metrics[3].view(-1)
        tb_writer.add_scalar(f'{prefix}/normalised_entropy', normalised_entropy.mean().item(), step)

    if len(metrics) > 4:
        log_likelihood = metrics[4].mean()
        tb_writer.add_scalar(f'{prefix}/log_likelihood', log_likelihood.item(), step)
        likelihood = torch.exp(-log_likelihood).item()
        tb_writer.add_scalar(f'{prefix}/likelihood', likelihood, step)
        perplexity = torch.exp(log_likelihood).item()
        tb_writer.add_scalar(f'{prefix}/perplexity', perplexity, step)

    if len(metrics) > 5:
        mcfaddens_pseudo_r2 = metrics[5].mean()
        tb_writer.add_scalar(f'{prefix}/mcfaddens_pseudo_r2', mcfaddens_pseudo_r2.item(), step)

    if len(metrics) > 6:
        tb_writer.add_scalar(f'{prefix}/top1_accuracy', metrics[6].mean().item(), step)
        tb_writer.add_scalar(f'{prefix}/top5_accuracy', metrics[7].mean().item(), step)
        tb_writer.add_scalar(f'{prefix}/top20_accuracy', metrics[8].mean().item(), step)

    if len(metrics) > 9:
        tb_writer.add_scalar(f'{prefix}/load_balancing_loss', metrics[9].mean().item(), step)

    if len(metrics) > 10:
        tb_writer.add_scalar(f'{prefix}/alternate_load_balancing_loss', metrics[10].mean().item(), step)

    return loss


def evaluate(model_engine, eval_dataloader, tb_writer, step):
    if is_main_process():
        print('Running eval')
        start = time.time()
    iterator = iter(eval_dataloader)
    all_metrics = None
    while True:
        metrics = model_engine.eval_batch(iterator)
        eval_dataloader.sync_epoch()
        if all_metrics is None:
            all_metrics = [[] for _ in range(len(metrics))]
        if eval_dataloader.epoch == 2:
            break
        for i, metric in enumerate(metrics):
            all_metrics[i].append(metric)

    eval_dataloader.reset()
    eval_metrics = [torch.cat(metric_list) for metric_list in all_metrics]
    loss = None
    if is_main_process():
        duration = time.time() - start
        tb_writer.add_scalar('eval/eval_time_sec', duration, step)
        loss = write_metrics(tb_writer, f'eval', eval_metrics, step)
    return loss


def parse_layers_to_transform(config):
    """Parse layers_to_transform config into list of layer numbers."""
    layers_to_transform = []
    if layers_spec := config.get('layers_to_transform', None):
        parts = layers_spec.split(',')
        for part in parts:
            start, stop = part.split(':')
            layers_to_transform.extend(range(int(start), int(stop)+1))
    return layers_to_transform


def create_model(config, model_type):
    """Create the base transformer model with appropriate quantization."""
    if config.get('full_fine_tune', False):
        quantization_config = None
    else:
        quantization_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            llm_int8_skip_modules=["lm_head"]
        )       

    if model_type == 'llama':
        model = llama_pipe.LlamaForCausalLMPipe(config, quantization_config=quantization_config)
    elif model_type == 'qwen2':
        model = llama_pipe.Qwen2ForCausalLMPipe(config, quantization_config=quantization_config)
    elif model_type == 'cohere':
        model = llama_pipe.CohereForCausalLMPipe(config, quantization_config=quantization_config)
    elif model_type == 'phi3':
        model = llama_pipe.Phi3ForCausalLMPipe(config, quantization_config=quantization_config)
    elif model_type == 'gemma2':
        model = llama_pipe.Gemma2ForCausalLMPipe(config, quantization_config=quantization_config)
    elif model_type == 'mistral':
        model = llama_pipe.MistralForCausalLMPipe(config, quantization_config=quantization_config)
    else:
        raise NotImplementedError()
    
    return model


def create_pipeline_model(model, config):
    """Create pipeline model from base model for distributed training."""
    # CAREFUL! The "primary" layers of the model have to have 'decoderlayer' in them for
    # activation checkpointing to automatically work correctly.
    layers = model.to_layer_specs()
    checkpointable_layers = set()
    for layer in layers:
        if isinstance(layer, LayerSpec) and 'decoderlayer' in layer.typename.__name__.lower():
            checkpointable_layers.add(layer.typename.__name__)
    checkpointable_layers = list(checkpointable_layers)

    pipeline_model = engine.CustomPipelineModule(
        layers=layers,
        num_stages=config.get('pipeline_stages', 1),
        activation_checkpoint_interval=1,
        checkpointable_layers=checkpointable_layers,
        activation_checkpoint_func=unsloth_utils.unsloth_checkpoint,
        partition_method='estimated_size',
        use_column_major_topology=config.get('use_column_major_topology', False)
    )

    return pipeline_model


def create_lora_config(config, target_modules, layers_to_transform):
    """Create LoRA configuration."""
    return LoraConfig(
        r=config['lora_rank'],
        lora_alpha=config['lora_alpha'],
        target_modules=target_modules if target_modules else 'all-linear',
        modules_to_save=config['modules_to_save'] if 'modules_to_save' in config else [],
        lora_dropout=config['lora_dropout'] if 'lora_dropout' in config else 0,
        layers_to_transform=layers_to_transform if layers_to_transform else None,
        bias='none',
        task_type='CAUSAL_LM',
        use_dora=config.get('use_dora', False)
    )


def apply_lora_adapters(model, config, lora_config):
    """Apply LoRA configuration to model."""
    lora_model = get_peft_model(model, lora_config)
    # If the underlying weights are floats, the lora weights have already been
    # cast to the same dtype, so we need to change the dtype here.
    for p in lora_model.parameters():
        if p.requires_grad:
            p.data = p.data.to(torch.bfloat16)

    lora_model.model.config.use_cache = False
    for name, p in lora_model.named_parameters():
        p.original_name = name
        

def configure_full_fine_tuning(model, config, target_modules, layers_to_transform):
    """Setup full fine-tuning by setting requires_grad on parameters."""
    for name, p in model.named_parameters():
        p.original_name = name
    
    for name, p in model.named_parameters():
        should_train = True
        if target_modules and not any(target in name for target in target_modules):
            should_train = False
            print(f'not training {name} because it is not present in target_modules')
        elif layers_to_transform and 'model.layers.' in name:
            layer_num = int(name.split('model.layers.')[1].split('.')[0])
            if layer_num not in layers_to_transform:
                should_train = False
                print(f'not training {name} because layer {layer_num} is not in layers_to_transform')
        p.requires_grad = should_train


if __name__ == '__main__':
    with open(args.config) as f:
        config = toml.load(f)

    deepspeed.init_distributed(timeout=timedelta(hours=2))

    with open(os.path.join(config['model'], 'config.json')) as f:
        model_config = json.load(f)
        model_type = model_config.get('model_type', 'llama')

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config['model'], local_files_only=True, model_max_length=int(1e30),
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_data, eval_data = load_datasets(config, tokenizer)

    # if this is a new run, create a new dir for it
    if not args.resume_from_checkpoint and is_main_process():
        run_dir = os.path.join(config['output_dir'], datetime.now(timezone.utc).strftime('%Y%m%d_%H-%M-%S'))
        os.makedirs(run_dir, exist_ok=True)
        shutil.copy(args.config, run_dir)
        shutil.copy(args.deepspeed_config, run_dir)

    # wait for all processes then get the most recent dir (may have just been created)
    deepspeed.comm.barrier()
    run_dir = get_most_recent_run_dir(config['output_dir'])

    # Ugly hack to move quantized models from GPU to CPU, and back to GPU again without triggering re-quantization
    bnb_cuda_old = bitsandbytes.nn.modules.Params4bit.cuda
    def bnb_cuda_hijack(self, device):
        if getattr(self, 'already_quantized', False):
            self.data = self.data.to(device)
            self.quant_state.to(device)
            return self
        self.already_quantized = True
        return bnb_cuda_old(self, device)
    bitsandbytes.nn.modules.Params4bit.cuda = bnb_cuda_hijack

    model = create_model(config, model_type)
    pipeline_model = create_pipeline_model(model, config)
    
    target_modules = config['target_modules'] if 'target_modules' in config else []
    layers_to_transform = parse_layers_to_transform(config)
    
    if config.get('full_fine_tune', False):
        lora_config = None
        configure_full_fine_tuning(model, config, target_modules, layers_to_transform)
    else:
        lora_config = create_lora_config(config, target_modules, layers_to_transform)
        apply_lora_adapters(model, config, lora_config)

    parameters_to_train = [p for p in pipeline_model.parameters() if p.requires_grad]

    def get_optimizer(model_parameters):
        optim_config = config['optimizer']
        optimizer_kwargs = {
            "params": model_parameters,
            "lr": optim_config['lr'],
            "betas": (optim_config.get('beta1', 0.9), optim_config.get('beta2', 0.99)),
            "weight_decay": optim_config.get('weight_decay', 0.01),
            "eps": optim_config.get('eps', 1e-6),
            "kahan_sum": True
        }
        return optimi.AdamW(**optimizer_kwargs)            

    model_engine, optimizer = engine.initialize(
        args=args,
        model=pipeline_model,
        model_parameters=parameters_to_train,
        optimizer=get_optimizer,
    )

    model_engine.communication_data_type = torch.bfloat16

    train_dataloader = dataloader.PipelineDataLoader(
        train_data,
        tokenizer,
        model_engine.train_micro_batch_size_per_gpu(),
        model_engine.gradient_accumulation_steps(),
        model_engine.grid.get_data_parallel_world_size(),
        model_engine.grid.get_data_parallel_rank(),
        group_by_length=False,
        batch_size_tokens=None,
    )
    model_engine.set_dataloader(train_dataloader)
    steps_per_epoch = len(train_dataloader) // model_engine.gradient_accumulation_steps()
    model_engine.total_steps = steps_per_epoch * config['epochs']

    # handle Deepspeed optimizer wrapper (e.g. BF16_Optimizer)
    optimizer = getattr(optimizer, 'optimizer', optimizer)
   
    # see: https://github.com/tdrussell/qlora-pipe/pull/35#issuecomment-2495460307
    def make_rms_ratio_fn(beta):
        def rms_ratio_fn(step):
            return torch.sqrt(torch.tensor((1 - beta**step)/(1 + beta**step))).item()
        return rms_ratio_fn
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=make_rms_ratio_fn(config['optimizer']['beta2'])
    )
        
    model_engine.lr_scheduler = lr_scheduler

    step = 1
    if args.resume_from_checkpoint:
        load_path, client_state = model_engine.load_checkpoint(
            run_dir,
            load_module_strict=False,
            load_optimizer_states=True
        )
        deepspeed.comm.barrier()  # just so the print below doesn't get swamped
        assert load_path is not None
        train_dataloader.load_state_dict(client_state['custom_loader'])
        step = client_state['step'] + 1
        del client_state
        if is_main_process():
            print(f'Resuming training from checkpoint. Resuming at epoch: {train_dataloader.epoch}, step: {step}')

    # Eval dataset doesn't need to repeat; we just use this to track "epoch" so we know when we're done iterating over it.
    eval_dataloader = dataloader.PipelineDataLoader(
        eval_data,
        tokenizer,
        model_engine.train_micro_batch_size_per_gpu(),
        model_engine.gradient_accumulation_steps(),
        model_engine.grid.get_data_parallel_world_size(),
        model_engine.grid.get_data_parallel_rank(),
        shuffle=False,
        group_by_length=False,
        batch_size_tokens=None,
    )

    tb_writer = SummaryWriter(log_dir=run_dir) if is_main_process() else None

    epoch = train_dataloader.epoch

    saver = Saver(model_engine, pipeline_model, train_dataloader, lora_config, run_dir, args, config)

    epoch = train_dataloader.epoch

    loss = evaluate(model_engine, eval_dataloader, tb_writer, 0)
    saver.append_eval_results(loss, save_best=False)

    while True:
        gc.collect()
        torch.cuda.empty_cache()
        metrics = model_engine.train_batch()
        train_dataloader.sync_epoch()

        new_epoch = saver.process_epoch(epoch, step)
        finished_epoch = True if new_epoch != epoch else False

        if is_main_process():
            write_metrics(tb_writer, 'train', metrics, step)
            tb_writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], step)
            tb_writer.add_scalar('train/epoch', step/steps_per_epoch, step)

        if step % config['eval_steps'] == 0:
            loss = evaluate(model_engine, eval_dataloader, tb_writer, step)
            saver.append_eval_results(loss)

        saver.process_step(step)

        if finished_epoch:
            epoch = new_epoch
            if epoch is None:
                break

        step += 1

    if is_main_process():
        print('TRAINING COMPLETE!')