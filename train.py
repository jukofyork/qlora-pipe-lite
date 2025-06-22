import argparse
import os
from datetime import datetime, timedelta, timezone
import shutil
import json

import torch
import transformers
import deepspeed
import toml
import bitsandbytes

from dataset_utils import load_datasets
import dataloader
from utils import is_main_process
import engine
from training.model_factory import (
    create_model, 
    create_pipeline_model, 
    create_lora_config, 
    apply_lora_adapters, 
    configure_full_fine_tuning, 
    parse_layers_to_transform
)
from training.utils import get_most_recent_run_dir, get_optimizer, make_rms_ratio_fn
from training.trainer import Trainer

parser = argparse.ArgumentParser()
parser.add_argument('--config',
                    help='Path to TOML configuration file.')
parser.add_argument('--local_rank', type=int, default=-1,
                    help='local rank passed from distributed launcher')
parser.add_argument('--resume_from_checkpoint', action='store_true', default=None,
                    help='resume training from the most recent checkpoint')
parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()


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

    # Create model and pipeline
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

    model_engine, optimizer = engine.initialize(
        args=args,
        model=pipeline_model,
        model_parameters=parameters_to_train,
        optimizer=lambda params: get_optimizer(params, config),
    )

    model_engine.communication_data_type = torch.bfloat16

    train_dataloader = dataloader.PipelineDataLoader(
        train_data,
        model_engine.train_micro_batch_size_per_gpu(),
        model_engine.gradient_accumulation_steps(),
        model_engine.grid.get_data_parallel_world_size(),
        model_engine.grid.get_data_parallel_rank(),
    )
    model_engine.set_dataloader(train_dataloader)
    steps_per_epoch = len(train_dataloader) // model_engine.gradient_accumulation_steps()
    model_engine.total_steps = steps_per_epoch * config.get('epochs', 1)
    
    # handle Deepspeed optimizer wrapper (e.g. BF16_Optimizer)
    optimizer = getattr(optimizer, 'optimizer', optimizer)
   
    # see: https://github.com/tdrussell/qlora-pipe/pull/35#issuecomment-2495460307
    model_engine.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=make_rms_ratio_fn(config.get('beta2', 0.99))
    )
        
    # Eval dataset doesn't need to repeat; we just use this to track "epoch" so we know when we're done iterating over it.
    eval_dataloader = dataloader.PipelineDataLoader(
        eval_data,
        model_engine.train_micro_batch_size_per_gpu(),
        model_engine.gradient_accumulation_steps(),
        model_engine.grid.get_data_parallel_world_size(),
        model_engine.grid.get_data_parallel_rank(),
        shuffle=False,
    )

    # Create trainer
    trainer = Trainer(
        config=config,
        model_engine=model_engine,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        run_dir=run_dir,
        pipeline_model=pipeline_model,
        args=args,
        lora_config=lora_config
    )
    
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

    # Start training
    trainer.train(start_step=step)
