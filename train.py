from datetime import datetime, timedelta, timezone
import argparse
import deepspeed
import os
import shutil
import toml

import transformers

from dataset_utils import load_datasets
from training.model_factory import setup_model_and_engine, get_model_type
from training.trainer import Trainer
from utils import is_main_process
import dataloader

parser = argparse.ArgumentParser()
parser.add_argument('--config',
                    help='Path to TOML configuration file.')
parser.add_argument('--local_rank', type=int, default=-1,
                    help='local rank passed from distributed launcher')
parser.add_argument('--resume_from_checkpoint', action='store_true', default=None,
                    help='resume training from the most recent checkpoint')
parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()

def setup_distributed_training(config):
    """Initialize distributed training and return run directory."""
    deepspeed.init_distributed(timeout=timedelta(hours=2))

    # Ensure output directory exists
    if is_main_process():
        os.makedirs(config['output_dir'], exist_ok=True)

    # Create new run directory for fresh training runs
    if not args.resume_from_checkpoint and is_main_process():
        run_dir = os.path.join(config['output_dir'], datetime.now(timezone.utc).strftime('%Y%m%d_%H-%M-%S'))
        os.makedirs(run_dir, exist_ok=True)
        shutil.copy(args.config, run_dir)
        shutil.copy(args.deepspeed_config, run_dir)

    # Synchronize all processes before determining run directory
    deepspeed.comm.barrier()

    # Get most recent run directory
    existing_runs = list(sorted(glob.glob(os.path.join(config['output_dir'], '*'))))
    if not existing_runs:
        if args.resume_from_checkpoint:
            raise RuntimeError(f"Cannot resume: no existing run directories found in {config['output_dir']}")
        else:
            raise RuntimeError(f"No run directories found in {config['output_dir']} (this shouldn't happen)")

    return existing_runs[-1]

def setup_tokenizer(config):
    """Load and configure tokenizer."""
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config['model'], local_files_only=True, model_max_length=int(1e30),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def setup_dataloaders(train_data, eval_data, model_engine):
    """Create train and eval dataloaders for pipeline parallelism."""
    train_dataloader = dataloader.PipelineDataLoader(
        train_data,
        model_engine.train_micro_batch_size_per_gpu(),
        model_engine.gradient_accumulation_steps(),
        model_engine.grid.get_data_parallel_world_size(),
        model_engine.grid.get_data_parallel_rank(),
    )

    eval_dataloader = dataloader.PipelineDataLoader(
        eval_data,
        model_engine.train_micro_batch_size_per_gpu(),
        model_engine.gradient_accumulation_steps(),
        model_engine.grid.get_data_parallel_world_size(),
        model_engine.grid.get_data_parallel_rank(),
        shuffle=False,
    )

    return train_dataloader, eval_dataloader

if __name__ == '__main__':
    # Load configuration
    with open(args.config) as f:
        config = toml.load(f)

    # Setup distributed training and data
    run_dir = setup_distributed_training(config)
    model_type = get_model_type(config)
    tokenizer = setup_tokenizer(config)
    train_data, eval_data = load_datasets(config, tokenizer)

    # Setup model and training engine
    model_engine, pipeline_model, lora_config, optimizer = setup_model_and_engine(config, model_type, args)

    # Setup data loading
    train_dataloader, eval_dataloader = setup_dataloaders(train_data, eval_data, model_engine)

    # Configure training schedule
    model_engine.set_dataloader(train_dataloader)
    steps_per_epoch = len(train_dataloader) // model_engine.gradient_accumulation_steps()
    model_engine.total_steps = steps_per_epoch * config.get('epochs', 1)

    # Initialize and start training
    trainer = Trainer(
        model_engine=model_engine,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        run_dir=run_dir,
        pipeline_model=pipeline_model,
        args=args,
        lora_config=lora_config,
        model_dir=config['model'],
        epochs=config.get('epochs', 1),
        evals_per_run=config.get('evals_per_run', 10),
        checkpoint_interval=config.get('checkpoint_interval', 60),
        max_checkpoints=config.get('max_checkpoints', -1),
        resume_from_checkpoint=args.resume_from_checkpoint
    )

    trainer.train()