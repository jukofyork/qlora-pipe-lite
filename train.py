from datetime import datetime, timedelta, timezone
import argparse
import deepspeed
import glob
import math
import os
import shutil
import sys
import toml

import transformers

from constants import DEEPSPEED_TIMEOUT_HOURS
from pipeline import dataloader
from training.model_factory import setup_model_and_engine
from training.trainer import Trainer
from utils.dataset_utils import load_datasets
from utils.utils import is_main_process

parser = argparse.ArgumentParser()
parser.add_argument('--config',
                    help='Path to TOML configuration file.')
parser.add_argument('--local_rank', type=int, default=-1,
                    help='local rank passed from distributed launcher')
parser.add_argument('--resume_from_checkpoint', action='store_true', default=None,
                    help='resume training from the most recent checkpoint')
parser.add_argument("--add-prefix-space", action="store_true", default=False,
                    help="Add prefix space when tokenizing")
parser.add_argument("--trust-remote-code", action="store_true",
                    help="Allow custom code execution when loading models with non-standard architectures")
parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()

def setup_run_directory(config):
    """Create or determine the run directory for this training session."""

    # Ensure output directory exists
    if is_main_process():
        os.makedirs(config['output_dir'], exist_ok=True)

    # Create new run directory for fresh training runs
    if not args.resume_from_checkpoint and is_main_process():
        run_dir = os.path.join(config['output_dir'], datetime.now(timezone.utc).strftime('%Y%m%d_%H-%M-%S'))
        os.makedirs(run_dir, exist_ok=True)
        shutil.copy(args.config, run_dir)

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

if __name__ == '__main__':
    # Load configuration
    with open(args.config) as f:
        config = toml.load(f)

    # Increase deepspeed timeout to avoid timeouts during dataset loading
    deepspeed.init_distributed(timeout=timedelta(hours=DEEPSPEED_TIMEOUT_HOURS))

    # Setup run directory for this training session
    run_dir = setup_run_directory(config)

    # Load and configure tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config['model_dir'],
        local_files_only=True,
        model_max_length=sys.maxsize,
        add_prefix_space=args.add_prefix_space,
        trust_remote_code=args.trust_remote_code,
    )

    # Load, convert and split the datasets
    train_data, eval_data = load_datasets(config, tokenizer, run_dir)

    # Setup model and training engine
    model_engine, pipeline_model, lora_config, optimizer = setup_model_and_engine(config, args)

    # Create training dataloader with data parallelism and gradient accumulation
    train_dataloader = dataloader.PipelineDataLoader(
        train_data,
        model_engine.train_micro_batch_size_per_gpu(),
        model_engine.gradient_accumulation_steps(),
        model_engine.grid.get_data_parallel_world_size(),
        model_engine.grid.get_data_parallel_rank(),
    )

    # Get the (optional) separate evaluation gradient accumulation setting
    eval_gradient_accumulation_steps = config.get(
        'eval_gradient_accumulation_steps',
        config.get('gradient_accumulation_steps', 1)
    )

    # Create evaluation dataloader with evaluation gradient accumulation setting and no shuffling
    eval_dataloader = dataloader.PipelineDataLoader(
        eval_data,
        model_engine.train_micro_batch_size_per_gpu(),
        eval_gradient_accumulation_steps,
        model_engine.grid.get_data_parallel_world_size(),
        model_engine.grid.get_data_parallel_rank(),
        shuffle=False,
    )

    # Initialize trainer
    trainer = Trainer(
        config=config,
        model_engine=model_engine,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        run_dir=run_dir,
        pipeline_model=pipeline_model,
        args=args,
        lora_config=lora_config,
        optimizer=optimizer,
        eval_gradient_accumulation_steps=eval_gradient_accumulation_steps,
        resume_from_checkpoint=args.resume_from_checkpoint
    )

    # Start training
    trainer.train()