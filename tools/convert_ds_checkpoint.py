# Very hacky script to convert pipeline parallel Deepspeed checkpoints into a saved lora model.
# I originally wrote this because I screwed up the lora model saving initially, and needed a
# way to turn the training checkpoints into saved lora models to test them.

from glob import glob
from peft import LoraConfig
import json
import math
import os.path
import re
import toml
import torch

def parse_layers_to_transform(config):
    """Parse layers_to_transform config into list of layer numbers."""
    layers_to_transform = []
    if layers_spec := config.get('layers_to_transform', None):
        parts = layers_spec.split(',')
        for part in parts:
            start, stop = part.split(':')
            layers_to_transform.extend(range(int(start), int(stop) + 1))
    return layers_to_transform

def create_lora_config(config, target_modules, layers_to_transform):
    """Create LoRA configuration."""
    # Handle empty list case for Control Adapters
    use_dummy_target = target_modules == [] or target_modules is None
    actual_target_modules = ['none'] if use_dummy_target else (target_modules if target_modules else 'all-linear')

    lora_config = LoraConfig(
        r=config['lora_rank'],
        lora_alpha=config['lora_alpha'],
        lora_dropout=config.get('lora_dropout', 0.0),
        target_modules=actual_target_modules,
        layers_to_transform=layers_to_transform if layers_to_transform else None,
        task_type='CAUSAL_LM'
    )

    # Reset target_modules to empty list if we used dummy value
    if use_dummy_target:
        lora_config.target_modules = []

    return lora_config

def convert_ds_checkpoint_to_lora(ds_checkpoint_dir, config_path, lora_output_dir):
    # Load config
    with open(config_path) as f:
        config = toml.load(f)

    # Set lora_alpha to the "Rank-Stabilized LoRA" default, such that: scale = 1/sqrt(rank)
    if 'lora_alpha' not in config and 'lora_rank' in config:
        config['lora_alpha'] = round(math.sqrt(config['lora_rank']))

    # Convert checkpoint files
    layer_checkpoint_files = glob(os.path.join(ds_checkpoint_dir, 'layer_*-model_states.pt'))
    combined_state_dict = {}
    for path in layer_checkpoint_files:
        match = re.fullmatch('layer_(.+)-model_states.pt', os.path.basename(path))
        layer_idx = int(match.group(1)) - 2
        state_dict = torch.load(path, weights_only=True)
        for name, weight in state_dict.items():
            converted_name = f'base_model.model.model.layers.{layer_idx}.{name}'
            combined_state_dict[converted_name] = weight

    # Create LoRA config
    target_modules = config.get('target_modules', None)
    layers_to_transform = parse_layers_to_transform(config)
    lora_config = create_lora_config(config, target_modules, layers_to_transform)

    # Save files
    os.makedirs(lora_output_dir, exist_ok=True)
    torch.save(combined_state_dict, os.path.join(lora_output_dir, 'adapter_model.bin'))

    # Save adapter config as JSON
    with open(os.path.join(lora_output_dir, 'adapter_config.json'), 'w') as f:
        json.dump(lora_config.to_dict(), f, indent=2)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Input DeepSpeed checkpoint directory')
    parser.add_argument('--config', required=True, help='Training config TOML file')
    parser.add_argument('--output', required=True, help='Output LoRA directory')
    args = parser.parse_args()

    convert_ds_checkpoint_to_lora(args.input, args.config, args.output)