#!/usr/bin/env python3
"""
Convert DeepSpeed pipeline-parallel checkpoints into a LoRA adapter directory.

USAGE:
    python convert_ds_checkpoint.py --input /path/to/ds_ckpt --config /path/to/train_config.toml --output /path/to/adapter_out
"""

import argparse

from training.model_factory import convert_ds_checkpoint_to_lora

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Convert DS pipeline checkpoints to LoRA adapter")
    parser.add_argument('--input', required=True, help='Input DeepSpeed checkpoint directory')
    parser.add_argument('--config', required=True, help='Training config TOML file')
    parser.add_argument('--output', required=True, help='Output LoRA directory')
    args = parser.parse_args()

    convert_ds_checkpoint_to_lora(args.input, args.config, args.output)