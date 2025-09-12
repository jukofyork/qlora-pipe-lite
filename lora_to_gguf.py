#!/usr/bin/env python3
"""
Export a Hugging Face PEFT LoRA adapter to GGUF for llama.cpp.

NOTE: 'mixtral' model type is not yet supported (TODO).

USAGE:
    python lora_to_gguf.py --input /path/to/lora_adapter --output /path/to/adapter.gguf
                           [--arch llama] [--outtype F32]
"""

from pathlib import Path
from typing import List, Tuple
import argparse
import re
import torch

from training.model_factory import find_lora_weights, export_lora_gguf, load_lora_config, load_lora_state
import gguf

ALLOWED_ARCHS = {
    "llama",  # 'llama', 'mistral', 'mistral3'
    "command-r",  # 'cohere'
    "cohere2",  # 'cohere2'
    "gemma2",  # 'gemma2'
    "phi3",  # 'phi3'
    "qwen2",  # 'qwen2'
    "qwen3"  # 'qwen3'
}

OUTTYPE_MAP = {
    "F32": gguf.GGMLQuantizationType.F32,
    "F16": gguf.GGMLQuantizationType.F16,
    "BF16": gguf.GGMLQuantizationType.BF16,
    "Q8_0": gguf.GGMLQuantizationType.Q8_0,
}

def _map_hf_to_gguf_weight_base(arch: str, hf_base: str) -> str:
    """
    Map HF (PEFT) base path (without .lora_[AB].weight) to GGUF canonical weight base.
    Returns a string like: 'blk.{n}.attn_q' or 'blk.{n}.ffn_up', etc. Add '.weight' later.
    """
    # Expect 'model.' prefix - error if not present
    if not hf_base.startswith("model."):
        raise ValueError(f"Expected HF base key to start with 'model.': '{hf_base}'")
    hf_base = hf_base[len("model."):]

    # Extract layer number and component
    m = re.search(r"layers\.(\d+)\.(.*)$", hf_base)
    if not m:
        raise ValueError(f"Cannot extract layer index from LoRA key base: '{hf_base}'")
    layer = int(m.group(1))
    rest = m.group(2)

    # Handle attention projections
    if rest.startswith("self_attn."):
        proj = rest[len("self_attn."):]
        attn_map = {
            "q_proj": "attn_q",
            "k_proj": "attn_k",
            "v_proj": "attn_v",
            "o_proj": "attn_output",
        }
        if proj in attn_map:
            return f"blk.{layer}.{attn_map[proj]}"

    # Handle MLP projections
    if rest.startswith("mlp."):
        proj = rest[len("mlp."):]
        ffn_map = {
            "gate_proj": "ffn_gate",
            "up_proj": "ffn_up",
            "down_proj": "ffn_down",
        }
        if proj in ffn_map:
            return f"blk.{layer}.{ffn_map[proj]}"

    raise ValueError(f"Cannot map HF LoRA key base '{hf_base}' to GGUF canonical name for arch '{arch}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export PEFT LoRA adapter to GGUF for llama.cpp")
    parser.add_argument("--input", required=True, type=str,
                        help="Path to the LoRA adapter directory")
    parser.add_argument("--output", required=True, type=str,
                        help="Path to the output GGUF file")
    parser.add_argument("--arch", type=str, default="llama",
                        help="Target architecture name for GGUF (" + ", ".join(sorted(ALLOWED_ARCHS)) + ") (default: llama)")
    parser.add_argument("--outtype", type=str, default="F32",
                        help="Output tensor type for GGUF tensors (F32, F16, BF16, Q8_0) (default: F32)")

    args = parser.parse_args()

    arch = args.arch.lower()
    if arch not in ALLOWED_ARCHS:
        parser.error(f"--arch must be one of: {', '.join(sorted(ALLOWED_ARCHS))}")

    outtype = args.outtype.upper()
    if outtype not in OUTTYPE_MAP:
        parser.error(f"--outtype must be one of: {', '.join(OUTTYPE_MAP.keys())}")

    adapter_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    qtype = OUTTYPE_MAP[outtype]

    print(f"Loading LoRA adapter from: '{adapter_path}'")
    cfg = load_lora_config(adapter_path)
    alpha = cfg["lora_alpha"]
    lora_state = load_lora_state(adapter_path)

    # Collect base weight keys from LoRA A keys
    base_keys = set()
    for key in lora_state.keys():
        if key.endswith(".lora_A.weight"):
            # Convert "base_model.model.layers.0.self_attn.q_proj.lora_A.weight"
            # to "model.layers.0.self_attn.q_proj.weight"
            base_key = key.replace("base_model.", "").replace(".lora_A.weight", ".weight")
            base_keys.add(base_key)

    print(f"Converting {len(base_keys)} LoRA weight pairs to GGUF format...")

    # Build GGUF tensors list
    gguf_tensors: List[Tuple[str, torch.Tensor]] = []
    for base_key in base_keys:
        lora_A, lora_B = find_lora_weights(base_key, lora_state)
        base_without_weight = base_key.removesuffix('.weight')
        canonical_base = _map_hf_to_gguf_weight_base(arch, base_without_weight)
        name_a = f"{canonical_base}.weight.lora_a"
        name_b = f"{canonical_base}.weight.lora_b"

        gguf_tensors.append((name_a, lora_A.contiguous()))
        gguf_tensors.append((name_b, lora_B.contiguous()))

    # Sort for deterministic output
    gguf_tensors.sort(key=lambda x: x[0])

    export_lora_gguf(str(output_path), gguf_tensors, alpha=alpha, quant_type=qtype, architecture=arch)

    print(f"LoRA adapter exported to: '{output_path}'")