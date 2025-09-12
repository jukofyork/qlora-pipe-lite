"""
Control Adapter utilities.

Provides functions to:
- Inject Control Adapter parameters and patch layer forward logic
- Compute Control Adapter forward/inverse transforms
- Load/parse Control Adapter weights and keys
- Patch adapter_config.json for conversion
- Generate model weight keys and LoRA keys for conversion
- Load base model weights from safetensors shards
"""

from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple
import json
import re
import safetensors.torch
import torch

def apply_control_adapters(model, layers_to_transform, adapter_rank, adapter_dropout, adapter_dtype=torch.float32):
    """
    Inject Control Adapter parameters into decoder layers and patch their forward method.

    Parameters:
        model               : Pipeline model containing DecoderLayerPipe modules (with 'orig' submodule).
        layers_to_transform : list[int] | None, layer indices to transform; None transforms all decoder layers.
        adapter_rank        : int, the rank (r) for the adapter subspace (shape: hidden_size x r).
        adapter_dropout     : float in [0, 1], dropout probability applied to residual delta before projection.
        adapter_dtype       : torch.dtype for the adapter parameters (e.g., torch.float32).

    Behavior:
        - For each selected decoder layer:
          * Adds control_dropout module (Dropout or Identity)
          * Creates trainable parameters:
              - control_Q: [hidden_size, r], orthogonally initialized
              - control_S: [r], zeros initialized (log-parameterization of λ: 1 + λ = exp(S))
          * Casts control_Q and control_S to adapter_dtype
          * Patches layer.forward with a wrapper that:
              - Computes residual delta
              - Projects into Q-subspace
              - Applies class-conditional transform using expm1(S) and its inverse
              - Masks class 0 positions to zero contribution
              - Adds adapter contribution back to residual stream
        - Sets original_name for saving compatibility
        - Sets requires_grad True for control_Q/control_S and False for other parameters on the layer
    """

    def patch_decoder_layer_control_adapter(module):

        def control_adapter_forward(inputs):
            if not isinstance(inputs, (tuple, list)) or len(inputs) != 7:
                raise ValueError(
                    "expects 7-tuple: (hidden_states, attention_mask, cos, sin, cache_position, control_classes, labels)"
                )

            hidden_states, attention_mask, cos, sin, cache_position, control_classes, labels = inputs

            # Shift control_classes for causal LM: [control_classes[1:], 0_padding]
            batch_size, seq_len = control_classes.shape
            shift_control_classes = torch.cat([
                control_classes[:, 1:],
                torch.full((batch_size, 1), 0, device=control_classes.device, dtype=control_classes.dtype)
            ], dim=1)

            # Save input for residual computation
            input_hidden_states = hidden_states

            layer_output = module.orig(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=(cos, sin),
                cache_position=cache_position
            )[0]
            torch_result_dtype = layer_output.dtype

            # Compute residual delta, apply optional dropout, then cast to adapter dtype
            layer_delta = layer_output - input_hidden_states
            x = module.control_dropout(layer_delta).to(module.control_Q.dtype)

            # Project to adapter subspace via Q (semi-orthogonal columns: Q^T Q ≈ I_r)
            x_q = x @ module.control_Q  # [batch_size, seq_len, adapter_rank]

            # Per-rank eigenvalue offsets in the Q-subspace using a log-parameterization:
            # λ = exp(S) - 1, so 1 + λ = exp(S) > 0 and the forward operator is I + diag(λ)
            lambda_pos_coeff = torch.expm1(module.control_S)  # [adapter_rank]

            # Exact inverse for the -1 branch (exact if Q is orthonormal; otherwise approximate):
            # (I + Q diag(λ) Q^T)^{-1} - I = -Q diag(λ/(1+λ)) Q^T
            # Identity: λ' = -λ/(1+λ) = exp(-S) - 1
            lambda_neg_coeff = torch.expm1(-module.control_S)  # [adapter_rank]

            # +1 class delta: Q diag(λ) Q^T x
            out_pos = ((x_q * lambda_pos_coeff) @ module.control_Q.T)  # [batch_size, seq_len, hidden_size]

            # -1 class delta (exact inverse): Q diag(λ') Q^T x
            out_neg = ((x_q * lambda_neg_coeff) @ module.control_Q.T)  # [batch_size, seq_len, hidden_size]

            # Select +1 branch where class==1; otherwise use inverse branch (class -1)
            class1_mask = (shift_control_classes == 1).unsqueeze(-1)  # broadcast to [batch_size, seq_len, 1]
            adapter_output = torch.where(class1_mask, out_pos, out_neg)

            # 0 class: zero-out (no loss; labels = -100)
            class0_mask = (shift_control_classes == 0).unsqueeze(-1)  # broadcast to [batch_size, seq_len, 1]
            adapter_output = torch.where(class0_mask, torch.zeros_like(adapter_output), adapter_output)

            # Cast adapter contribution back to original dtype and add to the residual stream
            result = layer_output + adapter_output.to(torch_result_dtype)  # [batch_size, seq_len, hidden_size]

            return (result, attention_mask, cos, sin, cache_position, control_classes, labels)

        return control_adapter_forward

    for name, module in model.named_modules():
        if 'decoderlayerpipe' in module.__class__.__name__.lower():
            layer_idx = module.layer_idx
            should_transform = (layers_to_transform is None or layer_idx in layers_to_transform)

            if should_transform:
                device = next(module.orig.parameters()).device
                hidden_size = module.orig.hidden_size

                # Add dropout
                module.control_dropout = torch.nn.Dropout(p=adapter_dropout) if adapter_dropout > 0 else torch.nn.Identity()
                module.control_dropout = module.control_dropout.to(device)

                # Create Control Adapter parameters
                module.control_Q = torch.nn.Parameter(torch.empty(hidden_size, adapter_rank, device=device))
                module.control_S = torch.nn.Parameter(torch.empty(adapter_rank, device=device))

                # Add original_name for saving compatibility
                module.control_Q.original_name = f"base_model.model.model.layers.{layer_idx}.control_Q"
                module.control_S.original_name = f"base_model.model.model.layers.{layer_idx}.control_S"

                # Initialize
                torch.nn.init.orthogonal_(module.control_Q)
                torch.nn.init.zeros_(module.control_S)

                # Cast to the desired dtype
                module.control_Q.data = module.control_Q.data.to(adapter_dtype)
                module.control_S.data = module.control_S.data.to(adapter_dtype)

                # Replace forward with Control Adapter version
                module.forward = patch_decoder_layer_control_adapter(module)

    # Set original_name for all parameters (for saving compatibility)
    for name, p in model.named_parameters():
        if not hasattr(p, 'original_name'):
            p.original_name = name

    # Disable gradients for all base model parameters, enable only for Control Adapters
    for name, p in model.named_parameters():
        if 'control_Q' in name or 'control_S' in name:
            p.requires_grad = True
        else:
            p.requires_grad = False

def apply_control_adapter_transform(Q, S, device=None, inverse: bool=False):
    """Compute Control Adapter delta: Q diag(λ) Q^T (forward) or exact inverse.

    Definitions:
        - Forward delta: λ = exp(S) - 1,      result = Q diag(λ) Q^T
        - Exact inverse: λ' = exp(-S) - 1,    result = Q diag(λ') Q^T
          (equivalently (I + Q diag(λ) Q^T)^{-1} - I = -Q diag(λ/(1+λ)) Q^T)

    Args:
        Q (Tensor): [hidden_size, r], semi-orthogonal columns (Q^T Q ≈ I_r).
        S (Tensor): [r], log-parameterization of eigenvalue offsets.
        device (str|torch.device|None): device to run on; defaults to 'cuda' if available else 'cpu'.
        inverse (bool): when True, compute exact inverse delta; otherwise forward delta.

    Returns:
        Tensor: [hidden_size, hidden_size] delta on the specified device (float32).
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    Q_gpu = Q.to(device=device, dtype=torch.float32)
    S_gpu = S.to(device=device, dtype=torch.float32)

    coeff = torch.expm1(-S_gpu) if inverse else torch.expm1(S_gpu)
    result = (Q_gpu * coeff.unsqueeze(0)) @ Q_gpu.T
    return result

def load_control_adapter_weights(adapter_path: Path) -> Tuple[List[Tuple[str, torch.Tensor]], Dict[str, torch.Tensor]]:
    """Load control adapter weights and return (sorted key-value pairs, full state dict)."""

    def _find_and_load_adapter_weights(path: Path) -> Tuple[Dict[str, torch.Tensor], str]:
        """Find and load control adapter weights, returning weights and filename."""
        for filename in ['adapter_model.safetensors', 'adapter_model.bin']:
            filepath = path / filename
            if filepath.exists():
                if filename.endswith('.safetensors'):
                    weights = safetensors.torch.load_file(filepath)
                else:
                    weights = torch.load(filepath, map_location='cpu', weights_only=True)
                return weights, filename

        raise FileNotFoundError("No adapter_model.safetensors or adapter_model.bin found in adapter directory")

    def _extract_layer_num(item):
        """Extract layer number and parameter type from control adapter key for sorting."""
        key, _ = item
        match = re.search(r'layers\.(\d+)\.control_(Q|S)', key)
        if match:
            layer_num = int(match.group(1))
            param_type = match.group(2)
            return (layer_num, 0 if param_type == 'Q' else 1)
        return (float('inf'), 2)

    state_dict, _ = _find_and_load_adapter_weights(adapter_path)
    control_keys = list(state_dict.items())
    control_keys.sort(key=_extract_layer_num)
    return control_keys, state_dict

def parse_control_adapter_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[int, Dict[str, torch.Tensor]]:
    """Parse control adapter state dict into layer -> {Q, S} mapping."""
    control_adapters: Dict[int, Dict[str, torch.Tensor]] = {}

    for key, tensor in state_dict.items():
        match = re.search(r'layers\.(\d+)\.control_(Q|S)', key)
        if match:
            layer_idx = int(match.group(1))
            param_type = match.group(2)

            if layer_idx not in control_adapters:
                control_adapters[layer_idx] = {}
            control_adapters[layer_idx][param_type] = tensor
        else:
            raise ValueError(f"Could not parse control adapter key: {key}")

    return control_adapters

def copy_and_patch_adapter_config(input_path: Path, output_path: Path, args):
    """Copy adapter config and patch target_modules based on model type"""
    config_file = input_path / 'adapter_config.json'
    if not config_file.exists():
        raise FileNotFoundError(f"adapter_config.json not found in {input_path}")

    with open(config_file, 'r') as f:
        config = json.load(f)

    # Set target modules based on model type
    if getattr(args, 'mixtral', None):
        config['target_modules'] = ["w2"]
    elif getattr(args, 'cohere', None):
        config['target_modules'] = ["down_proj", "o_proj"]
    else:
        config['target_modules'] = ["down_proj"]

    with open(output_path / 'adapter_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    print("Updated and copied 'adapter_config.json'")

def generate_model_weight_keys(layer_idx: int, args) -> List[str]:
    """Generate model weight keys for a given layer based on model type."""
    base_key = f"model.layers.{layer_idx}"
    target_keys: List[str] = []

    if getattr(args, 'mixtral', None):
        for expert_idx in range(args.mixtral):
            target_keys.append(f"{base_key}.block_sparse_moe.experts.{expert_idx}.w2.weight")
    else:
        target_keys.append(f"{base_key}.mlp.down_proj.weight")
        if getattr(args, 'cohere', None):
            target_keys.append(f"{base_key}.self_attn.o_proj.weight")

    return target_keys

def generate_lora_key(layer_idx: int, target_key: str, adapter_type: str, args) -> str:
    """Generate LoRA A or B key for LoRA format."""
    base_key = f"base_model.model.model.layers.{layer_idx}"

    if getattr(args, 'mixtral', None):
        expert_match = re.search(r'experts\.(\d+)\.w2\.weight', target_key)
        if expert_match:
            expert_idx = expert_match.group(1)
            lora_base = f"{base_key}.block_sparse_moe.experts.{expert_idx}.w2"
        else:
            raise ValueError(f"Could not parse expert index from {target_key}")
    elif "o_proj" in target_key:
        lora_base = f"{base_key}.self_attn.o_proj"
    else:
        lora_base = f"{base_key}.mlp.down_proj"

    return f"{lora_base}.lora_{adapter_type}.weight"

def load_model_weights(model_path: Path) -> Dict[str, torch.Tensor]:
    """Load model weights from safetensors shards into CPU memory."""
    model_weights: Dict[str, torch.Tensor] = {}
    model_shards = list(model_path.glob('model*.safetensors'))
    if not model_shards:
        raise FileNotFoundError("No model*.safetensors files found in model directory")

    for shard in tqdm(model_shards, desc="Loading model shards"):
        with safetensors.torch.safe_open(shard, framework='pt', device='cpu') as f:
            for key in f.keys():
                model_weights[key] = f.get_tensor(key)

    return model_weights