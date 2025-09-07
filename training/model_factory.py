"""
Model factory utilities.

Provides functions to:
- Patch BitsAndBytes CUDA path to safely move 4-bit tensors across devices
- Detect model_type and instantiate base HF models (optionally 4-bit)
- Parse layers_to_transform
- Build LoRA configuration and apply LoRA adapters
- Configure full fine-tuning parameter selection
"""
from peft import LoraConfig, get_peft_model
import json
import os
import torch

import transformers

from models.causal_lm import (
    CohereForCausalLmPipe,
    Cohere2ForCausalLmPipe,
    Gemma2ForCausalLmPipe,
    LlamaForCausalLmPipe,
    MistralForCausalLmPipe,
    MixtralForCausalLmPipe,
    Qwen2ForCausalLmPipe,
    Phi3ForCausalLmPipe,
    Qwen3ForCausalLmPipe,
)
from utils.utils import DTYPE_MAP

def create_model(config, trust_remote_code=False):
    """
    Create the base transformer model with appropriate quantization for training mode.
    """
    with open(os.path.join(config['model_dir'], 'config.json')) as f:
        model_config = json.load(f)
        model_type = model_config.get('model_type', 'llama')

    if config.get('full_fine_tune', False) or not config.get('load_in_4bit', False):
        quantization_config = None
    else:
        no_quant_modules = ['lm_head']
        if model_type == 'mixtral':
            no_quant_modules.append('gate')
        quantization_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            llm_int8_skip_modules=no_quant_modules,
        )

    if model_type == 'cohere':
        model = CohereForCausalLmPipe(config, quantization_config=quantization_config, trust_remote_code=trust_remote_code)
    elif model_type == 'cohere2':
        model = Cohere2ForCausalLmPipe(config, quantization_config=quantization_config, trust_remote_code=trust_remote_code)
    elif model_type == 'gemma2':
        model = Gemma2ForCausalLmPipe(config, quantization_config=quantization_config, trust_remote_code=trust_remote_code)
    elif model_type == 'llama':
        model = LlamaForCausalLmPipe(config, quantization_config=quantization_config, trust_remote_code=trust_remote_code)
    elif model_type == 'mistral' or model_type == 'mistral3':
        model = MistralForCausalLmPipe(config, quantization_config=quantization_config, trust_remote_code=trust_remote_code)
    elif model_type == 'mixtral':
        model = MixtralForCausalLmPipe(config, quantization_config=quantization_config, trust_remote_code=trust_remote_code)
    elif model_type == 'phi3':
        model = Phi3ForCausalLmPipe(config, quantization_config=quantization_config, trust_remote_code=trust_remote_code)
    elif model_type == 'qwen2':
        model = Qwen2ForCausalLmPipe(config, quantization_config=quantization_config, trust_remote_code=trust_remote_code)
    elif model_type == 'qwen3':
        model = Qwen3ForCausalLmPipe(config, quantization_config=quantization_config, trust_remote_code=trust_remote_code)
    else:
        raise NotImplementedError()

    return model

def parse_layers_to_transform(config):
    """
    Parse a compact 'layers_to_transform' string into a list of layer indices.

    Expected format in config: "start1:stop1,start2:stop2,..."
    Example: "0:3,10:12" -> [0,1,2,3,10,11,12]

    Args:
        config (dict): Configuration with optional 'layers_to_transform' string.

    Returns:
        list[int]: Expanded list of layer indices (possibly empty).
    """
    layers_to_transform = []
    if layers_spec := config.get('layers_to_transform', None):
        parts = layers_spec.split(',')
        for part in parts:
            start, stop = part.split(':')
            layers_to_transform.extend(range(int(start), int(stop) + 1))
    return layers_to_transform

def configure_full_fine_tuning(model, config, target_modules, layers_to_transform):
    """
    Set requires_grad for parameters to enable full fine-tuning with optional scoping.

    Behavior:
        - Assign 'original_name' to parameters that don't have it (preserve HF names if present)
        - Enable training for all parameters by default
        - If target_modules is provided, only parameters whose names contain any target string are trained
        - If layers_to_transform is provided, restrict to selected transformer layers

    Args:
        model (torch.nn.Module): Pipeline model whose parameters to mark as trainable.
        config (dict): Unused here; kept for symmetry and future extension.
        target_modules (Optional[list[str]]): Substrings of parameter names that should be trainable.
        layers_to_transform (Optional[list[int]]): Specific transformer layer indices to train.
    """
    # Preserve existing original_name (from HF) if present; set only when missing
    for name, p in model.named_parameters():
        if not hasattr(p, 'original_name'):
            p.original_name = name

    for name, p in model.named_parameters():
        should_train = True
        if target_modules and not any(target in name for target in target_modules):
            should_train = False
            # log_all(f'not training {name} because it is not present in target_modules')
        elif layers_to_transform and 'model.layers.' in name:
            layer_idx = int(name.split('model.layers.')[1].split('.')[0])
            if layer_idx not in layers_to_transform:
                should_train = False
                # log_all(f'not training {name} because layer {layer_idx} is not in layers_to_transform')
        p.requires_grad = should_train

    # Fail-fast validation: if using tied layers (FFT with tied embeddings), both ends must be trainable.
    # DeepSpeed will all_reduce weight.grad for tied parameters; a frozen end (grad=None) would crash.
    tied_param_ids = set()
    # PipelineModule sets these only if TiedLayerSpec was used
    has_ties = hasattr(model, 'tied_modules') and hasattr(model, 'tied_weight_attrs') and len(model.tied_modules) > 0
    if has_ties:
        # Gather tied parameters present on this stage
        for key, tied_module in model.tied_modules.items():
            weight_attrs = model.tied_weight_attrs.get(key, [])
            for attr_name in weight_attrs:
                try:
                    # Recursive getattr matches DS internal logic
                    tied_param = model._recursive_getattr(tied_module, attr_name)
                    tied_param_ids.add(id(tied_param))
                except Exception:
                    pass

        # Validate that all tied params are trainable on this stage
        for pname, p in model.named_parameters():
            if id(p) in tied_param_ids and not p.requires_grad:
                raise ValueError(
                    f'Full fine-tuning with tied embeddings requires training all tied weights. '
                    f'Parameter "{pname}" is frozen by your target_modules/layers_to_transform selection. '
                    f'Please include both the embedding and the lm_head in training or remove the restriction.'
                )

def create_lora_config(config, target_modules, layers_to_transform):
    """
    Build a PEFT LoRA configuration compatible with this training setup.

    Behavior:
        - If target_modules is an empty list (Control Adapters path), we pass a dummy 'none'
          value to peft for compatibility, then reset to [] afterward.
        - Otherwise, use 'all-linear' when target_modules is None to target all linear modules.

    Args:
        config (dict): Must include 'lora_rank' and optional 'lora_dropout'.
        target_modules (Optional[list[str]]): Specific module name substrings or None/'all-linear'.
        layers_to_transform (Optional[list[int]]): Subset of transformer layers to target.

    Returns:
        peft.LoraConfig: Config object to be used by get_peft_model or Control Adapters.
    """
    # Handle empty list case for Control Adapters
    use_dummy_target = target_modules == []
    actual_target_modules = ['none'] if use_dummy_target else (target_modules if target_modules else 'all-linear')

    lora_config = LoraConfig(
        r=config['lora_rank'],
        lora_alpha=config['lora_rank'],  # NOTE: We fix lora_alpha = lora_rank and then just adjust learning rate...
        lora_dropout=config.get('lora_dropout', 0.0),
        target_modules=actual_target_modules,
        layers_to_transform=layers_to_transform if layers_to_transform else None,
        task_type='CAUSAL_LM'
    )

    # Reset target_modules to empty list if we used dummy value
    if use_dummy_target:
        lora_config.target_modules = []

    return lora_config

def apply_lora_adapters(model, config, lora_config):
    """
    Apply LoRA configuration to the base HF model via PEFT and fix trainable dtypes.

    Steps:
        - Wrap model with PEFT using lora_config
        - Cast trainable parameters to desired dtype as per config['lora_weight_dtype']
        - Disable KV cache for training
        - Set 'original_name' on parameters to support custom saving flows

    Args:
        model (transformers.PreTrainedModel): Base model to wrap with PEFT.
        config (dict): Training configuration containing 'lora_weight_dtype'.
        lora_config (peft.LoraConfig): LoRA configuration.
    """
    lora_model = get_peft_model(model, lora_config)
    # If the underlying weights are floats, the lora weights have already been
    # cast to the same dtype, so we need to change the dtype here.
    for p in lora_model.parameters():
        if p.requires_grad:
            p.data = p.data.to(DTYPE_MAP[config.get('lora_weight_dtype', 'float32')])

    lora_model.model.config.use_cache = False
    for name, p in lora_model.named_parameters():
        p.original_name = name