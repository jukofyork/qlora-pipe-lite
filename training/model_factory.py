"""
Model factory utilities.

Provides functions to:
- Patch BitsAndBytes CUDA path to safely move 4-bit tensors across devices
- Detect model_type and instantiate base HF models (optionally 4-bit)
- Parse layers_to_transform
- Build LoRA configuration and apply LoRA adapters
- Configure full fine-tuning parameter selection
- Convert DeepSpeed pipeline checkpoints to a LoRA adapter (utility)
"""
from glob import glob
from peft import LoraConfig, get_peft_model
import json
import os
import re
import toml
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

    Supported formats:
        - Ranges: "start1:stop1,start2:stop2,..."
          e.g., "0:3,10:12" -> [0,1,2,3,10,11,12]
        - Singles: "1,2,3" -> [1,2,3]
        - Mixed: "0:2, 5, 7:8" -> [0,1,2,5,7,8]

    Args:
        config (dict): Configuration with optional 'layers_to_transform' string.

    Returns:
        list[int]: Expanded list of layer indices (possibly empty).
    """
    layers_to_transform = []
    if layers_spec := config.get('layers_to_transform', None):
        for part in layers_spec.split(','):
            token = part.strip()
            if not token:
                continue
            if ':' in token:
                start_str, stop_str = token.split(':', 1)
                start = int(start_str.strip())
                stop = int(stop_str.strip())
                layers_to_transform.extend(range(start, stop + 1))
            else:
                layers_to_transform.append(int(token))
    return layers_to_transform

def configure_full_fine_tuning(model, config):
    """
    Set requires_grad for parameters to enable full fine-tuning with optional scoping.

    Behavior:
        - Assign 'original_name' to parameters that don't have it (preserve HF names if present)
        - Enable training for all parameters by default
        - If target_modules is provided, only parameters whose names contain any target string are trained
        - If layers_to_transform is provided, restrict to selected transformer layers

    Args:
        model (torch.nn.Module): Pipeline model whose parameters to mark as trainable.
        config (dict): Training configuration containing optional 'target_modules'
                       and 'layers_to_transform'.
    """
    target_modules = config.get('target_modules', None)
    layers_to_transform = parse_layers_to_transform(config)

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

def create_lora_config(config):
    """
    Build a PEFT LoRA configuration based on the training config.

    Behavior:
        - Requires 'lora_rank' in config.
        - Reads 'target_modules' and 'layers_to_transform' from config.
        - If use_control_adapters is True:
            * target_modules must not be set (asserts falsy).
            * target_modules is forced to [] (adapters apply to entire decoder layers).
        - If target_modules is None:
            * defaults to 'all-linear' (target all linear modules).
        - If target_modules is an empty list:
            * raises ValueError (no point training a LoRA with nothing to train).
        - layers_to_transform is parsed and passed through unchanged.

    Returns:
        peft.LoraConfig
    """
    if 'lora_rank' not in config:
        raise KeyError("Training config TOML must define 'lora_rank' to build adapter_config.json")

    target_modules = config.get('target_modules', None)
    layers_to_transform = parse_layers_to_transform(config)

    if config.get('use_control_adapters', False):
        assert not target_modules, "Control Adapters don't use target_modules - they apply to entire decoder layers"
        target_modules = []
    elif target_modules is None:
        target_modules = 'all-linear'
    elif isinstance(target_modules, list) and len(target_modules) == 0:
        raise ValueError("Empty target_modules specified for LoRA - no point in training a LoRA with nothing to train")

    lora_config = LoraConfig(
        r=config['lora_rank'],
        lora_alpha=config['lora_rank'],  # NOTE: Fix lora_alpha = lora_rank and adjust LR externally
        lora_dropout=config.get('lora_dropout', 0.0),
        target_modules=target_modules,
        layers_to_transform=layers_to_transform if layers_to_transform else None,
        task_type='CAUSAL_LM'
    )

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

def convert_ds_checkpoint_to_lora(ds_checkpoint_dir, config_path, lora_output_dir):
    """
    Convert pipeline-parallel DeepSpeed checkpoints into a saved LoRA/control-adapter directory.

    Filters out BitsAndBytes 4-bit buffers (absmax/quant_map/quant_state, etc.) and base model weights,
    keeping only adapter tensors:
      - LoRA: *.lora_A.weight / *.lora_B.weight (and optional modules_to_save.*)
      - Control Adapters: control_Q / control_S

    Saved keys are normalized to HF/PEFT-compatible names:
      - Strips leading 'orig.' indirection introduced by pipeline wrappers
      - Removes PEFT wrappers '.modules_to_save' and '.default'
    """
    # Load config
    with open(config_path) as f:
        config = toml.load(f)

    # Convert checkpoint files
    layer_checkpoint_files = glob(os.path.join(ds_checkpoint_dir, 'layer_*-model_states.pt'))
    if not layer_checkpoint_files:
        raise FileNotFoundError(
            f"No checkpoint files matching 'layer_*-model_states.pt' found in '{ds_checkpoint_dir}'"
        )

    def _wants_key(name: str) -> bool:
        # Control adapters
        if 'control_Q' in name or 'control_S' in name:
            return True
        # LoRA
        if '.lora_A.' in name or name.endswith('.lora_A.weight'):
            return True
        if '.lora_B.' in name or name.endswith('.lora_B.weight'):
            return True
        # Optional PEFT "modules_to_save" items
        if '.modules_to_save.' in name:
            return True
        # Everything else (base weights, bnb buffers) -> ignore
        return False

    def _normalize_name(name: str) -> str:
        # Remove the pipeline wrapper indirection
        if name.startswith('orig.'):
            name = name[len('orig.'):]
        # Strip PEFT wrappers to match how we save during training
        name = name.replace('.default', '').replace('.modules_to_save', '')
        return name

    combined_state_dict = {}
    kept, dropped = 0, 0

    for path in layer_checkpoint_files:
        basename = os.path.basename(path)
        match = re.fullmatch(r'layer_(\d+)-model_states\.pt', basename)
        if not match:
            raise ValueError(
                f"Unexpected checkpoint filename: '{basename}' "
                f"(expected 'layer_<N>-model_states.pt' with integer N)"
            )
        layer_idx = int(match.group(1)) - 2
        state_dict = torch.load(path, weights_only=True)

        for name, weight in state_dict.items():
            if not _wants_key(name):
                dropped += 1
                continue
            kept += 1
            inner_name = _normalize_name(name)
            converted_name = f'base_model.model.model.layers.{layer_idx}.{inner_name}'
            combined_state_dict[converted_name] = weight

    # Create LoRA config (also used to carry metadata alongside control adapters)
    lora_config = create_lora_config(config)

    # Save files
    os.makedirs(lora_output_dir, exist_ok=True)
    torch.save(combined_state_dict, os.path.join(lora_output_dir, 'adapter_model.bin'))
    lora_config.save_pretrained(lora_output_dir)

    print(f"Converted DS checkpoint -> adapter: kept {kept} tensors, dropped {dropped} non-adapter entries")