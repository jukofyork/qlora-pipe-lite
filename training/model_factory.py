from deepspeed.runtime.pipe.module import LayerSpec
from peft import LoraConfig, get_peft_model
import bitsandbytes
import json
import math
import optimi
import os
import torch

import transformers

from constants import (
    DEFAULT_BETA1,
    DEFAULT_BETA2,
    DEFAULT_EPS
)
from models import causal_lm_models
from pipeline import engine
from utils.unsloth_checkpoint import unsloth_checkpoint
from utils.utils import DTYPE_MAP, log_all

# Utility functions
def patch_bitsandbytes_cuda():
    """Ugly hack to move quantized models from GPU to CPU, and back to GPU again without triggering re-quantization"""
    bnb_cuda_old = bitsandbytes.nn.modules.Params4bit.cuda

    def bnb_cuda_hijack(self, device):
        if getattr(self, 'already_quantized', False):
            self.data = self.data.to(device)
            self.quant_state.to(device)
            return self
        self.already_quantized = True
        return bnb_cuda_old(self, device)

    bitsandbytes.nn.modules.Params4bit.cuda = bnb_cuda_hijack

def patch_decoder_layer_control_adapter(module, lora_config):
    """Create a new forward method that includes Control Adapter logic for DecoderLayerPipe."""

    control_adapter_type = getattr(lora_config, "control_adapter_type", "full")

    # Validate control adapter type from LoRA config
    allowed_types = {"full", "symmetrise", "antisymmetrise"}
    if control_adapter_type not in allowed_types:
        raise ValueError(f"Invalid control_adapter_type: {control_adapter_type}. Must be one of {sorted(allowed_types)}")

    def control_adapter_forward(inputs):
        hidden_states, attention_mask, cos, sin, control_classes, labels = inputs

        # Shift control_classes for causal LM: [control_classes[1:], 0_padding]
        batch_size, seq_len = control_classes.shape
        shift_control_classes = torch.cat([
            control_classes[:, 1:],
            torch.full((batch_size, 1), 0, device=control_classes.device, dtype=control_classes.dtype)
        ], dim=1)

        # Save input for residual computation
        input_hidden_states = hidden_states

        layer_output = module.orig(hidden_states, attention_mask=attention_mask, position_embeddings=(cos, sin))[0]
        torch_result_dtype = layer_output.dtype

        # Compute residual delta, apply optional dropout, then cast to adapter dtype
        layer_delta = layer_output - input_hidden_states
        x = module.control_dropout(layer_delta).to(module.control_A.weight.dtype)

        # Compute adapter output
        if control_adapter_type == "full":
            # Standard low-rank path: x -> A -> B -> scale
            adapter_output = module.control_B(module.control_A(x)) * module.control_scaling
        else:
            # Use low-rank paths with weights to support symmetric variants efficiently
            A_w = module.control_A.weight  # [r, H]
            B_w = module.control_B.weight  # [H, r]

            # y_full = s * (x @ (B @ A)^T) = s * ((x @ A^T) @ B^T)
            y_full = module.control_scaling * ((x @ A_w.transpose(-2, -1)) @ B_w.transpose(-2, -1))

            # y_swap = s * (x @ (B @ A)) = s * ((x @ B) @ A)
            y_swap = module.control_scaling * ((x @ B_w) @ A_w)

            if control_adapter_type == "symmetrise":
                adapter_output = 0.5 * (y_full + y_swap)
            elif control_adapter_type == "antisymmetrise":
                adapter_output = 0.5 * (y_full - y_swap)

        # Zero out any with class 0, as these won't have any loss calculated due to having label = -100
        class0_mask = (shift_control_classes == 0).unsqueeze(-1)  # broadcast to [B, S, 1]
        adapter_output = torch.where(class0_mask, torch.zeros_like(adapter_output), adapter_output)

        # For class +1 samples: add adapter_output as normal, For class -1 samples: add the negation of adapter_output.
        # NOTE: This approximates the 1st order Neumann series approximation to the inverse when ρ(A) << 1.
        negate_mask = (shift_control_classes == -1).unsqueeze(-1)  # broadcast to [B, S, 1]
        adapter_output = torch.where(negate_mask, -adapter_output, adapter_output)

        # Cast adapter contribution back to original dtype and add to the residual stream
        result = layer_output + adapter_output.to(torch_result_dtype)

        return (result, attention_mask, cos, sin, control_classes, labels)

    return control_adapter_forward

def get_model_type(config):
    """Extract model type from model config."""
    with open(os.path.join(config['model_dir'], 'config.json')) as f:
        model_config = json.load(f)
        return model_config.get('model_type', 'llama')

def parse_layers_to_transform(config):
    """Parse layers_to_transform config into list of layer numbers."""
    layers_to_transform = []
    if layers_spec := config.get('layers_to_transform', None):
        parts = layers_spec.split(',')
        for part in parts:
            start, stop = part.split(':')
            layers_to_transform.extend(range(int(start), int(stop) + 1))
    return layers_to_transform

# Optimizer and scheduler functions
def get_optimizer(model_parameters, config):
    """Create optimizer with configuration from config.

    Do NOT use AdamW or set the 'weight_decay' parameter:
    - We do our own "LoRA-specific" decoupled weight decay on the composite matrix AB now.
    - Full fine-tuning always uses bfloat16 and weight decay will underflow due to catastrophic cancellation.

    By default, optimi will automatically use Kahan summation for any layers training in low precision.
    """
    optimizer_kwargs = {
        "params": model_parameters,
        "lr": config['lr'],
        "betas": (config.get('beta1', DEFAULT_BETA1), config.get('beta2', DEFAULT_BETA2)),
        "eps": config.get('eps', DEFAULT_EPS)
    }
    return optimi.Adam(**optimizer_kwargs)

def get_lr_scheduler(optimizer, config):
    """Create learning rate scheduler with RMS ratio scaling.

    This is similar to RAdam (https://arxiv.org/abs/1908.03265), but using a scheduler instead.
    See: https://github.com/tdrussell/qlora-pipe/pull/35#issuecomment-2495460307
    """
    beta = config.get('beta2', DEFAULT_BETA2)

    def rms_ratio_fn(step):
        return torch.sqrt(torch.tensor((1 - beta ** step) / (1 + beta ** step))).item()

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=rms_ratio_fn)

# Model creation functions
def create_model(config, model_type, trust_remote_code=False):
    """Create the base transformer model with appropriate quantization."""
    if config.get('full_fine_tune', False) or not config.get('load_in_4bit', False):
        quantization_config = None
    else:
        no_quant_modules = ['lm_head']
        if model_type == 'mixtral':
            no_quant_modules.append('gate')
        quantization_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,  # Always use bfloat16 for QLoRA fine-tuning regardless
            bnb_4bit_quant_type="nf4",
            llm_int8_skip_modules=no_quant_modules,
        )

    if model_type == 'llama':
        model = causal_lm_models.LlamaForCausalLMPipe(
            config,
            quantization_config=quantization_config,
            trust_remote_code=trust_remote_code
        )
    elif model_type == 'mistral' or model_type == 'mistral3':
        model = causal_lm_models.MistralForCausalLMPipe(
            config,
            quantization_config=quantization_config,
            trust_remote_code=trust_remote_code
        )
    elif model_type == 'mixtral':
        model = causal_lm_models.MixtralForCausalLMPipe(
            config,
            quantization_config=quantization_config,
            trust_remote_code=trust_remote_code
        )
    elif model_type == 'qwen2':
        model = causal_lm_models.Qwen2ForCausalLMPipe(
            config,
            quantization_config=quantization_config,
            trust_remote_code=trust_remote_code
        )
    elif model_type == 'qwen3':
        model = causal_lm_models.Qwen3ForCausalLMPipe(
            config,
            quantization_config=quantization_config,
            trust_remote_code=trust_remote_code
        )
    elif model_type == 'phi3':
        model = causal_lm_models.Phi3ForCausalLMPipe(
            config,
            quantization_config=quantization_config,
            trust_remote_code=trust_remote_code
        )
    elif model_type == 'cohere':
        model = causal_lm_models.CohereForCausalLMPipe(
            config,
            quantization_config=quantization_config,
            trust_remote_code=trust_remote_code
        )
    elif model_type == 'cohere2':
        model = causal_lm_models.Cohere2ForCausalLMPipe(
            config,
            quantization_config=quantization_config,
            trust_remote_code=trust_remote_code
        )
    elif model_type == 'gemma2':
        model = causal_lm_models.Gemma2ForCausalLMPipe(
            config,
            quantization_config=quantization_config,
            trust_remote_code=trust_remote_code
        )
    else:
        raise NotImplementedError()

    return model

def create_pipeline_model(model, config):
    """Create pipeline model from base model for distributed training."""
    # The "primary" layers of the model must have 'decoderlayer' in their name for activation checkpointing to work
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
        activation_checkpoint_func=unsloth_checkpoint,
        partition_method='estimated_size',
        use_column_major_topology=config.get('use_column_major_topology', False)
    )

    return pipeline_model

# Training configuration functions

def create_lora_config(config, target_modules, layers_to_transform):
    """Create LoRA configuration."""
    # Handle empty list case for Control Adapters
    use_dummy_target = target_modules == []
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

    # Add control adapter type for Control Adapters
    if config.get('use_control_adapters', False):
        lora_config.control_adapter_type = config.get('control_adapter_type', 'full')

    return lora_config

def apply_lora_adapters(model, config, lora_config):
    """Apply LoRA configuration to model."""
    lora_model = get_peft_model(model, lora_config)
    # If the underlying weights are floats, the lora weights have already been
    # cast to the same dtype, so we need to change the dtype here.
    for p in lora_model.parameters():
        if p.requires_grad:
            p.data = p.data.to(DTYPE_MAP[config.get('lora_weight_dtype', 'float32')])

    lora_model.model.config.use_cache = False
    for name, p in lora_model.named_parameters():
        p.original_name = name

def apply_control_adapters(model, config, lora_config):
    """Apply Control Adapters using the LoraConfig object."""
    layers_to_transform = lora_config.layers_to_transform
    adapter_rank = lora_config.r
    adapter_alpha = lora_config.lora_alpha
    adapter_dropout_p = lora_config.lora_dropout

    layer_count = 0
    applied_count = 0

    for name, module in model.named_modules():
        if 'decoderlayerpipe' in module.__class__.__name__.lower():
            layer_idx = module.layer_idx
            should_transform = (layers_to_transform is None or layer_idx in layers_to_transform)

            if should_transform:
                device = next(module.orig.parameters()).device
                hidden_size = module.orig.hidden_size

                # Add dropout
                module.control_dropout = torch.nn.Dropout(p=adapter_dropout_p) if adapter_dropout_p > 0 else torch.nn.Identity()
                module.control_dropout = module.control_dropout.to(device)

                # Create Control Adapter layers (following PEFT pattern)
                module.control_A = torch.nn.Linear(hidden_size, adapter_rank, bias=False).to(device)
                module.control_B = torch.nn.Linear(adapter_rank, hidden_size, bias=False).to(device)

                # Store scaling as attribute (needed in forward)
                module.control_scaling = adapter_alpha / adapter_rank

                # Add original_name for saving compatibility
                module.control_A.weight.original_name = f"base_model.model.model.layers.{layer_idx}.control_A.weight"
                module.control_B.weight.original_name = f"base_model.model.model.layers.{layer_idx}.control_B.weight"

                # Initialize
                torch.nn.init.kaiming_uniform_(module.control_A.weight, a=math.sqrt(5))
                torch.nn.init.zeros_(module.control_B.weight)

                # Cast to the desired dtype
                lora_weight_dtype = DTYPE_MAP[config.get('lora_weight_dtype', 'float32')]
                module.control_A.weight.data = module.control_A.weight.data.to(lora_weight_dtype)
                module.control_B.weight.data = module.control_B.weight.data.to(lora_weight_dtype)

                # Store original forward method and replace with Control Adapter version
                module._original_forward = module.forward
                module.forward = patch_decoder_layer_control_adapter(module, lora_config)

                applied_count += 1
            # else:
            #    log_all(f'not training {layer_idx} because it is not in layers_to_transform')

            layer_count += 1

    # Set original_name for all parameters (for saving compatibility)
    for name, p in model.named_parameters():
        if not hasattr(p, 'original_name'):
            p.original_name = name

    # Disable gradients for all base model parameters, enable only for Control Adapters
    for name, p in model.named_parameters():
        if 'control_A' in name or 'control_B' in name:
            p.requires_grad = True
        else:
            p.requires_grad = False

    # log_all(f"Applied Control Adapters to {applied_count} of {layer_count} decoder layers")

def configure_full_fine_tuning(model, config, target_modules, layers_to_transform):
    """Setup full fine-tuning by setting requires_grad on parameters."""
    for name, p in model.named_parameters():
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

# Main setup function
def setup_model_and_engine(config, args):
    """Complete model setup including LoRA/full fine-tuning and engine initialization."""
    patch_bitsandbytes_cuda()

    # Create model and pipeline
    model_type = get_model_type(config)
    model = create_model(config, model_type, trust_remote_code=args.trust_remote_code)

    # Disable KV cache for all training modes
    if hasattr(model, "config"):
        model.config.use_cache = False

    # Setup training adapters
    target_modules = config.get('target_modules', None)
    layers_to_transform = parse_layers_to_transform(config)

    if config.get('full_fine_tune', False):
        lora_config = None
        pipeline_model = create_pipeline_model(model, config)
        configure_full_fine_tuning(pipeline_model, config, target_modules, layers_to_transform)  # Apply to pipeline_model
    elif config.get('use_control_adapters', False):
        assert not target_modules, "Control Adapters don't use target_modules - they apply to entire decoder layers"
        lora_config = create_lora_config(config, [], layers_to_transform)
        pipeline_model = create_pipeline_model(model, config)
        apply_control_adapters(pipeline_model, config, lora_config)  # Apply to pipeline_model
    else:
        lora_config = create_lora_config(config, target_modules, layers_to_transform)
        apply_lora_adapters(model, config, lora_config)  # Apply to base model (PEFT wraps HF modules)
        pipeline_model = create_pipeline_model(model, config)  # Build the pipeline AFTER PEFT wrapping

    parameters_to_train = [p for p in pipeline_model.parameters() if p.requires_grad]

    model_engine, optimizer = engine.initialize(
        config=config,
        args=args,
        model=pipeline_model,
        model_parameters=parameters_to_train,
        optimizer=lambda params: get_optimizer(params, config),
    )

    if lora_config is None:
        weight_dtype = torch.bfloat16  # Always use bfloat16 for full fine-tuning regardless
    else:
        weight_dtype = DTYPE_MAP[config.get('lora_weight_dtype', 'float32')]

    model_engine.communication_data_type = weight_dtype

    # Handle Deepspeed optimizer wrapper (e.g. BF16_Optimizer)
    optimizer = getattr(optimizer, 'optimizer', optimizer)
    model_engine.lr_scheduler = get_lr_scheduler(optimizer, config)

    return model_engine, pipeline_model, lora_config, optimizer