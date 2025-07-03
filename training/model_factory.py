from deepspeed.runtime.pipe.module import LayerSpec
from peft import LoraConfig, get_peft_model
import bitsandbytes
import json
import optimi
import os
import torch

import transformers

from constants import (
    DEFAULT_BETA1,
    DEFAULT_BETA2,
    DEFAULT_OPTIMIZER_EPS,
    DEFAULT_WEIGHT_DECAY,
    DEFAULT_LORA_DROPOUT
)
from models import causal_lm_models
from pipeline import engine
from utils.unsloth_checkpoint import unsloth_checkpoint
from utils.utils import log

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

def patch_decoder_layer_control_adapter(module):
    """Create a new forward method that includes Control Adapter logic for DecoderLayerPipe."""

    def control_adapter_forward(inputs):
        hidden_states, attention_mask, cos, sin, labels, sample_weights = inputs

        # Save input for residual computation
        input_hidden_states = hidden_states

        # Original layer computation (equivalent to base_layer)
        layer_output = module.orig(hidden_states, attention_mask=attention_mask, position_embeddings=(cos, sin))[0]
        torch_result_dtype = layer_output.dtype

        # Control Adapter computation (following PEFT pattern)
        layer_delta = layer_output - input_hidden_states

        # Cast input to match adapter dtype (following PEFT pattern)
        layer_delta = layer_delta.to(module.control_A.weight.dtype)

        # Apply Control Adapter: dropout -> A -> B -> scaling (following PEFT function call pattern)
        adapter_output = module.control_B(module.control_A(module.control_dropout(layer_delta))) * module.control_scaling

        # Use sample_weights sign for negate
        negate_mask = (sample_weights < 0).any(dim=-1, keepdim=True)
        while negate_mask.dim() < adapter_output.dim():
            negate_mask = negate_mask.unsqueeze(-1)

        # Apply negate logic
        adapter_contribution = torch.where(negate_mask, -adapter_output, adapter_output)
        result = layer_output + adapter_contribution

        # Cast back to original dtype (following PEFT pattern)
        result = result.to(torch_result_dtype)

        return (result, attention_mask, cos, sin, labels, sample_weights)

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
    """Create AdamW optimizer with configuration from config."""
    optimizer_kwargs = {
        "params": model_parameters,
        "lr": config['lr'],
        "betas": (config.get('beta1', DEFAULT_BETA1), config.get('beta2', DEFAULT_BETA2)),
        "weight_decay": config.get('weight_decay', DEFAULT_WEIGHT_DECAY),
        "eps": config.get('eps', DEFAULT_OPTIMIZER_EPS),
        "kahan_sum": True
    }
    return optimi.AdamW(**optimizer_kwargs)

def get_lr_scheduler(optimizer, config):
    """Create learning rate scheduler with RMS ratio scaling."""
    beta = config.get('beta2', DEFAULT_BETA2)

    # see: https://github.com/tdrussell/qlora-pipe/pull/35#issuecomment-2495460307
    def rms_ratio_fn(step):
        return torch.sqrt(torch.tensor((1 - beta ** step) / (1 + beta ** step))).item()

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=rms_ratio_fn)

# Model creation functions
def create_model(config, model_type):
    """Create the base transformer model with appropriate quantization."""
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

    if model_type == 'llama':
        model = causal_lm_models.LlamaForCausalLMPipe(config, quantization_config=quantization_config)
    elif model_type == 'mistral' or model_type == 'mistral3':
        model = causal_lm_models.MistralForCausalLMPipe(config, quantization_config=quantization_config)
    elif model_type == 'mixtral':
        model = causal_lm_models.MixtralForCausalLMPipe(config, quantization_config=quantization_config)
    elif model_type == 'qwen2':
        model = causal_lm_models.Qwen2ForCausalLMPipe(config, quantization_config=quantization_config)
    elif model_type == 'phi3':
        model = causal_lm_models.Phi3ForCausalLMPipe(config, quantization_config=quantization_config)
    elif model_type == 'cohere':
        model = causal_lm_models.CohereForCausalLMPipe(config, quantization_config=quantization_config)
    elif model_type == 'cohere2':
        model = causal_lm_models.Cohere2ForCausalLMPipe(config, quantization_config=quantization_config)
    elif model_type == 'gemma2':
        model = causal_lm_models.Gemma2ForCausalLMPipe(config, quantization_config=quantization_config)
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
    return LoraConfig(
        r=config['lora_rank'],
        lora_alpha=config.get('lora_alpha', round(config['lora_rank'] ** 0.5)),  # rslora: s = 1/sqrt(rank)
        lora_dropout=config.get('lora_dropout', DEFAULT_LORA_DROPOUT),
        target_modules=target_modules if target_modules else 'all-linear',
        layers_to_transform=layers_to_transform if layers_to_transform else None,
        task_type='CAUSAL_LM'
    )

def apply_lora_adapters(model, config, lora_config):
    """Apply LoRA configuration to model."""
    lora_model = get_peft_model(model, lora_config)
    # Cast LoRA weights to bfloat16 to match underlying model weights
    for p in lora_model.parameters():
        if p.requires_grad:
            p.data = p.data.to(torch.bfloat16)

    lora_model.model.config.use_cache = False
    for name, p in lora_model.named_parameters():
        p.original_name = name

def apply_control_adapters(model, config, lora_config):
    """Apply Control Adapters using the LoraConfig object."""
    layers_to_transform = lora_config.layers_to_transform
    adapter_rank = lora_config.r
    adapter_alpha = lora_config.lora_alpha
    adapter_dropout_p = lora_config.lora_dropout

    layer_idx = 0
    applied_count = 0

    for name, module in model.named_modules():
        if 'decoderlayerpipe' in module.__class__.__name__.lower():
            should_transform = (layers_to_transform is None or layer_idx in layers_to_transform)

            if should_transform:
                hidden_size = module.orig.hidden_size

                # Add dropout
                module.control_dropout = torch.nn.Dropout(p=adapter_dropout_p) if adapter_dropout_p > 0 else torch.nn.Identity()

                # Create Control Adapter layers (following PEFT pattern)
                module.control_A = torch.nn.Linear(hidden_size, adapter_rank, bias=False)
                module.control_B = torch.nn.Linear(adapter_rank, hidden_size, bias=False)

                # Store scaling as attribute (needed in forward)
                module.control_scaling = adapter_alpha / adapter_rank

                # Add original_name for saving compatibility
                module.control_A.weight.original_name = f"base_model.model.model.layers.{layer_idx}.control_adapter_A.weight"
                module.control_B.weight.original_name = f"base_model.model.model.layers.{layer_idx}.control_adapter_B.weight"

                # Initialize
                torch.nn.init.kaiming_uniform_(module.control_A.weight, a=5 ** 0.5)
                torch.nn.init.zeros_(module.control_B.weight)

                # Cast to bfloat16 like the original apply_lora_adapters
                module.control_A.weight.data = module.control_A.weight.data.to(torch.bfloat16)
                module.control_B.weight.data = module.control_B.weight.data.to(torch.bfloat16)

                # Store original forward method and replace with Control Adapter version
                module._original_forward = module.forward
                module.forward = patch_decoder_layer_control_adapter(module)

                applied_count += 1

            layer_idx += 1

    # Set original_name for all parameters (for saving compatibility)
    for name, p in model.named_parameters():
        if not hasattr(p, 'original_name'):
            p.original_name = name

    log(f"Applied Control Adapters to {applied_count} of {layer_idx} decoder layers")

def configure_full_fine_tuning(model, config, target_modules, layers_to_transform):
    """Setup full fine-tuning by setting requires_grad on parameters."""
    for name, p in model.named_parameters():
        p.original_name = name

    for name, p in model.named_parameters():
        should_train = True
        if target_modules and not any(target in name for target in target_modules):
            should_train = False
            log(f'not training {name} because it is not present in target_modules')
        elif layers_to_transform and 'model.layers.' in name:
            layer_num = int(name.split('model.layers.')[1].split('.')[0])
            if layer_num not in layers_to_transform:
                should_train = False
                log(f'not training {name} because layer {layer_num} is not in layers_to_transform')
        p.requires_grad = should_train

def setup_training_adapters(model, config):
    """Setup LoRA, Control Adapters, or full fine-tuning adapters."""
    target_modules = config.get('target_modules', [])
    layers_to_transform = parse_layers_to_transform(config)

    if config.get('full_fine_tune', False):
        lora_config = None
        configure_full_fine_tuning(model, config, target_modules, layers_to_transform)
    elif config.get('use_control_adapters', False):
        # Assert that target_modules is empty for Control Adapters
        assert not target_modules or target_modules == [], \
            "Control Adapters don't use target_modules - they apply to entire decoder layers"

        lora_config = create_lora_config(config, [], layers_to_transform)  # Empty target_modules
        apply_control_adapters(model, config, lora_config)
    else:
        lora_config = create_lora_config(config, target_modules, layers_to_transform)
        apply_lora_adapters(model, config, lora_config)

    return lora_config

# Main setup function
def setup_model_and_engine(config, args):
    """Complete model setup including LoRA/full fine-tuning and engine initialization."""
    patch_bitsandbytes_cuda()

    # Create model and pipeline
    model_type = get_model_type(config)
    model = create_model(config, model_type)
    pipeline_model = create_pipeline_model(model, config)

    # Setup training adapters (LoRA or full fine-tuning)
    lora_config = setup_training_adapters(model, config)

    parameters_to_train = [p for p in pipeline_model.parameters() if p.requires_grad]

    model_engine, optimizer = engine.initialize(
        config=config,
        args=args,
        model=pipeline_model,
        model_parameters=parameters_to_train,
        optimizer=lambda params: get_optimizer(params, config),
    )

    model_engine.communication_data_type = torch.bfloat16

    # Handle Deepspeed optimizer wrapper (e.g. BF16_Optimizer)
    optimizer = getattr(optimizer, 'optimizer', optimizer)
    model_engine.lr_scheduler = get_lr_scheduler(optimizer, config)

    return model_engine, pipeline_model, lora_config, optimizer