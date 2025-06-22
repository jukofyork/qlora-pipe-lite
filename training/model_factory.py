from deepspeed.runtime.pipe.module import LayerSpec
from peft import LoraConfig, get_peft_model
import bitsandbytes
import json
import optimi
import os
import torch

import transformers

from models import causal_lm_models
from utils import log
import engine
import unsloth_utils

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

def get_model_type(config):
    """Extract model type from model config."""
    with open(os.path.join(config['model'], 'config.json')) as f:
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
        "betas": (config.get('beta1', 0.9), config.get('beta2', 0.99)),
        "weight_decay": config.get('weight_decay', 0.0),
        "eps": config.get('eps', 1e-6),
        "kahan_sum": True
    }
    return optimi.AdamW(**optimizer_kwargs)

def get_lr_scheduler(optimizer, config):
    """Create learning rate scheduler with RMS ratio scaling."""
    beta = config.get('beta2', 0.99)

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
        activation_checkpoint_func=unsloth_utils.unsloth_checkpoint,
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
        lora_dropout=config['lora_dropout'] if 'lora_dropout' in config else 0.0,
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
    """Setup LoRA or full fine-tuning adapters."""
    target_modules = config.get('target_modules', [])
    layers_to_transform = parse_layers_to_transform(config)

    if config.get('full_fine_tune', False):
        lora_config = None
        configure_full_fine_tuning(model, config, target_modules, layers_to_transform)
    else:
        lora_config = create_lora_config(config, target_modules, layers_to_transform)
        apply_lora_adapters(model, config, lora_config)

    return lora_config

# Main setup function
def setup_model_and_engine(config, model_type, args):
    """Complete model setup including LoRA/full fine-tuning and engine initialization."""
    patch_bitsandbytes_cuda()

    # Create model and pipeline
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