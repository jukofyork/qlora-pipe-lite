from deepspeed.runtime.pipe.module import LayerSpec
from peft import LoraConfig, get_peft_model
import torch

import transformers

from models import causal_lm_models
import engine
import unsloth_utils

def parse_layers_to_transform(config):
    """Parse layers_to_transform config into list of layer numbers."""
    layers_to_transform = []
    if layers_spec := config.get('layers_to_transform', None):
        parts = layers_spec.split(',')
        for part in parts:
            start, stop = part.split(':')
            layers_to_transform.extend(range(int(start), int(stop) + 1))
    return layers_to_transform

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
    # CAREFUL! The "primary" layers of the model have to have 'decoderlayer' in them for
    # activation checkpointing to automatically work correctly.
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
    # If the underlying weights are floats, the lora weights have already been
    # cast to the same dtype, so we need to change the dtype here.
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
            print(f'not training {name} because it is not present in target_modules')
        elif layers_to_transform and 'model.layers.' in name:
            layer_num = int(name.split('model.layers.')[1].split('.')[0])
            if layer_num not in layers_to_transform:
                should_train = False
                print(f'not training {name} because layer {layer_num} is not in layers_to_transform')
        p.requires_grad = should_train
