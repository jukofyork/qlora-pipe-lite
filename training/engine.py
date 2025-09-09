from deepspeed import comm as dist
from deepspeed.runtime.config import DeepSpeedConfig
from deepspeed.runtime.pipe.engine import PipelineEngine
from deepspeed.runtime.pipe.module import PipelineModule, LayerSpec
from deepspeed.runtime.pipe.topology import ProcessTopology, PipeDataParallelTopology
import bitsandbytes
import optimi
import sys
import torch

from constants import (
    DEFAULT_BETA1,
    DEFAULT_BETA2,
    DEFAULT_EPS
)
from training.control_adapters import apply_control_adapters
from training.model_factory import (
    create_model,
    configure_full_fine_tuning,
    create_lora_config,
    apply_lora_adapters
)
from utils.unsloth_checkpoint import unsloth_checkpoint
from utils.utils import DTYPE_MAP

class Engine:
    """
    Orchestrates end-to-end setup of a DeepSpeed PipelineEngine.

    Behavior:
    - Patches BitsAndBytes CUDA path for safe device transfers (4-bit models)
    - Loads base HF model (optionally 4-bit quantized) based on model_type
    - Disables KV cache for training
    - Applies exactly one training mode:
      * Full fine-tuning (FFT) with optional target/layer scoping
      * Control Adapters applied to pipeline decoder layers
      * LoRA via PEFT (default), applied to base HF model before pipeline wrapping
    - Creates PipelineModule with activation checkpointing and desired topology
    - Initializes DeepSpeed PipelineEngine with optimizer and RMS-ratio LR scheduler
    - Exposes engine, pipeline model, optional LoRA config, and optimizer handles

    Parameters:
        config : dict-like training configuration
        args   : argparse.Namespace passed through to DeepSpeed PipelineEngine

    Attributes:
        pipeline_engine : deepspeed.runtime.pipe.engine.PipelineEngine
        pipeline_model  : deepspeed.runtime.pipe.module.PipelineModule
        lora_config     : Optional[peft.LoraConfig] (None for Full FT)
        optimizer       : torch.optim.Optimizer (possibly unwrapped from DeepSpeed)
    """

    def __init__(self, config, args):
        self._patch_bitsandbytes_cuda()
        self._patch_ds_tied_broadcast_no_grad()

        # Create model and pipeline
        model = create_model(config, trust_remote_code=args.trust_remote_code)

        # Disable KV cache for all training modes
        if hasattr(model, "config"):
            model.config.use_cache = False

        if config.get('full_fine_tune', False):
            lora_config = None
            pipeline_model = self._create_pipeline_model(model, config)
            configure_full_fine_tuning(pipeline_model, config)
        elif config.get('use_control_adapters', False):
            lora_config = create_lora_config(config)
            pipeline_model = self._create_pipeline_model(model, config)
            apply_control_adapters(
                pipeline_model,
                lora_config.layers_to_transform,
                lora_config.r,
                lora_config.lora_dropout,
                DTYPE_MAP[config.get('lora_weight_dtype', 'float32')]
            )
        else:
            lora_config = create_lora_config(config)
            apply_lora_adapters(model, config, lora_config)  # Apply to base model (PEFT wraps HF modules)
            pipeline_model = self._create_pipeline_model(model, config)  # Build the pipeline AFTER PEFT wrapping

        parameters_to_train = [p for p in pipeline_model.parameters() if p.requires_grad]

        # Initialize DeepSpeed engine
        assert pipeline_model is not None, "deepspeed.initialize requires a model"

        ds_config = {
            'train_micro_batch_size_per_gpu': 1,
            'gradient_accumulation_steps': config.get('gradient_accumulation_steps', 1),
            'gradient_clipping': 1.0,
            'steps_per_print': sys.maxsize  # We do our own printing in Trainer._print_train_progress()
        }

        mpu = pipeline_model.mpu()
        config_class = DeepSpeedConfig(ds_config, mpu)
        pipeline_engine = PipelineEngine(
            args=args,
            model=pipeline_model,
            optimizer=lambda params: self._get_optimizer(params, config),
            model_parameters=parameters_to_train,
            mpu=mpu,
            config=ds_config,
            config_class=config_class
        )

        optimizer = pipeline_engine.optimizer

        if lora_config is None:
            weight_dtype = torch.bfloat16  # Always use bfloat16 for full fine-tuning regardless
        else:
            weight_dtype = DTYPE_MAP[config.get('lora_weight_dtype', 'float32')]

        pipeline_engine.communication_data_type = weight_dtype

        # Handle Deepspeed optimizer wrapper (e.g. BF16_Optimizer)
        optimizer = getattr(optimizer, 'optimizer', optimizer)
        pipeline_engine.lr_scheduler = self._get_lr_scheduler(optimizer, config)

        self.pipeline_engine = pipeline_engine
        self.pipeline_model = pipeline_model
        self.lora_config = lora_config
        self.optimizer = optimizer

    def _patch_bitsandbytes_cuda(self):
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

    def _patch_ds_tied_broadcast_no_grad(self):
        """Ensure DeepSpeed's tied-weight broadcasts are not recorded by autograd."""
        if getattr(PipelineModule, "_qp_tied_broadcast_patched", False):
            return

        assert hasattr(PipelineModule, "_synchronize_tied_weights"), \
            "DeepSpeed PipelineModule missing _synchronize_tied_weights"

        original = PipelineModule._synchronize_tied_weights

        def _wrapped(self, *args, **kwargs):
            with torch.no_grad():
                return original(self, *args, **kwargs)

        PipelineModule._synchronize_tied_weights = _wrapped
        PipelineModule._qp_tied_broadcast_patched = True

    def _create_pipeline_model(self, model, config):
        """
        Create a PipelineModule from a base model for distributed training.

        Configuration:
            - Activation checkpointing of LayerSpec names containing 'decoderlayer'
              via unsloth_checkpoint (case-insensitive)
            - Partitioning controlled by config['partition_method']:
                * 'uniform'      : balances number of layers per stage (default)
                * 'parameters'   : balances trainable parameter counts per stage
                * "type:[regex]" : balances layers whose class names match [regex] (case-insensitive), eg:
                                     "type:decoderlayer"        [matches DecoderLayerPipe]
                                     "type:^decoderlayerpipe$"  [exact match]
                                     "type:(decoderlayerpipe|embeddingpipe|lmheadpipe)"
            - Topology may be column-major or standard using PipeDataParallelTopology

        Args:
            model (torch.nn.Module): Base model supporting to_layer_specs().
            config (dict): Training configuration including 'pipeline_stages' and topology preferences.

        Returns:
            deepspeed.runtime.pipe.module.PipelineModule: The pipeline-wrapped model.
        """
        # The "primary" layers of the model must have 'decoderlayer' in their name for activation checkpointing to work
        layers = model.to_layer_specs()
        checkpointable_layers = set()
        for layer in layers:
            if isinstance(layer, LayerSpec) and 'decoderlayer' in layer.typename.__name__.lower():
                checkpointable_layers.add(layer.typename.__name__)
        checkpointable_layers = list(checkpointable_layers)

        # Set up topology
        use_column_major_topology = config.get('use_column_major_topology', False)
        num_stages = config.get('pipeline_stages', 1)
        world_size = dist.get_world_size()
        num_dp = world_size // num_stages

        if use_column_major_topology:

            class ColumnMajorParallelTopology(ProcessTopology):

                def __init__(self, num_pp, num_dp):
                    # Swap the axes and dims to change the rank mapping
                    super().__init__(axes=['data', 'pipe'], dims=[num_dp, num_pp])

            topology = ColumnMajorParallelTopology(num_pp=num_stages, num_dp=num_dp)
        else:
            topology = PipeDataParallelTopology(num_pp=num_stages, num_dp=num_dp)

        # Validate partition_method
        partition_method = config.get('partition_method', 'uniform')
        if not isinstance(partition_method, str):
            raise TypeError("partition_method must be a string")
        if partition_method not in ('uniform', 'parameters') and not partition_method.startswith('type:'):
            raise ValueError("Invalid partition_method, expected: 'uniform', 'parameters' or 'type:[regex]'")

        pipeline_model = PipelineModule(
            layers=layers,
            num_stages=num_stages,
            topology=topology,
            partition_method=partition_method,
            activation_checkpoint_interval=1,
            activation_checkpoint_func=unsloth_checkpoint,
            checkpointable_layers=checkpointable_layers
        )

        return pipeline_model

    def _get_optimizer(self, model_parameters, config):
        """Create optimizer with configuration from config.

        Do NOT use AdamW or set the 'weight_decay' parameter:
        - We do our own "LoRA-specific" decoupled weight decay on the composite matrix AB now.
        - Full fine-tuning always uses bfloat16 and weight decay will underflow due to catastrophic cancellation.

        By default, optimi will automatically use Kahan summation for any layers training in low precision.

        Args:
            model_parameters (iterable): Parameters to optimize.
            config (dict): Must include 'lr'; may include 'beta1', 'beta2', 'eps'.

        Returns:
            torch.optim.Optimizer: The initialized optimizer instance.
        """
        optimizer_kwargs = {
            "params": model_parameters,
            "lr": config['lr'],
            "betas": (config.get('beta1', DEFAULT_BETA1), config.get('beta2', DEFAULT_BETA2)),
            "eps": config.get('eps', DEFAULT_EPS)
        }
        return optimi.Adam(**optimizer_kwargs)

    def _get_lr_scheduler(self, optimizer, config):
        """Create learning rate scheduler with RMS ratio scaling.

        This is similar to RAdam (https://arxiv.org/abs/1908.03265), but using a scheduler instead.
        See: https://github.com/tdrussell/qlora-pipe/pull/35#issuecomment-2495460307

        Args:
            optimizer (torch.optim.Optimizer): Optimizer to schedule.
            config (dict): Uses 'beta2' to compute the ratio schedule.

        Returns:
            torch.optim.lr_scheduler.LambdaLR: Scheduler that applies RMS-ratio scaling.
        """
        beta = config.get('beta2', DEFAULT_BETA2)

        def rms_ratio_fn(step):
            return torch.sqrt(torch.tensor((1 - beta ** step) / (1 + beta ** step))).item()

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=rms_ratio_fn)