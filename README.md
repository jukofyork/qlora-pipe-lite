# qlora-pipe-lite

A streamlined fork of [qlora-pipe](https://github.com/tdrussell/qlora-pipe) focused on Control Adapter training.

**Key Features:**
- **Pipeline Parallelism**: Train models larger than single GPU memory using DeepSpeed
- **Supports**: LoRA, QLoRA, full fine-tuning, and Control Adapters.
- **Memory Optimization**: 4-bit quantization, activation checkpointing, tied weights
- **Multiple Training Modes**: LoRA, QLoRA, full fine-tuning, and Control Adapters
- **Production Ready**: Robust checkpointing, distributed training, comprehensive monitoring
- **Advanced Data Processing**: Flexible dataset handling with sophisticated preprocessing

See [here](docs/ControlAdapters.md) for Control Adapter documentation.

## Table of Contents

- [Quick Start](#quick-start)
- [Training Modes](#training-modes)
- [Configuration Reference](#configuration-reference)
- [Pipeline Parallelism](#pipeline-parallelism)
- [Data Processing](#data-processing)
- [Utility Tools](#utility-tools)
- [Monitoring and Analysis](#monitoring-and-analysis)
- [Advanced Topics](#advanced-topics)
- [Troubleshooting](#troubleshooting)
- [Credits](#credits)

## Quick Start

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU(s)
- DeepSpeed

### Installation

```bash
git clone https://github.com/jukofyork/qlora-pipe-lite
cd qlora-pipe-lite
pip install -r requirements.txt
```

### Basic LoRA Training

1. **Create a configuration file** (`config.toml`):

```toml
# Model and paths
model_dir = "/path/to/your/model"
output_dir = "./training_output"

# LoRA settings
lora_rank = 16
lora_dropout = 0.1
lora_weight_dtype = "float32"
lora_weight_decay = 10.0

# Training parameters
lr = 2e-5
epochs = 1
sequence_len = 4096
gradient_accumulation_steps = 8

# Pipeline parallelism
pipeline_stages = 1

# Dataset
[[datasets]]
dataset_path = "data/training_data.jsonl"
```

2. **Start training**:

```bash
# Single GPU
deepspeed --num_gpus=1 train.py --config config.toml

# Multi-GPU pipeline parallel
deepspeed --num_gpus=4 train.py --config config.toml

# Resume from checkpoint
deepspeed --num_gpus=4 train.py --config config.toml --resume_from_checkpoint
```

3. **Monitor progress**:

```bash
tensorboard --logdir ./training_output
```

### Quick Examples

**QLoRA (4-bit quantized LoRA)**:
```toml
load_in_4bit = true
lora_rank = 16
```

**Full Fine-tuning**:
```toml
full_fine_tune = true
target_modules = ["q_proj", "v_proj"]  # Optional: target specific modules
layers_to_transform = "16:31"          # Optional: target specific layers
```

## Training Modes

### LoRA (Low-Rank Adaptation)

LoRA adds trainable low-rank matrices to frozen base model weights: `W_new = W_original + BA`

**Configuration**:
```toml
lora_rank = 16                    # Adapter rank (bottleneck dimension)
lora_dropout = 0.1                # Dropout on adapter weights
lora_weight_dtype = "float32"     # Use float32 for stability with weight decay
lora_weight_decay = 10.0          # L2 regularization on composite matrix W=BA

# Optional targeting
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
layers_to_transform = "0:31"      # Apply to layers 0-31 (inclusive)
```

**Key Points**:
- Uses composite matrix regularization: `L = ½||BA||²_F` instead of separate `A`, `B` penalties
- Requires `float32` precision for effective weight decay (avoids catastrophic cancellation)
- Default `lora_alpha = lora_rank` (adjust learning rate instead of alpha)
- If `target_modules` is omitted, it defaults to `"all-linear"`; you may also explicitly set `target_modules = "all-linear"`.

### QLoRA (Quantized LoRA)

Combines LoRA with 4-bit quantization for maximum memory efficiency:

```toml
load_in_4bit = true               # Enable 4-bit quantization
lora_rank = 16
lora_weight_dtype = "float32"     # LoRA weights remain in float32
```

**Benefits**:
- Approximately 75% reduction in model weight memory usage
- Maintains training quality with float32 adapters
- Enables training of larger models on limited hardware

### Full Fine-tuning

Updates all model parameters with optional targeting:

```toml
full_fine_tune = true

# Optional: target specific modules/layers
target_modules = ["q_proj", "v_proj", "down_proj"]
layers_to_transform = "16:31"
```

**Key Points**:
- Uses `bfloat16` for memory efficiency
- Supports tied embedding weights with automatic gradient synchronization
- No weight decay (counterproductive for pretrained weights)

## Configuration Reference

### Core Settings

```toml
# Model and output paths
model_dir = "/path/to/model"
output_dir = "/path/to/output"

# Training parameters
lr = 2e-5                         # Learning rate
epochs = 1                        # Number of training epochs
sequence_len = 4096               # Fixed sequence length (must be multiple of 64)
gradient_accumulation_steps = 8   # Micro-batches per optimizer step

# Pipeline configuration
pipeline_stages = 1               # Number of pipeline stages (must divide world_size)
partition_method = "uniform"      # "uniform", "parameters", or "type:regex"
```

### Advanced Optimizer Settings

```toml
# Optimizer configuration (uses optimi.Adam with Kahan summation)
beta1 = 0.9                       # Default: 0.9
beta2 = 0.99                      # Default: 0.99  
eps = 1e-6                        # Default: 1e-6

# Note: Uses RMS ratio scheduling: sqrt((1-β₂^t)/(1+β₂^t))
```

### Evaluation and Checkpointing

```toml
# Evaluation settings
eval_fraction = 0.01                  # Fraction of data for evaluation
evals_per_epoch = 10                  # Evaluations per epoch
eval_gradient_accumulation_steps = 1  # Separate setting for evaluation

# Checkpointing
checkpoint_interval_hours = 1         # Time-based checkpoint frequency
max_checkpoints = 3                   # Maximum checkpoints to retain
```

### Memory and Performance Tuning

```toml
# Memory optimization
load_in_4bit = true               # Enable 4-bit quantization
use_column_major_topology = true  # Optimize for mixed interconnects

# Data processing
max_sequences = 1000000           # Limit total sequences
drop_tails = false                # Drop incomplete document tails
mix_datasets = false              # Allow cross-dataset sequence mixing
```

## Pipeline Parallelism

### Basic Concept

Pipeline parallelism splits the model vertically across GPUs:

```
GPU 0: [Embedding] → [Decoder Layers 0-7]
GPU 1: [Decoder Layers 8-15] → [Decoder Layers 16-23]  
GPU 2: [Decoder Layers 24-31] → [Norm] → [LM Head] → [Loss]
```

### Configuration

```toml
pipeline_stages = 3               # Split model across 3 GPUs
partition_method = "uniform"      # How to distribute layers

# Alternative partitioning strategies:
# partition_method = "parameters"           # Balance by parameter count
# partition_method = "type:decoderlayer"    # Balance by layer type
```

### Multi-Node Setup

For multi-node training, ensure:
1. **Shared filesystem**: All nodes can access the same model and data paths
2. **Network configuration**: Proper NCCL/MPI setup
3. **SSH access**: Passwordless SSH between nodes

```bash
# Multi-node example
deepspeed --num_gpus=8 --num_nodes=2 --master_addr=node0.cluster.local train.py --config config.toml
```

### Topology Optimization

For mixed interconnect environments (PCIe + InfiniBand):

```toml
use_column_major_topology = true
```

This optimization can be useful for LoRAs as it routes:
- **High-bandwidth activations** over fast PCIe/NVLink
- **Low-bandwidth gradients** over slower network interconnects

## Data Processing

### Dataset Configuration

Support for multiple formats and sophisticated preprocessing:

```toml
[[datasets]]
dataset_path = "data/train/*.jsonl"
max_tokens = 1000000             # Limit dataset size

[[datasets]]
dataset_path = "data/validation/*.txt"
```

### Supported Formats

- **Text files** (`.txt`): Raw text, one document per file
- **JSON/JSONL** (`.json`, `.jsonl`): Structured data with `"text"` field
- **Parquet** (`.parquet`): Columnar format with `"text"` field

### Advanced Processing Options

```toml
# Sequence initialization
sequence_prefix = "<BOS>"         # String to encode as prefix tokens
# sequence_prefix = 123           # Single token ID
# sequence_prefix = [123, 456]    # Multiple token IDs

# Document suffixes (applied during tokenization)
document_suffix = "<EOT>"         # String suffix before tokenization
# document_suffix = 123           # Token ID after tokenization
# document_suffix = [123, 456]    # Multiple token IDs

# Token masking (sets labels = -100 for loss exclusion)
mask_tokens = true                # Mask all special tokens
# mask_tokens = 123               # Mask specific token
# mask_tokens = [123, 456]        # Mask multiple tokens
```

### Distributed Data Loading

The framework implements distributed data loading with per-rank sampling, remainder truncation to ensure complete global batches, stateful iteration for checkpoint/resume support, and memory-efficient pinned memory allocation.

## Utility Tools

### Model Conversion and Merging

**Merge LoRA into base model**:
```bash
python merge_lora.py \
  --input /path/to/base_model \
  --adapter /path/to/lora_adapter \
  --output /path/to/merged_model \
  --scale 1.0
```

**Convert DeepSpeed checkpoints to LoRA**:
```bash
python convert_ds_checkpoint.py \
  --input /path/to/ds_checkpoint \
  --config /path/to/config.toml \
  --output /path/to/lora_adapter
```

## Monitoring and Analysis

### TensorBoard Metrics

Monitor training progress via TensorBoard:

```bash
tensorboard --logdir /path/to/output_dir --host 0.0.0.0
```

**Available metrics**:
- `eval/loss`: Evaluation loss with percentage changes
- `train/loss`: Training loss progression
- `train/lr`: Learning rate schedule
- `train/norms_{avg,min,max}`: LoRA adapter norm statistics
- `train/weight_decay_{avg,min,max}`: LoRA regularization effectiveness

### Training Progress Monitoring

Console output provides real-time metrics:

```
[2024-01-15 14:30:25.123] [INFO] [qlora-pipe-lite] step: 100 / 1000, loss: 2.3456, lr: 1.95e-05, throughput: 42.3 sequences/s, elapsed: 5m30s, eta: 45m12s
```

## Advanced Topics

### Custom Cross-Entropy Kernel

The framework includes [Unsloth](https://github.com/unslothai/unsloth)'s optimized Triton cross-entropy loss kernel (`kernels/cross_entropy_loss.py`) that handles large vocabularies (>65K tokens) via chunking, uses numerically stable LogSumExp implementation, and provides fused forward/backward passes for memory efficiency.

### Memory Optimization Techniques

The framework uses several memory optimization techniques:

**Activation Checkpointing**: Uses [Unsloth](https://github.com/unslothai/unsloth)'s CPU offloading strategy with automatic application to decoder layers. The system patches BitsAndBytes for safe CPU↔GPU transfers when using 4-bit quantization.

**Tied Weight Support**: Implements proper parameter sharing for embedding/output layers using DeepSpeed's `TiedLayerSpec` with automatic gradient synchronization across pipeline stages.

### Custom LoRA Weight Decay Implementation

This framework implements a novel composite matrix regularization that addresses fundamental issues with standard LoRA weight decay:

**The Problem**: Traditional weight decay applied to `A` and `B` separately fails because:
1. **Catastrophic cancellation**: Tiny decay amounts cancel to zero with `float16`/`bfloat16` precision
2. **Wrong target**: Should regularize the actual learned transformation `W = scale * B @ A`, not individual matrices

**The Solution**: Custom decoupled weight decay on the composite matrix:

### Regularization Mathematics

**LoRA Weight Decay**:
The composite matrix regularization prevents the catastrophic cancellation issue:

```
L_total = L_task + λ/2 ||BA||²_F

∂L/∂A = ∂L_task/∂A + λ B^T(BA)
∂L/∂B = ∂L_task/∂B + λ (BA)A^T
```

This approach:
- Uses `Adam` (not `AdamW`) for the primary loss
- Applies SGD-style decay to the composite matrix using the same learning rate
- **Requires `float32` precision** for adapter weights to avoid cancellation
- Directly penalizes the transformation that affects model behavior

## Troubleshooting

### Common Issues

**Out of Memory**: Enable 4-bit quantization (`load_in_4bit = true`), reduce sequence length/batch size, or increase pipeline stages.

**Poor Convergence**: Check learning rate (typical range: `1e-5` to `5e-5`), verify data quality and format, monitor for gradient explosion via sudden loss spikes.

**Pipeline Issues**: Ensure `pipeline_stages` divides `world_size` evenly, verify all nodes can access model/data paths, check network connectivity for multi-node setups.

**RTX 4000 Series GPUs** may need one or both of these environment variables setting:
```bash
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"
deepspeed --num_gpus=4 train.py --config config.toml
```

### Validation Steps

**Before training**: Verify dataset format and paths, test configuration with a small dataset, check GPU memory usage with target batch size.

**During training**: Monitor loss convergence, verify checkpoint saving/loading works correctly.

**After training**: Test inference with merged models, validate outputs on held-out data.

**For LoRA training specifically**: Monitor norm stability (should converge), watch for gradient explosion (sudden loss spikes), verify effective batch size via throughput metrics.


## Credits

- **[tdrussell](https://github.com/tdrussell)** - Original author of [qlora-pipe](https://github.com/tdrussell/qlora-pipe)
- **[Daniel Han-Chen & Unsloth team](https://github.com/unslothai/unsloth)** - Cross-entropy kernel and checkpoint utilities

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.