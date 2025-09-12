# Control Adapters

This document describes Control Adapters as implemented in `qlora-pipe-lite`. Control Adapters are a novel parameter-efficient fine-tuning method (PEFT) that applies multiplicative, class-conditional transformations to the output of LLM decoder layers.

## Table of Contents

- [Quick Start](#quick-start)
- [Overview](#overview)
- [Mathematical Foundation](#mathematical-foundation)
- [Configuration](#configuration)
- [Data and Classes](#data-and-classes)
- [Training Behaviour](#training-behaviour)
- [Analysis and Monitoring](#analysis-and-monitoring)
- [Conversion to LoRA](#conversion-to-lora)
- [Best Practices](#best-practices)
- [Files and Tools](#files-and-tools)

## Quick Start

### 1. Basic Configuration

```toml
# Enable Control Adapters
use_control_adapters = true
lora_rank = 16

# Basic training settings
model_dir = "/path/to/your/model"
output_dir = "./control_adapter_output"
lr = 2e-5
epochs = 1
sequence_len = 4096

# Your datasets with class labels
[[datasets]]
dataset_path = "data/positive_examples.jsonl"
control_class = 1    # Enhance this behaviour

[[datasets]]
dataset_path = "data/negative_examples.jsonl" 
control_class = -1   # Suppress this behaviour

[[datasets]]
dataset_path = "data/neutral_examples.jsonl"
control_class = 0    # Randomly assigned ±1 during preprocessing (regularisation technique - see below)
```

### 2. Run Training

```bash
# Single GPU
deepspeed --num_gpus=1 train.py --config config.toml

# Multi-GPU pipeline parallel
deepspeed --num_gpus=4 train.py --config config.toml

# Resume from checkpoint
deepspeed --num_gpus=4 train.py --config config.toml --resume_from_checkpoint
```

### 3. Analyse Results

```bash
python analyze_control_adapters.py --adapter ./control_adapter_output/epoch0
```

### 4. Monitor Training

```bash
tensorboard --logdir ./control_adapter_output --host 0.0.0.0
```

### 5. Convert for Deployment

```bash
python control_adapter_to_lora.py \
  --base /path/to/your/model \
  --adapter ./control_adapter_output/epoch0 \
  --output ./converted_lora
```

### 6. Optional: Export LoRA to GGUF (llama.cpp)

```bash
python lora_to_gguf.py \
  --input ./converted_lora \
  --output ./adapter.gguf \
  [--arch llama] \
  [--outtype F32]
```

## Overview

### What are Control Adapters?

Control Adapters are a parameter-efficient fine-tuning method that provides **multiplicative control** over LLM behaviour. Unlike additive methods like LoRA that add `W + BA` to weights, Control Adapters apply multiplicative transformations to the residual delta (the change produced by decoder layers) of the form `(I + Q diag(λ) Q^T) × delta` (ie: parameterised as a [spectral decomposition of a real symmetric matrix](https://en.wikipedia.org/wiki/Eigendecomposition_of_a_matrix#Real_symmetric_matrices)).

### Development and Motivation

Control Adapters evolved through several iterations to address fundamental challenges with multiplicative interventions in language models:

The initial concept was a "multiplicative LoRA" using general transformations `(I + AB^T) × delta`, but this proved too unconstrained and destabilised models. Adding bidirectional control with separate positive/negative classes helped, but using negated gradients for the negative class caused training instability due to the unbounded nature of maximising cross-entropy loss.

Early approaches explored first-order Neumann series approximations `(I - AB^T) × delta` for inverse transformations, but required keeping eigenvalues within a narrow range (`|λ| < 0.3`) for mathematical validity, making regularisation difficult. General factorisations like `Q diag(λ) P^T` offered more flexibility but disrupted critical model behaviours like end-of-line handling.

The current approach using symmetric matrices `Q diag(λ) Q^T` with log-parameterisation `λ = exp(S) - 1` emerged as the solution, balancing expressive power with the constraints necessary for stable language model training.

### Key Features

- **Bidirectional Control**: Forward transformations (class `+1`) and inverse transformations (class `-1`) using the same parameters
- **Class-Conditional**: Different behaviour based on control classes (`+1`, `-1`)
- **Randomised Regularisation**: Datasets assigned to class `0` are mapped randomly to class `±1` during preprocessing
- **Parameter Efficient**: Only `r×(H+1)` parameters per transformed layer (where `H` is hidden size, `r` is rank)
- **Multiplicative**: Transformations scale proportionally with activation magnitude

### When to Use Control Adapters

**Ideal for:**

- Behavioural steering (tone, style, personality, prose)
- Bidirectional learning (enhance/suppress behaviours using the same model)
- "Unlearning" specific behaviours
- Scenarios requiring precise control reversal

**Consider alternatives for:**

- General instruction following (use LoRA)
- Simple domain adaptation (use LoRA)
- Learning new concepts or knowledge (use full fine-tuning)

### Relationship to Control Vectors

Control Adapters are conceptually similar to [Control Vectors](https://github.com/jukofyork/control-vectors), which also steer model behaviour by intervening in the residual stream. However:

- **Control Vectors**: Compute steering directions analytically using eigenvector analysis of the symmetrised cross-covariance matrix. Applied via additive combination, where multiple vectors can interfere when used simultaneously.
- **Control Adapters**: Learn steering transformations through gradient-based training. More forgiving for "fuzzy" criteria like writing style. The `Q` matrix learns a task-specific subspace while `S` parameters provide per-direction scaling effects.
- **Complementary usage**: Control Vectors provide additive translation while Control Adapters provide multiplicative scaling along learned directions. Together they enable richer intervention patterns than either method alone.

## Mathematical Foundation

### Core Transformation

Control Adapters apply multiplicative transformations to the residual delta produced by each decoder layer:

```
layer_delta = layer_output - input_hidden_states
adapter_output = Q diag(λ) Q^T @ layer_delta
final_output = layer_output + adapter_output
```

Where:

- **`Q ∈ ℝ^{H×r}`**: Semi-orthogonal matrix spanning a learned subspace  
- **`λ ∈ ℝ^r`**: Per-direction eigenvalue offsets
- **`r`**: Adapter rank (typically 16-64)
- **`H`**: Hidden dimension

### Log-Parameterisation

Eigenvalues are derived from learnable parameters `S`:

```
λ = exp(S) - 1
```

This parameterisation provides several key advantages:

- **Mathematical stability**: `1 + λ = exp(S) > 0` always, ensuring well-conditioned transformations
- **Identity initialisation**: `S=0 → λ=0` → no change initially, allowing gradual learning
- **Natural inverses**: Exact bidirectionality via `exp(-S) - 1` using the same learned parameters
- **Smooth regularisation**: Symmetric behaviour in log-space for both forward and inverse directions
- **Spectral control**: Interpretable eigenvalue-based scaling along learned directions
- **Unbounded range**: Unlike constrained parameterisations, `S` can take any real value while keeping transformations stable

### Bidirectional Control

The key innovation is principled inverse transformations:

- **Forward (Class `+1`)**: `λ = exp(S) - 1`
- **Inverse (Class `-1`)**: `λ' = exp(-S) - 1 = -λ/(1+λ)`

This provides mathematical guarantees:

- Class `-1` exactly undoes Class `+1` transformations when `Q^T Q = I`
- Perfect bidirectional control using the same parameters
- In practice, effects are approximate due to operating on residual deltas and semi-orthogonality

### Orthogonality Constraint

The method maintains `Q^T Q ≈ I_r` through regularisation, ensuring:

- Stable matrix operations
- Predictable eigenvalue behaviour
- Numerical stability during training

## Configuration

### Basic Setup

```toml
# Enable Control Adapters
use_control_adapters = true

# Adapter parameters (reuses LoRA config names for consistency)
lora_rank = 16                    # Control Adapter rank
lora_dropout = 0.0                # Dropout on residual delta
lora_weight_dtype = "float32"     # Recommended for stability
```

### Advanced Options

```toml
# Regularisation
lora_weight_decay = 10.0          # L2 decay on S parameters (requires float32)
control_adapter_gamma = 0.5       # Orthogonality step size (0, 0.5]

# Layer targeting
layers_to_transform = "0:29"      # Transform layers 0-29 (inclusive)
# Formats: "0:10,20:25,30" or "1,3,5,7" or "0:31" etc.
```

### Important Notes

- **Float32 required**: If using `lora_weight_decay > 0`, you *must* set `lora_weight_dtype = "float32"`
- **Gamma bounds**: `control_adapter_gamma` must be in range `(0, 0.5]` for numerical stability
- **Layer selection**: Can target specific layers; omit `layers_to_transform` to use all

## Data and Classes

### Control Classes

Each example/document in your training data uses a control class:

- **Class `+1`**: Apply forward transformation (enhance behaviour)
- **Class `-1`**: Apply inverse transformation (suppress behaviour)  
- **Class `0`**: Randomised regulariser. During preprocessing, each example marked `0` is deterministically mapped to `+1` or `-1` (`≈ 50/50`). This injects controlled label noise to reduce overfitting and prevents the model from assuming controls are always active. There is no special "neutral" behaviour in training - class `0` becomes `±1` before training.

### Dataset Configuration

Set control classes per dataset:

```toml
# Formality control example
[[datasets]]
dataset_path = "data/formal_writing.jsonl"  # "Dear Sir/Madam, I write to inquire about..."
control_class = 1

[[datasets]] 
dataset_path = "data/casual_writing.jsonl"  # "Hey! Just wanted to ask about..."
control_class = -1

[[datasets]]
dataset_path = "data/neutral_writing.jsonl" # Mixed/uncurated; used as randomised regulariser
control_class = 0                           # Will be mapped to ±1 during preprocessing
```

### Why Use Class `0`?

Class `0` is a convenient switch to introduce randomised directionality without curating separate positive/negative datasets. Those examples are converted to `±1` at preprocessing, acting as a regulariser that improves generalisation and helps preserve base model behaviour.

**Recommendation**: Allocate roughly 10–30% of your total training examples as `control_class = 0` sources.

### More Examples

```toml
# Sentiment steering
[[datasets]]
dataset_path = "data/positive_reviews.jsonl"  # "This product exceeded my expectations!"
control_class = 1

[[datasets]]
dataset_path = "data/negative_reviews.jsonl"  # "This product was disappointing."
control_class = -1

[[datasets]]
dataset_path = "data/neutral_reviews.jsonl"   # "This product works as described."
control_class = 0                             # Randomised regulariser; mapped to ±1 during preprocessing
```

## Training Behaviour

### Forward Pass Process

For each decoder layer with Control Adapters:

1. **Compute residual delta**: `layer_delta = layer_output - input_hidden_states`
2. **Apply dropout** (if configured) and cast to adapter dtype
3. **Project to subspace**: `x_q = layer_delta @ Q`  
4. **Class-conditional transformation**:
   - Class `+1`: `λ = exp(S) - 1`
   - Class `-1`: `λ' = exp(-S) - 1`
5. **Reconstruct**: `adapter_output = (x_q * λ) @ Q^T`
6. **Add to residual stream**: `final_output = layer_output + adapter_output`

Note on causal alignment:
- The training pipeline uses causal language modelling. The control signal is shifted one token to align with next-token prediction (same mechanism as label shifting).
- The final position in a sequence is padded accordingly.

### Regularisation During Training

Control Adapters employ three regularisation mechanisms:

1. **Orthogonality maintenance**: `Q ← Q - γ Q (Q^T Q - I)` with `γ = control_adapter_gamma` (mandatory analytical regularisation)
2. **Eigenvalue shrinkage**: `S ← S - lr * lora_weight_decay * S` (optional analytical regularisation; prevents overfitting)
3. **Class randomisation**: Examples marked `control_class = 0` are randomly assigned `±1` during preprocessing, injecting controlled label noise to improve generalisation and prevent overfitting to always-active controls

## Analysis and Monitoring

### Built-in Metrics

During training, monitor these metrics via TensorBoard:

- **`train/norms_{avg,min,max}`**: Spectral norms (≈ `max |λ|`) per layer
- **`train/orthogonality_{avg,min,max}`**: `‖Q^T Q - I‖_F²` constraint satisfaction
- **`train/weight_decay_{avg,min,max}`**: Norm reduction from regularisation (if applied)

### Analysis Tool

Use the analysis tool for detailed post-training evaluation:

```bash
python analyze_control_adapters.py --adapter /path/to/adapter [--no-gpu]
```

This provides per-layer metrics including orthogonality errors, effective rank usage, and approximation quality.

### Interpreting Key Metrics

| Metric | Good Values | What It Means |
|--------|-------------|---------------|
| Orthogonality error | < 1.0 (excellent < 0.5) | `Q` matrix maintains good subspace properties |
| Effective rank | Close to adapter rank | Adapter is using its full capacity |
| Approximation errors | < 5% (poor > 20%) | Orthogonality assumption holds well |
| Condition number | < 100 (poor > 1000) | Numerically stable transformations |

## Conversion to LoRA

### Why Convert?

Control Adapters can be converted to standard additive LoRA format for:

- Deployment in existing inference frameworks
- Compatibility with LoRA merging tools
- Easier serving infrastructure

### Conversion Process

```bash
python control_adapter_to_lora.py \
  --base /path/to/base_model \
  --adapter /path/to/control_adapter \
  --output /path/to/lora_output \
  [--inverse] [--model-specific-flags]
```

NOTE: Targets `mlp.down_proj` by default; use `--cohere` or `--mixtral N` to include additional modules.

**Key options:**

- `--inverse`: Convert inverse branch (class `-1` behaviour) instead of forward branch (useful for testing!)
- `--cohere`: Also target `o_proj` layers (for `Cohere` models only)
- `--mixtral N`: Target `experts.{0..N-1}.w2` (for `Mixtral` models only)

### Conversion Math

The conversion uses an exact low-rank mapping to an additive LoRA:

1. Compute eigenvalue offsets: `λ = exp(S) - 1` (or `λ' = exp(-S) - 1` for `--inverse`)
2. Compute delta to base weight: `ΔW = (Q diag(λ) Q^T) @ W_base`
3. Exact LoRA factorisation:
   - `B = Q diag(λ)`  (shape `[H, r]`)
   - `A = Q^T W_base` (shape `[r, N]`)
   - Then `ΔW = B @ A` exactly, with rank `r` preserved (no SVD, no truncation).

### Deployment

**Merge LoRA into base model**:

Once conversion is complete, you can merge the LoRA using any standard LoRA-merging tool or the included `merge_lora.py` script:

```bash
# Merge into base model
python merge_lora.py \
  --input /path/to/base_model \
  --adapter /path/to/lora_output \
  --output /path/to/merged_model
```

Alternatively, you can use the [Memory-Efficient LoRA Merge](https://huggingface.co/spaces/jukofyork/merge-lora) Hugging Face space (useful for users with limited upload bandwidth who want to share their models publicly!).

**Export LoRA adapter to GGUF (llama.cpp)**:

```bash
python lora_to_gguf.py \
  --input /path/to/lora_output \
  --output /path/to/adapter.gguf \
  [--arch llama] \
  [--outtype F16]
```

NOTE: Mixtral is not yet supported by `lora_to_gguf.py` - use [convert_lora_to_gguf.py](https://github.com/ggml-org/llama.cpp/blob/master/convert_lora_to_gguf.py) instead.

## Best Practices

### Configuration

- **Use float32**: Keep `lora_weight_dtype = "float32"` (the default) for numerical stability
- **Start small**: Begin with rank 16-32; higher ranks need a *lot* more data
- **Layer selection**: Consider excluding the first (1-2) and last (1-2) layers, as these are more prone to training instabilities
- **Include randomised regulariser data**: Use 10–30% class `0` examples for stability and generalisation

### Training

- **Learning rate**: Start with `2e-4`, typical range is `1e-5` to `1e-3`
- **Weight decay**: Use moderate values (`1.0`-`20.0`) when using `float32`
- **Monitor orthogonality**: Keep orthogonality error well below `1.0` during training
- **Check both directions**: Test that forward/inverse behaviours work as expected via the `--inverse` option (see above)

### Data Preparation

- **Clear distinctions**: Ensure `+1`/`-1` examples show clearly different behaviours (eg: "good" prose / "bad" prose, etc)
- **Quality matters**: Fewer high-quality examples beat many poor ones
- **Balance classes**: Use roughly equal amounts of `+1` and `-1` examples
- **Include variety**: Class `0` should represent diverse "normal" behaviours (eg: general "instruction-following" data, etc)

### Debugging Common Issues

- **Persistent High orthogonality error (>1.0)**: Reduce learning rate or increase `control_adapter_gamma`
- **Sudden norm spikes**: Check for gradient explosion, reduce learning rate, increase `lora_weight_decay`
- **Poor effective rank (<50% of adapter rank)**: Try more training data, more diverse training data, or reduce `lora_rank`
- **High approximation errors (>20%)**: Reduce learning rate and/or increase regularisation parameters

## Files and Tools

### Core Implementation

- `training/control_adapters.py`: Main implementation and training logic
- `training/regularizer.py`: Orthogonality and weight decay regularisation

### Analysis Tools

- `analyze_control_adapters.py`: Comprehensive adapter analysis with per-layer metrics

### Conversion Tools

- `control_adapter_to_lora.py`: Convert to standard LoRA format for deployment
- `merge_lora.py`: Standard LoRA merging tool (use after conversion)
- `lora_to_gguf.py`: Export a LoRA adapter to GGUF for [llama.cpp](https://github.com/ggml-org/llama.cpp)

### Usage Summary

```bash
# Train Control Adapters
deepspeed --num_gpus=4 train.py --config config_control_adapter.toml

# Resume training
deepspeed --num_gpus=4 train.py --config config_control_adapter.toml --resume_from_checkpoint

# Analyse trained adapter
python analyze_control_adapters.py --adapter /path/to/adapter

# Convert to LoRA for deployment
python control_adapter_to_lora.py --base /path/to/model --adapter /path/to/adapter --output /path/to/lora

# Merge LoRA into base model
python merge_lora.py --input /path/to/model --adapter /path/to/lora --output /path/to/merged

# Export LoRA to GGUF for llama.cpp
python lora_to_gguf.py --input /path/to/lora --output /path/to/adapter.gguf
```