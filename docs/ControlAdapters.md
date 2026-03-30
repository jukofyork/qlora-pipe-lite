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

Control Adapters are a parameter-efficient fine-tuning method that provides **multiplicative control** over LLM behaviour. Unlike additive methods like LoRA that add `W + BA` to weights, Control Adapters apply multiplicative transformations to the residual delta (the change produced by decoder layers) of the form `(I + W) × delta`, where `W = BA` is a low-rank matrix similar to LoRA.

### Development and Motivation

Control Adapters evolved through several iterations to address fundamental challenges with multiplicative interventions in language models:

The initial concept was a "multiplicative LoRA" using general transformations `(I + AB^T) × delta`, but this proved too unconstrained and destabilised models. Adding bidirectional control with separate positive/negative classes helped, but using negated gradients for the negative class caused training instability due to the unbounded nature of maximising cross-entropy loss.

Early approaches explored first-order Neumann series approximations `(I - AB^T) × delta` for inverse transformations, but required keeping spectral norms within a narrow range (`‖W‖₂ < 0.3`) for mathematical validity, making regularisation difficult. Attempts to use spectral decompositions `Q diag(λ) Q^T` with log-parameterisation provided exact inverses but required complex orthogonality constraints and disrupted critical model behaviours.

The current approach using standard LoRA-like factorisation `W = BA` with Neumann series inverse approximation emerged as the solution, balancing expressive power with the constraints necessary for stable language model training while maintaining simple, familiar parameterisation.

### Key Features

- **Bidirectional Control**: Forward transformations (class `+1`) and inverse approximations (class `-1`) using the same parameters
- **Class-Conditional**: Different behaviour based on control classes (`+1`, `-1`)
- **Randomised Regularisation**: Datasets assigned to class `0` are mapped randomly to class `±1` during preprocessing
- **Parameter Efficient**: Only `r×(H×2)` parameters per transformed layer (where `H` is hidden size, `r` is rank)
- **Multiplicative**: Transformations scale proportionally with activation magnitude
- **LoRA-Compatible**: Uses standard LoRA factorisation structure (BA) for easy adaptation of existing code

### When to Use Control Adapters

**Ideal for:**

- Behavioural steering (tone, style, personality, prose)
- Bidirectional learning (enhance/suppress behaviours using the same model)
- "Unlearning" specific behaviours
- Scenarios requiring control reversal

**Consider alternatives for:**

- General instruction following (use LoRA)
- Simple domain adaptation (use LoRA)
- Learning new concepts or knowledge (use full fine-tuning)

### Relationship to Control Vectors

Control Adapters are conceptually similar to [Control Vectors](https://github.com/jukofyork/control-vectors), which also steer model behaviour by intervening in the residual stream. However:

- **Control Vectors**: Compute steering directions analytically using eigenvector analysis of the symmetrised cross-covariance matrix. Applied via additive combination, where multiple vectors can interfere when used simultaneously.
- **Control Adapters**: Learn steering transformations through gradient-based training. More forgiving for "fuzzy" criteria like writing style. The low-rank structure provides task-specific directional control with learned scaling.
- **Complementary usage**: Control Vectors provide additive translation while Control Adapters provide multiplicative scaling along learned directions. Together they enable richer intervention patterns than either method alone.

## Mathematical Foundation

### Core Transformation

Control Adapters apply multiplicative transformations to the residual delta produced by each decoder layer:

```
layer_delta = layer_output - input_hidden_states
adapter_output = B @ A @ dropout(layer_delta)
final_output = layer_output + adapter_output
```

Where:

- **`A ∈ ℝ^{r×H}`**: Down-projection matrix (hidden_size → adapter_rank)
- **`B ∈ ℝ^{H×r}`**: Up-projection matrix (adapter_rank → hidden_size)
- **`W = BA ∈ ℝ^{H×H}`**: Composite low-rank transformation matrix
- **`r`**: Adapter rank (typically 16-64)
- **`H`**: Hidden dimension

This structure is identical to standard LoRA, but applied multiplicatively to residual deltas rather than additively to weights.

### Neumann Series Inverse Approximation

For bidirectional control, we need to approximate `(I + W)^{-1}` for negative examples. The Neumann series provides:

```
(I + W)^{-1} = I - W + W² - W³ + ...
```

This series converges when the spectral norm `‖W‖₂ < 1`. Control Adapters use a **1st-order approximation**:

- **Forward (Class `+1`)**: Apply `W` directly: `(I + W) × delta ≈ delta + W × delta`
- **Inverse (Class `-1`)**: Apply `-W` for 1st-order inverse: `(I + W)^{-1} × delta ≈ delta - W × delta`

**Convergence and Accuracy:**

When `‖W‖₂ ≲ 0.25`, the 1st-order truncation error is `O(‖W‖₂²) ≤ 1-2%`. The training regularisation maintains this spectral norm bound to ensure:

- Convergence guarantee (`‖W‖₂ < 1`)
- Acceptable approximation error (`‖W‖₂ < 0.25`)

**Scalar Intuition:**

For scalar `δ` with `|δ| < 1`:
- True inverse: `1/(1 + δ)`
- 1st order: `1 - δ` (error ≈ `δ²`)
- Example with `δ = 0.1`: `1/1.1 ≈ 0.9091` vs `1 - 0.1 = 0.9000` (≈1% error)

### Norm Monitoring

Since computing `‖W‖₂` directly is expensive, we monitor the **Frobenius norm** `‖W‖_F` instead:

- For rank-`r` matrices: `‖W‖₂ ≤ ‖W‖_F ≤ √r · ‖W‖₂`
- Target: `‖W‖_F < 0.25√r` ensures `‖W‖₂ ≲ 0.25` with balanced singular values
- Regularisation maintains this bound via L2 weight decay on the composite matrix

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
lora_weight_decay = 10.0          # L2 decay on composite W = BA (requires float32)

# Layer targeting
layers_to_transform = "0:29"      # Transform layers 0-29 (inclusive)
# Formats: "0:10,20:25,30" or "1,3,5,7" or "0:31" etc.
```

### Important Notes

- **Float32 required**: If using `lora_weight_decay > 0`, you *must* set `lora_weight_dtype = "float32"`
- **Layer selection**: Can target specific layers; omit `layers_to_transform` to use all
- **Weight decay range**: Typical values 1.0-20.0; higher values enforce stricter norm bounds

## Data and Classes

### Control Classes

Each example/document in your training data uses a control class:

- **Class `+1`**: Apply forward transformation (enhance behaviour)
- **Class `-1`**: Apply inverse approximation (suppress behaviour)  
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
3. **Apply control adapter**: `adapter_output = B(A(dropout(layer_delta)))`
4. **Class-conditional transformation**:
   - **Class `+1`**: Add `adapter_output` to apply forward transformation `(I + W) × delta`
   - **Class `-1`**: Apply Neumann series inverse approximation `(I + W)^{-1} × delta`:
     - **1st-order** (default): Negate `adapter_output`, giving `(I - W) × delta`
     - **Higher orders**: Compute `I - W + W² - W³ + ...` up to `INVERSE_APPROXIMATION_SERIES_ORDER`
   - **Class `0`**: Zero out (padding tokens; labels = -100)
5. **Add to residual stream**: `final_output = layer_output + adapter_output`

Note on causal alignment:
- The training pipeline uses causal language modelling. The control signal is shifted one token to align with next-token prediction (same mechanism as label shifting).
- The final position in a sequence is padded accordingly.

### Regularisation During Training

Control Adapters employ two regularisation mechanisms:

1. **Weight decay**: L2 regularisation on composite matrix `W = BA` using `L = ½‖W‖_F²` (optional analytical regularisation; maintains spectral norm bounds for convergence)
2. **Class randomisation**: Examples marked `control_class = 0` are randomly assigned `±1` during preprocessing, injecting controlled label noise to improve generalisation and prevent overfitting to always-active controls

## Analysis and Monitoring

### Built-in Metrics

During training, monitor these metrics via TensorBoard:

- **`train/norms_{avg,min,max}`**: Frobenius norms `‖W‖_F` per layer
- **`train/weight_decay_{avg,min,max}`**: Norm reduction from regularisation (if applied)

### Analysis Tool

Use the analysis tool for detailed post-training evaluation:

```bash
python analyze_control_adapters.py --adapter /path/to/adapter [--no-gpu]
```

This provides per-layer metrics including spectral norms, Frobenius norms, effective rank, condition numbers, and convergence status.

### Interpreting Key Metrics

| Metric | Target Values | What It Means |
|--------|---------------|---------------|
| Spectral norm (`‖W‖₂`) | < 0.25 (must be < 1) | Ensures convergence and low approximation error |
| Frobenius norm (`‖W‖_F`) | < 0.25√r | Proxy for spectral norm (cheaper to compute during training) |
| Effective rank | Close to adapter rank | Adapter is using its full capacity |
| Condition number | < 100 (poor > 1000) | Numerically stable transformations |

**Warning indicators:**
- `‖W‖₂ ≥ 1`: Neumann series may diverge (critical!)
- `‖W‖₂ ≥ 0.25`: Approximation error exceeds 1-2% target
- Low effective rank: Potential rank collapse or underutilisation

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
  [--inverse] [--rank R] [--model-specific-flags]
```

NOTE: Targets `mlp.down_proj` by default; use `--cohere` or `--mixtral N` to include additional modules.

**Key options:**

- `--inverse`: Convert inverse branch (class `-1` behaviour) instead of forward branch (useful for testing!)
- `--rank R`: Override output rank (default: use adapter rank)
- `--cohere`: Also target `o_proj` layers (for `Cohere` models only)
- `--mixtral N`: Target `experts.{0..N-1}.w2` (for `Mixtral` models only)

### Conversion Math

The conversion computes the effect on model weights and approximates via SVD:

1. **Compute effect on weights**: 
   - Forward: `delta = W @ weight` where `W = BA`
   - Inverse: `delta = ((I + W)^{-1} - I) @ weight` (exact inverse)

2. **SVD approximation**: `delta ≈ U @ diag(√S) @ V^T`

3. **LoRA factorisation**:
   - `A_lora = diag(√S) @ V^T` (shape `[rank, output_size]`)
   - `B_lora = U @ diag(√S)` (shape `[hidden_size, rank]`)
   - Result: `delta ≈ B_lora @ A_lora` with specified rank

**Rank Selection:**
- Default: Uses original adapter rank
- Override with `--rank R` to reduce/increase rank
- Tool reports variance explained by chosen rank

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
- **Weight decay**: Use moderate values (`1.0`-`20.0`) to maintain norm bounds
- **Monitor norms**: Keep spectral norm well below `0.25` during training (ideally `0.15-0.20`)
- **Check both directions**: Test that forward/inverse behaviours work as expected via the `--inverse` option (see above)

### Data Preparation

- **Clear distinctions**: Ensure `+1`/`-1` examples show clearly different behaviours (eg: "good" prose / "bad" prose, etc)
- **Quality matters**: Fewer high-quality examples beat many poor ones
- **Balance classes**: Use roughly equal amounts of `+1` and `-1` examples
- **Include variety**: Class `0` should represent diverse "normal" behaviours (eg: general "instruction-following" data, etc)

### Debugging Common Issues

- **High spectral norms (>0.25)**: Increase `lora_weight_decay` or reduce learning rate
- **Norm spikes**: Check for gradient explosion, reduce learning rate, increase `lora_weight_decay`
- **Poor effective rank (<50% of adapter rank)**: Try more training data, more diverse training data, or reduce `lora_rank`
- **Convergence warnings (‖W‖₂ ≥ 1)**: Critical issue - increase weight decay significantly or reduce learning rate

## Files and Tools

### Core Implementation

- `training/control_adapters.py`: Main implementation and training logic
- `training/regularizer.py`: Weight decay regularisation for composite matrices

### Analysis Tools

- `analyze_control_adapters.py`: Comprehensive adapter analysis with per-layer metrics and convergence status

### Conversion Tools

- `control_adapter_to_lora.py`: Convert to standard LoRA format via SVD approximation
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