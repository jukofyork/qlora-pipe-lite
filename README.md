# qlora-pipe-lite
A streamlined fork of [qlora-pipe](https://github.com/tdrussell/qlora-pipe) by [tdrussell](https://github.com/tdrussell), focused on Control Adapter training.

**For full fine-tuning, LoRA, or QLoRA training, please use the original [qlora-pipe](https://github.com/tdrussell/qlora-pipe) repository which is more feature-complete and actively maintained.**

## About

Control Adapters are a new Parameter-Efficient Fine-Tuning ([PEFT](https://github.com/huggingface/peft)) method that applies multiplicative transformations (and their inverse) to the residual stream of whole (ie: attention + MLP) decoder blocks, enabling the same adapter to both enhance and suppress behaviors through positive and negative training examples.

## Features
- Pipeline parallel training, for efficiently training large models that cannot fit on one GPU
- Multi-node training support with improved data-parallel handling via shared network storage
- Supports Control Adapters (primary focus), QLoRA, LoRA, and full fine tuning
- Proper decoupled weight decay for LoRA parameters (applies weight decay to the full reconstructed weight matrix)
- Quantize weights using bitsandbytes
- Efficient model loading. Each process only loads the layers it needs, and quantizes and moves them to the GPU layer-by-layer. This means you can load a large model on a lot of GPUs even with limited system RAM.
- Support for "raw text" training using either a structured list of documents in a JSON file, or a single txt file
- Support for resuming training from a checkpoint, including the dataloader state, to easily allow training in a piecemeal fashion
- Useful metrics logged to Tensorboard
- Train on multiple datasets simultaneously, with support for contrastive datasets (eg: `class -1` and `class 1` for Control Adapters)
- Models currently supported: Llama, Mistral, Mixtral, Qwen, Cohere (Command R), Cohere 2 (Command-A), and Gemma 2

## Table of Contents

- [About](#about)
- [Features](#features)
- [Quick Start](#quick-start)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Basic Control Adapter Training](#basic-control-adapter-training)
  - [Quick QLoRA Example](#quick-qlora-example)
  - [Next Steps](#next-steps)
- [What are Control Adapters?](#what-are-control-adapters)
  - [Relationship to Control Vectors](#relationship-to-control-vectors)
  - [Mathematical Foundation](#mathematical-foundation)
  - [Convergence Requirements and the need for Weight Decay](#convergence-requirements-and-the-need-for-weight-decay)
  - [Conversion and Compatibility](#conversion-and-compatibility)
  - [Why Use Control Adapters?](#why-use-control-adapters)
- [Training](#training)
  - [Training Modes](#training-modes)
  - [Configuration Structure](#configuration-structure)
  - [Parallelism and Scaling](#parallelism-and-scaling)
  - [Advanced Configuration](#advanced-configuration)
  - [Example Training Commands](#example-training-commands)
  - [Monitoring Training](#monitoring-training)
- [A Note on LoRA Weight Decay](#a-note-on-lora-weight-decay)
- [Credits](#credits)
- [Contributing](#contributing)
- [License](#license)

## Quick Start

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU(s)
- 24GB+ VRAM recommended (or use QLoRA with `load_in_4bit = true`)

### Installation

```bash
git clone https://github.com/your-username/qlora-pipe-lite
cd qlora-pipe-lite
pip install -r requirements.txt
```

### Basic Control Adapter Training

1. **Create a configuration file** (`config.toml`):

```toml
# Model and output paths
model_dir = '/path/to/your/model'  # eg: Llama-3.1-8B
output_dir = './my_control_adapter'

# Control Adapter settings
use_control_adapters = true
lora_rank = 16

# For Llama-3.1-8B (32 layers total), skip last 2 layers
layers_to_transform = '0:29'       # Train layers 0-29 (30 layers total)

# Training parameters
lr = 2e-5
lora_weight_decay = 500.0
sequence_len = 4096
gradient_accumulation_steps = 64   # ie: ~250k tokens per step

# Datasets with class labels
[[datasets]]
dataset_path = 'data/positive_examples.json'
control_class = 1                  # Enhance this behavior

[[datasets]]
dataset_path = 'data/negative_examples.json'
control_class = -1                 # Suppress this behavior
```

2. **Prepare your data** in JSON format with a "text" field:

```json
[
  {"text": "Your training example text here..."},
  {"text": "Another example..."}
]
```

3. **Start training**:

```bash
# Single GPU
python train.py --config config.toml --num_gpus 1

# Multi-GPU with pipeline parallelism
python train.py --config config.toml --num_gpus 4

# Resume from checkpoint
python train.py --config config.toml --resume_from_checkpoint
```

4. **Monitor training** with TensorBoard:

```bash
tensorboard --host 0.0.0.0 --logdir="./my_control_adapter"
```

5. **Convert to standard LoRA format**:

After training completes, you can convert your Control Adapter to standard PEFT-compatible LoRA format:

```bash
# Multiplicative LoRA conversion (lossless, very fast, and preserves multiplicative behavior)
./tools/control_adapter_to_multiplicative_lora.py ./my_control_adapter/epoch1 ./my_multiplicative_lora

# Additive LoRA approximation (compatible with standard LoRA tools, but slow due to SVD decomposition)
./tools/control_adapter_to_additive_lora.py ./Llama-3.1-8B ./my_control_adapter/epoch1 ./my_lora
```

6. **Merge with base model**:

To create a standalone merged model, use the included merge tool:

```bash
# Merge multiplicative LoRA
./tools/merge_lora.py ./Llama-3.1-8B ./my_multiplicative_lora ./my_merged_model --multiplicative

# Merge additive LoRA
./tools/merge_lora.py ./Llama-3.1-8B ./my_lora ./my_merged_model
```

Alternatively, use my [huggingface space](https://huggingface.co/spaces/jukofyork/merge-lora) capable of merging in the cloud and saving on upload bandwidth.

7. **Convert to GGUF format** (optional):

```bash
# NOTE: Additive LoRA *ONLY* (multiplicative LoRA cannot be used with llama.cpp!)
./llama.cpp/convert_lora_to_gguf.py --base ./Llama-3.1-8B --outtype f32 --outfile ./my_lora.gguf ./my_lora
```

and then run [llama-server](https://github.com/ggml-org/llama.cpp/tree/master/tools/server) using the `--lora FNAME` option (or `--lora-scaled FNAME SCALE` if you want to adjust the effect of the LoRA).

### Quick QLoRA Example

For memory-constrained setups, use 4-bit quantization:

```toml
# Add to your config.toml
load_in_4bit = true
```

### Next Steps

- See [Training](#training) for detailed configuration options
- Check [example configs](examples/) for model-specific settings
- Read about [Convergence Requirements](#convergence-requirements-and-the-need-for-weight-decay) for Control Adapters
- Learn about [LoRA Weight Decay](#a-note-on-lora-weight-decay) implementation

## What are Control Adapters?

Control Adapters are a new form of PEFT adapter that simultaneously applies multiplicative (and inverse multiplicative) transformations to the residual stream (ie: the "delta" or change in hidden states) of whole decoder blocks (ie: attention + MLP). Unlike traditional LoRA adapters that add learned parameters directly to specific weight matrices, Control Adapters operate on the decoder layer level residual connections, enabling the same adapter to both enhance and suppress behaviors through positive and negative training examples; making them particularly effective for steering model behavior in specific directions.

**Key Characteristics:**
- **Residual-based**: Applied to the delta/change produced by each decoder block rather than absolute hidden state values
- **Multiplicative**: Uses `h' = (I + scale * B @ A) @ h` transformations instead of additive `h' = h + (scale * B @ A) @ h` transformations like normal LoRA adapters
- **Layer-wide**: Affects the entire residual stream rather than individual weight matrices
- **Class-aware**: Supports positive (ie: `class 1`) and negative (ie: `class -1`, aka "unlearning") training examples, with negative examples using inverse transformations of the same adapter `h' = (I + scale * B @ A)^{-1} @ h` to suppress rather than enhance behaviors
- **Optional neutral examples**: Can include neutral (ie: `class 0`) examples to prevent unwanted forgetting and provide additional regularization by isolating the specific behavioral "axis" being modified

### Relationship to Control Vectors

Control Adapters are conceptually similar to [Control Vectors](https://github.com/jukofyork/control-vectors), which also steer model behavior by intervening in the residual stream. However, there are important differences:

- **Control Vectors**: Compute steering directions analytically using eigenvector analysis to separate behavioral classes, but combine by adding vector directions which can cause interference when multiple vectors are applied simultaneously

- **Control Adapters**: Learn steering transformations through gradient-based training, making them more forgiving and suitable for "fuzzy" criteria like writing style rather than requiring carefully selected behavioral "axes". They can be thought of as having two components: the `A` vectors act as "signed direction detectors" (via dot product), whilst the `B` vectors provide the steering effect. Due to the very high dimensionality of the hidden states, multiple Control Adapters are less likely to interfere when combined.

#### Mathematical Relationship

Control Adapters can represent several classical transformations, including:

- [Orthogonal projection onto the null space](https://en.wikipedia.org/wiki/Projection_(linear_algebra)) (aka: 
["Abliteration"](https://www.lesswrong.com/posts/jGuXSZgv6qfdhMCuJ/refusal-in-llms-is-mediated-by-a-single-direction)):

```
I - u u^T, when B = -A
```

- [Householder transformations](https://en.wikipedia.org/wiki/Householder_transformation):

```
I - 2 * u u^T, when B = -2A
```

When combined with Control Vectors, they enable full [affine transformations](https://en.wikipedia.org/wiki/Affine_transformation) of the decoder layer outputs.

### Mathematical Foundation

Control Adapters apply multiplicative transformations to the residual stream using a (scaled) rank-decomposed matrix:

```
W = scale * B @ A, where scale = alpha / rank
```

#### For positive examples (`class +1`):

```
h' = (I + W) @ h
```

#### For negative examples (`class -1`):

```
h' = (I + W)^{-1} @ h ≈ (I - W + higher_order_terms) @ h
```

The inverse transformation uses a [Neumann series approximation](https://en.wikipedia.org/wiki/Neumann_series), allowing the same adapter parameters to both enhance desired behaviors (positive) and suppress undesired behaviors (negative) during training.

#### For (optional) neutral examples (`class 0`):

Neutral examples receive no transformation (`h' = h`) but contribute an auxiliary regularization loss that encourages the adapter to have minimal effect on these samples. This helps isolate the specific behavioral axis being modified and provides additional regularization beyond (decoupled) weight decay.

### Convergence Requirements and the need for Weight Decay

As mentioned above, Control Adapters rely on the [Neumann series approximation](https://en.wikipedia.org/wiki/Neumann_series) for computing matrix inverses during negative example training. For mathematical stability and convergence, the composite matrix `W = scale * B @ A` must satisfy specific norm constraints.

#### Convergence Condition

The Neumann series `(I + W)^{-1} = I - W + W^2 - W^3 + ...` converges when the [spectral radius](https://en.wikipedia.org/wiki/Spectral_radius) `ρ(W) < 1`.

A sufficient condition (ie: a stricter but easier-to-check requirement that guarantees convergence) is that the [spectral norm](https://en.wikipedia.org/wiki/Matrix_norm#Spectral_norm) `‖W‖₂ < 1`.

For rank-`r` matrices, the relationship between norms is:

```
‖W‖₂ ≤ ‖W‖_F ≤ √r · ‖W‖₂
```

Where `‖W‖_F` is the [Frobenius norm](https://en.wikipedia.org/wiki/Frobenius_norm).

Since the Frobenius norm is much cheaper to compute than the spectral norm (which requires SVD or power iteration per step), we monitor it instead.

#### Scalar Intuition

To understand why norm constraints matter, consider the simple scalar case: computing `1/(1 + δ)` using the series `1 - δ + δ² - δ³ + ...`. When `δ = 0.1`, the true value is `1/1.1 ≈ 0.9091`. The first-order approximation `1 - δ = 0.9000` has \~1% error, while the second-order `1 - δ + δ² = 0.9100` has \~0.1% error. However, if `δ = 2`, the series diverges completely since `|δ| ≥ 1`. The same principle applies to matrices: keeping `‖W‖₂ < 1` ensures the matrix inverse series converges, and smaller norms give better approximations with fewer terms.

#### Practical Guidelines

To ensure convergence with acceptable approximation error:

- **Target**: Keep `‖W‖_F ≲ 0.25√r` (approximately 0.2-0.3 times `√rank`)
- **Result**: This guarantees `‖W‖₂ ≲ 0.2-0.3`, giving truncation errors ≤ 1-2%
- **Monitoring**: The training logs show `norm_avg` and `norm_max` values for `‖W‖_F`

#### Why Weight Decay is Essential

Unlike traditional LoRA where weight decay is optional, Control Adapters **require** proper regularization to maintain norm constraints:

1. **Prevents divergence**: Without regularization, norms can grow beyond convergence limits
2. **Uniform scaling**: Decoupled weight decay on the composite matrix `W` properly scales all singular values, unlike naive decay on `A` and `B` separately
3. **Stability**: Maintains the mathematical guarantees needed for reliable inverse approximations

**Recommendation**: Always use `lora_weight_dtype = "float32"` with `lora_weight_decay > 0` for Control Adapters. The `bfloat16` option should only be used when memory constraints are severe, as it disables the essential weight decay regularization.

For typical setups:
- **Rank 16**: Target `‖W‖_F < 1.0`, and use `lora_weight_decay` of `100-500`
- **Rank 64**: Target `‖W‖_F < 4.0`, and use `lora_weight_decay` of `500` or more

Monitor the `norm_max` values in TensorBoard to ensure they stay well below these thresholds throughout training.

### Conversion and Compatibility

Control Adapters can be converted to standard [PEFT](https://github.com/huggingface/peft) compatible LoRAs easily:

1. **Multiplicative LoRA**: Lossless conversion that preserves the multiplicative nature by distributing the matrix products to specific linear layers that write to the residual stream (eg: `down_proj`, but also `o_proj` or `w2` for some models...)
2. **Additive LoRA**: SVD-based approximation that converts the multiplicative effect into standard (additive) LoRA format

This flexibility allows Control Adapters to be used with existing LoRA-compatible inference frameworks while maintaining their unique training advantages.

See the [tools](tools) folder for the conversion scripts. Note that the Cohere and Mixtral models require the `--cohere` and `--mixtral` command line options respectively, due to their different residual stream writing mechanisms (ie: via `o_proj` and `w2`).

### Why Use Control Adapters?

Control Adapters excel in scenarios where you need:
- **Precise behavioral steering** with "fuzzy" criteria (eg: writing style, tone) rather than requiring carefully crafted behavioral axes like Control Vectors
- **Compositional control** - multiple Control Adapters can be combined with minimal interference due to high-dimensional space and the A/B decomposition
- **Bidirectional training** with positive and negative examples to both enhance and suppress behaviors using the same parameters
- **Smaller datasets** - more forgiving than traditional methods and are effective with less well-structured data
- **Mathematical expressiveness** - can easily learn classical transformations like "abliteration" or Householder transforms
- **Complementary to Control Vectors** - work together to enable full affine transformations of decoder outputs

**Best used for**: Behavioral modification, style transfer, and "unlearning" specific behaviors.

**NOTE**: For general fine-tuning tasks like instruction following or domain adaptation, traditional LoRA or QLoRA may be more appropriate. Control Adapters are specifically designed for behavioral steering applications.

## Training

Supports multiple training modes: **Control Adapters**, **LoRA**, **QLoRA**, and **full fine-tuning**. The training system uses **DeepSpeed** for efficient large model training with pipeline parallelism, data parallelism, and memory optimization techniques like 4-bit quantization.

### Training Modes

#### Full Fine-Tuning

Loads and updates all model parameters using `bfloat16`:

```toml
full_fine_tune = true
```

**NOTE**: Weight decay is not supported for full fine-tuning due to:

1. **Catastrophic cancellation** with `bfloat16` precision (see [LoRA Weight Decay Implementation](#lora-weight-decay-implementation) below)
2. **Poor inductive bias** - pulling pretrained weights toward zero is counterproductive (unlike LoRA where a zero Bayesian prior is beneficial)

#### LoRA (Low-Rank Adaptation)

Trains low-rank adapter matrices on top of frozen base model weights:

```toml
lora_rank = 64
lora_alpha = 64                # Default: sqrt(rank) for rsLoRA scaling (see: https://arxiv.org/abs/2312.03732)
lora_dropout = 0.05            # Default: 0.0
lora_weight_decay = 10.0       # Default: 0.0, requires float32 adapters
lora_weight_dtype = "float32"  # Default: "float32", use "bfloat16" to save memory (disables weight decay)
```

**NOTE**: The `lora_weight_dtype` parameter controls the precision of LoRA adapter parameters. When set to its default `float32`, it enables `lora_weight_decay` for proper regularization. When set to `bfloat16`, weight decay is automatically disabled due to catastrophic cancellation (see [A Note on LoRA Weight Decay](#a-note-on-lora-weight-decay) below). For Control Adapters, `float32` and weight decay are essential for mathematical stability - see [Convergence Requirements](#convergence-requirements-and-the-need-for-weight-decay) for details.

#### QLoRA (Quantized LoRA)

LoRA (or Control Adapter) training on 4-bit quantized base models for memory efficiency:

```toml
load_in_4bit = true
```

#### Control Adapters

Contrastive training using LoRA-like (multiplicative) transformations on decoder block outputs:

```toml
use_control_adapters = true
control_class0_lambda = 0.0    # Auxiliary loss weight for neutral examples (default: 0.0)
```

Control Adapters support **class-aware training** with positive, negative, and (optional) neutral examples:

```toml
[[datasets]]
dataset_path = '/path/to/positive_examples/*.json'
control_class = 1   # Enhance behaviors (default)

[[datasets]]
dataset_path = '/path/to/negative_examples/*.json'
control_class = -1  # Suppress/unlearn behaviors

[[datasets]]
dataset_path = '/path/to/neutral_examples/*.json'
control_class = 0   # Behavior we wish to preserve
```

### Configuration Structure

Training configurations use TOML files with the following main sections:

#### Model and Output

```toml
model_dir = '/path/to/model'
output_dir = '/path/to/output'
```

**NOTE**: For multi-node training, ensure paths use the same mount point across all nodes.

#### Optimizer Settings

```toml
lr = 5e-5
epochs = 1                     # Default: 1
beta1 = 0.9                    # Default: 0.9
beta2 = 0.99                   # Default: 0.99
eps = 1e-6                     # Default: 1e-6
```

**NOTE**: Uses the [**optimi Adam optimizer**](https://optimi.benjaminwarner.dev/optimizers/adam/) with [Kahan summation](https://en.wikipedia.org/wiki/Kahan_summation_algorithm) automatically applied for low-precision parameters.

#### Training Parameters

```toml
sequence_len = 32768
gradient_accumulation_steps = 32      # Controls effective batch size
eval_gradient_accumulation_steps = 1  # Default: same as gradient_accumulation_steps
pipeline_stages = 1                   # Must evenly divide world_size
```

**NOTE**: Reducing `eval_gradient_accumulation_steps` can help drop fewer examples when creating equal-sized evaluation batches.

#### Dataset Configuration

Multiple datasets with optional limits and class labels:

```toml
[[datasets]]
dataset_path = 'raw_text_data/*.txt'
max_sequences = 10000          # Optional limit
drop_tails = false             # Drop partial sequences at document ends

[[datasets]]
dataset_path = 'structured_data/*.json'
```

**Supported formats**: `.txt` (raw text), `.json`, `.jsonl`, `.parquet` (structured with "text" field)

**NOTE**: For multi-node training, ensure paths use the same mount point across all nodes. Using shared dataset paths also enables cache reuse, where secondary nodes can load preprocessed data created by the main node instead of reprocessing datasets independently.

### Parallelism and Scaling

#### Pipeline Parallelism

Splits the model across multiple GPUs vertically (by layers):

```toml
pipeline_stages = 2            # Divide model across 2 GPUs
```

#### Data Parallelism

Automatically configured based on available GPUs:
- **Total GPUs**: `world_size`
- **Data parallel instances**: `world_size / pipeline_stages`
- **Effective batch size**: `gradient_accumulation_steps × (world_size / pipeline_stages)`

#### Hybrid Parallelism Optimization

For LoRA training with mixed interconnects:

```toml
use_column_major_topology = true
```

This optimization:

- Sends high-volume hidden states over **PCIe/NVLink**
- Sends low-volume LoRA gradients over **Ethernet/InfiniBand**

### Advanced Configuration

#### Layer and Module Targeting

```toml
# Target specific modules (not available with Control Adapters - they operate per layer)
target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj']

# Target specific layers (works with all training types, can combine with target_modules)
layers_to_transform = '16:31'  # Layers 16-31 (inclusive:inclusive, with first layer = 0 / last layer = n-1)
```

#### Checkpointing

```toml
checkpoint_interval_hours = 1  # Default: 1
max_checkpoints = 3            # Default: 3
```

Use the `--resume_from_checkpoint` flag to continue training from the most recent checkpoint:

```bash
python train.py --config config.toml --resume_from_checkpoint
```

**NOTE**: The system automatically resumes from the latest checkpoint in the output directory, including dataloader state for seamless continuation.

#### Evaluation

```toml
eval_fraction = 0.01           # Default: 0.01
evals_per_run = 5              # Default: 11 (eg: 1 at start, 1 at end, and 9 more evenly spaced)
```

#### LoRA Weight Decay Implementation

```toml
lora_weight_decay = 10.0       # Default: 0.0
```

See the [A Note on LoRA Weight Decay](#a-note-on-lora-weight-decay) section below.

### Example Training Commands

```bash
# Single GPU LoRA Training
python train.py --config examples/config.toml --num_gpus 1
```

```bash
# Multi-GPU Pipeline Parallel Training
python train.py --config examples/config_qwq.toml --num_gpus 4
```

```bash
# Multi-Node Training
python train.py --config config.toml --num_gpus 8 --master_addr node0.example.com
```

**NOTES**:

- The `--num_gpus` option may not be required depending on your setup
- RTX 4000 series GPUs may need `NCCL_P2P_DISABLE="1" NCCL_IB_DISABLE="1"` environment variables setting
- For multi-node training guidance, see the [Stanford DeepSpeed tutorial](https://nlp.stanford.edu/mistral/tutorials/deepspeed.html) and how to setup [passwordless SSH](https://wiki.debian.org/Setup%20SSH%20Passwordless%20Login)

### Monitoring Training

You can use [TensorBoard](https://www.tensorflow.org/tensorboard) to visualize training and evaluation metrics:

```bash
tensorboard --host 0.0.0.0 --logdir="/path/to/output_dir"
```

Then open `http://localhost:6006` in your browser.

## A Note on LoRA Weight Decay

Traditional weight decay applied directly to LoRA parameters (`lora_A` and `lora_B`) has significant limitations that this implementation addresses through **decoupled weight decay on the composite matrix**.

### The Problem with standard weight decay

1. **Catastrophic Cancellation**: Standard optimizer weight decay fails with `float16`/`bfloat16` precision because the tiny decay amounts cancel out to zero when applied to the relatively large parameter tensors!

2. **Incorrect Regularization Target**: Applying weight decay to `A` and `B` matrices separately doesn't properly regularize the actual learned transformation `W = scale * B @ A` that affects the model's behavior!

### Solution: Custom 'composite' decoupled weight decay + use `float32` adapters only

Instead of regularizing `A` and `B` independently, we apply decoupled weight decay to the composite matrix `W = scale * B @ A` using manually calculated gradients. This directly penalizes the composite transformation that actually affects model behavior:

```
L' = L + λ · ½||W||²_F
```

```
∂L'/∂A = λ · scale · B^T W
∂L'/∂B = λ · scale · W A^T
```

- Use `Adam` (**NOT** `AdamW`) with the primary loss `L`
- Use SGD (without momentum) for the regularization term using the same learning rate (ie: decoupled, pseudo-`AdamW`)
- Always use `float32` precision for `A` and `B` matrices to avoid catastrophic cancellation
- Use the `lora_weight_decay` parameter in your `config.toml` file to control the regularization strength

## Credits

- **[tdrussell](https://github.com/tdrussell)** - Original author of [qlora-pipe](https://github.com/tdrussell/qlora-pipe), which this project is forked from.
- **[Daniel Han-Chen & the Unsloth team](https://github.com/unslothai/unsloth)** - For [`cross_entropy_loss.py`](kernels/cross_entropy_loss.py) and [`unsloth_checkpoint.py`](utils/unsloth_checkpoint.py).

## Contributing

Contributions to this project are welcome. Please feel free to fork the repository and submit pull requests.

## License

This project is licensed under the MIT license - see the [LICENSE](LICENSE) file for details.