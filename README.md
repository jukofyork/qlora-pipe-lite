# qlora-pipe-lite
A streamlined fork of [qlora-pipe](https://github.com/tdrussell/qlora-pipe) by tdrussell, focused on Control Adapter training.

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
- Train on multiple datasets simultaneously, with support for contrastive datasets (eg: class -1 and class 1 for Control Adapters)
- Models currently supported: Llama, Mistral, Mixtral, Qwen, Cohere (Command R), Cohere 2 (Command-A), and Gemma 2

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
["Abliteration"](https://www.lesswrong.com/posts/jGuXSZgv6qfdhMCuJ/refusal-in-llms-is-mediated-by-a-single-direction): `I - u u^T` when `B = -A`
- [Householder transformations](https://en.wikipedia.org/wiki/Householder_transformation): `I - 2 * u u^T` when `B = -2A`

When combined with Control Vectors, they enable full [affine transformations](https://en.wikipedia.org/wiki/Affine_transformation) of the decoder layer outputs.

### Mathematical Foundation

Control Adapters apply multiplicative transformations to the residual stream using a rank-decomposed matrix `scale * B @ A`:

#### For positive examples (`class +1`):

```
h' = (I + scale * B @ A) @ h
```

#### For negative examples (`class -1`):

```
h' = (I + scale * B @ A)^{-1} @ h ≈ (I - scale * B @ A + higher_order_terms) @ h
```

The inverse transformation uses a [Neumann series approximation](https://en.wikipedia.org/wiki/Neumann_series), allowing the same adapter parameters to both enhance desired behaviors (positive) and suppress undesired behaviors (negative) during training.

#### For neutral examples (`class 0`) - optional:

Neutral examples receive no transformation (`h' = h`) but contribute an auxiliary regularization loss that encourages the adapter to have minimal effect on these samples. This helps isolate the specific behavioral axis being modified and provides additional regularization beyond standard L2-regularization and/or decoupled weight decay.

### Conversion and Compatibility

Control Adapters can be converted to standard [PEFT](https://github.com/huggingface/peft) compatible LoRAs easily:

1. **Multiplicative LoRA**: Lossless conversion that preserves the multiplicative nature by distributing the matrix products to specific linear layers that write to the residual stream (eg: `down_proj`, but also `o_proj` or `w2` for some models...)
2. **Additive LoRA**: SVD-based approximation that converts the multiplicative effect into standard (additive) LoRA format

This flexibility allows Control Adapters to be used with existing LoRA-compatible inference frameworks while maintaining their unique training advantages.

### Why Use Control Adapters?

Control Adapters excel in scenarios where you need:
- **Precise behavioral steering** with "fuzzy" criteria (eg: writing style, tone) rather than requiring carefully crafted behavioral axes like Control Vectors
- **Compositional control** - multiple Control Adapters can be combined with minimal interference due to high-dimensional space and the A/B decomposition
- **Bidirectional training** with positive and negative examples to both enhance and suppress behaviors using the same parameters
- **Smaller datasets** - more forgiving than traditional methods and work well with less structured data
- **Mathematical expressiveness** - can easily learn classical transformations like "abliteration" or Householder transforms
- **Complementary to Control Vectors** - work together to enable full affine transformations of decoder outputs

**Best used for**: Behavioral modification, style transfer, and "unlearning" specific behaviors.

**NOTE**: For general fine-tuning tasks like instruction following or domain adaptation, traditional LoRA or QLoRA may be more appropriate. Control Adapters are specifically designed for behavioral steering applications.

## A note on LoRA weight decay

Traditional weight decay applied directly to LoRA parameters (`lora_A` and `lora_B`) has significant limitations that this implementation addresses through **decoupled weight decay on the composite matrix**.

### The Problem with Standard Weight Decay

1. **Catastrophic Cancellation**: Standard optimizer weight decay fails with `float16`/`bfloat16` precision because the tiny decay amounts cancel out to zero when applied to the (relatively) large parameter tensors!

2. **Incorrect Regularization Target**: Applying weight decay to `A` and `B` matrices separately doesn't properly regularize the actual learned transformation `W = scale * B @ A` that affects the model's behavior!

### Solution: Custom "composite" decoupled weight decay + use `float32` adapters only

Instead of regularizing `A` and `B` independently, we apply decoupled weight decay to the composite matrix `W = scale * B @ A` using manually calculated gradients. This directly penalizes the composite transformation that actually affects model behavior:

```
L' = L + λ · ½||W||²_F
```

```
∂L'/∂A = λ · scale · B^T W
∂L'/∂B = λ · scale · W A^T
```

- Use `Adam` (**NOT** `AdamW`) for the primary loss `L`
- Use SGD (without momentum) for the regularization term using the same learning rate (ie: decoupled, pseudo-`AdamW`)
- Always use `float32` precision for `A` and `B` matrices to avoid catastrophic cancellation
- Use the `lora_weight_decay` parameter in your `config.toml` file to control the regularization strength

## Credits

- **[tdrussell](https://github.com/tdrussell)** - Original author of [qlora-pipe](https://github.com/tdrussell/qlora-pipe), which this project is forked from.
- **[Daniel Han-Chen & the Unsloth team](https://github.com/unslothai/unsloth)** - For [`cross_entropy_loss.py`](kernels/cross_entropy_loss.py) and [`unsloth_checkpoint.py`](utils/unsloth_checkpoint.py).

## Contributing

Contributions to this project are welcome. Please feel free to fork the repository and submit pull requests.

## License

This project is licensed under the Apache-2.0 license - see the [LICENSE](LICENSE) file for details.