# Optimizer defaults
DEFAULT_BETA1 = 0.9
DEFAULT_BETA2 = 0.99
DEFAULT_EPS = 1e-6

# NOTE: This must be > 0 to maintain semi-orthogonality, and <= 0.5 to avoid overshooting (see code comment).
DEFAULT_CONTROL_ADAPTER_GAMMA = 0.5

# NOTE: This must strictly be > 1.0 to facilitate any training, and <= ~2.0 for stable/sensible inverse operation.
DEFAULT_CONTROL_ADAPTER_SIGMA = 2.0

# Evaluation defaults
DEFAULT_EVALS_PER_EPOCH = 10
DEFAULT_EVAL_FRACTION = 0.01

# Checkpoint defaults
DEFAULT_CHECKPOINT_INTERVAL_HOURS = 1
DEFAULT_MAX_CHECKPOINTS = 3

# Misc constants
DEEPSPEED_TIMEOUT_HOURS = 6
DATASET_MAP_BATCH_SIZE = 10