# Optimizer defaults
DEFAULT_BETA1 = 0.9
DEFAULT_BETA2 = 0.99
DEFAULT_EPS = 1e-6

# Evaluation defaults
DEFAULT_EVALS_PER_EPOCH = 10
DEFAULT_EVAL_FRACTION = 0.01

# Checkpoint defaults
DEFAULT_CHECKPOINT_INTERVAL_HOURS = 1
DEFAULT_MAX_CHECKPOINTS = 3

# Misc constants
DEEPSPEED_TIMEOUT_HOURS = 6
DATASET_MAP_BATCH_SIZE = 10

# For the inverse approximation (I + W)^{-1}, when ‖W‖₂ ≲ 0.2–0.3, the 1st-order truncation
# error O(‖W‖₂²) ≤ 1–2%. Going to order 2 halves the error (O(‖W‖₂³)) but doubles the matmul cost.
# Order 3+ yields less than 0.1% improvement in the intended ‖W‖₂ norm range.
NEUMANN_SERIES_ORDER = 2