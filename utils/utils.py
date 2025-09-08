from contextlib import contextmanager
from datetime import datetime
from deepspeed import comm as dist
import os
import shutil
import sys
import time
import torch

DTYPE_MAP = {'float32': torch.float32, 'float16': torch.float16, 'bfloat16': torch.bfloat16}

def is_main_process():
    """Check if current process is the main process (rank 0)."""
    return dist.get_rank() == 0

@contextmanager
def main_process_first():
    """Run wrapped context so that rank 0 executes first before other ranks, then all ranks wait for completion."""
    if not is_main_process():  # other ranks wait first
        dist.barrier()
    yield
    if is_main_process():  # then rank 0 waits after it has run the context
        dist.barrier()
    dist.barrier()  # All ranks wait here until everyone finishes

def log(msg, main_process_only=True):
    """Print timestamped log message with qlora-pipe-lite prefix."""
    if not main_process_only or is_main_process():
        formatted_msg = f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]}] [INFO] [qlora-pipe-lite] {msg}\n'
        sys.stdout.write(formatted_msg)
        sys.stdout.flush()

def log_all(msg):
    """Print timestamped log message from all processes."""
    log(msg, main_process_only=False)

def seconds_to_time_str(seconds):
    """Convert seconds to human-readable string (e.g., '2d5h30m', '2h30m', '45s')."""
    seconds = int(seconds)
    if seconds >= 86400:
        return f'{seconds // 86400}d{(seconds % 86400) // 3600}h{(seconds % 3600) // 60}m'
    elif seconds >= 3600:
        return f'{seconds // 3600}h{(seconds % 3600) // 60}m'
    elif seconds >= 60:
        return f'{seconds // 60}m{seconds % 60}s'
    else:
        return f'{seconds}s'

def format_percentage(value, decimal_places=1):
    """Format percentage, treating values that round to 0.0% as positive."""
    formatted = f"{value:.{decimal_places}%}"
    return "0.0%" if formatted == "-0.0%" else formatted

def safe_rmtree(dir_path, max_retries=5, initial_wait_seconds=1):
    """Remove directory tree with exponential backoff retries on failure."""
    for attempt in range(max_retries + 1):
        try:
            shutil.rmtree(dir_path)
            return
        except OSError as e:
            if attempt == max_retries:
                raise e
            time.sleep(initial_wait_seconds * 2 ** attempt)