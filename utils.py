from contextlib import contextmanager
from datetime import datetime
from deepspeed import comm as dist
import os
import shutil
import time

def is_main_process():
    """Check if current process is the main process (rank 0)."""
    return dist.get_rank() == 0

@contextmanager
def zero_first(is_main):
    """Run wrapped context so that rank 0 executes first before other ranks."""
    if not is_main:  # other ranks wait first
        dist.barrier()
    yield
    if is_main:  # then rank 0 waits after it has run the context
        dist.barrier()
    dist.barrier()  # All ranks wait here until everyone finishes

def log(msg):
    """Print timestamped log message with qlora-pipe prefix."""
    if is_main_process():
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]}] [INFO] [qlora-pipe] {msg}')

def eta_str(eta):
    """Convert ETA seconds to human-readable string (e.g., '2h30m', '45s')."""
    eta = int(eta)
    if eta > 3600:
        return f'{eta // 3600}h{(eta % 3600) // 60}m'
    return f'{eta // 60}m{eta % 60}s' if eta > 60 else f'{eta}s'

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