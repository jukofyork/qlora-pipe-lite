from contextlib import contextmanager
from datetime import datetime
from deepspeed import comm as dist
import glob
import os
import shutil
import time

def is_main_process():
    return dist.get_rank() == 0

@contextmanager
def zero_first(is_main):
    """
    runs the wrapped context so that rank 0 runs first before other ranks
    """
    if not is_main:  # other ranks wait first
        dist.barrier()
    yield
    if is_main:  # then rank 0 waits after it has run the context
        dist.barrier()

# Simplified logger-like printer.
def log(msg):
    print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]}] [INFO] [qlora-pipe] {msg}')

def eta_str(eta):
    eta = int(eta)
    if eta > 3600:
        return f'{eta // 3600}h{(eta % 3600) // 60}m'
    return f'{eta // 60}m{eta % 60}s' if eta > 60 else f'{eta}s'

def get_most_recent_run_dir(output_dir):
    return list(sorted(glob.glob(os.path.join(output_dir, '*'))))[-1]

# Attempt to remove a directory tree using exponential backoff for retries (default max wait = 31s)
def safe_rmtree(dir_path, max_retries=5, initial_wait_seconds=1):
    for attempt in range(max_retries + 1):
        try:
            shutil.rmtree(dir_path)
            return
        except OSError as e:
            if attempt == max_retries:
                raise e
            time.sleep(initial_wait_seconds * 2 ** attempt)