from datetime import datetime
import sys
import os.path
import torch
from deepspeed import comm as dist
from contextlib import contextmanager

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