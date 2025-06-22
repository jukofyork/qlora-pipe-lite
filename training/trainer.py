import time
import gc
import torch
from torch.utils.tensorboard import SummaryWriter

from saver import save_checkpoint, prune_checkpoints, save_model
from utils import is_main_process


class Trainer:
    def __init__(self, config, model_engine, train_dataloader, eval_dataloader, run_dir, pipeline_model, args, lora_config):
        self.config = config
        self.model_engine = model_engine
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.run_dir = run_dir
        self.pipeline_model = pipeline_model
        self.args = args
        self.lora_config = lora_config
        
        self.tb_writer = SummaryWriter(log_dir=run_dir) if is_main_process() else None
        self.last_checkpoint_time = time.time() if is_main_process() else None
        
        # Calculate evaluation step indices to use across the entire run
        evals_per_run = config.get('evals_per_run', 10)
        self.eval_step_indices = set()
        for i in range(1, evals_per_run):
            step_in_run = round(i * model_engine.total_steps / (evals_per_run - 1))
            self.eval_step_indices.add(step_in_run)

    def evaluate(self, step):
        orig_micro_batches = self.model_engine.micro_batches
        self.model_engine.micro_batches = 1
        iterator = iter(self.eval_dataloader)
        all_metrics = None
        while True:
            metrics = self.model_engine.eval_batch(iterator)
            self.eval_dataloader.sync_epoch()
            if all_metrics is None:
                all_metrics = [[] for _ in range(len(metrics))]
            if self.eval_dataloader.epoch == 2:
                break
            for i, metric in enumerate(metrics):
                all_metrics[i].append(metric)

        self.eval_dataloader.reset()
        self.model_engine.micro_batches = orig_micro_batches
        eval_metrics = [torch.cat(metric_list) for metric_list in all_metrics]
        loss = None
        if is_main_process():
            loss = self._write_metrics(f'eval', eval_metrics, step)
        return loss

    def _should_checkpoint(self):
        """Check if it's time to checkpoint based on time interval."""
        if not is_main_process():
            return False
        
        current_time = time.time()
        if (current_time - self.last_checkpoint_time) / 60 >= self.config.get('checkpoint_interval', 60):
            self.last_checkpoint_time = current_time
            return True
        return False

    def _save_checkpoint(self, step):
        """Save checkpoint and broadcast decision to all processes."""
        do_checkpoint = self._should_checkpoint()
        
        # Broadcast decision to ensure all processes checkpoint together
        result = [do_checkpoint]
        torch.distributed.broadcast_object_list(result, src=0)
        do_checkpoint = result[0]
        
        if do_checkpoint:
            save_checkpoint(self.model_engine, self.train_dataloader, self.run_dir, step)
            if is_main_process():
                prune_checkpoints(self.run_dir, self.config.get('max_checkpoints', -1))


    def _write_metrics(self, prefix, metrics, step):
        loss = metrics[0].mean().item()
        self.tb_writer.add_scalar(f'{prefix}/loss', loss, step)
        return loss

    def train(self, start_step=1):
        """Main training loop."""
        step = start_step
        
        # Initial evaluation
        last_eval_loss = self.evaluate(step - 1)
        if is_main_process():
            print(f'- Initial evaluation loss: {last_eval_loss:.4f}')
        
        while self.train_dataloader.epoch <= self.config.get('epochs', 1):
            # Memory cleanup before each training step
            gc.collect()
            torch.cuda.empty_cache()
            
            # Forward/backward pass and optimizer step
            metrics = self.model_engine.train_batch()
            self.train_dataloader.sync_epoch()
            
            # Log training metrics to tensorboard
            if is_main_process():
                self._write_metrics('train', metrics, step)
            
            # Periodic evaluation based on global step
            if step in self.eval_step_indices:
                loss = self.evaluate(step)
                if is_main_process():
                    percent_change = (loss / last_eval_loss - 1) * 100
                    print(f'- Step {step} evaluation loss: {loss:.4f} (last: {last_eval_loss:.4f}, Δ: {percent_change:.2f}%)')
                last_eval_loss = loss
            
            # Time-based checkpointing
            self._save_checkpoint(step)
            
            step += 1
        
        # Save final model when training completes
        save_model(self.model_engine, self.pipeline_model, self.args, self.lora_config, self.config, self.run_dir, 'final')
        
        if is_main_process():
            print('TRAINING COMPLETE!')
