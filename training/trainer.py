import time
import gc
import torch
from torch.utils.tensorboard import SummaryWriter

from training.checkpoint_manager import load_checkpoint, save_checkpoint, prune_checkpoints
from training.model_saver import save_lora, save_full_model
from utils import is_main_process


class Trainer:
    def __init__(
        self,
        model_engine,
        train_dataloader,
        eval_dataloader,
        run_dir,
        pipeline_model,
        args,
        lora_config,
        model_dir,
        epochs=1,
        evals_per_run=10,
        checkpoint_interval=60,
        max_checkpoints=-1,
        resume_from_checkpoint=False
    ):
        self.model_engine = model_engine
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.run_dir = run_dir
        self.pipeline_model = pipeline_model
        self.args = args
        self.lora_config = lora_config
        self.model_dir = model_dir
        self.epochs = epochs
        self.checkpoint_interval = checkpoint_interval
        self.max_checkpoints = max_checkpoints
        self.resume_from_checkpoint = resume_from_checkpoint
        
        self.tb_writer = SummaryWriter(log_dir=run_dir) if is_main_process() else None
        self.last_checkpoint_time = time.time() if is_main_process() else None
        
        # Calculate evaluation step indices to use across the entire run
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

    def train(self):
        """Main training loop."""
        step = 1
        
        if self.resume_from_checkpoint:
            resumed_step = load_checkpoint(self.model_engine, self.train_dataloader, self.run_dir)
            if resumed_step is not None:
                step = resumed_step
        
        # Initial evaluation
        last_eval_loss = self.evaluate(step - 1)
        if is_main_process():
            print(f'- Initial evaluation loss: {last_eval_loss:.4f}')
        
        while self.train_dataloader.epoch <= self.epochs:
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
                    self._print_eval_progress(step, loss, last_eval_loss)
                last_eval_loss = loss
            
            # Time-based checkpointing
            self._save_checkpoint(step)
            
            step += 1
        
        self._save_model('final')
        
        if is_main_process():
            print('TRAINING COMPLETE!')

    def _write_metrics(self, prefix, metrics, step):
        loss = metrics[0].mean().item()
        self.tb_writer.add_scalar(f'{prefix}/loss', loss, step)
        return loss

    def _print_eval_progress(self, step, current_loss, last_loss):
        """Print evaluation progress with percentage change."""
        percent_change = (current_loss / last_loss - 1) * 100
        print(f'- Step {step} evaluation loss: {current_loss:.4f} (last: {last_loss:.4f}, Δ: {percent_change:.2f}%)')

    def _should_checkpoint(self):
        """Check if it's time to checkpoint based on time interval."""
        if not is_main_process():
            return False
        
        current_time = time.time()
        if (current_time - self.last_checkpoint_time) / 60 >= self.checkpoint_interval:
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
                prune_checkpoints(self.run_dir, self.max_checkpoints)


    def _save_model(self, name):
        """Save the trained model (LoRA adapters or full model)."""
        if self.lora_config is None:
            save_full_model(self.model_engine, self.pipeline_model, self.model_dir, self.run_dir, name)
        else:
            save_lora(self.model_engine, self.pipeline_model, self.lora_config, self.run_dir, name)