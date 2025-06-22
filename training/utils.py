import os
import glob
import torch
import optimi


def get_most_recent_run_dir(output_dir):
    return list(sorted(glob.glob(os.path.join(output_dir, '*'))))[-1]


def write_metrics(tb_writer, prefix, metrics, step):
    loss = metrics[0].mean().item()
    tb_writer.add_scalar(f'{prefix}/loss', loss, step)
    return loss


def get_optimizer(model_parameters, config):
    optimizer_kwargs = {
        "params": model_parameters,
        "lr": config['lr'],
        "betas": (config.get('beta1', 0.9), config.get('beta2', 0.99)),
        "weight_decay": config.get('weight_decay', 0.0),
        "eps": config.get('eps', 1e-6),
        "kahan_sum": True
    }
    return optimi.AdamW(**optimizer_kwargs)


def make_rms_ratio_fn(beta):
    def rms_ratio_fn(step):
        return torch.sqrt(torch.tensor((1 - beta**step)/(1 + beta**step))).item()
    return rms_ratio_fn
