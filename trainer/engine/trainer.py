import math
import time
from pathlib import Path

import torch
from tqdm import tqdm

from trainer.loss.cross_entropy import cross_entropy
from trainer.utils.gradient_clip import gradient_clipping


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        scheduler=None,
        train_loader=None,
        valid_loader=None,
        device="cpu",
        grad_clip=None,
        log_interval=10,
        eval_interval=500,
        eval_steps=50,
        checkpoint_dir=None,
        use_wandb=False,
        wandb_project=None,
        wandb_run_name=None,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = torch.device(device)
        self.grad_clip = grad_clip
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.eval_steps = eval_steps
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
        
        if self.checkpoint_dir is not None:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.use_wandb = use_wandb
        if use_wandb:
            import wandb
            self.wandb = wandb
            wandb.init(project=wandb_project, name=wandb_run_name)
        else:
            self.wandb = None
        self.global_step = 0


    def train(self, max_steps: int):
        self.model.train()
        data_iter = iter(self.train_loader)
        pbar = tqdm(range(max_steps), desc="Training")
        running_loss = 0.0
        
        for _ in pbar:
            try:
                input_ids, labels = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                input_ids, labels = next(data_iter)
            input_ids = input_ids.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            start_time = time.perf_counter()
            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(input_ids)
            loss = cross_entropy(logits, labels)
            loss.backward()
            grad_norm = None
            if self.grad_clip is not None:
                grad_norm = gradient_clipping(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            elapsed = time.perf_counter() - start_time
            self.global_step += 1
            loss_value = loss.item()
            running_loss += loss_value
            ppl = math.exp(min(loss_value, 20))
            tokens_per_second = input_ids.numel() / elapsed
            lr = self._get_lr()
            metrics = {
                "train/loss": loss_value,
                "train/ppl": ppl,
                "train/lr": lr,
                "train/tokens_per_sec": tokens_per_second,
            }
            if grad_norm is not None:
                metrics["train/grad_norm"] = grad_norm
            if self.global_step % self.log_interval == 0:
                pbar.set_postfix(
                    loss=f"{loss_value:.4f}",
                    ppl=f"{ppl:.2f}",
                    lr=f"{lr:.2e}",
                )
                self._log(metrics)
            if (
                self.valid_loader is not None
                and self.global_step % self.eval_interval == 0
            ):
                valid_metrics = self.evaluate()
                self._log(valid_metrics)
                pbar.set_postfix(
                    loss=f"{loss_value:.4f}",
                    val_loss=f"{valid_metrics['valid/loss']:.4f}",
                    ppl=f"{valid_metrics['valid/ppl']:.2f}",
                )
                self.save_checkpoint()
        
        self.save_checkpoint()
        return {"train_loss": running_loss / max_steps}


    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        total_loss = 0.0
        steps = 0
        for input_ids, labels in self.valid_loader:
            input_ids = input_ids.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            logits = self.model(input_ids)
            loss = cross_entropy(logits, labels)
            total_loss += loss.item()
            steps += 1
            if steps >= self.eval_steps:
                break
        loss = total_loss / max(steps, 1)
        ppl = math.exp(min(loss, 20))
        self.model.train()
        return {"valid/loss": loss, "valid/ppl": ppl}


    def save_checkpoint(self):
        if self.checkpoint_dir is None:
            return
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "global_step": self.global_step,
        }
        if self.scheduler is not None:
            checkpoint["scheduler"] = self.scheduler.state_dict()
        path = self.checkpoint_dir / f"checkpoint_{self.global_step}.pt"
        torch.save(checkpoint, path)


    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.scheduler is not None and "scheduler" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.global_step = checkpoint["global_step"]


    def _get_lr(self):
        if self.scheduler is not None:
            return self.scheduler.get_lr()
        return self.optimizer.param_groups[0]["lr"]


    def _log(self, metrics):
        if self.wandb is not None:
            self.wandb.log(metrics, step=self.global_step)