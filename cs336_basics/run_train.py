from cs336_basics.transformer import *
from pathlib import Path

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    vocab_size: int = 10_000  # must match your tokenizer's vocab
    context_length: int = 64
    d_model: int = 128
    num_layers: int = 2
    num_heads: int = 4
    d_ff: int = 344
    rope_theta: float = 10_000.0


@dataclass
class AdamWConfig:
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0


@dataclass
class ScheduleConfig:
    warmup_steps: int = 100
    max_steps: int = 200
    max_lr: float = 3e-3
    min_lr: float = 3e-4


@dataclass
class OptimConfig:
    adamw: AdamWConfig = field(default_factory=AdamWConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    grad_max_l2_norm: float = 1.0


@dataclass
class DataConfig:
    train_filepath: str = "data/tinystories/valid_tokens.npy"
    checkpoint_path: str = "checkpoints/tinystories.pt"
    batch_size: int = 8


model_cfg = ModelConfig()
optim_cfg = OptimConfig()
data_cfg = DataConfig()
device = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    Path("checkpoints").mkdir(parents=True, exist_ok=True)

    # dataset
    dataset = np.load(data_cfg.train_filepath, mmap_mode="r")

    # model, optimizer
    model = TransformerLM(**vars(model_cfg)).to(device)
    opt = AdamW(params=model.parameters(), **vars(optim_cfg.adamw))

    for step in range(optim_cfg.schedule.max_steps):
        opt.zero_grad()

        # data
        inputs, targets = get_batch(
            dataset=dataset,
            batch_size=data_cfg.batch_size,
            context_length=model_cfg.context_length,
            device=device,
        )

        # forward
        logits = model(inputs)

        # loss
        loss = cross_entropy(
            logits.view(-1, logits.size(-1)),  # (batch * context_length, vocab)
            targets.view(-1),  # (batch * context_length)
        )
        print(loss.cpu().item())

        # backward
        loss.backward()  # Run backward pass, which computes gradients
        gradient_clipping(model.parameters(), optim_cfg.grad_max_l2_norm)

        # step
        lr = cosine_schedule(
            step,
            warmup_iters=optim_cfg.schedule.warmup_steps,
            cosine_cycle_iters=optim_cfg.schedule.max_steps,
            max_lr=optim_cfg.schedule.max_lr,
            min_lr=optim_cfg.schedule.min_lr,
        )
        for group in opt.param_groups:
            group["lr"] = lr
        opt.step()  # Run optimizer step

        # save checkpoint
        if step == optim_cfg.schedule.max_steps - 1:
            save_checkpoint(model, opt, step, data_cfg.checkpoint_path)


if __name__ == "__main__":
    main()
