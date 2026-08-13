import math
from torch.optim.lr_scheduler import LambdaLR


def build_scheduler(
    optimizer,
    total_steps,
    warmup_ratio=0.05,
):
    """
    Linear warmup + linear decay.

    lr:
        0
        ↑
        │     /\
        │    /  \
        │   /    \
        │  /      \
        └──────────────→ step
          warmup
    """

    warmup_steps = int(
        total_steps * warmup_ratio
    )

    def lr_lambda(step):

        if step < warmup_steps:

            return float(step + 1) / float(
                max(1, warmup_steps)
            )

        remaining_steps = (
            total_steps - step
        )

        decay_steps = (
            total_steps - warmup_steps
        )

        return max(
            0.0,
            float(remaining_steps)
            / float(max(1, decay_steps)),
        )

    scheduler = LambdaLR(
        optimizer,
        lr_lambda,
    )

    return scheduler