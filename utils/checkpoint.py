from pathlib import Path

import torch


def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    epoch,
    metrics,
    config,
):
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    state = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": (
            scheduler.state_dict()
            if scheduler is not None
            else None
        ),
        "metrics": metrics,
        "config": config,
    }

    torch.save(
        state,
        path,
    )