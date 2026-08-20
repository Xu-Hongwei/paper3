import argparse
from pathlib import Path

import torch

from datasets import create_dataset, create_loader
from evaluation import evaluate_retrieval
from models import CLIPRetrieval
from utils import load_config, set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate CLIP retrieval checkpoint on RSITR."
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["val", "test"],
    )
    return parser.parse_args()


def load_checkpoint(model, checkpoint_path):
    """严格加载当前 CLIPRetrieval checkpoint。"""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict, strict=True)

    if isinstance(checkpoint, dict):
        print(
            f"Checkpoint epoch: "
            f"{checkpoint.get('epoch', 'unknown')}"
        )

    return checkpoint


def print_metrics(metrics, split):
    print("\n" + "=" * 72)
    print(f"{split.upper()} RETRIEVAL RESULT")
    print("=" * 72)

    print("\nImage -> Text")
    print(f"R@1  : {metrics['i2t_r1']:.2f}")
    print(f"R@5  : {metrics['i2t_r5']:.2f}")
    print(f"R@10 : {metrics['i2t_r10']:.2f}")
    print(f"Mean : {metrics['i2t_mean']:.2f}")
    print(f"MedR : {metrics['i2t_medr']:.2f}")
    if "i2t_meanr" in metrics:
        print(f"MeanR: {metrics['i2t_meanr']:.2f}")

    print("\nText -> Image")
    print(f"R@1  : {metrics['t2i_r1']:.2f}")
    print(f"R@5  : {metrics['t2i_r5']:.2f}")
    print(f"R@10 : {metrics['t2i_r10']:.2f}")
    print(f"Mean : {metrics['t2i_mean']:.2f}")
    print(f"MedR : {metrics['t2i_medr']:.2f}")
    if "t2i_meanr" in metrics:
        print(f"MeanR: {metrics['t2i_meanr']:.2f}")

    print(f"\nmR   : {metrics['mR']:.2f}")
    print("=" * 72)


def main():
    args = parse_args()
    config = load_config(args.config)

    seed = int(config.get("seed", 42))
    set_seed(seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 72)
    print("CLIP RETRIEVAL TEST")
    print("=" * 72)
    print(f"Config     : {args.config}")
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Split      : {args.split}")
    print(f"Device     : {device}")

    model = CLIPRetrieval(config["model"])
    load_checkpoint(model, args.checkpoint)
    model = model.to(device).eval()

    dataset = create_dataset(
        config["dataset"],
        evaluate=True,
        eval_split=args.split,
        eval_transform=model.backbone.preprocess_val,
    )

    train_cfg = config["training"]
    eval_batch_size = int(
        train_cfg.get("eval_batch_size", 128)
    )
    text_batch_size = int(
        train_cfg.get("text_batch_size", 256)
    )
    num_workers = int(
        train_cfg.get("num_workers", 8)
    )

    loader = create_loader(
        dataset,
        batch_size=eval_batch_size,
        num_workers=num_workers,
        is_train=False,
        pin_memory=True,
    )

    print(f"Images     : {len(dataset)}")
    print(f"Captions   : {len(dataset.text)}")
    print(f"Image batch: {eval_batch_size}")
    print(f"Text batch : {text_batch_size}")

    metrics, scores = evaluate_retrieval(
        model=model,
        data_loader=loader,
        dataset=dataset,
        device=device,
        text_batch_size=text_batch_size,
    )

    expected_shape = (
        len(dataset),
        len(dataset.text),
    )
    if tuple(scores.shape) != expected_shape:
        raise RuntimeError(
            "Unexpected similarity matrix shape: "
            f"{tuple(scores.shape)} != {expected_shape}"
        )

    print_metrics(metrics, args.split)


if __name__ == "__main__":
    main()
