import argparse
from pathlib import Path

import torch
import yaml

from datasets import create_dataset, create_loader
from evaluation.retrieval import evaluate_retrieval
from models import CLIPRetrieval


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    return checkpoint


def print_metrics(metrics):
    print("\nImage -> Text")
    print(f"R@1  : {metrics['i2t_r1']:.2f}")
    print(f"R@5  : {metrics['i2t_r5']:.2f}")
    print(f"R@10 : {metrics['i2t_r10']:.2f}")
    print(f"Mean  : {metrics['i2t_mean']:.2f}")
    print(f"MedR  : {metrics['i2t_medr']:.2f}")
    print(f"MeanR : {metrics['i2t_meanr']:.2f}")

    print("\nText -> Image")
    print(f"R@1  : {metrics['t2i_r1']:.2f}")
    print(f"R@5  : {metrics['t2i_r5']:.2f}")
    print(f"R@10 : {metrics['t2i_r10']:.2f}")
    print(f"Mean  : {metrics['t2i_mean']:.2f}")
    print(f"MedR  : {metrics['t2i_medr']:.2f}")
    print(f"MeanR : {metrics['t2i_meanr']:.2f}")

    print(f"\nmR    : {metrics['mR']:.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Clean CLIP on RSITR test split."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 72)
    print("CLEAN CLIP RETRIEVAL TEST")
    print("=" * 72)
    print(f"Config     : {args.config}")
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Split      : {args.split}")
    print(f"Device     : {device}")

    model = CLIPRetrieval(config["model"]).to(device)
    checkpoint = load_checkpoint(
        model,
        Path(args.checkpoint),
        device,
    )
    model.eval()

    dataset = create_dataset(
        config["dataset"],
        evaluate=True,
        eval_split=args.split,
        eval_transform=model.backbone.preprocess_val,
    )

    eval_batch_size = int(
        config["training"].get("eval_batch_size", 128)
    )
    text_batch_size = int(
        config["training"].get("text_batch_size", 256)
    )
    num_workers = int(
        config["training"].get("num_workers", 4)
    )

    loader = create_loader(
        dataset,
        batch_size=eval_batch_size,
        num_workers=num_workers,
        is_train=False,
    )

    print(f"Images     : {len(dataset)}")
    print(f"Captions   : {len(dataset.text)}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")

    metrics, _ = evaluate_retrieval(
        model=model,
        data_loader=loader,
        dataset=dataset,
        device=device,
        text_batch_size=text_batch_size,
    )

    print_metrics(metrics)
    print("=" * 72)


if __name__ == "__main__":
    main()
