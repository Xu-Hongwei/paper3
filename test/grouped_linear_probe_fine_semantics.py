import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import CLIPRetrieval


def parse_args():
    parser = argparse.ArgumentParser(
        description="Grouped linear probe: can GDINO-CLIP visual features decode text-defined fine clusters?"
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--cache", type=str, required=True)
    parser.add_argument("--samples-csv", type=str, required=True)
    parser.add_argument("--classes", nargs="+", default=["aircraft"])
    parser.add_argument("--min-area", type=float, default=0.002)
    parser.add_argument("--max-area", type=float, default=0.35)
    parser.add_argument("--cluster-ks", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--text-batch-size", type=int, default=256)
    parser.add_argument("--kmeans-iters", type=int, default=80)
    parser.add_argument("--kmeans-restarts", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/grouped_linear_probe_aircraft",
    )
    return parser.parse_args()


def load_checkpoint(model, path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint.get("model", checkpoint), strict=True)
    return checkpoint


def load_samples(path, classes):
    groups = {name: [] for name in classes}
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row["class"] not in groups:
                continue
            groups[row["class"]].append({
                "class": row["class"],
                "image_id": int(row["image_id"]),
                "dataset_index": int(row["dataset_index"]),
                "phrase": row["phrase"],
                "entity": row["entity"],
                "caption": row["caption"],
            })
    return groups


def cache_key(class_name, image_id):
    return f"{class_name}:{int(image_id)}"


@torch.no_grad()
def encode_phrases(model, groups, device, batch_size):
    phrases, seen = [], set()
    for samples in groups.values():
        for sample in samples:
            key = sample["phrase"].strip().lower()
            if key not in seen:
                seen.add(key)
                phrases.append(sample["phrase"].strip())

    features = []
    for start in range(0, len(phrases), batch_size):
        features.append(
            model.backbone.encode_text(
                phrases[start:start + batch_size],
                normalize=True,
            ).cpu()
        )
    features = torch.cat(features).float()
    return {phrase.lower(): features[i] for i, phrase in enumerate(phrases)}


def visual_variants(record, min_area, max_area):
    feats = record.get("instance_features")
    areas = record.get("area_ratios", [])

    if feats is None or len(areas) == 0:
        return None

    feats = F.normalize(feats.float(), dim=-1)
    valid = [
        i for i, area in enumerate(areas)
        if min_area <= float(area) <= max_area
    ]
    if not valid:
        return None

    valid_feats = feats[valid]
    return {
        "raw_mean": F.normalize(feats.mean(dim=0), dim=0),
        "filtered_top1": F.normalize(valid_feats[0], dim=0),
        "filtered_mean": F.normalize(valid_feats.mean(dim=0), dim=0),
    }


def spherical_kmeans(x, k, seed, max_iters, restarts):
    x = F.normalize(x.float(), dim=-1)
    n = len(x)
    best = None

    for restart in range(restarts):
        generator = torch.Generator().manual_seed(seed + restart * 1009 + k)
        centers = x[torch.randperm(n, generator=generator)[:k]].clone()
        labels = None

        for _ in range(max_iters):
            new_labels = torch.argmax(x @ centers.t(), dim=1)
            if labels is not None and torch.equal(new_labels, labels):
                break
            labels = new_labels

            updated = []
            for c in range(k):
                mask = labels == c
                if mask.any():
                    center = x[mask].mean(dim=0)
                else:
                    center = x[
                        torch.randint(0, n, (1,), generator=generator).item()
                    ]
                updated.append(F.normalize(center, dim=0))
            centers = torch.stack(updated)

        objective = float((x * centers[labels]).sum().item())
        if best is None or objective > best[0]:
            best = (objective, labels.clone(), centers.clone())

    return best[1], best[2]


def assign_text_labels(samples, phrase_features, k, seed, args):
    unique_phrases = sorted({
        sample["phrase"].strip().lower() for sample in samples
    })
    unique_features = torch.stack([
        phrase_features[phrase] for phrase in unique_phrases
    ])
    _, centers = spherical_kmeans(
        unique_features,
        k,
        seed,
        args.kmeans_iters,
        args.kmeans_restarts,
    )

    sample_features = torch.stack([
        phrase_features[sample["phrase"].strip().lower()]
        for sample in samples
    ])
    labels = torch.argmax(sample_features @ centers.t(), dim=1)
    return labels, centers, unique_phrases


def build_group_stratified_folds(samples, labels, requested_folds, seed):
    phrase_to_indices = defaultdict(list)
    for i, sample in enumerate(samples):
        phrase_to_indices[sample["phrase"].strip().lower()].append(i)

    label_to_groups = defaultdict(list)
    for phrase, indices in phrase_to_indices.items():
        group_labels = {int(labels[i]) for i in indices}
        if len(group_labels) != 1:
            raise RuntimeError(f"同一 phrase 出现多个 text cluster: {phrase}")
        label_to_groups[next(iter(group_labels))].append(phrase)

    min_groups = min(len(groups) for groups in label_to_groups.values())
    n_folds = min(requested_folds, min_groups)
    if n_folds < 2:
        return [], {
            "requested_folds": requested_folds,
            "effective_folds": n_folds,
            "groups_per_class": {
                str(label): len(groups)
                for label, groups in sorted(label_to_groups.items())
            },
        }

    rng = random.Random(seed)
    fold_groups = [set() for _ in range(n_folds)]

    for label in sorted(label_to_groups):
        groups = list(label_to_groups[label])
        rng.shuffle(groups)
        for i, phrase in enumerate(groups):
            fold_groups[i % n_folds].add(phrase)

    folds = []
    all_indices = set(range(len(samples)))

    for test_phrases in fold_groups:
        test_indices = sorted(
            i for phrase in test_phrases for i in phrase_to_indices[phrase]
        )
        train_indices = sorted(all_indices - set(test_indices))
        folds.append((train_indices, test_indices))

    meta = {
        "requested_folds": requested_folds,
        "effective_folds": n_folds,
        "groups_per_class": {
            str(label): len(groups)
            for label, groups in sorted(label_to_groups.items())
        },
        "num_unique_phrases": len(phrase_to_indices),
    }
    return folds, meta


def confusion_matrix(y_true, y_pred, num_classes):
    matrix = torch.zeros((num_classes, num_classes), dtype=torch.long)
    for true, pred in zip(y_true.tolist(), y_pred.tolist()):
        matrix[int(true), int(pred)] += 1
    return matrix


def classification_metrics(y_true, y_pred, num_classes):
    cm = confusion_matrix(y_true, y_pred, num_classes).float()
    accuracy = float((y_true == y_pred).float().mean().item())

    recalls, f1s = [], []
    for c in range(num_classes):
        tp = cm[c, c]
        fn = cm[c].sum() - tp
        fp = cm[:, c].sum() - tp

        recall = tp / (tp + fn).clamp_min(1.0)
        precision = tp / (tp + fp).clamp_min(1.0)
        f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-8)

        recalls.append(float(recall.item()))
        f1s.append(float(f1.item()))

    return {
        "accuracy": accuracy,
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
    }


def train_linear_probe(
    x_train, y_train, x_test, num_classes,
    device, seed, epochs, lr, weight_decay,
):
    torch.manual_seed(seed)
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_test = x_test.to(device)

    classifier = nn.Linear(x_train.shape[1], num_classes).to(device)

    counts = torch.bincount(y_train, minlength=num_classes).float()
    weights = counts.sum() / counts.clamp_min(1.0)
    weights = weights / weights.mean()

    optimizer = torch.optim.AdamW(
        classifier.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    classifier.train()
    for _ in range(epochs):
        logits = classifier(x_train)
        loss = F.cross_entropy(logits, y_train, weight=weights)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    classifier.eval()
    with torch.no_grad():
        return torch.argmax(classifier(x_test), dim=1).cpu()


def majority_predict(y_train, test_size):
    counts = torch.bincount(y_train)
    majority = int(torch.argmax(counts).item())
    return torch.full((test_size,), majority, dtype=torch.long)


def zero_shot_text_proto_predict(x_visual, text_centers):
    return torch.argmax(
        F.normalize(x_visual.float(), dim=-1)
        @ F.normalize(text_centers.float(), dim=-1).t(),
        dim=1,
    )


def aggregate_fold_metrics(fold_results):
    keys = ["accuracy", "balanced_accuracy", "macro_f1"]
    output = {}
    for key in keys:
        values = [result[key] for result in fold_results]
        output[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "folds": values,
        }
    return output


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    model = CLIPRetrieval(config["model"])
    checkpoint = load_checkpoint(model, args.checkpoint)
    model = model.to(device).eval()

    cache_blob = torch.load(
        args.cache,
        map_location="cpu",
        weights_only=False,
    )
    cache = cache_blob["records"]
    groups = load_samples(args.samples_csv, args.classes)
    phrase_features = encode_phrases(
        model,
        groups,
        device,
        args.text_batch_size,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "metadata": {
            "purpose": (
                "Diagnostic: determine whether text-defined fine semantics are linearly "
                "decodable from GDINO-selected CLIP visual features."
            ),
            "checkpoint": args.checkpoint,
            "checkpoint_epoch": checkpoint.get("epoch", None),
            "cache": args.cache,
            "samples_csv": args.samples_csv,
            "area_range": [args.min_area, args.max_area],
            "visual_modes": ["raw_mean", "filtered_top1", "filtered_mean"],
            "text_cluster_basis": "unique fine phrases",
            "split": "group-stratified CV by exact fine phrase",
        },
        "classes": {},
    }

    rows = []

    print("=" * 116)
    print("GROUPED LINEAR PROBE: VISUAL FEATURE -> TEXT-DEFINED FINE CLUSTER")
    print("=" * 116)
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Epoch      : {checkpoint.get('epoch', 'unknown')}")
    print(f"Area range : [{args.min_area}, {args.max_area}]")
    print("Text KMeans: UNIQUE fine phrases only")
    print("CV split   : exact fine phrase groups never cross train/test")
    print("=" * 116)

    for class_name in args.classes:
        prepared = []

        for sample in groups[class_name]:
            record = cache.get(cache_key(class_name, sample["image_id"]))
            if record is None:
                continue

            variants = visual_variants(
                record,
                args.min_area,
                args.max_area,
            )
            if variants is None:
                continue

            prepared.append({
                **sample,
                "raw_mean": variants["raw_mean"],
                "filtered_top1": variants["filtered_top1"],
                "filtered_mean": variants["filtered_mean"],
            })

        print(f"\n{class_name}: common samples={len(prepared)}/{len(groups[class_name])}")

        class_summary = {
            "common_samples": len(prepared),
            "total_samples": len(groups[class_name]),
            "k_results": {},
        }

        if len(prepared) < 4:
            summary["classes"][class_name] = class_summary
            continue

        for k in args.cluster_ks:
            unique_phrase_count = len({
                sample["phrase"].strip().lower()
                for sample in prepared
            })
            if k > unique_phrase_count:
                continue

            labels, text_centers, unique_phrases = assign_text_labels(
                prepared,
                phrase_features,
                k,
                args.seed + k * 17,
                args,
            )

            folds, fold_meta = build_group_stratified_folds(
                prepared,
                labels,
                args.folds,
                args.seed + k * 101,
            )

            cluster_sample_counts = torch.bincount(
                labels,
                minlength=k,
            ).tolist()
            cluster_phrase_counts = [0] * k
            seen = set()
            for sample, label in zip(prepared, labels.tolist()):
                phrase = sample["phrase"].strip().lower()
                key = (phrase, int(label))
                if key not in seen:
                    seen.add(key)
                    cluster_phrase_counts[int(label)] += 1

            print(
                f"\n  K={k}: samples={len(prepared)}, unique_phrases={len(unique_phrases)}, "
                f"sample_counts={cluster_sample_counts}, phrase_counts={cluster_phrase_counts}, "
                f"folds={fold_meta['effective_folds']}"
            )

            if not folds:
                print("    Skip: grouped CV 无法保证每个 cluster 至少有两个 phrase groups。")
                class_summary["k_results"][str(k)] = {
                    "status": "skipped_insufficient_phrase_groups",
                    "cluster_sample_counts": cluster_sample_counts,
                    "cluster_phrase_counts": cluster_phrase_counts,
                    "fold_meta": fold_meta,
                }
                continue

            k_summary = {
                "cluster_sample_counts": cluster_sample_counts,
                "cluster_phrase_counts": cluster_phrase_counts,
                "fold_meta": fold_meta,
                "modes": {},
            }

            for mode_index, mode in enumerate(
                ["raw_mean", "filtered_top1", "filtered_mean"]
            ):
                x = F.normalize(
                    torch.stack([sample[mode] for sample in prepared]).float(),
                    dim=-1,
                )

                probe_results = []
                majority_results = []
                zero_shot_results = []

                for fold_id, (train_idx, test_idx) in enumerate(folds):
                    train_idx = torch.tensor(train_idx, dtype=torch.long)
                    test_idx = torch.tensor(test_idx, dtype=torch.long)

                    x_train = x[train_idx]
                    y_train = labels[train_idx]
                    x_test = x[test_idx]
                    y_test = labels[test_idx]

                    pred = train_linear_probe(
                        x_train,
                        y_train,
                        x_test,
                        k,
                        device,
                        args.seed + 1000 * mode_index + 100 * k + fold_id,
                        args.epochs,
                        args.lr,
                        args.weight_decay,
                    )
                    probe_results.append(
                        classification_metrics(y_test, pred, k)
                    )

                    majority_pred = majority_predict(
                        y_train,
                        len(test_idx),
                    )
                    majority_results.append(
                        classification_metrics(
                            y_test,
                            majority_pred,
                            k,
                        )
                    )

                    zero_shot_pred = zero_shot_text_proto_predict(
                        x_test,
                        text_centers,
                    )
                    zero_shot_results.append(
                        classification_metrics(
                            y_test,
                            zero_shot_pred,
                            k,
                        )
                    )

                probe = aggregate_fold_metrics(probe_results)
                majority = aggregate_fold_metrics(majority_results)
                zero_shot = aggregate_fold_metrics(zero_shot_results)

                mode_result = {
                    "linear_probe": probe,
                    "majority_baseline": majority,
                    "zero_shot_text_prototype": zero_shot,
                    "uniform_random_accuracy": 1.0 / k,
                }
                k_summary["modes"][mode] = mode_result

                rows.append({
                    "class": class_name,
                    "k": k,
                    "mode": mode,
                    "common_samples": len(prepared),
                    "unique_phrases": len(unique_phrases),
                    "effective_folds": fold_meta["effective_folds"],
                    "probe_accuracy_mean": probe["accuracy"]["mean"],
                    "probe_accuracy_std": probe["accuracy"]["std"],
                    "probe_balanced_accuracy_mean": probe["balanced_accuracy"]["mean"],
                    "probe_balanced_accuracy_std": probe["balanced_accuracy"]["std"],
                    "probe_macro_f1_mean": probe["macro_f1"]["mean"],
                    "probe_macro_f1_std": probe["macro_f1"]["std"],
                    "majority_accuracy_mean": majority["accuracy"]["mean"],
                    "majority_balanced_accuracy_mean": majority["balanced_accuracy"]["mean"],
                    "zero_shot_accuracy_mean": zero_shot["accuracy"]["mean"],
                    "zero_shot_balanced_accuracy_mean": zero_shot["balanced_accuracy"]["mean"],
                    "uniform_random_accuracy": 1.0 / k,
                })

                print(
                    f"    [{mode:<15}] "
                    f"Probe Acc={probe['accuracy']['mean']:.3f}±{probe['accuracy']['std']:.3f} | "
                    f"BalAcc={probe['balanced_accuracy']['mean']:.3f}±{probe['balanced_accuracy']['std']:.3f} | "
                    f"MacroF1={probe['macro_f1']['mean']:.3f} | "
                    f"Majority BalAcc={majority['balanced_accuracy']['mean']:.3f} | "
                    f"ZeroShot BalAcc={zero_shot['balanced_accuracy']['mean']:.3f} | "
                    f"Random={1.0 / k:.3f}"
                )

            class_summary["k_results"][str(k)] = k_summary

        summary["classes"][class_name] = class_summary

    summary_path = output_dir / "grouped_linear_probe_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    metrics_path = output_dir / "grouped_linear_probe_metrics.csv"
    if rows:
        with metrics_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print("\n" + "=" * 116)
    print("DECISION RULE")
    print("=" * 116)
    print("1. 重点看 Balanced Accuracy，不只看普通 Accuracy。")
    print("2. 若 Linear Probe 明显高于 Majority / 1-K chance，而 ZeroShot 仍低：")
    print("   -> fine semantics 在视觉 feature 中可线性解码，但原生 Text-Visual 轴未对齐。")
    print("   -> 下一步优先 Text-Conditioned Visual Organization / learned projection。")
    print("3. 若 Linear Probe 也接近 baseline：")
    print("   -> 当前 CLIP local visual feature 缺少足够 fine semantics。")
    print("   -> 下一步优先 Grounding-to-CLIP Distillation / local representation adaptation。")
    print("4. K=4 优先作为主判定；K=2 看粗粒度可解码性，K=8 只作为更难的辅助诊断。")
    print("-" * 116)
    print(f"Summary: {summary_path}")
    print(f"Metrics: {metrics_path}")
    print("=" * 116)


if __name__ == "__main__":
    main()
