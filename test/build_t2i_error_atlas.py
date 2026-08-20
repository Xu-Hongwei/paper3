import argparse
import csv
import html
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import create_dataset, create_loader
from evaluation import evaluate_retrieval
from models import CLIPRetrieval
from utils import load_config, set_seed


FAILURE_LABELS = [
    "entity_presence",
    "attribute",
    "relation_spatial",
    "count_quantity",
    "scene_layout",
    "visual_appearance_confusion",
    "caption_underspecified",
    "semantic_false_negative",
    "annotation_noise",
    "other",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="构建 RSICD Clean CLIP T2I Top-1 错误样例图谱。"
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--output-dir", type=str, default="outputs/t2i_error_atlas")
    parser.add_argument("--image-batch-size", type=int, default=None)
    parser.add_argument("--text-batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--near-count", type=int, default=15)
    parser.add_argument("--moderate-count", type=int, default=15)
    parser.add_argument("--severe-count", type=int, default=10)
    parser.add_argument("--confident-count", type=int, default=10)
    parser.add_argument("--thumb-size", type=int, default=320)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(checkpoint.get("model", checkpoint), strict=True)
    return checkpoint


def resolve_image_path(image_root, image_ref):
    image_root = Path(image_root)
    direct = image_root / image_ref
    if direct.is_file():
        return direct

    flat = image_root / Path(image_ref).name
    if flat.is_file():
        return flat

    raise FileNotFoundError(
        f"Image not found: ref={image_ref!r}, tried={direct} and {flat}"
    )


def image_captions(dataset, image_id):
    return [dataset.text[text_id] for text_id in dataset.img2txt[int(image_id)]]


def compute_t2i_cases(scores_i2t, dataset, topk):
    scores_t2i = scores_i2t.T
    cases = []

    for text_id, scores in enumerate(scores_t2i):
        gt_image_id = int(dataset.txt2img[text_id])
        order = np.argsort(scores)[::-1]
        rank_lookup = np.empty_like(order)
        rank_lookup[order] = np.arange(len(order))
        gt_rank0 = int(rank_lookup[gt_image_id])

        if gt_rank0 == 0:
            continue

        top_ids = [int(x) for x in order[:topk]]
        top_scores = [float(scores[x]) for x in top_ids]
        top1_image_id = top_ids[0]
        gt_score = float(scores[gt_image_id])
        top1_score = float(scores[top1_image_id])

        cases.append({
            "text_id": int(text_id),
            "query": dataset.text[text_id],
            "gt_image_id": gt_image_id,
            "gt_rank": gt_rank0 + 1,
            "gt_score": gt_score,
            "top1_image_id": top1_image_id,
            "top1_score": top1_score,
            "wrong_minus_gt": top1_score - gt_score,
            "top_image_ids": top_ids,
            "top_scores": top_scores,
        })

    return cases


def evenly_pick(cases, count):
    if count <= 0 or not cases:
        return []
    if len(cases) <= count:
        return list(cases)

    indices = np.linspace(0, len(cases) - 1, count).round().astype(int)
    return [cases[i] for i in indices]


def stratified_sample(cases, near_count, moderate_count, severe_count, confident_count):
    near = sorted(
        [x for x in cases if 2 <= x["gt_rank"] <= 5],
        key=lambda x: (x["gt_rank"], -x["wrong_minus_gt"]),
    )
    moderate = sorted(
        [x for x in cases if 6 <= x["gt_rank"] <= 10],
        key=lambda x: (x["gt_rank"], -x["wrong_minus_gt"]),
    )
    severe = sorted(
        [x for x in cases if x["gt_rank"] > 10],
        key=lambda x: (-x["gt_rank"], -x["wrong_minus_gt"]),
    )

    selected = []
    used = set()

    def add(items, bucket):
        for item in items:
            if item["text_id"] in used:
                continue
            copied = dict(item)
            copied["sample_bucket"] = bucket
            selected.append(copied)
            used.add(item["text_id"])

    add(evenly_pick(near, near_count), "near_rank_2_5")
    add(evenly_pick(moderate, moderate_count), "moderate_rank_6_10")
    add(evenly_pick(severe, severe_count), "severe_rank_gt10")

    remaining = [x for x in cases if x["text_id"] not in used]
    confident = sorted(
        remaining,
        key=lambda x: x["wrong_minus_gt"],
        reverse=True,
    )[:confident_count]
    add(confident, "confident_wrong")

    selected.sort(
        key=lambda x: (
            {
                "near_rank_2_5": 0,
                "moderate_rank_6_10": 1,
                "severe_rank_gt10": 2,
                "confident_wrong": 3,
            }[x["sample_bucket"]],
            x["text_id"],
        )
    )
    return selected


def save_thumbnail(src, dst, max_size):
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src).convert("RGB") as image:
        image.thumbnail((max_size, max_size))
        image.save(dst, quality=92)


def materialize_case_assets(dataset, cases, output_dir, thumb_size):
    assets_dir = output_dir / "assets"
    for case_index, case in enumerate(cases, start=1):
        case_dir = assets_dir / f"case_{case_index:03d}"
        case["case_id"] = case_index

        gt_path = resolve_image_path(
            dataset.image_root,
            dataset.image[case["gt_image_id"]],
        )
        gt_thumb = case_dir / "gt.jpg"
        save_thumbnail(gt_path, gt_thumb, thumb_size)
        case["gt_thumb"] = gt_thumb.relative_to(output_dir).as_posix()
        case["gt_image_ref"] = dataset.image[case["gt_image_id"]]
        case["gt_captions"] = image_captions(dataset, case["gt_image_id"])

        candidates = []
        for rank, (image_id, score) in enumerate(
            zip(case["top_image_ids"], case["top_scores"]),
            start=1,
        ):
            image_path = resolve_image_path(
                dataset.image_root,
                dataset.image[image_id],
            )
            thumb = case_dir / f"top{rank}.jpg"
            save_thumbnail(image_path, thumb, thumb_size)
            candidates.append({
                "rank": rank,
                "image_id": image_id,
                "image_ref": dataset.image[image_id],
                "score": score,
                "thumb": thumb.relative_to(output_dir).as_posix(),
                "captions": image_captions(dataset, image_id),
                "is_gt": int(image_id == case["gt_image_id"]),
            })

        case["candidates"] = candidates


def write_cases_json(cases, path):
    with path.open("w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)


def write_cases_csv(cases, path):
    rows = []
    for case in cases:
        row = {
            "case_id": case["case_id"],
            "sample_bucket": case["sample_bucket"],
            "text_id": case["text_id"],
            "query": case["query"],
            "gt_image_id": case["gt_image_id"],
            "gt_image_ref": case["gt_image_ref"],
            "gt_rank": case["gt_rank"],
            "gt_score": case["gt_score"],
            "top1_image_id": case["top1_image_id"],
            "top1_image_ref": case["candidates"][0]["image_ref"],
            "top1_score": case["top1_score"],
            "wrong_minus_gt": case["wrong_minus_gt"],
            "gt_captions": " || ".join(case["gt_captions"]),
            "top1_captions": " || ".join(case["candidates"][0]["captions"]),
        }

        for candidate in case["candidates"]:
            rank = candidate["rank"]
            row[f"top{rank}_image_id"] = candidate["image_id"]
            row[f"top{rank}_image_ref"] = candidate["image_ref"]
            row[f"top{rank}_score"] = candidate["score"]

        rows.append(row)

    if not rows:
        return

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_annotation_csv(cases, path):
    fieldnames = [
        "case_id",
        "sample_bucket",
        "text_id",
        "gt_rank",
        "wrong_minus_gt",
        "query",
        "primary_failure_type",
        "secondary_failure_type",
        "same_coarse_category",
        "wrong_image_semantically_valid",
        "gt_caption_under_specified",
        "annotation_noise",
        "notes",
    ]

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for case in cases:
            writer.writerow({
                "case_id": case["case_id"],
                "sample_bucket": case["sample_bucket"],
                "text_id": case["text_id"],
                "gt_rank": case["gt_rank"],
                "wrong_minus_gt": case["wrong_minus_gt"],
                "query": case["query"],
                "primary_failure_type": "",
                "secondary_failure_type": "",
                "same_coarse_category": "",
                "wrong_image_semantically_valid": "",
                "gt_caption_under_specified": "",
                "annotation_noise": "",
                "notes": "",
            })


def list_html(items):
    return "<ul>" + "".join(
        f"<li>{html.escape(str(x))}</li>" for x in items
    ) + "</ul>"


def build_html(cases, metrics, output_path):
    labels = "".join(
        f'<option value="{html.escape(label)}">{html.escape(label)}</option>'
        for label in FAILURE_LABELS
    )

    case_blocks = []
    for case in cases:
        candidates_html = []
        for candidate in case["candidates"]:
            gt_badge = '<span class="gt-badge">GT</span>' if candidate["is_gt"] else ""
            candidates_html.append(
                f"""
                <div class="candidate">
                  <div class="candidate-title">
                    Top-{candidate['rank']} &nbsp; score={candidate['score']:.4f} {gt_badge}
                  </div>
                  <img src="{html.escape(candidate['thumb'])}">
                  <div class="small">image_id={candidate['image_id']}</div>
                  <div class="small">{html.escape(str(candidate['image_ref']))}</div>
                  <details>
                    <summary>5 captions</summary>
                    {list_html(candidate['captions'])}
                  </details>
                </div>
                """
            )

        case_blocks.append(
            f"""
            <section class="case" data-case="{case['case_id']}">
              <h2>Case {case['case_id']:03d} · {html.escape(case['sample_bucket'])}</h2>
              <div class="query"><b>Query:</b> {html.escape(case['query'])}</div>
              <div class="meta">
                text_id={case['text_id']} · GT rank={case['gt_rank']} ·
                GT score={case['gt_score']:.4f} · Top1 score={case['top1_score']:.4f} ·
                Wrong−GT={case['wrong_minus_gt']:+.4f}
              </div>

              <h3>Ground Truth</h3>
              <div class="gt-panel">
                <img src="{html.escape(case['gt_thumb'])}">
                <div>
                  <div class="small">image_id={case['gt_image_id']}</div>
                  <div class="small">{html.escape(str(case['gt_image_ref']))}</div>
                  {list_html(case['gt_captions'])}
                </div>
              </div>

              <h3>Retrieved Top-{len(case['candidates'])}</h3>
              <div class="candidates">
                {''.join(candidates_html)}
              </div>

              <div class="annotation">
                <label>Primary
                  <select data-field="primary_failure_type">
                    <option value=""></option>{labels}
                  </select>
                </label>
                <label>Secondary
                  <select data-field="secondary_failure_type">
                    <option value=""></option>{labels}
                  </select>
                </label>
                <label>Same coarse category
                  <select data-field="same_coarse_category">
                    <option value=""></option><option>yes</option><option>no</option><option>uncertain</option>
                  </select>
                </label>
                <label>Wrong image semantically valid
                  <select data-field="wrong_image_semantically_valid">
                    <option value=""></option><option>yes</option><option>no</option><option>uncertain</option>
                  </select>
                </label>
                <label>Caption under-specified
                  <select data-field="gt_caption_under_specified">
                    <option value=""></option><option>yes</option><option>no</option><option>uncertain</option>
                  </select>
                </label>
                <label>Annotation noise
                  <select data-field="annotation_noise">
                    <option value=""></option><option>yes</option><option>no</option><option>uncertain</option>
                  </select>
                </label>
                <label class="notes-label">Notes
                  <textarea data-field="notes" rows="3"></textarea>
                </label>
              </div>
            </section>
            """
        )

    doc = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>RSICD T2I Error Atlas</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; background: #f5f5f5; color: #222; }}
header {{ position: sticky; top: 0; background: white; padding: 12px 16px; z-index: 10; border-bottom: 1px solid #ccc; }}
.case {{ background: white; padding: 18px; margin: 20px 0; border-radius: 8px; }}
.query {{ font-size: 19px; margin: 10px 0; }}
.meta, .small {{ color: #555; font-size: 13px; }}
.gt-panel {{ display: flex; gap: 18px; align-items: flex-start; }}
.gt-panel img {{ max-width: 320px; max-height: 320px; }}
.candidates {{ display: flex; gap: 12px; flex-wrap: wrap; }}
.candidate {{ width: 240px; border: 1px solid #ddd; padding: 8px; }}
.candidate img {{ width: 100%; max-height: 240px; object-fit: contain; background: #eee; }}
.candidate-title {{ font-weight: 700; margin-bottom: 6px; }}
.gt-badge {{ background: #222; color: white; padding: 2px 5px; border-radius: 3px; }}
.annotation {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-top: 18px; padding-top: 14px; border-top: 1px solid #ddd; }}
.annotation label {{ display: flex; flex-direction: column; gap: 4px; font-size: 13px; }}
.notes-label {{ grid-column: 1 / -1; }}
textarea, select {{ font-size: 14px; padding: 5px; }}
button {{ margin-right: 8px; padding: 8px 12px; }}
ul {{ margin-top: 4px; }}
</style>
</head>
<body>
<header>
  <b>RSICD Clean CLIP · T2I Error Atlas</b>
  <span> · cases={len(cases)} · T2I R@1={metrics.get('t2i_r1', float('nan')):.2f}</span>
  <div style="margin-top:8px">
    <button onclick="saveAll()">Save to browser</button>
    <button onclick="exportCSV()">Export annotations CSV</button>
  </div>
</header>
{''.join(case_blocks)}
<script>
const fields = [
  "primary_failure_type","secondary_failure_type","same_coarse_category",
  "wrong_image_semantically_valid","gt_caption_under_specified",
  "annotation_noise","notes"
];

function saveAll() {{
  const data = {{}};
  document.querySelectorAll(".case").forEach(section => {{
    const id = section.dataset.case;
    data[id] = {{}};
    fields.forEach(field => {{
      const el = section.querySelector(`[data-field="${{field}}"]`);
      data[id][field] = el ? el.value : "";
    }});
  }});
  localStorage.setItem("rsicd_t2i_error_atlas", JSON.stringify(data));
  alert("Saved in browser localStorage.");
}}

function loadAll() {{
  const raw = localStorage.getItem("rsicd_t2i_error_atlas");
  if (!raw) return;
  const data = JSON.parse(raw);
  document.querySelectorAll(".case").forEach(section => {{
    const item = data[section.dataset.case] || {{}};
    fields.forEach(field => {{
      const el = section.querySelector(`[data-field="${{field}}"]`);
      if (el && item[field] !== undefined) el.value = item[field];
    }});
  }});
}}

function csvEscape(v) {{
  const s = String(v ?? "");
  return '"' + s.replaceAll('"', '""') + '"';
}}

function exportCSV() {{
  saveAll();
  const raw = JSON.parse(localStorage.getItem("rsicd_t2i_error_atlas") || "{{}}");
  const rows = [["case_id", ...fields]];
  document.querySelectorAll(".case").forEach(section => {{
    const id = section.dataset.case;
    const item = raw[id] || {{}};
    rows.push([id, ...fields.map(f => item[f] || "")]);
  }});
  const csv = rows.map(row => row.map(csvEscape).join(",")).join("\\n");
  const blob = new Blob(["\\ufeff" + csv], {{type:"text/csv;charset=utf-8"}});
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "manual_annotations_export.csv";
  a.click();
  URL.revokeObjectURL(url);
}}

loadAll();
</script>
</body>
</html>
"""
    output_path.write_text(doc, encoding="utf-8")


def main():
    args = parse_args()
    config = load_config(args.config)
    seed = int(args.seed if args.seed is not None else config.get("seed", 42))
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPRetrieval(config["model"])
    checkpoint = load_checkpoint(model, args.checkpoint)
    model = model.to(device).eval()

    dataset = create_dataset(
        config["dataset"],
        evaluate=True,
        eval_split=args.split,
        eval_transform=model.backbone.preprocess_val,
    )

    image_batch_size = int(
        args.image_batch_size
        or config["training"].get("eval_batch_size", 128)
    )
    text_batch_size = int(
        args.text_batch_size
        or config["training"].get("text_batch_size", 256)
    )
    num_workers = int(
        args.num_workers
        if args.num_workers is not None
        else config["training"].get("num_workers", 4)
    )

    loader = create_loader(
        dataset,
        batch_size=image_batch_size,
        num_workers=num_workers,
        is_train=False,
        pin_memory=True,
    )

    print("=" * 92)
    print("RSICD BASELINE T2I ERROR ATLAS")
    print("=" * 92)
    print(f"Split      : {args.split}")
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Epoch      : {checkpoint.get('epoch', 'unknown')}")
    print(f"Images     : {len(dataset)}")
    print(f"Captions   : {len(dataset.text)}")
    print(f"Device     : {device}")

    metrics, scores = evaluate_retrieval(
        model=model,
        data_loader=loader,
        dataset=dataset,
        device=device,
        text_batch_size=text_batch_size,
    )

    cases = compute_t2i_cases(scores, dataset, args.topk)
    selected = stratified_sample(
        cases,
        args.near_count,
        args.moderate_count,
        args.severe_count,
        args.confident_count,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    materialize_case_assets(
        dataset,
        selected,
        output_dir,
        args.thumb_size,
    )

    write_cases_json(selected, output_dir / "t2i_error_cases.json")
    write_cases_csv(selected, output_dir / "t2i_error_cases.csv")
    write_annotation_csv(selected, output_dir / "manual_annotations.csv")
    build_html(selected, metrics, output_dir / "index.html")

    bucket_counts = {}
    for case in selected:
        bucket_counts[case["sample_bucket"]] = (
            bucket_counts.get(case["sample_bucket"], 0) + 1
        )

    summary = {
        "split": args.split,
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "metrics": metrics,
        "num_all_t2i_errors": len(cases),
        "num_selected": len(selected),
        "bucket_counts": bucket_counts,
        "selection": {
            "near_rank_2_5": args.near_count,
            "moderate_rank_6_10": args.moderate_count,
            "severe_rank_gt10": args.severe_count,
            "confident_wrong": args.confident_count,
        },
        "failure_labels": FAILURE_LABELS,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 92)
    print("ATLAS READY")
    print("=" * 92)
    print(f"T2I R@1          : {metrics['t2i_r1']:.2f}")
    print(f"All Top-1 errors : {len(cases)}")
    print(f"Selected cases   : {len(selected)}")
    for bucket, count in bucket_counts.items():
        print(f"{bucket:<22}: {count}")
    print("-" * 92)
    print(f"HTML             : {output_dir / 'index.html'}")
    print(f"Cases CSV        : {output_dir / 't2i_error_cases.csv'}")
    print(f"Annotation CSV   : {output_dir / 'manual_annotations.csv'}")
    print(f"JSON             : {output_dir / 't2i_error_cases.json'}")
    print("=" * 92)


if __name__ == "__main__":
    main()
