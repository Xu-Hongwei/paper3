"""Build RSITMD split files for the paper3 pipeline (official protocol).

Official RSITMD split (embedded in the authors' dataset.json):
    train 4,291 / test 452  (4,743 images total, 5 captions each, NO val).

Because the paper3 pipeline requires a validation set for checkpoint
selection, we carve a *stratified* validation set from the official
training set with a fixed random seed (default: 856 images, ~20% of the
official training set, stratified over the 33 filename-prefix categories).
The official 452-image test set is used verbatim, so reported test
numbers remain comparable with all published RSITMD results.

Notes:
  - RSITMD captions are stored verbatim (identical to dataset.json "raw"),
    matching the existing data/rsitmd files.
  - RSITMD contains no degenerate captions (unlike RSICD), so no caption
    filtering is needed; the loading layer guard remains a no-op safety net.
  - `label` is a deterministic category index (sorted category names);
    `label_name` is the filename-prefix category. Neither is consumed by
    the training/eval code (only `image_id`/`label_name` are read by
    tools/structured_semantics).

Outputs (paper3 pipeline formats, same as the existing files):
  rsitmd_train_official.json : list of {image, caption, image_id, label_name, label}
  rsitmd_val_official.json   : list of {image, caption: [5], label_name, label}
  rsitmd_test_official.json  : list of {image, caption: [5]}

Usage:
  python tools/build_rsitmd_split.py [--val-size 856] [--seed 42] [--out-dir data/rsitmd]
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.utils import pre_caption

SRC = r"E:\datasets\remote_sensing_captioning\RSITMD\dataset.json"
IMG_ROOT = r"E:\datasets\remote_sensing_captioning\RSITMD\images"
MAX_WORDS = 30


def category_of(filename: str) -> str:
    base = os.path.basename(filename)
    assert base.endswith(".tif") and "_" in base, f"unexpected filename: {base}"
    return base.split("_")[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-size", type=int, default=856,
                        help="validation image count carved from official train")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for the stratified val carve")
    parser.add_argument("--out-dir", default=r"E:\paper3\data\rsitmd")
    args = parser.parse_args()

    with open(SRC, encoding="utf-8") as f:
        data = json.load(f)

    images = data["images"]
    print(f"total images: {len(images)}")

    # --- 1. integrity checks on the official source -------------------------
    split_counts = {}
    for img in images:
        s = img["split"]
        split_counts[s] = split_counts.get(s, 0) + 1
        assert len(img["sentences"]) == 5, (
            f"image {img['filename']} has {len(img['sentences'])} sentences"
        )
    print(f"official split counts: {split_counts}")
    assert set(split_counts) == {"train", "test"}
    assert sum(split_counts.values()) == len(images)

    train_imgs = [i for i in images if i["split"] == "train"]
    test_imgs = [i for i in images if i["split"] == "test"]
    assert len(train_imgs) == 4291 and len(test_imgs) == 452

    # --- 2. categories (33 filename-prefix categories) ----------------------
    categories = sorted({category_of(i["filename"]) for i in images})
    print(f"categories: {len(categories)}")
    cat_idx = {c: i for i, c in enumerate(categories)}

    # --- 3. stratified val carve from official train ------------------------
    rng = random.Random(args.seed)
    by_cat = {}
    for img in train_imgs:
        by_cat.setdefault(category_of(img["filename"]), []).append(img)

    total_train = len(train_imgs)
    quota = {c: int(len(v) * args.val_size / total_train) for c, v in by_cat.items()}
    assigned = sum(quota.values())
    # largest-remainder adjustment to hit exactly args.val_size
    remainder_order = sorted(
        by_cat,
        key=lambda c: len(by_cat[c]) * args.val_size / total_train - quota[c],
        reverse=True,
    )
    i = 0
    while assigned < args.val_size:
        c = remainder_order[i % len(remainder_order)]
        quota[c] += 1
        assigned += 1
        i += 1

    val_imgs = []
    for c, pool in by_cat.items():
        val_imgs.extend(rng.sample(pool, quota[c]))
    val_names = {img["filename"] for img in val_imgs}
    print(f"val carve: {len(val_names)} images (target {args.val_size}, seed {args.seed})")

    train_names = {img["filename"] for img in train_imgs} - val_names
    print(f"train after carve: {len(train_names)} images")

    # --- 4. build outputs ----------------------------------------------------
    def make_train_entry(img, raw):
        cat = category_of(img["filename"])
        return {
            "image": f"train/{img['filename']}",
            "caption": raw,
            "image_id": img["imgid"],
            "label_name": cat,
            "label": cat_idx[cat],
        }

    def make_eval_entry(img, with_labels):
        entry = {
            "image": f"{img['split']}/{img['filename']}",
            "caption": [s["raw"] for s in img["sentences"]],
        }
        if with_labels:
            cat = category_of(img["filename"])
            entry["label_name"] = cat
            entry["label"] = cat_idx[cat]
        return entry

    train = []
    for img in train_imgs:
        if img["filename"] in train_names:
            for s in img["sentences"]:
                train.append(make_train_entry(img, s["raw"]))

    val = [make_eval_entry(img, True) for img in val_imgs]
    test = [make_eval_entry(img, False) for img in test_imgs]

    print(f"train entries: {len(train)} (unique images: {len(train_names)})")
    print(f"val images: {len(val)}")
    print(f"test images: {len(test)}")

    # --- 5. verification -----------------------------------------------------
    def base_set(entries):
        return {os.path.basename(e["image"]) for e in entries}

    st, sv, ste = base_set(train), base_set(val), base_set(test)
    assert len(st) == len(train_names) and st == train_names
    assert not (st & sv) and not (st & ste) and not (sv & ste)
    print("overlap train∩val / train∩test / val∩test: 0 / 0 / 0  [OK]")

    # image resolvability against flat images folder
    for name, entries in (("train", train), ("val", val), ("test", test)):
        missing = [
            e["image"]
            for e in entries
            if not os.path.isfile(os.path.join(IMG_ROOT, os.path.basename(e["image"])))
        ]
        print(f"{name}: unresolvable image files = {len(missing)}")
        assert not missing

    # no degenerate captions (pre_caption must never fail)
    def scan_bad(entries):
        bad = 0
        for e in entries:
            caps = e["caption"] if isinstance(e["caption"], list) else [e["caption"]]
            for c in caps:
                try:
                    pre_caption(c, MAX_WORDS)
                except ValueError:
                    bad += 1
        return bad

    for name, entries in (("train", train), ("val", val), ("test", test)):
        bad = scan_bad(entries)
        print(f"{name}: degenerate captions = {bad}")
        assert bad == 0

    # image_id == official imgid
    imgid_of = {i["filename"]: i["imgid"] for i in images}
    for e in train:
        assert e["image_id"] == imgid_of[os.path.basename(e["image"])]
    print("train image_id == official imgid  [OK]")

    # test equivalence with the previous test file
    old_test_path = os.path.join(args.out_dir, "rsitmd_test.json")
    if os.path.exists(old_test_path):
        with open(old_test_path, encoding="utf-8") as f:
            old_test = json.load(f)
        old_map = {os.path.basename(e["image"]): e["caption"] for e in old_test}
        new_map = {os.path.basename(e["image"]): e["caption"] for e in test}
        same = sum(1 for b in old_map if b in new_map and old_map[b] == new_map[b])
        print(
            f"test identical to previous file: "
            f"{set(old_map) == set(new_map)}; identical caption lists: {same}/{len(old_map)}"
        )

    # overlap with the previous (non-reproducible) carve, for the record
    old_val_path = os.path.join(args.out_dir, "rsitmd_val.json")
    if os.path.exists(old_val_path):
        with open(old_val_path, encoding="utf-8") as f:
            old_val = json.load(f)
        old_val_names = {os.path.basename(e["image"]) for e in old_val}
        print(
            f"val overlap with previous carve: "
            f"{len(old_val_names & val_names)}/{len(val_names)}"
        )

    # --- 6. write outputs ----------------------------------------------------
    os.makedirs(args.out_dir, exist_ok=True)
    for name, obj in (
        ("rsitmd_train_official.json", train),
        ("rsitmd_val_official.json", val),
        ("rsitmd_test_official.json", test),
    ):
        out = os.path.join(args.out_dir, name)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=1)
        print(f"wrote {out} ({os.path.getsize(out)} bytes)")


if __name__ == "__main__":
    main()
