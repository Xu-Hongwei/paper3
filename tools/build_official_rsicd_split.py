"""Build official-split RSICD annotation files for the paper3 pipeline.

Source of truth: the official RSICD annotation file (authors' dataset.json),
which embeds the official image-level split:
    train 8,734 / val 1,094 / test 1,093  (10,921 images total).

Correct-split principles enforced here:
  1. Image-level split: all 5 captions of an image stay in the same split.
  2. Exclusive and complete: every image belongs to exactly one split.
  3. Official split used verbatim (no re-randomization) for comparability.
  4. Deterministic + verifiable: image references resolve against the flat
     images folder; captions stored verbatim (= dataset.json "raw").
  5. Degenerate captions (those that fail pre_caption, e.g. "." or " .")
     are filtered out at generation time, using the exact same rule as the
     dataset loading layer (datasets.re_dataset). The loading layer keeps
     its own guard as a safety net.

Outputs (paper3 pipeline formats, same as the existing files):
  rsicd_train_official.json : list of {image, caption, image_id, label_name, label}
  rsicd_val_official.json   : list of {image, caption: [<=5]}
  rsicd_test_official.json  : list of {image, caption: [<=5]}
"""
import json
import os
import sys

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)

from datasets.utils import pre_caption

SRC = r"E:\datasets\remote_sensing_captioning\RSICD\dataset.json"
IMG_ROOT = r"E:\datasets\remote_sensing_captioning\RSICD\images"
OUT_DIR = r"E:\paper3\data\rsicd"

with open(SRC, encoding="utf-8") as f:
    data = json.load(f)

images = data["images"]
print(f"total images: {len(images)}")

# --- 1. integrity checks on the official source -----------------------------
splits = {}
for img in images:
    s = img["split"]
    splits[s] = splits.get(s, 0) + 1
    assert len(img["sentences"]) == 5, (
        f"image {img['filename']} has {len(img['sentences'])} sentences"
    )
print(f"official split counts: {splits}")
assert sum(splits.values()) == len(images)
assert set(splits) == {"train", "val", "test"}

# --- 2. category bookkeeping (31 categories from filename prefixes) ---------
def category_of(filename):
    base = os.path.basename(filename)
    if base[:1].isdigit():
        return None  # numeric-named images (original RSICD test pool)
    return base.split("_")[0]

categories = sorted({
    category_of(i["filename"])
    for i in images
    if category_of(i["filename"]) is not None
})
print(f"categories: {len(categories)}")
cat_idx = {c: i for i, c in enumerate(categories)}

# --- degenerate-caption filter (same rule as dataset loading layer) ---------
MAX_WORDS = 30


def is_valid_caption(raw: str) -> bool:
    try:
        pre_caption(raw, MAX_WORDS)
        return True
    except ValueError:
        return False


# --- 3. build outputs -------------------------------------------------------
train, val, test = [], [], []
skipped = {"train": 0, "val": 0, "test": 0}
empty_images = {"train": [], "val": [], "test": []}
for img in images:
    raws = [s["raw"] for s in img["sentences"]]
    base = os.path.basename(img["filename"])
    cat = category_of(img["filename"])
    split = img["split"]

    valid_raws = [r for r in raws if is_valid_caption(r)]
    skipped[split] += len(raws) - len(valid_raws)
    if not valid_raws:
        empty_images[split].append(base)

    if split == "train":
        for raw in valid_raws:
            train.append({
                "image": f"train/{base}",
                "caption": raw,
                "image_id": img["imgid"],
                "label_name": cat if cat is not None else base,
                "label": cat_idx[cat] if cat is not None else -1,
            })
    else:
        entry = {"image": f"{split}/{base}", "caption": valid_raws}
        (val if split == "val" else test).append(entry)

print(f"skipped degenerate captions: {skipped}")
for split_name in ("train", "val", "test"):
    if empty_images[split_name]:
        print(
            f"WARNING: images with 0 valid captions in {split_name}: "
            f"{empty_images[split_name]}"
        )
        if split_name != "train":
            # an eval image with no captions would break the eval dataset
            raise RuntimeError(
                f"{split_name} image(s) with no valid captions: "
                f"{empty_images[split_name]}"
            )

print(
    f"train entries: {len(train)} "
    f"(unique images: {len({e['image'] for e in train})})"
)
print(f"val images: {len(val)}")
print(f"test images: {len(test)}")

# --- 4. verification --------------------------------------------------------
def missing_images(entries):
    missing = []
    for e in entries:
        base = os.path.basename(e["image"])
        if not os.path.isfile(os.path.join(IMG_ROOT, base)):
            missing.append(e["image"])
    return missing

for name, entries in (("train", train), ("val", val), ("test", test)):
    missing = missing_images(entries)
    print(f"{name}: unresolvable image files = {len(missing)}")
    if missing[:5]:
        print("  e.g.", missing[:5])

def base_set(entries):
    return {os.path.basename(e["image"]) for e in entries}

st, sv, ste = base_set(train), base_set(val), base_set(test)
print(
    f"overlap train∩val: {len(st & sv)}, "
    f"train∩test: {len(st & ste)}, val∩test: {len(sv & ste)}"
)
assert not (st & sv) and not (st & ste) and not (sv & ste)

# no degenerate captions may remain in the outputs
def scan_bad(entries):
    bad = 0
    for e in entries:
        caps = e["caption"] if isinstance(e["caption"], list) else [e["caption"]]
        for c in caps:
            if not is_valid_caption(c):
                bad += 1
    return bad

for name, entries in (("train", train), ("val", val), ("test", test)):
    bad = scan_bad(entries)
    print(f"{name}: degenerate captions remaining = {bad}")
    assert bad == 0

# test content equivalence with the previous test file
old_test_path = os.path.join(OUT_DIR, "rsicd_test.json")
if os.path.exists(old_test_path):
    with open(old_test_path, encoding="utf-8") as f:
        old_test = json.load(f)
    old_map = {os.path.basename(e["image"]): e["caption"] for e in old_test}
    new_map = {os.path.basename(e["image"]): e["caption"] for e in test}
    same = sum(
        1 for b in old_map if b in new_map and old_map[b] == new_map[b]
    )
    print(
        f"test image-set identical to previous file: "
        f"{set(old_map) == set(new_map)}; "
        f"identical caption lists: {same}/{len(old_map)}"
    )

# --- 5. write outputs -------------------------------------------------------
os.makedirs(OUT_DIR, exist_ok=True)
for name, obj in (
    ("rsicd_train_official.json", train),
    ("rsicd_val_official.json", val),
    ("rsicd_test_official.json", test),
):
    out = os.path.join(OUT_DIR, name)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    print(f"wrote {out} ({os.path.getsize(out)} bytes)")
