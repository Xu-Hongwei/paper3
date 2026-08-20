import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


# 高精度别名：只做“显式文本支持”诊断，不尝试推断隐式场景语义。
# key 使用当前 RSICD 文件名解析出的类别字符串。
CATEGORY_ALIASES = {
    "airport": ["airport", "airports"],
    "bareland": ["bare land", "bareland", "barren land"],
    "baseballfield": ["baseball field", "baseball fields", "baseballfield", "baseballfields"],
    "beach": ["beach", "beaches"],
    "bridge": ["bridge", "bridges"],
    "center": ["center", "centre", "central area"],
    "church": ["church", "churches"],
    "commercial": ["commercial area", "commercial areas", "commercial district", "commercial districts"],
    "denseresidential": ["dense residential", "densely residential", "dense residential area", "dense residential areas"],
    "desert": ["desert", "deserts"],
    "farmland": ["farmland", "farmlands", "farm land", "farm lands", "farm", "farms"],
    "forest": ["forest", "forests"],
    "industrial": ["industrial area", "industrial areas", "industrial zone", "industrial zones"],
    "meadow": ["meadow", "meadows"],
    "mediumresidential": ["medium residential", "medium-density residential", "medium density residential"],
    "mountain": ["mountain", "mountains", "mountainous area", "mountainous areas"],
    "park": ["park", "parks"],
    "parkinglot": ["parking lot", "parking lots", "parkinglot", "parkinglots"],
    "playground": ["playground", "playgrounds"],
    "pond": ["pond", "ponds"],
    "port": ["port", "ports", "harbor", "harbour", "harbors", "harbours"],
    "railwaystation": ["railway station", "railway stations", "train station", "train stations"],
    "resort": ["resort", "resorts"],
    "river": ["river", "rivers"],
    "school": ["school", "schools", "campus", "campuses"],
    "sparseresidential": ["sparse residential", "sparsely residential", "sparse residential area", "sparse residential areas"],
    "square": ["square", "squares", "plaza", "plazas"],
    "stadium": ["stadium", "stadiums", "stadia"],
    "storagetanks": ["storage tank", "storage tanks", "storagetank", "storagetanks"],
    "viaduct": ["viaduct", "viaducts", "overpass", "overpasses"],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="统计跨类别 T2I 错误中 GT / Pred 类别是否被 caption 显式支持。"
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="analyze_t2i_category_errors_known_unknown.py 生成的 t2i_all_queries_with_category.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/t2i_text_supported_cross_category",
    )
    parser.add_argument(
        "--dedup",
        action="store_true",
        help="按 (GT image, normalized caption) 去重。建议打开。",
    )
    parser.add_argument(
        "--print-top",
        type=int,
        default=20,
        help="每类四象限最多打印多少个高频 confusion pair。",
    )
    return parser.parse_args()


def normalize_text(text):
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def phrase_present(text, phrase):
    """
    使用 token 边界匹配，避免 park 错匹配 parking。
    """
    text = f" {normalize_text(text)} "
    phrase = normalize_text(phrase)
    if not phrase:
        return False
    pattern = r"(?<![a-z0-9])" + re.escape(phrase) + r"(?![a-z0-9])"
    return re.search(pattern, text) is not None


def fallback_aliases(label):
    """
    对字典之外的类别只做保守 fallback：
    原 label 本身 + 下划线替空格。
    不做语义猜测。
    """
    label = str(label).lower().strip()
    aliases = {label, label.replace("_", " ")}
    return sorted(x for x in aliases if x)


def get_aliases(label):
    label = str(label).lower().strip()
    return CATEGORY_ALIASES.get(label, fallback_aliases(label))


def explicit_support(query, label):
    aliases = get_aliases(label)
    hits = [alias for alias in aliases if phrase_present(query, alias)]
    return bool(hits), hits


def read_cross_category_errors(path):
    rows = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("exact_match") != "0":
                continue
            if row.get("category_resolvable") != "1":
                continue
            if row.get("class_match") != "0":
                continue

            row["text_id"] = int(row["text_id"])
            row["gt_image_id"] = int(row["gt_image_id"])
            row["pred_image_id"] = int(row["pred_image_id"])
            row["gt_score"] = float(row["gt_score"])
            row["pred_score"] = float(row["pred_score"])
            row["pred_minus_gt"] = float(row["pred_minus_gt"])
            rows.append(row)
    return rows


def deduplicate(rows):
    seen = set()
    output = []
    dropped = 0

    for row in rows:
        key = (row["gt_image_id"], normalize_text(row["query"]))
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        output.append(row)

    return output, dropped


def quadrant_name(gt_supported, pred_supported):
    if gt_supported and not pred_supported:
        return "GT_yes_Pred_no"
    if gt_supported and pred_supported:
        return "GT_yes_Pred_yes"
    if not gt_supported and pred_supported:
        return "GT_no_Pred_yes"
    return "GT_no_Pred_no"


def analyze(rows):
    quadrant_counter = Counter()
    confusion_by_quadrant = defaultdict(Counter)
    annotated = []

    for row in rows:
        gt_supported, gt_hits = explicit_support(row["query"], row["gt_label"])
        pred_supported, pred_hits = explicit_support(row["query"], row["pred_label"])
        quadrant = quadrant_name(gt_supported, pred_supported)

        quadrant_counter[quadrant] += 1
        confusion_by_quadrant[quadrant][
            (row["gt_label"], row["pred_label"])
        ] += 1

        enriched = dict(row)
        enriched.update({
            "gt_explicit_supported": int(gt_supported),
            "pred_explicit_supported": int(pred_supported),
            "support_quadrant": quadrant,
            "gt_alias_hits": " | ".join(gt_hits),
            "pred_alias_hits": " | ".join(pred_hits),
        })
        annotated.append(enriched)

    return annotated, quadrant_counter, confusion_by_quadrant


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_quadrant_summary(counter, total):
    order = [
        "GT_yes_Pred_no",
        "GT_yes_Pred_yes",
        "GT_no_Pred_yes",
        "GT_no_Pred_no",
    ]
    print("\nFOUR-QUADRANT EXPLICIT SUPPORT")
    print("-" * 104)
    for name in order:
        count = counter[name]
        print(f"{name:<22} {count:>4} / {total:<4} = {count / max(total, 1):6.2%}")


def main():
    args = parse_args()
    raw_rows = read_cross_category_errors(args.csv)

    if args.dedup:
        rows, dropped = deduplicate(raw_rows)
    else:
        rows, dropped = list(raw_rows), 0

    annotated, quadrant_counter, confusion_by_quadrant = analyze(rows)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    write_csv(output_dir / "cross_category_explicit_support.csv", annotated)

    quadrant_rows = []
    for quadrant in [
        "GT_yes_Pred_no",
        "GT_yes_Pred_yes",
        "GT_no_Pred_yes",
        "GT_no_Pred_no",
    ]:
        for (gt, pred), count in confusion_by_quadrant[quadrant].most_common():
            quadrant_rows.append({
                "quadrant": quadrant,
                "gt_label": gt,
                "pred_label": pred,
                "count": count,
            })
    write_csv(output_dir / "quadrant_confusion_pairs.csv", quadrant_rows)

    total = len(rows)
    summary = {
        "metadata": {
            "input_csv": args.csv,
            "deduplicated": args.dedup,
            "dedup_key": "(gt_image_id, normalized_caption)",
            "raw_cross_category_errors": len(raw_rows),
            "deduplicated_cross_category_errors": total,
            "duplicates_removed": dropped,
            "method": "conservative lexical alias matching only",
            "important_note": (
                "GT_no / Pred_no means not explicitly lexicalized, not semantically absent. "
                "This diagnostic does not infer implicit scene semantics."
            ),
        },
        "quadrants": {
            name: {
                "count": quadrant_counter[name],
                "fraction": quadrant_counter[name] / max(total, 1),
            }
            for name in [
                "GT_yes_Pred_no",
                "GT_yes_Pred_yes",
                "GT_no_Pred_yes",
                "GT_no_Pred_no",
            ]
        },
        "top_confusions_by_quadrant": {
            quadrant: [
                {"gt": gt, "pred": pred, "count": count}
                for (gt, pred), count in confusion_by_quadrant[quadrant].most_common(args.print_top)
            ]
            for quadrant in [
                "GT_yes_Pred_no",
                "GT_yes_Pred_yes",
                "GT_no_Pred_yes",
                "GT_no_Pred_no",
            ]
        },
        "category_aliases": CATEGORY_ALIASES,
    }

    with (output_dir / "explicit_support_summary.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=" * 104)
    print("TEXT-SUPPORTED CROSS-CATEGORY ERROR ANALYSIS")
    print("=" * 104)
    print(f"Input cross-category errors : {len(raw_rows)}")
    print(f"Dedup enabled               : {args.dedup}")
    print(f"Duplicates removed          : {dropped}")
    print(f"Cases analyzed              : {total}")
    print("Support definition           : conservative explicit lexical mention")
    print("=" * 104)

    print_quadrant_summary(quadrant_counter, total)

    print("\nTOP CONFUSIONS BY QUADRANT")
    print("=" * 104)
    for quadrant in [
        "GT_yes_Pred_no",
        "GT_yes_Pred_yes",
        "GT_no_Pred_yes",
        "GT_no_Pred_no",
    ]:
        print(f"\n[{quadrant}]")
        print("-" * 104)
        pairs = confusion_by_quadrant[quadrant].most_common(args.print_top)
        if not pairs:
            print("<none>")
            continue
        for (gt, pred), count in pairs:
            print(f"{gt:<30} -> {pred:<30} {count:>4}")

    print("\nINTERPRETATION")
    print("=" * 104)
    print("GT_yes / Pred_no  : 高价值候选真实错误——caption 显式支持 GT，而未显式支持 Pred。")
    print("GT_yes / Pred_yes : 两种语义都被文本支持——更像 semantic competition / weighting。")
    print("GT_no  / Pred_yes : 高度疑似 caption-label mismatch / secondary-entity dominance。")
    print("GT_no  / Pred_no  : 文本未直接说出两类——更偏隐式场景推理、欠描述或 taxonomy ambiguity。")
    print("注意：这里的 no 仅表示“未显式词法出现”，不能解释为语义不存在。")
    print("-" * 104)
    print(f"Summary : {output_dir / 'explicit_support_summary.json'}")
    print(f"Cases   : {output_dir / 'cross_category_explicit_support.csv'}")
    print(f"Pairs   : {output_dir / 'quadrant_confusion_pairs.csv'}")
    print("=" * 104)


if __name__ == "__main__":
    main()
