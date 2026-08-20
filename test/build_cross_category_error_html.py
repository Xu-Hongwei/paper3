import argparse
import csv
import html
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="生成 RSICD T2I 跨类别错误可视化 HTML。")
    parser.add_argument("--csv", type=str, required=True, help="t2i_all_queries_with_category.csv")
    parser.add_argument("--test-json", type=str, required=True, help="rsicd_test.json")
    parser.add_argument("--image-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs/t2i_cross_category_html")
    parser.add_argument("--top-pairs", type=int, default=20)
    parser.add_argument("--max-per-pair", type=int, default=30)
    parser.add_argument("--thumb-size", type=int, default=420)
    return parser.parse_args()


def read_cross_category_rows(path):
    rows = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("exact_match") != "0":
                continue
            if row.get("category_resolvable") != "1":
                continue
            if row.get("class_match") != "0":
                continue

            row["gt_image_id"] = int(row["gt_image_id"])
            row["pred_image_id"] = int(row["pred_image_id"])
            row["gt_score"] = float(row["gt_score"])
            row["pred_score"] = float(row["pred_score"])
            row["pred_minus_gt"] = float(row["pred_minus_gt"])
            rows.append(row)
    return rows


def load_test_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("rsicd_test.json 顶层应为 list。")
    return data


def resolve_image_path(image_root, image_ref):
    root = Path(image_root)
    direct = root / image_ref
    if direct.is_file():
        return direct

    flat = root / Path(image_ref).name
    if flat.is_file():
        return flat

    raise FileNotFoundError(f"找不到图像: {image_ref}")


def save_thumb(src, dst, max_size):
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src).convert("RGB") as image:
        image.thumbnail((max_size, max_size))
        image.save(dst, quality=92)


def get_captions(ann):
    captions = ann.get("caption", [])
    if isinstance(captions, str):
        return [captions]
    return list(captions)


def esc(value):
    return html.escape(str(value))


def captions_html(captions):
    return "<ul>" + "".join(f"<li>{esc(x)}</li>" for x in captions) + "</ul>"


def attach_assets(rows, anns, image_root, output_dir, thumb_size):
    assets_dir = output_dir / "assets"
    thumb_cache = {}

    def get_thumb(image_id):
        if image_id in thumb_cache:
            return thumb_cache[image_id]

        ann = anns[image_id]
        src = resolve_image_path(image_root, ann["image"])
        dst = assets_dir / f"img_{image_id:04d}.jpg"
        save_thumb(src, dst, thumb_size)

        rel = dst.relative_to(output_dir).as_posix()
        thumb_cache[image_id] = rel
        return rel

    for row in rows:
        gt_id = row["gt_image_id"]
        pred_id = row["pred_image_id"]

        row["gt_thumb"] = get_thumb(gt_id)
        row["pred_thumb"] = get_thumb(pred_id)
        row["gt_image_ref"] = anns[gt_id]["image"]
        row["pred_image_ref"] = anns[pred_id]["image"]
        row["gt_captions"] = get_captions(anns[gt_id])
        row["pred_captions"] = get_captions(anns[pred_id])


def make_case(case_id, row):
    gt = row["gt_label"]
    pred = row["pred_label"]

    return f'''
    <article class="case"
        data-pair="{esc(gt)}|||{esc(pred)}"
        data-gt="{esc(gt)}"
        data-pred="{esc(pred)}"
        data-query="{esc(row["query"].lower())}"
        data-margin="{row["pred_minus_gt"]:.8f}">
      <div class="case-head">
        <div>
          <div class="case-id">CASE {case_id:03d}</div>
          <div class="pair-title">
            <span class="gt-tag">{esc(gt)}</span>
            <span class="arrow">→</span>
            <span class="pred-tag">{esc(pred)}</span>
          </div>
        </div>
        <div class="score-box">
          <div>Pred {row["pred_score"]:.4f}</div>
          <div>GT {row["gt_score"]:.4f}</div>
          <b>Δ {row["pred_minus_gt"]:+.4f}</b>
        </div>
      </div>

      <div class="query">{esc(row["query"])}</div>

      <div class="compare">
        <section class="image-panel gt-panel">
          <div class="panel-title">Ground Truth · {esc(gt)}</div>
          <img src="{esc(row["gt_thumb"])}" loading="lazy">
          <div class="img-ref">{esc(row["gt_image_ref"])}</div>
          <details>
            <summary>GT captions</summary>
            {captions_html(row["gt_captions"])}
          </details>
        </section>

        <section class="image-panel pred-panel">
          <div class="panel-title">Wrong Top-1 · {esc(pred)}</div>
          <img src="{esc(row["pred_thumb"])}" loading="lazy">
          <div class="img-ref">{esc(row["pred_image_ref"])}</div>
          <details>
            <summary>Pred captions</summary>
            {captions_html(row["pred_captions"])}
          </details>
        </section>
      </div>

      <div class="audit">
        <label>人眼是否认为类别确实错？
          <select>
            <option></option>
            <option>是</option>
            <option>否，两类都合理</option>
            <option>不确定</option>
          </select>
        </label>

        <label>文本是否有明确纠错线索？
          <select>
            <option></option>
            <option>有</option>
            <option>没有</option>
            <option>不确定</option>
          </select>
        </label>

        <label>主要原因
          <select>
            <option></option>
            <option>主体被背景压制</option>
            <option>类别边界相近</option>
            <option>上下文共现</option>
            <option>文本欠描述</option>
            <option>数据集标签歧义</option>
            <option>其他</option>
          </select>
        </label>

        <textarea placeholder="备注：为什么你认为会错？"></textarea>
      </div>
    </article>
    '''


def build_html(rows, output_path, top_pairs, max_per_pair):
    pair_counter = Counter((x["gt_label"], x["pred_label"]) for x in rows)
    top = pair_counter.most_common(top_pairs)

    rows_by_pair = defaultdict(list)
    for row in rows:
        rows_by_pair[(row["gt_label"], row["pred_label"])].append(row)

    for key in rows_by_pair:
        rows_by_pair[key].sort(key=lambda x: x["pred_minus_gt"], reverse=True)

    pair_options = ['<option value="all">全部混淆对</option>']
    pair_chips = []

    for rank, ((gt, pred), count) in enumerate(top, start=1):
        pair_key = f"{gt}|||{pred}"
        pair_options.append(
            f'<option value="{esc(pair_key)}">{esc(gt)} → {esc(pred)} ({count})</option>'
        )
        pair_chips.append(
            f'''
            <button class="pair-chip" data-pair="{esc(pair_key)}">
              <span class="rank">#{rank}</span>
              <span class="pair">{esc(gt)} → {esc(pred)}</span>
              <span class="count">{count}</span>
            </button>
            '''
        )

    case_blocks = []
    case_id = 0
    for (gt, pred), _ in top:
        for row in rows_by_pair[(gt, pred)][:max_per_pair]:
            case_id += 1
            case_blocks.append(make_case(case_id, row))

    html_doc = f'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RSICD Cross-Category Error Atlas</title>
<style>
:root {{
  --bg:#f4f6f8;
  --card:#ffffff;
  --line:#dce1e6;
  --text:#17212b;
  --muted:#6b7785;
  --gt:#146c43;
  --pred:#b42318;
  --accent:#1f5eff;
}}
* {{ box-sizing:border-box; }}
body {{
  margin:0;
  font-family:Inter,Segoe UI,Arial,sans-serif;
  background:var(--bg);
  color:var(--text);
}}
.topbar {{
  position:sticky;
  top:0;
  z-index:20;
  background:rgba(255,255,255,.96);
  border-bottom:1px solid var(--line);
  padding:16px 24px;
  backdrop-filter:blur(8px);
}}
.topline {{ display:flex; align-items:center; gap:18px; flex-wrap:wrap; }}
h1 {{ font-size:22px; margin:0; }}
.stat {{ color:var(--muted); }}
.controls {{ display:flex; gap:10px; margin-top:12px; flex-wrap:wrap; }}
select,input {{
  padding:9px 11px;
  border:1px solid var(--line);
  border-radius:8px;
  background:white;
  min-width:220px;
}}
.wrapper {{ max-width:1500px; margin:auto; padding:24px; }}
.pair-grid {{
  display:grid;
  grid-template-columns:repeat(auto-fit,minmax(240px,1fr));
  gap:10px;
  margin-bottom:24px;
}}
.pair-chip {{
  display:flex;
  align-items:center;
  gap:8px;
  border:1px solid var(--line);
  background:white;
  padding:11px 12px;
  border-radius:10px;
  cursor:pointer;
  text-align:left;
}}
.pair-chip:hover {{ border-color:var(--accent); }}
.rank {{ color:var(--muted); font-size:12px; }}
.pair {{ flex:1; font-weight:600; }}
.count {{ background:#eef3ff; color:#1746a2; border-radius:999px; padding:2px 8px; }}
.case {{
  background:var(--card);
  border:1px solid var(--line);
  border-radius:14px;
  margin:0 0 22px;
  padding:18px;
  box-shadow:0 4px 18px rgba(20,30,40,.05);
}}
.case-head {{ display:flex; justify-content:space-between; gap:20px; align-items:flex-start; }}
.case-id {{ font-size:12px; color:var(--muted); margin-bottom:7px; }}
.pair-title {{ font-size:22px; font-weight:700; display:flex; align-items:center; gap:9px; }}
.gt-tag {{ color:var(--gt); }}
.pred-tag {{ color:var(--pred); }}
.arrow {{ color:#8b949e; }}
.score-box {{ font-size:13px; text-align:right; line-height:1.5; color:var(--muted); }}
.score-box b {{ color:var(--pred); }}
.query {{
  margin:15px 0 18px;
  padding:13px 15px;
  border-left:4px solid var(--accent);
  background:#f7f9ff;
  border-radius:7px;
  font-size:18px;
  line-height:1.5;
}}
.compare {{ display:grid; grid-template-columns:1fr 1fr; gap:18px; }}
.image-panel {{ border:1px solid var(--line); border-radius:10px; padding:12px; }}
.panel-title {{ font-weight:700; margin-bottom:10px; }}
.gt-panel .panel-title {{ color:var(--gt); }}
.pred-panel .panel-title {{ color:var(--pred); }}
.image-panel img {{
  width:100%;
  height:390px;
  object-fit:contain;
  background:#eef1f4;
  border-radius:7px;
  display:block;
}}
.img-ref {{ color:var(--muted); font-size:12px; margin-top:7px; }}
details {{ margin-top:10px; }}
summary {{ cursor:pointer; font-weight:600; }}
li {{ margin:5px 0; line-height:1.35; }}
.audit {{
  display:grid;
  grid-template-columns:repeat(3,1fr);
  gap:10px;
  margin-top:16px;
  border-top:1px solid var(--line);
  padding-top:14px;
}}
.audit label {{ font-size:13px; display:flex; flex-direction:column; gap:5px; }}
.audit select {{ min-width:0; width:100%; }}
.audit textarea {{
  grid-column:1/-1;
  min-height:70px;
  border:1px solid var(--line);
  border-radius:8px;
  padding:9px;
}}
.empty {{ display:none; text-align:center; color:var(--muted); padding:50px; }}
@media (max-width:900px) {{
  .compare,.audit {{ grid-template-columns:1fr; }}
  .image-panel img {{ height:300px; }}
}}
</style>
</head>
<body>
<div class="topbar">
  <div class="topline">
    <h1>RSICD · Cross-Category Error Atlas</h1>
    <div class="stat">跨类别错误总数：<b>{len(rows)}</b></div>
    <div class="stat">展示 Top-{top_pairs} confusion pairs</div>
  </div>

  <div class="controls">
    <select id="pairFilter">{''.join(pair_options)}</select>
    <input id="searchBox" placeholder="搜索 query / 类别">
    <select id="sortMode">
      <option value="default">默认顺序</option>
      <option value="margin_desc">按 Δ 从大到小</option>
      <option value="margin_asc">按 Δ 从小到大</option>
    </select>
  </div>
</div>

<main class="wrapper">
  <h2>最高频混淆对</h2>
  <div class="pair-grid">{''.join(pair_chips)}</div>

  <div id="cases">{''.join(case_blocks)}</div>
  <div id="empty" class="empty">没有匹配的样例</div>
</main>

<script>
const pairFilter = document.getElementById("pairFilter");
const searchBox = document.getElementById("searchBox");
const sortMode = document.getElementById("sortMode");
const cases = document.getElementById("cases");
const empty = document.getElementById("empty");

function applyFilter() {{
  const pair = pairFilter.value;
  const q = searchBox.value.trim().toLowerCase();
  let visible = [];

  [...cases.children].forEach(card => {{
    const pairOk = pair === "all" || card.dataset.pair === pair;
    const text = (card.dataset.query + " " + card.dataset.gt + " " + card.dataset.pred).toLowerCase();
    const queryOk = !q || text.includes(q);
    const show = pairOk && queryOk;

    card.style.display = show ? "" : "none";
    if (show) visible.push(card);
  }});

  if (sortMode.value !== "default") {{
    visible.sort((a,b) => {{
      const da = parseFloat(a.dataset.margin);
      const db = parseFloat(b.dataset.margin);
      return sortMode.value === "margin_desc" ? db-da : da-db;
    }});
    visible.forEach(card => cases.appendChild(card));
  }}

  empty.style.display = visible.length ? "none" : "block";
}}

pairFilter.addEventListener("change", applyFilter);
searchBox.addEventListener("input", applyFilter);
sortMode.addEventListener("change", applyFilter);

document.querySelectorAll(".pair-chip").forEach(btn => {{
  btn.addEventListener("click", () => {{
    pairFilter.value = btn.dataset.pair;
    applyFilter();
    window.scrollTo({{top:0, behavior:"smooth"}});
  }});
}});
</script>
</body>
</html>
'''

    output_path.write_text(html_doc, encoding="utf-8")


def main():
    args = parse_args()

    rows = read_cross_category_rows(args.csv)
    anns = load_test_json(args.test_json)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    attach_assets(rows, anns, args.image_root, output_dir, args.thumb_size)
    build_html(rows, output_dir / "index.html", args.top_pairs, args.max_per_pair)

    pair_counter = Counter((x["gt_label"], x["pred_label"]) for x in rows)
    with (output_dir / "cross_category_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "num_cross_category_errors": len(rows),
                "num_confusion_pairs": len(pair_counter),
                "top_pairs": [
                    {"gt": gt, "pred": pred, "count": count}
                    for (gt, pred), count in pair_counter.most_common(args.top_pairs)
                ],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("=" * 92)
    print("CROSS-CATEGORY ERROR HTML READY")
    print("=" * 92)
    print(f"Cross-category errors : {len(rows)}")
    print(f"Confusion pairs       : {len(pair_counter)}")
    print(f"HTML                  : {output_dir / 'index.html'}")
    print(f"Assets                : {output_dir / 'assets'}")
    print("=" * 92)


if __name__ == "__main__":
    main()
