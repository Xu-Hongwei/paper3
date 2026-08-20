import argparse
import csv
import html
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


CAUSES = [
    "context_dominance",
    "fine_category_boundary",
    "co_occurrence_confusion",
    "scale_or_saliency",
    "caption_ambiguity",
    "annotation_ambiguity",
    "other",
]


def parse_args():
    parser = argparse.ArgumentParser(description="生成 GT_yes_Pred_no 高价值跨类别错误 Atlas。")
    parser.add_argument("--support-csv", type=str, required=True)
    parser.add_argument("--test-json", type=str, required=True)
    parser.add_argument("--image-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs/t2i_gt_yes_pred_no_atlas")
    parser.add_argument("--top-pairs", type=int, default=20)
    parser.add_argument("--max-per-pair", type=int, default=30)
    parser.add_argument("--thumb-size", type=int, default=420)
    return parser.parse_args()


def esc(x):
    return html.escape(str(x))


def load_cases(path):
    rows = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("support_quadrant") != "GT_yes_Pred_no":
                continue
            row["text_id"] = int(row["text_id"])
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


def captions_of(ann):
    captions = ann.get("caption", [])
    if isinstance(captions, str):
        return [captions]
    return list(captions)


def attach_assets(rows, anns, image_root, output_dir, thumb_size):
    assets_dir = output_dir / "assets"
    cache = {}

    def thumb(image_id):
        if image_id in cache:
            return cache[image_id]
        ann = anns[image_id]
        src = resolve_image_path(image_root, ann["image"])
        dst = assets_dir / f"img_{image_id:04d}.jpg"
        save_thumb(src, dst, thumb_size)
        rel = dst.relative_to(output_dir).as_posix()
        cache[image_id] = rel
        return rel

    for row in rows:
        gt_id = row["gt_image_id"]
        pred_id = row["pred_image_id"]
        row["gt_thumb"] = thumb(gt_id)
        row["pred_thumb"] = thumb(pred_id)
        row["gt_image_ref"] = anns[gt_id]["image"]
        row["pred_image_ref"] = anns[pred_id]["image"]
        row["gt_captions"] = captions_of(anns[gt_id])
        row["pred_captions"] = captions_of(anns[pred_id])


def list_html(items):
    return "<ul>" + "".join(f"<li>{esc(x)}</li>" for x in items) + "</ul>"


def highlight_hits(text, hits):
    safe = esc(text)
    hit_list = [x.strip() for x in str(hits).split("|") if x.strip()]
    for hit in sorted(hit_list, key=len, reverse=True):
        pattern = re.compile(re.escape(hit), flags=re.I)
        safe = pattern.sub(lambda m: f'<mark class="gt-hit">{m.group(0)}</mark>', safe)
    return safe


def case_html(case_id, row):
    query_html = highlight_hits(row["query"], row.get("gt_alias_hits", ""))
    cause_options = "".join(f'<option value="{c}">{c}</option>' for c in CAUSES)

    return f'''
<article class="case"
  data-pair="{esc(row["gt_label"])}|||{esc(row["pred_label"])}"
  data-gt="{esc(row["gt_label"])}"
  data-pred="{esc(row["pred_label"])}"
  data-margin="{row["pred_minus_gt"]:.8f}"
  data-query="{esc(row["query"].lower())}">
  <div class="case-head">
    <div>
      <div class="case-id">CASE {case_id:03d}</div>
      <div class="pair-title">
        <span class="gt-tag">{esc(row["gt_label"])}</span>
        <span class="arrow">→</span>
        <span class="pred-tag">{esc(row["pred_label"])}</span>
      </div>
    </div>
    <div class="score-box">
      <div>Pred {row["pred_score"]:.4f}</div>
      <div>GT {row["gt_score"]:.4f}</div>
      <b>Δ {row["pred_minus_gt"]:+.4f}</b>
    </div>
  </div>

  <div class="support-strip">
    <span class="yes">GT explicit: {esc(row.get("gt_alias_hits", ""))}</span>
    <span class="no">Pred explicit: none</span>
  </div>

  <div class="query">{query_html}</div>

  <div class="compare">
    <section class="image-panel gt-panel">
      <div class="panel-title">Ground Truth · {esc(row["gt_label"])}</div>
      <img src="{esc(row["gt_thumb"])}" loading="lazy">
      <div class="img-ref">{esc(row["gt_image_ref"])}</div>
      <details>
        <summary>GT captions</summary>
        {list_html(row["gt_captions"])}
      </details>
    </section>

    <section class="image-panel pred-panel">
      <div class="panel-title">Wrong Top-1 · {esc(row["pred_label"])}</div>
      <img src="{esc(row["pred_thumb"])}" loading="lazy">
      <div class="img-ref">{esc(row["pred_image_ref"])}</div>
      <details>
        <summary>Pred captions</summary>
        {list_html(row["pred_captions"])}
      </details>
    </section>
  </div>

  <div class="audit">
    <label>人眼确认 Top-1 确实错误？
      <select data-field="human_confirmed">
        <option value=""></option>
        <option value="yes">yes</option>
        <option value="no">no / both plausible</option>
        <option value="uncertain">uncertain</option>
      </select>
    </label>

    <label>GT 词是否足以区分类别？
      <select data-field="gt_cue_discriminative">
        <option value=""></option>
        <option value="yes">yes</option>
        <option value="no">no</option>
        <option value="uncertain">uncertain</option>
      </select>
    </label>

    <label>主要原因
      <select data-field="primary_cause">
        <option value=""></option>
        {cause_options}
      </select>
    </label>

    <label>错误是否像“背景压过目标”？
      <select data-field="context_over_target">
        <option value=""></option>
        <option value="yes">yes</option>
        <option value="no">no</option>
        <option value="uncertain">uncertain</option>
      </select>
    </label>

    <label>是否值得模型纠正？
      <select data-field="worth_fixing">
        <option value=""></option>
        <option value="yes">yes</option>
        <option value="no">no</option>
        <option value="uncertain">uncertain</option>
      </select>
    </label>

    <label>备注
      <textarea data-field="notes" rows="3"></textarea>
    </label>
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
    blocks = []
    case_id = 0

    for rank, ((gt, pred), count) in enumerate(top, start=1):
        pair_key = f"{gt}|||{pred}"
        pair_options.append(
            f'<option value="{esc(pair_key)}">{esc(gt)} → {esc(pred)} ({count})</option>'
        )
        pair_chips.append(
            f'<button class="pair-chip" data-pair="{esc(pair_key)}">'
            f'<span class="rank">#{rank}</span>'
            f'<span class="pair">{esc(gt)} → {esc(pred)}</span>'
            f'<span class="count">{count}</span></button>'
        )

        for row in rows_by_pair[(gt, pred)][:max_per_pair]:
            case_id += 1
            blocks.append(case_html(case_id, row))

    doc = f'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GT-supported Cross-category Errors</title>
<style>
:root {{
  --bg:#f4f6f8; --card:#fff; --line:#dce1e6; --text:#18212a;
  --muted:#6b7785; --gt:#137547; --pred:#c12b1f; --accent:#315efb;
}}
* {{ box-sizing:border-box; }}
body {{
  margin:0; background:var(--bg); color:var(--text);
  font-family:Inter,Segoe UI,Arial,sans-serif;
}}
.topbar {{
  position:sticky; top:0; z-index:10; padding:16px 24px;
  background:rgba(255,255,255,.96); border-bottom:1px solid var(--line);
}}
.topline {{ display:flex; align-items:center; gap:18px; flex-wrap:wrap; }}
h1 {{ margin:0; font-size:22px; }}
.stat {{ color:var(--muted); }}
.controls {{ display:flex; gap:10px; flex-wrap:wrap; margin-top:12px; }}
select,input,textarea,button {{
  border:1px solid var(--line); border-radius:8px; padding:8px 10px; background:#fff;
}}
.controls select,.controls input {{ min-width:220px; }}
.wrapper {{ max-width:1500px; margin:auto; padding:24px; }}
.pair-grid {{
  display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr));
  gap:10px; margin-bottom:24px;
}}
.pair-chip {{
  display:flex; gap:8px; align-items:center; border:1px solid var(--line);
  background:#fff; border-radius:10px; padding:10px 12px; cursor:pointer;
}}
.rank {{ color:var(--muted); font-size:12px; }}
.pair {{ flex:1; font-weight:600; }}
.count {{ background:#eef3ff; color:#214b9a; border-radius:999px; padding:2px 8px; }}
.case {{
  background:var(--card); border:1px solid var(--line); border-radius:14px;
  padding:18px; margin-bottom:22px; box-shadow:0 4px 16px rgba(20,30,40,.05);
}}
.case-head {{ display:flex; justify-content:space-between; gap:20px; }}
.case-id {{ font-size:12px; color:var(--muted); }}
.pair-title {{ display:flex; gap:8px; align-items:center; font-size:22px; font-weight:700; margin-top:5px; }}
.gt-tag {{ color:var(--gt); }}
.pred-tag {{ color:var(--pred); }}
.arrow {{ color:#8a949e; }}
.score-box {{ text-align:right; color:var(--muted); font-size:13px; line-height:1.5; }}
.score-box b {{ color:var(--pred); }}
.support-strip {{ display:flex; gap:8px; margin:12px 0 6px; flex-wrap:wrap; }}
.support-strip span {{ border-radius:999px; padding:5px 9px; font-size:12px; font-weight:600; }}
.support-strip .yes {{ background:#e8f7ef; color:var(--gt); }}
.support-strip .no {{ background:#fdecec; color:var(--pred); }}
.query {{
  margin:10px 0 18px; padding:13px 15px; background:#f7f9ff;
  border-left:4px solid var(--accent); border-radius:7px; font-size:18px; line-height:1.5;
}}
.gt-hit {{ background:#fff0a8; border-radius:3px; padding:0 2px; }}
.compare {{ display:grid; grid-template-columns:1fr 1fr; gap:18px; }}
.image-panel {{ border:1px solid var(--line); border-radius:10px; padding:12px; }}
.panel-title {{ font-weight:700; margin-bottom:10px; }}
.gt-panel .panel-title {{ color:var(--gt); }}
.pred-panel .panel-title {{ color:var(--pred); }}
.image-panel img {{
  width:100%; height:390px; object-fit:contain; background:#edf1f4; border-radius:7px;
}}
.img-ref {{ color:var(--muted); font-size:12px; margin-top:6px; }}
details {{ margin-top:9px; }}
summary {{ cursor:pointer; font-weight:600; }}
li {{ margin:4px 0; }}
.audit {{
  display:grid; grid-template-columns:repeat(3,1fr); gap:10px;
  margin-top:16px; padding-top:14px; border-top:1px solid var(--line);
}}
.audit label {{ font-size:13px; display:flex; flex-direction:column; gap:5px; }}
.audit select,.audit textarea {{ width:100%; min-width:0; }}
.empty {{ display:none; text-align:center; color:var(--muted); padding:40px; }}
@media (max-width:900px) {{
  .compare,.audit {{ grid-template-columns:1fr; }}
  .image-panel img {{ height:300px; }}
}}
</style>
</head>
<body>
<div class="topbar">
  <div class="topline">
    <h1>RSICD · GT Explicitly Supported Errors</h1>
    <div class="stat">GT_yes / Pred_no：<b>{len(rows)}</b></div>
    <div class="stat">只看高价值候选真实错误</div>
  </div>
  <div class="controls">
    <select id="pairFilter">{''.join(pair_options)}</select>
    <input id="searchBox" placeholder="搜索 query / 类别">
    <select id="sortMode">
      <option value="default">默认</option>
      <option value="margin_desc">Δ 从大到小</option>
      <option value="margin_asc">Δ 从小到大</option>
    </select>
    <button onclick="saveAnnotations()">保存标注</button>
    <button onclick="exportCSV()">导出 CSV</button>
  </div>
</div>

<main class="wrapper">
  <h2>最高频 GT-supported confusion pairs</h2>
  <div class="pair-grid">{''.join(pair_chips)}</div>
  <div id="cases">{''.join(blocks)}</div>
  <div id="empty" class="empty">没有匹配样例</div>
</main>

<script>
const fields = [
  "human_confirmed","gt_cue_discriminative","primary_cause",
  "context_over_target","worth_fixing","notes"
];
const cases = document.getElementById("cases");
const pairFilter = document.getElementById("pairFilter");
const searchBox = document.getElementById("searchBox");
const sortMode = document.getElementById("sortMode");
const empty = document.getElementById("empty");

function applyFilter() {{
  const pair = pairFilter.value;
  const q = searchBox.value.trim().toLowerCase();
  let visible = [];

  [...cases.children].forEach(card => {{
    const pairOk = pair === "all" || card.dataset.pair === pair;
    const text = (card.dataset.query + " " + card.dataset.gt + " " + card.dataset.pred).toLowerCase();
    const show = pairOk && (!q || text.includes(q));
    card.style.display = show ? "" : "none";
    if (show) visible.push(card);
  }});

  if (sortMode.value !== "default") {{
    visible.sort((a,b) => {{
      const da = parseFloat(a.dataset.margin), db = parseFloat(b.dataset.margin);
      return sortMode.value === "margin_desc" ? db-da : da-db;
    }});
    visible.forEach(card => cases.appendChild(card));
  }}
  empty.style.display = visible.length ? "none" : "block";
}}

function saveAnnotations() {{
  const data = {{}};
  [...cases.children].forEach((card, i) => {{
    data[i+1] = {{}};
    fields.forEach(field => {{
      const el = card.querySelector(`[data-field="${{field}}"]`);
      data[i+1][field] = el ? el.value : "";
    }});
  }});
  localStorage.setItem("gt_yes_pred_no_annotations", JSON.stringify(data));
  alert("已保存到浏览器 localStorage");
}}

function loadAnnotations() {{
  const raw = localStorage.getItem("gt_yes_pred_no_annotations");
  if (!raw) return;
  const data = JSON.parse(raw);
  [...cases.children].forEach((card, i) => {{
    const item = data[i+1] || {{}};
    fields.forEach(field => {{
      const el = card.querySelector(`[data-field="${{field}}"]`);
      if (el && item[field] !== undefined) el.value = item[field];
    }});
  }});
}}

function csvEscape(v) {{
  const s = String(v ?? "");
  return '"' + s.replaceAll('"','""') + '"';
}}

function exportCSV() {{
  saveAnnotations();
  const raw = JSON.parse(localStorage.getItem("gt_yes_pred_no_annotations") || "{{}}");
  const rows = [["case_id","gt","pred","query","margin",...fields]];
  [...cases.children].forEach((card, i) => {{
    const item = raw[i+1] || {{}};
    rows.push([
      i+1, card.dataset.gt, card.dataset.pred, card.dataset.query,
      card.dataset.margin, ...fields.map(f => item[f] || "")
    ]);
  }});
  const csv = rows.map(r => r.map(csvEscape).join(",")).join("\\n");
  const blob = new Blob(["\\ufeff"+csv], {{type:"text/csv;charset=utf-8"}});
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "gt_yes_pred_no_manual_annotations.csv";
  a.click();
  URL.revokeObjectURL(url);
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

loadAnnotations();
</script>
</body>
</html>
'''
    output_path.write_text(doc, encoding="utf-8")


def main():
    args = parse_args()
    rows = load_cases(args.support_csv)
    anns = load_test_json(args.test_json)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    attach_assets(rows, anns, args.image_root, output_dir, args.thumb_size)
    build_html(rows, output_dir / "index.html", args.top_pairs, args.max_per_pair)

    pair_counter = Counter((x["gt_label"], x["pred_label"]) for x in rows)
    summary = {
        "num_gt_yes_pred_no": len(rows),
        "num_confusion_pairs": len(pair_counter),
        "top_pairs": [
            {"gt": gt, "pred": pred, "count": count}
            for (gt, pred), count in pair_counter.most_common(args.top_pairs)
        ],
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=" * 96)
    print("GT_yes / Pred_no ERROR ATLAS READY")
    print("=" * 96)
    print(f"Cases           : {len(rows)}")
    print(f"Confusion pairs : {len(pair_counter)}")
    print(f"HTML            : {output_dir / 'index.html'}")
    print(f"Summary         : {output_dir / 'summary.json'}")
    print("=" * 96)


if __name__ == "__main__":
    main()
