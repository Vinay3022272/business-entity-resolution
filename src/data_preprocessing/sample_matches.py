import os
import sys
import csv
import json
import random
import pandas as pd
from typing import List, Dict, Any, Optional

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

def sample_and_fetch_matches(
    dataset_dir: str = "dataset/train",
    n_samples: int = 500,
    random_state: int = 42,
    save_html_path: Optional[str] = "sample_500_matches.html",
    save_json_path: Optional[str] = "sample_500_matches.json"
) -> List[Dict[str, Any]]:
    """
    Randomly samples `n_samples` Source 1 records that have matches in ground truth,
    extracts the corresponding records from Source 1, Source 2, and Source 3,
    and returns a structured comparison list.
    """
    gt_path = os.path.join(dataset_dir, "train_ground_truth.tsv")
    s1_path = os.path.join(dataset_dir, "train_source1.tsv")
    s2_path = os.path.join(dataset_dir, "train_source2.tsv")
    s3_path = os.path.join(dataset_dir, "train_source3.tsv")

    print(f"[*] Step 1/4: Reading ground truth from {gt_path}...")
    gt_df = pd.read_csv(gt_path, sep="\t", quoting=csv.QUOTE_NONE, dtype=str)
    
    # Filter for entities with matching records
    gt_with_matches = gt_df[gt_df["matched_entity_ids"].notna() & (gt_df["matched_entity_ids"] != "")].copy()
    sample_gt = gt_with_matches.sample(n=min(n_samples, len(gt_with_matches)), random_state=random_state)
    
    s1_target_ids = set(sample_gt["source1_entity_id"])
    s2_target_ids = set()
    s3_target_ids = set()
    
    s1_to_matches = {}
    for _, row in sample_gt.iterrows():
        s1_id = row["source1_entity_id"]
        matched_raw = str(row["matched_entity_ids"]).strip()
        matched_list = [x.strip() for x in matched_raw.split(",") if x.strip()]
        s1_to_matches[s1_id] = matched_list
        for m_id in matched_list:
            if m_id.startswith("S2-"):
                s2_target_ids.add(m_id)
            elif m_id.startswith("S3-"):
                s3_target_ids.add(m_id)

    print(f"    Sampled {len(s1_target_ids)} S1 records -> {len(s2_target_ids)} S2 records, {len(s3_target_ids)} S3 records")

    # Fast single-pass file extractor (streaming)
    def fetch_details(tsv_path: str, target_ids: set) -> Dict[str, Dict[str, str]]:
        records = {}
        if not target_ids or not os.path.exists(tsv_path):
            return records
        with open(tsv_path, "r", encoding="utf-8", errors="ignore") as f:
            next(f, None)  # Skip header
            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                if parts and parts[0] in target_ids:
                    records[parts[0]] = {
                        "entity_id": parts[0],
                        "business_name": parts[1] if len(parts) > 1 else "",
                        "business_address": parts[2] if len(parts) > 2 else "",
                        "country": parts[3] if len(parts) > 3 else ""
                    }
                    if len(records) == len(target_ids):
                        break
        return records

    print(f"[*] Step 2/4: Extracting S1 records...")
    s1_details = fetch_details(s1_path, s1_target_ids)

    print(f"[*] Step 3/4: Extracting S2 & S3 matching records...")
    s2_details = fetch_details(s2_path, s2_target_ids)
    s3_details = fetch_details(s3_path, s3_target_ids)

    # Combine into rich comparison objects
    results = []
    for s1_id, match_ids in s1_to_matches.items():
        s1_rec = s1_details.get(s1_id, {"entity_id": s1_id, "business_name": "N/A", "business_address": "N/A", "country": "N/A"})
        
        s2_matches = []
        s3_matches = []
        for m_id in match_ids:
            if m_id.startswith("S2-") and m_id in s2_details:
                s2_matches.append(s2_details[m_id])
            elif m_id.startswith("S3-") and m_id in s3_details:
                s3_matches.append(s3_details[m_id])

        results.append({
            "source1": s1_rec,
            "source2_matches": s2_matches,
            "source3_matches": s3_matches,
            "total_matches": len(s2_matches) + len(s3_matches)
        })

    print(f"[*] Step 4/4: Formatting results...")

    # Optional JSON save
    if save_json_path:
        with open(save_json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"    Saved JSON to {save_json_path}")

    # Optional interactive HTML dashboard save
    if save_html_path:
        generate_html_report(results, save_html_path)
        print(f"    Saved interactive HTML report to {save_html_path}")

    return results

def generate_html_report(results: List[Dict[str, Any]], html_path: str):
    """Generates an aesthetic HTML page to easily browse and search sampled matches."""
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Entity Resolution Sample Matches (500 Samples)</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 24px; }
  .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 24px; border-bottom: 1px solid #334155; padding-bottom: 16px; }
  h1 { font-size: 24px; color: #38bdf8; margin: 0; }
  .search-box { padding: 10px 16px; width: 320px; border-radius: 8px; border: 1px solid #475569; background: #1e293b; color: #fff; font-size: 14px; }
  .card { background: #1e293b; border-radius: 12px; border: 1px solid #334155; margin-bottom: 20px; padding: 18px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.3); }
  .s1-header { background: #0284c7; color: white; padding: 6px 12px; border-radius: 6px; font-weight: 600; display: inline-block; font-size: 13px; }
  .badge-country { background: #475569; color: #e2e8f0; padding: 2px 8px; border-radius: 4px; font-size: 12px; margin-left: 8px; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 14px; }
  .match-section { background: #0f172a; border-radius: 8px; padding: 12px; border: 1px solid #334155; }
  .match-title { font-weight: 600; font-size: 13px; margin-bottom: 8px; }
  .badge-s2 { color: #4ade80; }
  .badge-s3 { color: #f472b6; }
  .match-item { background: #1e293b; border-left: 3px solid #64748b; padding: 8px 12px; margin-bottom: 8px; border-radius: 4px; font-size: 13px; }
  .name { font-weight: 600; color: #f1f5f9; font-size: 14px; }
  .addr { color: #94a3b8; font-size: 12px; margin-top: 4px; }
  .id { font-family: monospace; font-size: 11px; color: #64748b; }
</style>
</head>
<body>
<div class="header">
  <div>
    <h1>🔍 Entity Resolution: Sample Matching Pairs</h1>
    <div style="color: #94a3b8; font-size: 13px; margin-top: 4px;">Showing 500 sampled Source 1 entities and corresponding ground truth matches from Source 2 & Source 3</div>
  </div>
  <input type="text" id="search" class="search-box" placeholder="Search business name or ID..." onkeyup="filterCards()">
</div>
<div id="container">
"""
    for item in results:
        s1 = item["source1"]
        s1_name = str(s1.get("business_name", "")).replace("<", "&lt;").replace(">", "&gt;")
        s1_addr = str(s1.get("business_address", "")).replace("<", "&lt;").replace(">", "&gt;")
        search_str = f"{s1_name} {s1.get('entity_id', '')}".lower().replace('"', '&quot;')
        
        html_content += f"""
<div class="card" data-search="{search_str}">
  <div style="display: flex; justify-content: space-between; align-items: center;">
    <div>
      <span class="s1-header">REFERENCE S1: {s1.get('entity_id')}</span>
      <span class="badge-country">{s1.get('country', '')}</span>
    </div>
    <span style="font-size: 12px; color: #94a3b8;">Total Matches: <b>{item['total_matches']}</b></span>
  </div>
  <div style="margin-top: 8px;">
    <div class="name" style="font-size: 16px; color: #38bdf8;">{s1_name}</div>
    <div class="addr" style="font-size: 13px; color: #cbd5e1;">📍 {s1_addr}</div>
  </div>
  <div class="grid">
    <div class="match-section">
      <div class="match-title badge-s2">📦 Source 2 Matches ({len(item['source2_matches'])})</div>
"""
        if not item["source2_matches"]:
            html_content += '<div style="color: #64748b; font-size: 12px;">(No matches)</div>'
        else:
            for s2 in item["source2_matches"]:
                name = str(s2.get("business_name", "")).replace("<", "&lt;").replace(">", "&gt;")
                addr = str(s2.get("business_address", "")).replace("<", "&lt;").replace(">", "&gt;")
                html_content += f"""
      <div class="match-item" style="border-left-color: #4ade80;">
        <div class="id">{s2.get('entity_id')} [{s2.get('country', '')}]</div>
        <div class="name">{name}</div>
        <div class="addr">📍 {addr}</div>
      </div>"""

        html_content += f"""
    </div>
    <div class="match-section">
      <div class="match-title badge-s3">🏬 Source 3 Matches ({len(item['source3_matches'])})</div>
"""
        if not item["source3_matches"]:
            html_content += '<div style="color: #64748b; font-size: 12px;">(No matches)</div>'
        else:
            for s3 in item["source3_matches"]:
                name = str(s3.get("business_name", "")).replace("<", "&lt;").replace(">", "&gt;")
                addr = str(s3.get("business_address", "")).replace("<", "&lt;").replace(">", "&gt;")
                html_content += f"""
      <div class="match-item" style="border-left-color: #f472b6;">
        <div class="id">{s3.get('entity_id')} [{s3.get('country', '')}]</div>
        <div class="name">{name}</div>
        <div class="addr">📍 {addr}</div>
      </div>"""

        html_content += """
    </div>
  </div>
</div>
"""
    html_content += """
</div>
<script>
function filterCards() {
  const query = document.getElementById('search').value.toLowerCase();
  const cards = document.querySelectorAll('.card');
  cards.forEach(c => {
    const text = c.getAttribute('data-search');
    c.style.display = text.includes(query) ? 'block' : 'none';
  });
}
</script>
</body>
</html>
"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

def print_readable_samples(results: List[Dict[str, Any]], num_to_display: int = 5):
    """Prints formatted cards in the terminal or Jupyter notebook console."""
    sep = "=" * 80
    sub_sep = "-" * 80
    for idx, item in enumerate(results[:num_to_display], 1):
        s1 = item["source1"]
        print(f"\n{sep}")
        print(f"[{idx}/{len(results)}] REFERENCE SOURCE 1: {s1.get('entity_id')} ({s1.get('country', '')})")
        print(f" Name    : {s1.get('business_name', '')}")
        print(f" Address : {s1.get('business_address', '')}")
        print(f"{sub_sep}")
        
        # Source 2
        s2_list = item["source2_matches"]
        print(f" -> SOURCE 2 MATCHES ({len(s2_list)}):")
        if not s2_list:
            print("    (None)")
        for s2 in s2_list:
            print(f"    * [{s2.get('entity_id')}] {s2.get('business_name', '')}")
            print(f"      Address: {s2.get('business_address', '')}")

        # Source 3
        s3_list = item["source3_matches"]
        print(f" -> SOURCE 3 MATCHES ({len(s3_list)}):")
        if not s3_list:
            print("    (None)")
        for s3 in s3_list:
            print(f"    * [{s3.get('entity_id')}] {s3.get('business_name', '')}")
            print(f"      Address: {s3.get('business_address', '')}")
    print(f"\n{sep}\n")

if __name__ == "__main__":
    results = sample_and_fetch_matches(n_samples=500, random_state=42)
    print("\n--- Displaying 5 Sample Comparisons ---")
    print_readable_samples(results, num_to_display=5)
