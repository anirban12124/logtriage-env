import json
from collections import Counter

tasks = {
    "EASY":   ("data/logs/easy_log.json",   "data/ground_truth/easy_gt.json"),
    "MEDIUM": ("data/logs/medium_log.json", "data/ground_truth/medium_gt.json"),
    "HARD":   ("data/logs/hard_log.json",   "data/ground_truth/hard_gt.json"),
}

for task_name, (log_path, gt_path) in tasks.items():
    logs = json.load(open(log_path))
    gt   = json.load(open(gt_path))

    gt_ids = set(gt["annotations"].keys())
    total  = len(logs)

    # Severity counts
    sev_counts = Counter(l["severity"] for l in logs)
    # Service counts
    svc_counts = Counter(l["service"] for l in logs)

    # GT vs non-GT
    gt_logs   = [l for l in logs if l["id"] in gt_ids]
    noise     = [l for l in logs if l["id"] not in gt_ids]

    # GT severity breakdown
    gt_sev = Counter(l["severity"] for l in gt_logs)

    # Signal ratio
    signal_ratio = len(gt_ids) / total * 100

    print(f"\n{'='*55}")
    print(f"  {task_name}  —  {total} total logs")
    print(f"{'='*55}")

    print(f"\n  SEVERITY DISTRIBUTION:")
    for sev in ["DEBUG", "INFO", "WARN", "ERROR", "FATAL"]:
        n = sev_counts.get(sev, 0)
        bar = "#" * (n * 30 // total) if total else ""
        print(f"    {sev:<6} {n:>4}  ({n/total*100:5.1f}%)  {bar}")

    print(f"\n  SERVICE DISTRIBUTION:")
    for svc, n in svc_counts.most_common():
        print(f"    {svc:<25} {n:>4}  ({n/total*100:5.1f}%)")

    print(f"\n  GROUND TRUTH ({len(gt_ids)} signals, signal ratio={signal_ratio:.1f}%):")
    for lid, cat in gt["annotations"].items():
        entry = next(l for l in logs if l["id"] == lid)
        print(f"    {lid}  [{entry['severity']:<5}] [{cat:<25}]  {entry['message'][:55]}")

    print(f"\n  GT severity breakdown: {dict(gt_sev)}")
    print(f"  Noise logs:  {len(noise)} ({len(noise)/total*100:.1f}%)")
    print(f"  GT logs:     {len(gt_ids)} ({signal_ratio:.1f}%)")

    if gt.get("correlations"):
        print(f"\n  CORRELATIONS ({len(gt['correlations'])} pairs):")
        for pair in gt["correlations"]:
            print(f"    {pair[0]} -> {pair[1]}")

    print(f"\n  EXPECTED SEVERITY: {gt['severity']}")
    print(f"  KEY FINDINGS:")
    for kf in gt["key_findings"]:
        print(f"    - {kf}")
