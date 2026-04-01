import json

# Verify easy
logs = json.load(open("data/logs/easy_log.json"))
gt   = json.load(open("data/ground_truth/easy_gt.json"))
print(f"EASY: {len(logs)} logs, GT annotations: {list(gt['annotations'].keys())}")
for lid, cat in gt["annotations"].items():
    entry = next(l for l in logs if l["id"] == lid)
    print(f"  {lid} [{cat}] sev={entry['severity']} | {entry['message'][:70]}")

# Verify medium
logs = json.load(open("data/logs/medium_log.json"))
gt   = json.load(open("data/ground_truth/medium_gt.json"))
print(f"\nMEDIUM: {len(logs)} logs, GT annotations: {list(gt['annotations'].keys())}")
for lid, cat in gt["annotations"].items():
    entry = next(l for l in logs if l["id"] == lid)
    print(f"  {lid} [{cat}] sev={entry['severity']} | {entry['message'][:70]}")
print("  CORRELATIONS:", gt["correlations"])
print("  SEVERITY:", gt["severity"])

# Verify hard
logs = json.load(open("data/logs/hard_log.json"))
gt   = json.load(open("data/ground_truth/hard_gt.json"))
print(f"\nHARD: {len(logs)} logs, GT annotations: {list(gt['annotations'].keys())}")
for lid, cat in gt["annotations"].items():
    entry = next(l for l in logs if l["id"] == lid)
    print(f"  {lid} [{cat}] sev={entry['severity']} | {entry['message'][:70]}")
print("  CORRELATIONS:", gt["correlations"])
print("  SEVERITY:", gt["severity"])

print("\n=== SIZE CHECK ===")
for f in ["data/logs/easy_log.json", "data/logs/medium_log.json", "data/logs/hard_log.json"]:
    import os
    print(f"  {f}: {os.path.getsize(f):,} bytes")

print("\nAll datasets verified OK.")
