import csv
from collections import defaultdict
import numpy as np

data = defaultdict(list)
with open("results.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        script = row["Script"]
        acc = float(row["Top-1"])
        data[script].append(acc)

print("\n========== Final Results (Mean ± Std) ==========")
print(f"{'Method':<35} {'Top-1 (%)':<20}")
print("-" * 55)
for script, accs in data.items():
    mean = np.mean(accs)
    std = np.std(accs, ddof=1) if len(accs) > 1 else 0.0
    name = script.replace(".py", "").replace("_", " ").title()
    print(f"{name:<35} {mean:.2f} ± {std:.2f}")