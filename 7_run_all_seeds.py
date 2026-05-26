import subprocess
import csv
import sys
import os
from datetime import datetime

seeds = [42, 123, 999]
scripts = [
    "2_train_teacher.py",
    "3_train_student_baseline.py",
    "4_train_kd_standard.py",
    "5_train_kd_hybrid.py",
    "6_ablation.py"
]

results = []
log_file = "run_log.txt"

def run_script(seed, script):
    cmd = ["python", script, str(seed)]
    print(f"\n{'='*60}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting: {script} (seed={seed})")
    print(f"{'='*60}")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        errors='replace',
        bufsize=1
    )

    output_lines = []
    for line in proc.stdout:
        print(line, end='')
        output_lines.append(line)

    proc.wait()

    if proc.returncode != 0:
        print(f"\n❌ ERROR: {script} returned code {proc.returncode}. Aborting.")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"\n[{datetime.now()}] {script} (seed={seed}) FAILED with code {proc.returncode}\n")
            f.write("".join(output_lines[-200:]))
        sys.exit(1)

    return "".join(output_lines)

def extract_top1(output):
    for line in output.splitlines():
        if "Top-1:" in line:
            try:
                top1_str = line.split(":")[1].split(",")[0].strip().replace("%", "")
                return float(top1_str)
            except:
                continue
    return None

for seed in seeds:
    print(f"\n{'#'*60}")
    print(f"# Running SEED {seed}")
    print(f"{'#'*60}")
    for script in scripts:
        output = run_script(seed, script)
        top1 = extract_top1(output)
        if top1 is not None:
            results.append({"Seed": seed, "Script": script, "Top-1": top1})
            print(f"✅ Extracted Top-1: {top1:.2f}%")
        else:
            print(f"⚠️ Warning: Could not extract Top-1 accuracy from {script}. Proceeding anyway.")

with open("results.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["Seed", "Script", "Top-1"])
    writer.writeheader()
    writer.writerows(results)

print(f"\n🏁 All experiments finished. {len(results)} results saved to results.csv")